"""
x3b_extended.py - RF5/RF6/RF7: X3 extended.
 * 50 graphs x 100 requests per cell (was 10 x 100) -> 5,000 trials/cell, same 24 cells, same seed derivation family (master 20261113, stream 'x3b').
 * budget-aware multipath baselines (not just hop-minimal MP-SPF):
     MP-MinW   : min-cost-flow t+1 disjoint paths on links with r >= R_MIN, edge cost = w (minimise total defect)
     MP-MaxR   : same filter, edge cost = 1/r (rate-greedy), then checked
     MP-SPFf   : min-hop t+1 disjoint paths on the filtered graph (r >= R_MIN and w <= Delta), then checked
   all are scored with the same compliance rule as BMR-MP; BMR-MP maximises rate over compliant plans, so each baseline's rate <= BMR-MP by construction
   (the interesting numbers are HOW MUCH the exact solver adds, and how often each baseline is non-compliant).
 * refusal attribution for BMR-MP: topology (no t+1 disjoint paths in G), rate filter (none in G restricted to r>=R_MIN), budget (none within per-path Delta).
 * rate decomposition of  mean(B2) - mean(BMR-MP):  lost deliveries vs lower rate when both deliver vs gains where only MP delivers.
 * reconciliation with X2: MP-SPF delivered vs compliant share.
Output: ../x3b_out/x3b_rows.csv.gz, x3b_cells.csv, x3b_paired.csv, x3b_decomp.csv, x3b_meta.json
"""
import sys, os, json, time, itertools, platform
import numpy as np, networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_COMPROMISED
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp, plan_rate
from q6_common import make_topology, attach_certs, R_MIN
from x2_collusion import suurballe_like
from x3_reduced import plan_ok, boot

OUT = os.path.join(os.path.dirname(HERE), "x3b_out"); os.makedirs(OUT, exist_ok=True)
MASTER = 20261113
GRAPHS, PER, N = int(os.environ.get("X3B_GRAPHS", 50)), 100, 30


def mc_plan(G, s, d, K, cost):
    """K internally vertex-disjoint paths minimising total edge cost (min-cost flow, unit node capacities); direct link = plan on its own."""
    if G.has_edge(s, d): return [[s, d]]
    F = nx.DiGraph()
    for v in G.nodes(): F.add_edge(("i", v), ("o", v), capacity=(K if v in (s, d) else 1), weight=0)
    SC = 10 ** 9
    for u, v, a in G.edges(data=True):
        c = int(round(cost(a) * SC))
        for x, y in ((u, v), (v, u)):
            if y == s or x == d: continue
            F.add_edge(("o", x), ("i", y), capacity=1, weight=c)
    if s not in G or d not in G: return None
    F.nodes[("o", s)]["demand"] = -K; F.nodes[("i", d)]["demand"] = K
    try: flow = nx.min_cost_flow(F)
    except (nx.NetworkXUnfeasible, nx.NodeNotFound): return None
    nxt = {}
    for (kind, x), dd in flow.items():
        if kind != "o": continue
        for (k2, y), f in dd.items():
            if f > 0 and k2 == "i" and y != x: nxt.setdefault(x, []).append(y)
    paths = []
    for _ in range(K):
        p = [s]
        while p[-1] != d: p.append(nxt[p[-1]].pop())
        paths.append(p)
    return paths


def filt(G, pred):
    H = nx.Graph(); H.add_nodes_from(G.nodes())
    for u, v, a in G.edges(data=True):
        if pred(a): H.add_edge(u, v, **a)
    return H


def one_cell(spec):
    topo, Delta, cond, t, seed = spec
    rng = np.random.default_rng(seed); rows = []; gl = []
    while len(gl) < GRAPHS:
        G = make_topology(topo, N, int(rng.integers(1e9)))
        if nx.is_connected(G): gl.append(G)
    for gi, G0 in enumerate(gl):
        edges = list(G0.edges()); dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a][b] >= 2]
        for k in range(PER):
            s, d = pairs[int(rng.integers(len(pairs)))]; sd = int(rng.integers(1e9))
            sim = LinkSim(edges, 10_000, delta=1e-6, ell=10_000, phi=0.99, seed=sd)
            if cond == "compromised":
                for i in np.random.default_rng(sd).choice(len(edges), max(1, round(0.2 * len(edges))), replace=False): sim.set_true_S(edges[i], S_COMPROMISED)
            G = G0.copy(); attach_certs(G, sim, epoch=0)
            out = dict(topo=topo, Delta=Delta, cond=cond, t=t, graph=gi, req=k, s=s, d=d)
            r = bmr_route(G, s, d, R_MIN, Delta, 0, 10, m_min=0)
            out.update(B2_deliv=int(r is not None), B2_rate=(r[0] if r else 0.0))
            # plan baselines
            Gr = filt(G, lambda a: a["r"] >= R_MIN)
            Gb = filt(G, lambda a: a["r"] >= R_MIN and a["w"] <= Delta + 1e-15)
            cand = {"MPSPF": suurballe_like(G, s, d, t + 1),
                    "MPSPFf": suurballe_like(Gb, s, d, t + 1) if (s in Gb and d in Gb) else None,
                    "MPMinW": mc_plan(Gr, s, d, t + 1, lambda a: a["w"]),
                    "MPMaxR": mc_plan(Gr, s, d, t + 1, lambda a: 1.0 / (a["r"] + 1e-9))}
            for nm, pl in cand.items():
                ok = plan_ok(G, pl, s, d, t, Delta)
                out.update({f"{nm}_deliv": int(pl is not None), f"{nm}_compliant": int(ok), f"{nm}_rate": (plan_rate(G, pl) if ok else 0.0)})
            res = route_bmr_mp(G, s, d, t, Delta, R_MIN, "ilp", ilp_time=10); pl = res["plan"]
            out.update(BMRMP_deliv=int(pl is not None), BMRMP_rate=(plan_rate(G, pl) if pl else 0.0), BMRMP_status=res["status"],
                       BMRMP_direct=int(bool(pl) and len(pl) == 1 and len(pl[0]) == 2))
            # refusal attribution (ground truth at the same certificates)
            topo_ok = suurballe_like(G, s, d, t + 1) is not None
            rate_ok = (suurballe_like(Gr, s, d, t + 1) is not None) if (s in Gr and d in Gr) else False
            out.update(stage_topology=int(topo_ok), stage_rate=int(rate_ok), stage_budget=int(pl is not None))
            rows.append(out)
    return rows


def main():
    t0 = time.time(); rng = np.random.default_rng(np.random.SeedSequence([MASTER, 33]))
    specs = []
    for topo, Delta, cond in itertools.product(["ER", "BA", "grid", "ring"], [1e-3, 3e-5], ["clean", "compromised"]):
        specs.append((topo, Delta, cond, 1, int(rng.integers(1e9))))
    for topo, Delta, cond in itertools.product(["ER", "BA"], [1e-3, 3e-5], ["clean", "compromised"]):
        specs.append((topo, Delta, cond, 2, int(rng.integers(1e9))))
    rows = []
    with Pool(max(1, os.cpu_count() - 1)) as p:
        for i, r in enumerate(p.imap(one_cell, specs)):
            rows += r; print(f"cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "x3b_rows.csv.gz"), index=False)
    cells, paired, decomp = [], [], []
    for (topo, Delta, cond, t), g in df.groupby(["topo", "Delta", "cond", "t"]):
        c = dict(topo=topo, Delta=Delta, cond=cond, t=t, trials=len(g), graphs=g.graph.nunique(), B2_deliv_pct=100 * g.B2_deliv.mean(), B2_rate=g.B2_rate.mean(),
                 BMRMP_deliv_pct=100 * g.BMRMP_deliv.mean(), BMRMP_rate=g.BMRMP_rate.mean(), BMRMP_nonexact=int((g.BMRMP_status != "ok").sum()))
        for nm in ("MPSPF", "MPSPFf", "MPMinW", "MPMaxR"):
            c[f"{nm}_deliv_pct"] = 100 * g[f"{nm}_deliv"].mean(); c[f"{nm}_compliant_pct"] = 100 * g[f"{nm}_compliant"].mean(); c[f"{nm}_rate"] = g[f"{nm}_rate"].mean()
        cells.append(c)
        for a, b, lab in (("BMRMP", "B2", "BMR-MP - B2"), ("BMRMP", "MPSPF", "BMR-MP - MP-SPF"), ("BMRMP", "MPSPFf", "BMR-MP - MP-SPF(filtered)"),
                          ("BMRMP", "MPMinW", "BMR-MP - MP-MinW"), ("BMRMP", "MPMaxR", "BMR-MP - MP-MaxR")):
            da = g[f"{a}_rate"] - g[f"{b}_rate"]; arr = {gi: x.values for gi, x in da.groupby(g.graph)}; lo, hi = boot(arr)
            paired.append(dict(topo=topo, Delta=Delta, cond=cond, t=t, contrast=lab, mean_diff=float(da.mean()), ci_lo=lo, ci_hi=hi))
        n = len(g)
        both = (g.B2_deliv == 1) & (g.BMRMP_deliv == 1); onlyb2 = (g.B2_deliv == 1) & (g.BMRMP_deliv == 0); onlymp = (g.B2_deliv == 0) & (g.BMRMP_deliv == 1)
        decomp.append(dict(topo=topo, Delta=Delta, cond=cond, t=t, trials=n,
                           refuse_B2_pct=100 * (g.B2_deliv == 0).mean(), refuse_MP_pct=100 * (g.BMRMP_deliv == 0).mean(),
                           refuse_MP_topology_pct=100 * (g.stage_topology == 0).mean(),
                           refuse_MP_rate_filter_pct=100 * ((g.stage_topology == 1) & (g.stage_rate == 0)).mean(),
                           refuse_MP_budget_pct=100 * ((g.stage_rate == 1) & (g.stage_budget == 0)).mean(),
                           rate_gap=g.B2_rate.mean() - g.BMRMP_rate.mean(),
                           gap_from_lost_deliveries=g.loc[onlyb2, "B2_rate"].sum() / n,
                           gap_from_lower_rate_when_both=(g.loc[both, "B2_rate"] - g.loc[both, "BMRMP_rate"]).sum() / n,
                           gap_from_mp_only=-g.loc[onlymp, "BMRMP_rate"].sum() / n,
                           n_both=int(both.sum()), n_only_b2=int(onlyb2.sum()), n_only_mp=int(onlymp.sum())))
    pd.DataFrame(cells).to_csv(os.path.join(OUT, "x3b_cells.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(OUT, "x3b_paired.csv"), index=False)
    pd.DataFrame(decomp).to_csv(os.path.join(OUT, "x3b_decomp.csv"), index=False)
    json.dump(dict(seed=MASTER, graphs=GRAPHS, per=PER, cells=len(specs), seconds=round(time.time() - t0, 1), python=platform.python_version(),
                   machine=platform.platform(), ortools=__import__("ortools").__version__), open(os.path.join(OUT, "x3b_meta.json"), "w"), indent=1)
    print("done", len(df), "rows", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
