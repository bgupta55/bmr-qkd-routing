"""
tier3_sweep.py - Tier 3 (X3 full): the 200-cell routing sweep rerun with multipath plans (revisionplan_v1 X3).
Design (revisionplan_v1 sec. 2.5 X3): the q6 design with the relay-mode axis REPLACED by the collusion bound t.
  base cells (100): synthetic {ER, BA, WS, grid, ring} x N {10,20,30,50} x n_audit {500, 1e4} x {clean, compromised(20 % links, S=2.418)} = 80
                    + SNDlib {abilene, polska, nobel-germany, geant, germany50} x n_audit x cond = 20
  x t in {1, 2} -> 200 cells.  1000 trials per cell = 10 graphs x 100 requests (hop distance >= 2); same certificate draw and same trial for every router.
  Delta = 0.5 (n_audit=500) and 1e-3 (n_audit=1e4), delta_cert = 1e-6, ell = 1e4, phi = 0.99, R_MIN = 0.01 (as in experiments/q6_sweep.py).
Routers (t = 0 single path routers deliver by one path; none of them resists a colluding relay if m >= 2, Lemma A):
  B0-SPF, B1a, B1b, B2 (= BMR-single, floor off), B3, B5, B6, B7, B8     all scored with the budget-only compliance rule of q6 (no hop floor)
  BMRs(2t+1)   the paper's R1 router with hop floor 2t+1
  MP-SPF       min-hop t+1 disjoint paths, budget-blind; scored for compliance
  MP-SPFf      min-hop t+1 disjoint paths on the links with r >= R_MIN and w <= Delta (budget-aware heuristic)
  BMR-MP       exact plan solver (CP-SAT), per-path budget, rate >= R_MIN
Metrics per trial: delivered, compliant, rate (= min link rate over the plan if compliant, else 0), hop-keys consumed,
  exposure to a WORST-CASE coalition of t relays (single path: 1 if delivered and m >= 2; plans: 0 by Theorem B / X1 / X2), refusal attribution for BMR-MP.
Output: ../tier3_out/t3_rows.csv.gz (per trial), t3_cells.csv, t3_paired.csv, t3_meta.json
Env: T3_GRAPHS (10), T3_PER (100), T3_LIMIT (all) for smoke tests.
"""
import sys, os, json, time, itertools, platform
import numpy as np, networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_COMPROMISED
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp, plan_rate
from q6_common import make_topology, load_sndlib, attach_certs, evaluate, ROUTERS, R_MIN
from x2_collusion import suurballe_like
from x3_reduced import plan_ok, boot

OUT = os.path.join(os.path.dirname(HERE), "tier3_out"); os.makedirs(OUT, exist_ok=True)
MASTER = 20261113
GRAPHS, PER = int(os.environ.get("T3_GRAPHS", 10)), int(os.environ.get("T3_PER", 100))
DELTA_BY_N = {500: 0.5, 10_000: 1e-3}
SYN = ["ER", "BA", "WS", "grid", "ring"]; SIZES = [10, 20, 30, 50]
REAL = ["abilene", "polska", "nobel-germany", "geant", "germany50"]
SINGLE = [(n, f) for n, f in ROUTERS if n not in ("BMR",)]          # drop the fixed-floor BMR; BMRs(2t+1) is run separately
TAG = {"B0-SPF": "B0", "B1a-Plambda(0.01)": "B1a", "B1b-Plambda(1)": "B1b", "B2-BMR-NoFloor": "B2", "B3-Zhang-style": "B3",
       "B5-Rate-greedy": "B5", "B6-Qpath-style": "B6", "B7-Qcast-style": "B7", "B8-QBER-monitor": "B8"}


def filt(G, pred):
    H = nx.Graph(); H.add_nodes_from(G.nodes())
    for u, v, a in G.edges(data=True):
        if pred(a): H.add_edge(u, v, **a)
    return H


def one_cell(spec):
    topo, n, na, cond, t, seed = spec
    Delta = DELTA_BY_N[na]; fcomp = 0.2 if cond == "compromised" else 0.0
    rng = np.random.default_rng(seed); rows = []
    if topo in REAL:
        gl = [load_sndlib(os.path.join(HERE, "data", "topologies", topo + ".txt"))] * GRAPHS
    else:
        gl = []
        while len(gl) < GRAPHS:
            G = make_topology(topo, n, int(rng.integers(1e9)))
            if nx.is_connected(G): gl.append(G)
    for gi, G0 in enumerate(gl):
        edges = list(G0.edges()); dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a][b] >= 2]
        for k in range(PER):
            s, d = pairs[int(rng.integers(len(pairs)))]; sd = int(rng.integers(1e9))
            sim = LinkSim(edges, na, delta=1e-6, ell=10_000, phi=0.99, seed=sd)
            if fcomp > 0:
                kk = max(1, int(round(fcomp * len(edges))))
                for i in np.random.default_rng(sd).choice(len(edges), kk, replace=False): sim.set_true_S(edges[i], S_COMPROMISED)
            G = G0.copy(); attach_certs(G, sim, epoch=0)
            out = dict(topo=topo, n=n, n_audit=na, cond=cond, t=t, Delta=Delta, graph=gi, req=k, s=s, d=d)
            # single-path routers (budget-only compliance, as in q6)
            for rn, fn in SINGLE:
                p = fn(G, s, d, Delta); f, _, _, h, _, cb, rb = evaluate(G, p, Delta); tg = TAG[rn]
                out.update({f"{tg}_deliv": f, f"{tg}_compl": cb, f"{tg}_rate": rb, f"{tg}_keys": (h if f else 0), f"{tg}_exp": int(bool(f) and h >= 2)})
            r = bmr_route(G, s, d, R_MIN, Delta, 0, 10, m_min=2 * t + 1)
            if r is None: out.update(BMRs_deliv=0, BMRs_compl=0, BMRs_rate=0.0, BMRs_keys=0, BMRs_exp=0)
            else:
                m = len(r[1]) - 1; out.update(BMRs_deliv=1, BMRs_compl=1, BMRs_rate=r[0], BMRs_keys=m, BMRs_exp=int(m >= 2))
            # plan baselines
            Gb = filt(G, lambda a: a["r"] >= R_MIN and a["w"] <= Delta + 1e-15)
            sp = suurballe_like(G, s, d, t + 1)
            spf = suurballe_like(Gb, s, d, t + 1) if (s in Gb and d in Gb) else None
            for nm, pl in (("MPSPF", sp), ("MPSPFf", spf)):
                ok = plan_ok(G, pl, s, d, t, Delta)
                out.update({f"{nm}_deliv": int(pl is not None), f"{nm}_compl": int(ok), f"{nm}_rate": (plan_rate(G, pl) if ok else 0.0),
                            f"{nm}_keys": (sum(len(p) - 1 for p in pl) if pl else 0), f"{nm}_exp": 0})
            res = route_bmr_mp(G, s, d, t, Delta, R_MIN, "ilp", ilp_time=10); pl = res["plan"]
            out.update(MP_deliv=int(pl is not None), MP_compl=int(pl is not None), MP_rate=(plan_rate(G, pl) if pl else 0.0),
                       MP_keys=(sum(len(p) - 1 for p in pl) if pl else 0), MP_exp=0, MP_status=res["status"])
            Gr = filt(G, lambda a: a["r"] >= R_MIN)
            out.update(stage_topology=int(sp is not None),
                       stage_rate=int((suurballe_like(Gr, s, d, t + 1) is not None) if (s in Gr and d in Gr) else 0),
                       stage_budget=int(pl is not None))
            rows.append(out)
    return rows


def main():
    t0 = time.time(); rng = np.random.default_rng(np.random.SeedSequence([MASTER, 3]))
    specs = []
    for t in (1, 2):
        for topo in SYN:
            for n in SIZES:
                for na in (500, 10_000):
                    for cond in ("clean", "compromised"):
                        specs.append((topo, n, na, cond, t, int(rng.integers(1e9))))
        for topo in REAL:
            for na in (500, 10_000):
                for cond in ("clean", "compromised"):
                    specs.append((topo, 0, na, cond, t, int(rng.integers(1e9))))
    assert len(specs) == 200 or os.environ.get("T3_LIMIT")
    lim = int(os.environ.get("T3_LIMIT", len(specs)))
    if lim < len(specs):   # smoke subset: spread over the list
        specs = specs[::max(1, len(specs) // lim)][:lim]
    print(f"cells {len(specs)} graphs {GRAPHS} per {PER}", flush=True)
    rows = []
    with Pool(max(1, os.cpu_count() - 1)) as p:
        for i, r in enumerate(p.imap_unordered(one_cell, specs, chunksize=1)):
            rows += r
            if (i + 1) % 10 == 0: print(f"cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "t3_rows.csv.gz"), index=False)
    print("rows written", len(df), f"{time.time()-t0:.0f}s", flush=True)
    tags = list(TAG.values()) + ["BMRs", "MPSPF", "MPSPFf", "MP"]
    cells, paired = [], []
    for (topo, n, na, cond, t), g in df.groupby(["topo", "n", "n_audit", "cond", "t"]):
        c = dict(topo=topo, n=n, n_audit=na, cond=cond, t=t, Delta=DELTA_BY_N[na], trials=len(g))
        for tg in tags:
            c[f"{tg}_deliv_pct"] = 100 * g[f"{tg}_deliv"].mean(); c[f"{tg}_compl_pct"] = 100 * g[f"{tg}_compl"].mean()
            c[f"{tg}_rate"] = g[f"{tg}_rate"].mean(); c[f"{tg}_exp_pct_of_all"] = 100 * g[f"{tg}_exp"].mean()
            dl = g[g[f"{tg}_deliv"] == 1]; c[f"{tg}_keys_per_deliv"] = dl[f"{tg}_keys"].mean() if len(dl) else float("nan")
        c["MP_nonexact"] = int((g.MP_status != "ok").sum())
        c["ref_MP_topology_pct"] = 100 * (g.stage_topology == 0).mean()
        c["ref_MP_ratefilter_pct"] = 100 * ((g.stage_topology == 1) & (g.stage_rate == 0)).mean()
        c["ref_MP_budget_pct"] = 100 * ((g.stage_rate == 1) & (g.stage_budget == 0)).mean()
        cells.append(c)
        for b in ("B2", "B3", "B5", "B8", "BMRs", "MPSPF", "MPSPFf"):
            da = g["MP_rate"] - g[f"{b}_rate"]; lo, hi = boot({gi: x.values for gi, x in da.groupby(g.graph)}, B=1000)
            paired.append(dict(topo=topo, n=n, n_audit=na, cond=cond, t=t, contrast=f"BMR-MP - {b}", mean_diff=float(da.mean()), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(cells).to_csv(os.path.join(OUT, "t3_cells.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(OUT, "t3_paired.csv"), index=False)
    json.dump(dict(seed=MASTER, graphs=GRAPHS, per=PER, cells=len(specs), rows=len(df), seconds=round(time.time() - t0, 1), python=platform.python_version(),
                   machine=platform.platform(), ortools=__import__("ortools").__version__), open(os.path.join(OUT, "t3_meta.json"), "w"), indent=1)
    print("done", len(df), "rows", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
