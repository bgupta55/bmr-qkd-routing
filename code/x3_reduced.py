"""
x3_reduced.py - X3 (A-lite): reduced router comparison with multipath plans.
Cells: topo in {ER, BA, grid, ring} (N=30), n_audit=1e4, Delta in {1e-3, 3e-5}, cond in {clean, compromised(20 % links at S=2.418)}, t=1;
plus t=2 for ER and BA  -> 16 + 8 = 24 cells. 1000 trials per cell = 10 graphs x 100 requests (hop distance >= 2), same trial stream for all routers.
Routers (one certificate draw per trial, shared):
  B2         single path, floor off (= BMR-single, t=0)       delivered by one path; exposure: any relay exposes the key (Lemma A) if m>=2
  BMR-s(t)   single path, hop floor 2t+1 (paper's R1 router)  same exposure
  MP-SPF     Suurballe-like min-hop t+1 disjoint paths (ignores rate/budget), scored for compliance
  BMR-MP     exact plan solver (CP-SAT), per-path budget, rate >= R_MIN
Compliance (plan): every link rate >= R_MIN, every path sum(w) <= Delta, paths internally disjoint, t+1 paths (direct link alone is a plan).
rate = min link rate over the plan if compliant else 0. Collusion exposure by theory of the X1-verified protocols (P2 for single paths, P4 for plans);
X2 verified this against the GF(2) checker (12,000 plan instances, 0 disagreements).
Output: x3_out/x3_rows.csv.gz (per trial), x3_cells.csv, x3_paired.csv, x3_meta.json. Statistics: two-stage cluster bootstrap (graphs, then requests within graph), 2000 reps.
"""
import sys, os, math, csv, gzip, json, time, itertools, platform
import numpy as np, networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_COMPROMISED
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp, plan_valid, plan_rate
from q6_common import make_topology, attach_certs, R_MIN
from x2_collusion import suurballe_like

OUT = os.path.join(os.path.dirname(HERE), "x3_out"); os.makedirs(OUT, exist_ok=True)
MASTER = 20261113
GRAPHS, PER = 10, 100
N = 30


def plan_ok(G, plan, s, d, t, Delta):
    if plan is None: return False
    if len(plan) == 1 and len(plan[0]) == 2:
        a = G[plan[0][0]][plan[0][1]]; return a["r"] >= R_MIN and a["w"] <= Delta + 1e-15
    return plan_valid(G, plan, s, d, t, Delta, R_MIN)


def one_cell(spec):
    topo, Delta, cond, t, seed = spec
    rng = np.random.default_rng(seed)
    rows = []
    gl = []
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
                for i in np.random.default_rng(sd).choice(len(edges), max(1, round(0.2 * len(edges))), replace=False):
                    sim.set_true_S(edges[i], S_COMPROMISED)
            G = G0.copy(); attach_certs(G, sim, epoch=0)
            out = dict(topo=topo, Delta=Delta, cond=cond, t=t, graph=gi, req=k, s=s, d=d)
            # single-path routers
            for name, mmin in (("B2", 0), ("BMRs", 2 * t + 1)):
                r = bmr_route(G, s, d, R_MIN, Delta, 0, 10, m_min=mmin)
                if r is None:
                    out.update({f"{name}_deliv": 0, f"{name}_rate": 0.0, f"{name}_hops": 0, f"{name}_exposed": 0, f"{name}_keys": 0})
                else:
                    m = len(r[1]) - 1
                    out.update({f"{name}_deliv": 1, f"{name}_rate": r[0], f"{name}_hops": m, f"{name}_exposed": int(m >= 2), f"{name}_keys": m})
            # plan routers
            sp = suurballe_like(G, s, d, t + 1)
            ok = plan_ok(G, sp, s, d, t, Delta)
            out.update(MPSPF_deliv=int(sp is not None), MPSPF_compliant=int(ok), MPSPF_rate=(plan_rate(G, sp) if ok else 0.0),
                       MPSPF_keys=(sum(len(p) - 1 for p in sp) if sp else 0))
            res = route_bmr_mp(G, s, d, t, Delta, R_MIN, "ilp", ilp_time=10)
            pl = res["plan"]
            out.update(BMRMP_deliv=int(pl is not None), BMRMP_rate=(plan_rate(G, pl) if pl else 0.0),
                       BMRMP_keys=(sum(len(p) - 1 for p in pl) if pl else 0), BMRMP_npaths=(len(pl) if pl else 0),
                       BMRMP_status=res["status"], BMRMP_direct=int(bool(pl) and len(pl) == 1 and len(pl[0]) == 2))
            out["MP_exposed"] = 0
            rows.append(out)
    return rows


def boot(arr, B=2000, seed=7):
    """arr: dict graph -> np.array of paired differences; two-stage cluster bootstrap of the grand mean."""
    rng = np.random.default_rng(seed)
    gs = list(arr.values()); G = len(gs); means = []
    for _ in range(B):
        idx = rng.integers(0, G, G); tot = []
        for i in idx:
            x = gs[i]; tot.append(x[rng.integers(0, len(x), len(x))].mean())
        means.append(np.mean(tot))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    t0 = time.time()
    rng = np.random.default_rng(MASTER)
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
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "x3_rows.csv.gz"), index=False)
    cells, paired = [], []
    for (topo, Delta, cond, t), g in df.groupby(["topo", "Delta", "cond", "t"]):
        c = dict(topo=topo, Delta=Delta, cond=cond, t=t, trials=len(g))
        for nm in ("B2", "BMRs"):
            c[f"{nm}_deliv_pct"] = 100 * g[f"{nm}_deliv"].mean(); c[f"{nm}_rate"] = g[f"{nm}_rate"].mean()
            c[f"{nm}_exposed_pct_of_all"] = 100 * g[f"{nm}_exposed"].mean(); c[f"{nm}_keys"] = g.loc[g[f"{nm}_deliv"] == 1, f"{nm}_keys"].mean()
        c["MPSPF_compliant_pct"] = 100 * g.MPSPF_compliant.mean(); c["MPSPF_rate"] = g.MPSPF_rate.mean()
        c["MPSPF_keys"] = g.loc[g.MPSPF_deliv == 1, "MPSPF_keys"].mean()
        c["BMRMP_deliv_pct"] = 100 * g.BMRMP_deliv.mean(); c["BMRMP_rate"] = g.BMRMP_rate.mean(); c["BMRMP_exposed_pct_of_all"] = 0.0
        c["BMRMP_keys"] = g.loc[g.BMRMP_deliv == 1, "BMRMP_keys"].mean(); c["BMRMP_direct_pct_of_deliv"] = 100 * g.loc[g.BMRMP_deliv == 1, "BMRMP_direct"].mean() if g.BMRMP_deliv.sum() else float("nan")
        c["BMRMP_nonexact_status"] = int((g.BMRMP_status != "ok").sum())
        cells.append(c)
        for a, b, lab in (("BMRMP", "B2", "BMR-MP - B2"), ("BMRMP", "BMRs", "BMR-MP - BMR-single(floor 2t+1)"), ("BMRMP", "MPSPF", "BMR-MP - MP-SPF"), ("BMRs", "B2", "BMR-single - B2")):
            da = g[f"{a}_rate"] - g[f"{b}_rate"]
            arr = {gi: x.values for gi, x in da.groupby(g.graph)}
            lo, hi = boot(arr)
            paired.append(dict(topo=topo, Delta=Delta, cond=cond, t=t, contrast=lab, mean_diff=float(da.mean()), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(cells).to_csv(os.path.join(OUT, "x3_cells.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(OUT, "x3_paired.csv"), index=False)
    json.dump(dict(seed=MASTER, graphs=GRAPHS, per=PER, cells=len(specs), seconds=round(time.time() - t0, 1), python=platform.python_version(),
                   machine=platform.platform(), ortools=__import__("ortools").__version__), open(os.path.join(OUT, "x3_meta.json"), "w"), indent=1)
    print("done", len(df), "rows", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
