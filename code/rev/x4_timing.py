"""X4 timing: BMR-MP on sparse random graphs (ER with mean degree 5, N in {30,50,100,150}), t in {1,2}, 500 requests per cell.
Certificate-derived attributes (n_audit=1e4, Delta=1e-3, 20 % links compromised at S=2.418). Solvers: exact enumeration (cap 20,000 candidate
paths, 0.5 s deadline per feasibility call (binary search makes up to ~log2(#rates) calls)) and CP-SAT (1 worker, 10 s limit per call). Also single-path BMR (Yen, t=0, floor off) for reference.
Run on 6 worker processes (so wall times include some contention); per-request wall time of the full binary-search routing call."""
import sys, os, time, csv, platform, json
import numpy as np, networkx as nx
from multiprocessing import Pool
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_COMPROMISED
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp, plan_valid
from q6_common import attach_certs
OUT = os.path.join(os.path.dirname(HERE), "x4_out")
REQ = int(os.environ.get("X4T_REQ", 500)); MASTER = 20261113; ENUM_N = 40

def one(args):
    N, t, i = args
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, 41, N, t, i]))
    for _ in range(100):
        G = nx.gnp_random_graph(N, 5.0 / (N - 1), seed=int(rng.integers(1e9)))
        if nx.is_connected(G): break
    s, d = [int(x) for x in rng.choice(N, 2, replace=False)]
    edges = list(G.edges())
    sim = LinkSim(edges, 10_000, seed=int(rng.integers(1e9)))
    for j in rng.choice(len(edges), max(1, round(.2 * len(edges))), replace=False): sim.set_true_S(edges[j], S_COMPROMISED)
    attach_certs(G, sim, 0)
    D = 1e-3
    t0 = time.perf_counter(); r0 = bmr_route(G, s, d, 0.01, D, 0, 10, m_min=0); t_yen = time.perf_counter() - t0
    if i < ENUM_N:
        t0 = time.perf_counter(); ex = route_bmr_mp(G, s, d, t, D, 0.01, "enum", cap=20_000, deadline_s=0.5); t_en = time.perf_counter() - t0
    else:
        ex = dict(rho=None, status="skipped", plan=None); t_en = float("nan")
    t0 = time.perf_counter(); il = route_bmr_mp(G, s, d, t, D, 0.01, "ilp", ilp_time=10); t_il = time.perf_counter() - t0
    agree = (ex["status"] == "skipped") or (ex["rho"] is None and il["rho"] is None) or (ex["rho"] is not None and il["rho"] is not None and abs(ex["rho"] - il["rho"]) < 1e-12)
    return dict(N=N, t=t, i=i, edges=len(edges), single_found=int(r0 is not None), t_single=t_yen, t_enum=t_en, t_ilp=t_il,
                enum_status=ex["status"], ilp_status=il["status"], enum_rho=ex["rho"], ilp_rho=il["rho"], agree=int(agree),
                mp_found=int(il["rho"] is not None))

if __name__ == "__main__":
    t0 = time.time()
    jobs = [(N, t, i) for N in (30, 50, 100, 150) for t in (1, 2) for i in range(REQ)]
    with Pool(6) as p: rows = p.map(one, jobs, chunksize=10)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "x4_timing_rows.csv"), index=False)
    out = []
    for (N, t), g in df.groupby(["N", "t"]):
        q = lambda c, p: float(np.percentile(g[c], p)) * 1e3
        both = g[(g.enum_status == "ok") & (g.ilp_status == "ok")]
        g_en = g[g.enum_status != "skipped"]
        out.append(dict(N=N, t=t, n=len(g), mean_edges=round(g.edges.mean(), 1), feasible_pct=round(100 * g.mp_found.mean(), 1),
                        single_found_pct=round(100 * g.single_found.mean(), 1),
                        yen_med_ms=round(q("t_single", 50), 2), yen_p95_ms=round(q("t_single", 95), 2),
                        enum_n=len(g_en), enum_med_ms=round(float(np.percentile(g_en.t_enum, 50)) * 1e3, 2), enum_p95_ms=round(float(np.percentile(g_en.t_enum, 95)) * 1e3, 2),
                        enum_nonok=int((g_en.enum_status != "ok").sum()),
                        ilp_med_ms=round(q("t_ilp", 50), 2), ilp_p95_ms=round(q("t_ilp", 95), 2), ilp_max_ms=round(g.t_ilp.max() * 1e3, 1),
                        ilp_nonok=int((g.ilp_status != "ok").sum()),
                        enum_vs_ilp_disagree_where_both_ok=int((1 - both.agree).sum()), n_both_ok=len(both)))
    o = pd.DataFrame(out); o.to_csv(os.path.join(OUT, "x4_timing_summary.csv"), index=False)
    print(o.to_string()); print("total", round(time.time() - t0), "s", platform.platform())
