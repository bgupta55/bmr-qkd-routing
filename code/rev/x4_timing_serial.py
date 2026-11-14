"""
x4_timing_serial.py - RF9: X4 timing WITHOUT contention. One process, nothing else running (the caller must make sure of that).
Same instances as rev/x4_timing.py (same SeedSequence([MASTER, 41, N, t, i])), first 100 requests per cell, Yen single-path and CP-SAT only
(exact enumeration already shown to be unusable at N>=30). CP-SAT: 1 worker, 10 s limit. Per-request wall time of the whole binary-search call.
Output: ../x4_out/x4_timing_serial_{rows,summary}.csv
"""
import sys, os, time, platform
import numpy as np, networkx as nx
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_COMPROMISED
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp
from q6_common import attach_certs
OUT = os.path.join(os.path.dirname(HERE), "x4_out"); MASTER = 20261113
REQ = int(os.environ.get("X4S_REQ", 100))


def one(N, t, i):
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, 41, N, t, i]))
    for _ in range(100):
        G = nx.gnp_random_graph(N, 5.0 / (N - 1), seed=int(rng.integers(1e9)))
        if nx.is_connected(G): break
    s, d = [int(x) for x in rng.choice(N, 2, replace=False)]
    edges = list(G.edges()); sim = LinkSim(edges, 10_000, seed=int(rng.integers(1e9)))
    for j in rng.choice(len(edges), max(1, round(.2 * len(edges))), replace=False): sim.set_true_S(edges[j], S_COMPROMISED)
    attach_certs(G, sim, 0); D = 1e-3
    t0 = time.perf_counter(); bmr_route(G, s, d, 0.01, D, 0, 10, m_min=0); ty = time.perf_counter() - t0
    t0 = time.perf_counter(); il = route_bmr_mp(G, s, d, t, D, 0.01, "ilp", ilp_time=10); ti = time.perf_counter() - t0
    return dict(N=N, t=t, i=i, t_yen=ty, t_ilp=ti, status=il["status"], found=int(il["rho"] is not None))


if __name__ == "__main__":
    t0 = time.time(); rows = []
    for N in (30, 50, 100, 150):
        for t in (1, 2):
            for i in range(REQ): rows.append(one(N, t, i))
            print(N, t, f"{time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "x4_timing_serial_rows.csv"), index=False)
    o = df.groupby(["N", "t"]).agg(n=("i", "size"), feasible_pct=("found", lambda x: round(100 * x.mean(), 1)),
                                   yen_med_ms=("t_yen", lambda x: round(1e3 * np.median(x), 2)), yen_p95_ms=("t_yen", lambda x: round(1e3 * np.percentile(x, 95), 2)),
                                   ilp_med_ms=("t_ilp", lambda x: round(1e3 * np.median(x), 1)), ilp_p95_ms=("t_ilp", lambda x: round(1e3 * np.percentile(x, 95), 1)),
                                   ilp_max_ms=("t_ilp", lambda x: round(1e3 * x.max(), 1)), nonok=("status", lambda x: int((x != "ok").sum()))).reset_index()
    o.to_csv(os.path.join(OUT, "x4_timing_serial_summary.csv"), index=False)
    print(o.to_string()); print("total", round(time.time() - t0), "s", platform.platform(), "python", platform.python_version())
