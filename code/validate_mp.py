"""
validate_mp.py - X4: validation of BMR-MP (t+1 internally disjoint paths, per-path budget) against independent brute force and CP-SAT.

Per t in {1,2}, N in 5..9: 2000 instances (=10,000 per t, 20,000 in total). Half certificate-derived link attributes
(LinkSim + Algorithm 1; n_audit in {500, 10^4}; 20 % links compromised at S=2.418; Delta = 0.5 / 1e-3 respectively), half uniform random
(rates from a discrete grid so that ties occur, weights ~ U(0, 0.6*Delta)). Random connected graph, p in {0.3, 0.5}. Random pair s != d.
Compared: (feasible?, optimal bottleneck rho) between brute force, exact enumeration, CP-SAT; every returned plan is re-validated by an
independent checker. Aggregate necessary filter evaluated at R_MIN and at the median rate level: false rejects must be 0; over-accepts reported.
Output: x4_out/x4_validation_rows.csv, x4_validation_summary.txt
"""
import sys, os, math, time, json, csv, platform
import numpy as np
import networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_COMPROMISED
from bmr.bmr_mp import (route_bmr_mp, brute_force_best_plan, plan_valid, plan_rate, plan_feasible_enum, plan_feasible_agg,
                        filter_rate)
from q6_common import attach_certs

OUT = os.path.join(os.path.dirname(HERE), "x4_out"); os.makedirs(OUT, exist_ok=True)
MASTER = 20261113
PER_N = int(os.environ.get("X4_PER_N", 2000))
R_MIN = 0.01


def make_instance(t, N, i):
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, 4, t, N, i]))
    p = [0.3, 0.5][int(rng.integers(2))]
    for _ in range(200):
        G = nx.gnp_random_graph(N, p, seed=int(rng.integers(1e9)))
        if nx.is_connected(G):
            break
    else:
        G = nx.path_graph(N)
    kind = "cert" if i % 2 == 0 else "uniform"
    s, d = [int(x) for x in rng.choice(N, 2, replace=False)]
    if kind == "cert":
        na = [500, 10_000][int(rng.integers(2))]
        Delta = 0.5 if na == 500 else 1e-3
        edges = list(G.edges())
        sim = LinkSim(edges, na, delta=1e-6, ell=10_000, phi=0.99, seed=int(rng.integers(1e9)))
        k = max(1, round(0.2 * len(edges)))
        for j in rng.choice(len(edges), k, replace=False):
            sim.set_true_S(edges[j], S_COMPROMISED)
        attach_certs(G, sim, epoch=0)
    else:
        Delta = 1.0
        for u, v in G.edges():
            G[u][v]["r"] = float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.3, 0.2]))
            G[u][v]["w"] = float(rng.uniform(0, 0.6 * Delta))
            G[u][v]["T"] = 0
    return G, s, d, Delta, kind


def run_one(args):
    t, N, i = args
    G, s, d, Delta, kind = make_instance(t, N, i)
    bf = brute_force_best_plan(G, s, d, t, Delta, R_MIN)
    t0 = time.perf_counter(); ex = route_bmr_mp(G, s, d, t, Delta, R_MIN, solver="enum", cap=10 ** 7); t_ex = time.perf_counter() - t0
    t0 = time.perf_counter(); il = route_bmr_mp(G, s, d, t, Delta, R_MIN, solver="ilp", ilp_time=30); t_il = time.perf_counter() - t0
    rb = None if bf is None else float(bf)
    re_ = ex["rho"]; ri = il["rho"]
    ok_ex = (rb is None and re_ is None) or (rb is not None and re_ is not None and abs(rb - re_) < 1e-12)
    ok_il = (rb is None and ri is None) or (rb is not None and ri is not None and abs(rb - ri) < 1e-12)
    valid_ex = (ex["plan"] is None) or (plan_valid(G, ex["plan"], s, d, t, Delta, R_MIN) and abs(plan_rate(G, ex["plan"]) - re_) < 1e-12)
    valid_il = (il["plan"] is None) or (plan_valid(G, il["plan"], s, d, t, Delta, R_MIN) and abs(plan_rate(G, il["plan"]) - ri) < 1e-12)
    H = filter_rate(G, R_MIN)
    levels = [R_MIN]
    rates = sorted({a["r"] for _, _, a in H.edges(data=True)})
    if rates:
        levels.append(rates[len(rates) // 2])
    agg = []
    for lv in levels:
        Hl = filter_rate(G, lv)
        pl, st = plan_feasible_enum(Hl, s, d, t, Delta, cap=10 ** 7)
        agg.append((int(pl is not None), int(plan_feasible_agg(Hl, s, d, t, Delta))))
    return dict(t=t, N=N, i=i, kind=kind, edges=G.number_of_edges(), Delta=Delta, bf_rho=rb, enum_rho=re_, ilp_rho=ri,
                feasible=int(rb is not None), match_enum=int(ok_ex), match_ilp=int(ok_il), plan_valid_enum=int(valid_ex),
                plan_valid_ilp=int(valid_il), enum_status=ex["status"], ilp_status=il["status"], t_enum=t_ex, t_ilp=t_il,
                agg_exact1=agg[0][0], agg_filt1=agg[0][1], agg_exact2=agg[-1][0], agg_filt2=agg[-1][1])


def main():
    t0 = time.time()
    jobs = [(t, N, i) for t in (1, 2) for N in range(5, 10) for i in range(PER_N)]
    rows = []
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for k, r in enumerate(pool.imap(run_one, jobs, chunksize=20)):
            rows.append(r)
            if k % 2000 == 0:
                print(f"{k}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "x4_validation_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    lines = []

    def P(s):
        print(s); lines.append(s)
    P(f"X4 validation: {len(df)} instances, {time.time()-t0:.0f}s, master seed {MASTER}, python {platform.python_version()}, {platform.platform()}")
    for t in (1, 2):
        x = df[df.t == t]
        P(f"\n t={t}: instances {len(x)}  feasible {int(x.feasible.sum())} ({100*x.feasible.mean():.1f}%)  "
          f"mismatch(enum vs brute) {int((1-x.match_enum).sum())}  mismatch(ILP vs brute) {int((1-x.match_ilp).sum())}  "
          f"invalid plans enum {int((1-x.plan_valid_enum).sum())} ilp {int((1-x.plan_valid_ilp).sum())}  "
          f"enum status!=ok {int((x.enum_status!='ok').sum())}  ilp status!=ok {int((x.ilp_status!='ok').sum())}")
        for kind in ("cert", "uniform"):
            y = x[x.kind == kind]
            P(f"    {kind:8s}: n={len(y)} feasible={int(y.feasible.sum())} mismatches enum/ilp = {int((1-y.match_enum).sum())}/{int((1-y.match_ilp).sum())}")
        for lv, (e, fcol) in {"R_MIN": ("agg_exact1", "agg_filt1"), "median-rate": ("agg_exact2", "agg_filt2")}.items():
            ex_inf = x[x[e] == 0]; ex_f = x[x[e] == 1]
            P(f"    aggregate filter @ {lv}: exact-feasible {len(ex_f)} (filter false-rejects {int((ex_f[fcol]==0).sum())}); "
              f"exact-infeasible {len(ex_inf)} (filter over-accepts {int((ex_inf[fcol]==1).sum())} = {100*(ex_inf[fcol]==1).mean():.2f}%)")
    for N in range(5, 10):
        x = df[df.N == N]
        P(f" N={N}: n={len(x)} feasible={int(x.feasible.sum())} mismatches {int((1-x.match_enum).sum())}/{int((1-x.match_ilp).sum())}")
    open(os.path.join(OUT, "x4_validation_summary.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
