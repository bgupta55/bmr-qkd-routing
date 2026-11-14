"""
x2b_random_collusion.py - RF2: X2 re-measured with metrics that are NOT tautological.

X2 reports "share of delivered keys that SOME worst-case coalition of <= t relays can expose"; for single paths this is 1 for every
route with m>=2, so the percentage is just the share of non-adjacent pairs. Here, on the SAME trials as X2 (same seed, same RNG flow,
same graphs, certificates and (s,d) pairs) we report, for each router, the exposure under two probabilistic adversaries and on the
non-adjacent subset only:
  (a) random coalition: t distinct nodes drawn uniformly from V \ {s,d}; P(expose) is computed EXACTLY (hypergeometric for a single
      path; exact enumeration of all C(N-2,t) coalitions with the GF(2)-verified rule 'plan exposed iff coalition hits every path' for plans);
  (b) independent compromise: each node in V \ {s,d} is compromised independently with probability p in {0.02, 0.05, 0.10}; P(expose) exact
      (single path: 1-(1-p)^(m-1); plan: prod_i (1-(1-p)^(m_i-1)); no bound on the number of compromised nodes).
Single-path routers use delivery P2 (any compromised relay exposes: X1-verified). Plans use P4 (X1-verified: exposed iff every path is hit).
For cross-checking, 300 random plan instances per topology are re-checked with the GF(2) checker for the t-coalition rule.
Output: ../x2b_out/x2b_rows.csv, x2b_summary.csv
"""
import sys, os, itertools, time, csv, math
import numpy as np, networkx as nx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import x2_collusion as X2
import symbolic_exposure as X
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp

OUT = os.path.join(os.path.dirname(HERE), "x2b_out"); os.makedirs(OUT, exist_ok=True)
P_LIST = (0.02, 0.05, 0.10)
CHECK_PER_TOPO = 300


def relays_of(path): return set(path[1:-1])


def p_random_t_single(N, m, t):
    k = m - 1                                    # relays on the path
    pool = N - 2
    return 0.0 if k == 0 else 1.0 - math.comb(pool - k, t) / math.comb(pool, t)


def p_random_t_plan(nodes, s, d, plan, t):
    if len(plan) == 1 and len(plan[0]) == 2: return 0.0
    cand = [v for v in nodes if v not in (s, d)]; rel = [relays_of(p) for p in plan]
    hit = tot = 0
    for C in itertools.combinations(cand, t):
        tot += 1; C = set(C)
        hit += all(r & C for r in rel)
    return hit / tot


def p_indep(path_or_plan, p, is_plan):
    if not is_plan:
        return 1 - (1 - p) ** (len(path_or_plan) - 2)
    if len(path_or_plan) == 1 and len(path_or_plan[0]) == 2: return 0.0
    out = 1.0
    for q in path_or_plan: out *= 1 - (1 - p) ** (len(q) - 2)
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(X2.SEED)
    rows = []; checks = {"n": 0, "mismatch": 0}
    for topo in ["ER", "BA", "ring", "hub"]:
        nchk = 0
        for tr in range(X2.TRIALS):
            G = X2.certify(X2.make_graph(topo, rng), rng)
            nodes = list(G.nodes()); N = len(nodes)
            i, j = [int(x) for x in rng.choice(len(nodes), 2, replace=False)]
            s, d = nodes[i], nodes[j]
            adj = int(G.has_edge(s, d))
            base = X2.routers_single(G, s, d)
            for t in (1, 2):
                single = dict(base)
                r = bmr_route(G, s, d, X2.R_MIN, X2.DELTA, 0, 100, m_min=2 * t + 1); single[f"BMR-single(t={t})"] = r[1] if r else None
                mp_spf = X2.suurballe_like(G, s, d, t + 1)
                res = route_bmr_mp(G, s, d, t, X2.DELTA, X2.R_MIN, "ilp", ilp_time=10)
                items = [(rn, p, False) for rn, p in single.items()] + [("MP-SPF", mp_spf, True), ("BMR-MP", res["plan"], True)]
                for rn, obj, is_plan in items:
                    rec = dict(topo=topo, trial=tr, t=t, router=rn, adjacent=adj, delivered=int(obj is not None))
                    if obj is not None:
                        if is_plan:
                            rec["pexp_randt"] = p_random_t_plan(nodes, s, d, obj, t)
                            if nchk < CHECK_PER_TOPO and t == 1 and not (len(obj) == 1 and len(obj[0]) == 2):
                                nchk += 1
                                inst = X.P4_xor_sharing(obj); cand = [v for v in nodes if v not in (s, d)]
                                for v in cand:
                                    e = inst.world.coalition_space([v]).contains(inst.key) if v in inst.relays else False
                                    th = all(relays_of(q) & {v} for q in obj)
                                    checks["n"] += 1; checks["mismatch"] += int(e != th)
                        else:
                            rec["pexp_randt"] = p_random_t_single(N, len(obj) - 1, t)
                        for pp in P_LIST: rec[f"pexp_p{pp}"] = p_indep(obj, pp, is_plan)
                    rows.append(rec)
        print(topo, f"{time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "x2b_rows.csv"), index=False)
    out = []
    for (topo, t, rn), g in df.groupby(["topo", "t", "router"]):
        dl = g[g.delivered == 1]; na = dl[dl.adjacent == 0]
        rec = dict(topo=topo, t=t, router=rn, trials=len(g), adjacent_pairs_pct=round(100 * g.adjacent.mean(), 1), delivered_pct=round(100 * g.delivered.mean(), 1),
                   delivered_nonadj_pct=round(100 * len(na) / max(1, (g.adjacent == 0).sum()), 1))
        for c in ["pexp_randt"] + [f"pexp_p{p}" for p in P_LIST]:
            rec[c + "_nonadj_pct"] = round(100 * na[c].mean(), 2) if len(na) else float("nan")
            rec[c + "_all_pct"] = round(100 * g[c].fillna(0).mean(), 2)
        out.append(rec)
    o = pd.DataFrame(out); o.to_csv(os.path.join(OUT, "x2b_summary.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
    print(o[["topo", "t", "router", "adjacent_pairs_pct", "delivered_nonadj_pct", "pexp_randt_nonadj_pct", "pexp_p0.02_nonadj_pct", "pexp_p0.05_nonadj_pct", "pexp_p0.1_nonadj_pct"]].to_string())
    print("GF(2) cross-check of the plan rule (single compromised node):", checks)
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
