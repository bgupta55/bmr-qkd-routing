"""
run_x7_x8.py - Implementation of X7 (Achievable collusion level t per request and fallback policy)
and X8 (Per-topology panels and confidence intervals).
Seed: 20261113
Outputs written to ../x7_out/ and ../x8_out/
"""
import sys, os, json, time, math, itertools, platform, gzip, csv
import numpy as np, networkx as nx, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_13NOV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

from bmr.sim import LinkSim, S_COMPROMISED
from bmr.route import bmr_route
from bmr.bmr_mp import plan_feasible_ilp, plan_feasible_agg, filter_rate, route_bmr_mp, plan_rate
from q6_common import make_topology, load_sndlib, attach_certs, R_MIN
from x2_collusion import suurballe_like
import x2_collusion as X2

X7_OUT = os.path.join(ROOT_13NOV, "x7_out")
X8_OUT = os.path.join(ROOT_13NOV, "x8_out")
os.makedirs(X7_OUT, exist_ok=True)
os.makedirs(X8_OUT, exist_ok=True)

MASTER = 20261113
GRAPHS = int(os.environ.get("T3_GRAPHS", 10))
PER = int(os.environ.get("T3_PER", 100))
DELTA_BY_N = {500: 0.5, 10_000: 1e-3}
SYN = ["ER", "BA", "WS", "grid", "ring"]
SIZES = [10, 20, 30, 50]
REAL = ["abilene", "polska", "nobel-germany", "geant", "germany50"]
P_LIST = (0.02, 0.05, 0.10)


def relays_of(path):
    return set(path[1:-1])


def safe_suurballe(graph, s, d, K):
    if s not in graph or d not in graph:
        return None
    return suurballe_like(graph, s, d, K)


def p_random_t_single(N, m, t):
    k = m - 1
    pool = N - 2
    if pool < t or k == 0:
        return 0.0
    return 1.0 - math.comb(pool - k, t) / math.comb(pool, t)


def p_random_t_plan(nodes, s, d, plan, t):
    if len(plan) == 1 and len(plan[0]) == 2:
        return 0.0
    cand = [v for v in nodes if v not in (s, d)]
    rel = [relays_of(p) for p in plan]
    if len(cand) < t:
        return 0.0
    hit = tot = 0
    for C in itertools.combinations(cand, t):
        tot += 1
        C = set(C)
        hit += all(r & C for r in rel)
    return hit / tot if tot > 0 else 0.0


def p_indep(path_or_plan, p, is_plan):
    if not is_plan:
        return 1.0 - (1.0 - p) ** max(0, len(path_or_plan) - 2)
    if len(path_or_plan) == 1 and len(path_or_plan[0]) == 2:
        return 0.0
    out = 1.0
    for q in path_or_plan:
        out *= (1.0 - (1.0 - p) ** max(0, len(q) - 2))
    return out


def boot_cluster(arr_by_graph, B=2000, seed=MASTER):
    """Two-stage cluster bootstrap resampling over graph clusters."""
    rng = np.random.default_rng(seed)
    gs = list(arr_by_graph.values())
    G = len(gs)
    if G == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(B):
        idx = rng.integers(0, G, G)
        tot = []
        for i in idx:
            x = gs[i]
            if len(x) > 0:
                tot.append(x[rng.integers(0, len(x), len(x))].mean())
        if tot:
            means.append(np.mean(tot))
    if not means:
        return float("nan"), float("nan")
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def process_cell(spec):
    """
    spec: (cell_idx, topo, n, na, cond, t_req, seed)
    """
    cell_idx, topo, n, na, cond, t_req, seed = spec
    Delta = DELTA_BY_N[na]
    fcomp = 0.2 if cond == "compromised" else 0.0
    rng = np.random.default_rng(seed)
    rows = []

    if topo in REAL:
        gl = [load_sndlib(os.path.join(HERE, "data", "topologies", topo + ".txt"))] * GRAPHS
    else:
        gl = []
        while len(gl) < GRAPHS:
            G = make_topology(topo, n, int(rng.integers(1e9)))
            if nx.is_connected(G):
                gl.append(G)

    for gi, G0 in enumerate(gl):
        edges = list(G0.edges())
        nodes = list(G0.nodes())
        N_nodes = len(nodes)
        dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a][b] >= 2]

        for k in range(PER):
            s, d = pairs[int(rng.integers(len(pairs)))]
            sd = int(rng.integers(1e9))
            sim = LinkSim(edges, na, delta=1e-6, ell=10_000, phi=0.99, seed=sd)
            if fcomp > 0:
                kk = max(1, int(round(fcomp * len(edges))))
                for i in np.random.default_rng(sd).choice(len(edges), kk, replace=False):
                    sim.set_true_S(edges[i], S_COMPROMISED)
            G = G0.copy()
            attach_certs(G, sim, epoch=0)
            Gr = filter_rate(G, R_MIN)

            # 1. t_max_topology
            if G.has_edge(s, d):
                tmax_topo = float("inf")
            else:
                tmax_topo = nx.node_connectivity(G, s, d) - 1

            # 2. t_max_offered
            tmax_off = -1

            # t = 0 feasibility test (single path budget-only)
            p0 = None
            if s in Gr and d in Gr:
                if Gr.has_edge(s, d) and Gr[s][d]["w"] <= Delta + 1e-15:
                    p0 = [s, d]
                    tmax_off = 0
                else:
                    try:
                        p_cand = nx.shortest_path(Gr, s, d, weight="w")
                        w_cand = sum(Gr[u][v]["w"] for u, v in zip(p_cand[:-1], p_cand[1:]))
                        if w_cand <= Delta + 1e-15:
                            p0 = p_cand
                            tmax_off = 0
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

            # t = 1, 2, 3 feasibility test
            plans_by_t = {}
            if tmax_off == 0:
                for t_val in (1, 2, 3):
                    if tmax_topo < t_val:
                        break
                    if safe_suurballe(Gr, s, d, t_val + 1) is None:
                        break
                    if not plan_feasible_agg(Gr, s, d, t_val, Delta):
                        break
                    pl, st = plan_feasible_ilp(Gr, s, d, t_val, Delta, time_limit=5.0)
                    if pl is None:
                        break
                    plans_by_t[t_val] = pl
                    tmax_off = t_val

            # 3. Limiter decomposition
            # For t = 1:
            if tmax_off >= 1:
                limiter_1 = "served"
            elif safe_suurballe(G, s, d, 2) is None:
                limiter_1 = "topology"
            elif safe_suurballe(Gr, s, d, 2) is None:
                limiter_1 = "rate"
            else:
                limiter_1 = "budget"

            # For t = 2:
            if tmax_off >= 2:
                limiter_2 = "served"
            elif safe_suurballe(G, s, d, 3) is None:
                limiter_2 = "topology"
            elif safe_suurballe(Gr, s, d, 3) is None:
                limiter_2 = "rate"
            else:
                limiter_2 = "budget"

            # 4. Fallback Policy: BMR-MP-fallback(t) for requested t in {1, 2}
            # Requested t = 1
            if tmax_off >= 1:
                offered_t_for_req1 = 1
            elif tmax_off == 0:
                offered_t_for_req1 = 0
            else:
                offered_t_for_req1 = -1

            # Requested t = 2
            if tmax_off >= 2:
                offered_t_for_req2 = 2
            elif tmax_off == 1:
                offered_t_for_req2 = 1
            elif tmax_off == 0:
                offered_t_for_req2 = 0
            else:
                offered_t_for_req2 = -1

            # Evaluate rate and true-secure rate for the relevant levels:
            # Rate at t=0 (B2)
            rate_t0 = 0.0
            ts_rate_t0 = 0.0
            path_t0 = p0
            if path_t0:
                rate_t0 = min(Gr[u][v]["r"] for u, v in zip(path_t0[:-1], path_t0[1:]))
                is_clean_t0 = all(sim.true_S[sim._key((u, v))] != S_COMPROMISED for u, v in zip(path_t0[:-1], path_t0[1:]))
                ts_rate_t0 = rate_t0 if is_clean_t0 else 0.0

            # Single path exposure
            hops_t0 = (len(path_t0) - 1) if path_t0 else 0
            exp_wc_t0 = int(hops_t0 >= 2) if path_t0 else 0
            exp_p02_t0 = p_indep(path_t0, 0.02, False) if path_t0 else 0.0
            exp_p05_t0 = p_indep(path_t0, 0.05, False) if path_t0 else 0.0
            exp_p10_t0 = p_indep(path_t0, 0.10, False) if path_t0 else 0.0
            exp_rand_t1_t0 = p_random_t_single(N_nodes, hops_t0, 1) if path_t0 else 0.0
            exp_rand_t2_t0 = p_random_t_single(N_nodes, hops_t0, 2) if path_t0 else 0.0

            # Plan evaluation for t=1
            rate_t1 = 0.0
            ts_rate_t1 = 0.0
            plan_t1 = plans_by_t.get(1)
            exp_wc_req1_t1 = 0
            exp_wc_req2_t1 = 0
            exp_p02_t1 = 0.0
            exp_p05_t1 = 0.0
            exp_p10_t1 = 0.0
            exp_rand_req1_t1 = 0.0
            exp_rand_req2_t1 = 0.0

            if plan_t1:
                rate_t1 = plan_rate(G, plan_t1)
                clean_t1 = any(all(sim.true_S[sim._key((u, v))] != S_COMPROMISED for u, v in zip(p[:-1], p[1:])) for p in plan_t1)
                ts_rate_t1 = rate_t1 if clean_t1 else 0.0
                exp_wc_req1_t1 = 0  # theorem B
                # If requested was t=2 but offered was t=1:
                rel_p = [relays_of(p) for p in plan_t1]
                exp_wc_req2_t1 = 1 if all(len(r) > 0 for r in rel_p) else 0
                exp_p02_t1 = p_indep(plan_t1, 0.02, True)
                exp_p05_t1 = p_indep(plan_t1, 0.05, True)
                exp_p10_t1 = p_indep(plan_t1, 0.10, True)
                exp_rand_req1_t1 = p_random_t_plan(nodes, s, d, plan_t1, 1)
                exp_rand_req2_t1 = p_random_t_plan(nodes, s, d, plan_t1, 2)

            # Plan evaluation for t=2
            rate_t2 = 0.0
            ts_rate_t2 = 0.0
            plan_t2 = plans_by_t.get(2)
            exp_wc_req2_t2 = 0
            exp_p02_t2 = 0.0
            exp_p05_t2 = 0.0
            exp_p10_t2 = 0.0
            exp_rand_req2_t2 = 0.0

            if plan_t2:
                rate_t2 = plan_rate(G, plan_t2)
                clean_t2 = any(all(sim.true_S[sim._key((u, v))] != S_COMPROMISED for u, v in zip(p[:-1], p[1:])) for p in plan_t2)
                ts_rate_t2 = rate_t2 if clean_t2 else 0.0
                exp_wc_req2_t2 = 0  # theorem B
                exp_p02_t2 = p_indep(plan_t2, 0.02, True)
                exp_p05_t2 = p_indep(plan_t2, 0.05, True)
                exp_p10_t2 = p_indep(plan_t2, 0.10, True)
                exp_rand_req2_t2 = p_random_t_plan(nodes, s, d, plan_t2, 2)

            # Assemble row record
            rec = {
                "cell_idx": cell_idx,
                "topo": topo,
                "n": n,
                "n_audit": na,
                "cond": cond,
                "cell_t_spec": t_req,
                "Delta": Delta,
                "graph": gi,
                "req": k,
                "s": s,
                "d": d,
                "tmax_topo": tmax_topo,
                "tmax_offered": tmax_off,
                "limiter_t1": limiter_1,
                "limiter_t2": limiter_2,
                # Requested t = 1
                "fb_req1_offered_t": offered_t_for_req1,
                "fb_req1_rate": (rate_t1 if offered_t_for_req1 == 1 else (rate_t0 if offered_t_for_req1 == 0 else 0.0)),
                "fb_req1_ts_rate": (ts_rate_t1 if offered_t_for_req1 == 1 else (ts_rate_t0 if offered_t_for_req1 == 0 else 0.0)),
                "fb_req1_exp_wc": (0 if offered_t_for_req1 == 1 else (exp_wc_t0 if offered_t_for_req1 == 0 else 0)),
                "fb_req1_exp_rand": (exp_rand_req1_t1 if offered_t_for_req1 == 1 else (exp_rand_t1_t0 if offered_t_for_req1 == 0 else 0.0)),
                "fb_req1_exp_p02": (exp_p02_t1 if offered_t_for_req1 == 1 else (exp_p02_t0 if offered_t_for_req1 == 0 else 0.0)),
                "fb_req1_exp_p05": (exp_p05_t1 if offered_t_for_req1 == 1 else (exp_p05_t0 if offered_t_for_req1 == 0 else 0.0)),
                "fb_req1_exp_p10": (exp_p10_t1 if offered_t_for_req1 == 1 else (exp_p10_t0 if offered_t_for_req1 == 0 else 0.0)),
                # Requested t = 2
                "fb_req2_offered_t": offered_t_for_req2,
                "fb_req2_rate": (rate_t2 if offered_t_for_req2 == 2 else (rate_t1 if offered_t_for_req2 == 1 else (rate_t0 if offered_t_for_req2 == 0 else 0.0))),
                "fb_req2_ts_rate": (ts_rate_t2 if offered_t_for_req2 == 2 else (ts_rate_t1 if offered_t_for_req2 == 1 else (ts_rate_t0 if offered_t_for_req2 == 0 else 0.0))),
                "fb_req2_exp_wc": (0 if offered_t_for_req2 == 2 else (exp_wc_req2_t1 if offered_t_for_req2 == 1 else (exp_wc_t0 if offered_t_for_req2 == 0 else 0))),
                "fb_req2_exp_rand": (exp_rand_req2_t2 if offered_t_for_req2 == 2 else (exp_rand_req2_t1 if offered_t_for_req2 == 1 else (exp_rand_t2_t0 if offered_t_for_req2 == 0 else 0.0))),
                "fb_req2_exp_p02": (exp_p02_t2 if offered_t_for_req2 == 2 else (exp_p02_t1 if offered_t_for_req2 == 1 else (exp_p02_t0 if offered_t_for_req2 == 0 else 0.0))),
                "fb_req2_exp_p05": (exp_p05_t2 if offered_t_for_req2 == 2 else (exp_p05_t1 if offered_t_for_req2 == 1 else (exp_p05_t0 if offered_t_for_req2 == 0 else 0.0))),
                "fb_req2_exp_p10": (exp_p10_t2 if offered_t_for_req2 == 2 else (exp_p10_t1 if offered_t_for_req2 == 1 else (exp_p10_t0 if offered_t_for_req2 == 0 else 0.0))),
                # Baselines for comparison and CI
                "b2_rate": rate_t0,
                "b2_deliv": int(tmax_off >= 0),
                "mp_t1_deliv": int(tmax_off >= 1),
                "mp_t1_rate": rate_t1,
                "mp_t2_deliv": int(tmax_off >= 2),
                "mp_t2_rate": rate_t2,
            }
            rows.append(rec)
    return rows


def generate_specs():
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, 3]))
    specs = []
    cell_idx = 0
    for t in (1, 2):
        for topo in SYN:
            for n in SIZES:
                for na in (500, 10_000):
                    for cond in ("clean", "compromised"):
                        specs.append((cell_idx, topo, n, na, cond, t, int(rng.integers(1e9))))
                        cell_idx += 1
        for topo in REAL:
            for na in (500, 10_000):
                for cond in ("clean", "compromised"):
                    specs.append((cell_idx, topo, 0, na, cond, t, int(rng.integers(1e9))))
                    cell_idx += 1
    return specs


def run_x8_exposure_bootstrap():
    """Compute bootstrap confidence intervals for X2b exposure data."""
    x2b_rows_path = os.path.join(ROOT_13NOV, "x2b_out", "x2b_rows.csv")
    if not os.path.exists(x2b_rows_path):
        print("x2b_rows.csv not found, skipping x2b bootstrap")
        return []

    df_x2b = pd.read_csv(x2b_rows_path)
    # df_x2b has: topo, trial, t, router, adjacent, delivered, pexp_randt, pexp_p0.02, pexp_p0.05, pexp_p0.1
    # Trials are 0..1499. For cluster bootstrap, we can treat blocks of 150 trials (or trials) as clusters,
    # or simple bootstrap over 1500 trials with B=2000.
    rng = np.random.default_rng(MASTER)
    records = []
    routers = ["SPF", "B2-BMR-NoFloor", "BMR-single(t=1)", "BMR-single(t=2)", "MP-SPF", "BMR-MP"]
    for topo in ["ER", "BA", "hub", "ring"]:
        for t in (1, 2):
            sub = df_x2b[(df_x2b.topo == topo) & (df_x2b.t == t)]
            for r in routers:
                r_sub = sub[sub.router == r]
                if len(r_sub) == 0:
                    continue
                # non-adjacent delivered subset
                na_sub = r_sub[(r_sub.adjacent == 0) & (r_sub.delivered == 1)]
                deliv_pct = 100.0 * r_sub.delivered.mean()
                adj_pct = 100.0 * r_sub.adjacent.mean()

                for p_val in P_LIST:
                    col = f"pexp_p{p_val}"
                    if len(na_sub) > 0 and col in na_sub:
                        vals = na_sub[col].values * 100.0
                        mean_val = float(vals.mean())
                        # bootstrap over trials
                        B = 2000
                        b_means = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(B)]
                        ci_lo, ci_hi = float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5))
                    else:
                        mean_val = float("nan")
                        ci_lo, ci_hi = float("nan"), float("nan")

                    records.append({
                        "metric_group": "X2b_exposure_nonadjacent",
                        "topo": topo,
                        "t": t,
                        "router": r,
                        "p": p_val,
                        "mean": mean_val,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "ci_str": f"{mean_val:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(mean_val) else "N/A"
                    })
    return records


def main():
    t0 = time.time()
    specs = generate_specs()
    assert len(specs) == 200, f"Expected 200 specs, got {len(specs)}"

    limit = os.environ.get("X7_LIMIT")
    if limit:
        lim = int(limit)
        specs = specs[::max(1, len(specs) // lim)][:lim]
        print(f"Running SMOKE subset: {len(specs)} cells")
    else:
        print(f"Running FULL suite: {len(specs)} cells, 10 graphs x 100 requests = {len(specs)*1000} trials", flush=True)

    workers = max(1, os.cpu_count() - 1)
    print(f"Starting multiprocessing pool with {workers} workers...", flush=True)

    all_rows = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(process_cell, specs, chunksize=1)):
            all_rows.extend(r)
            if (i + 1) % 10 == 0 or (i + 1) == len(specs):
                print(f"Completed {i+1}/{len(specs)} cells ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    print(f"Collected {len(df)} total request rows in {time.time()-t0:.1f}s. Saving rows...", flush=True)
    df.to_csv(os.path.join(X7_OUT, "x7_rows.csv.gz"), index=False, compression="gzip")

    # -------------------------------------------------------------
    # Summary 1: x7_summary.csv (t_max_offered distribution & limiter decomposition)
    # -------------------------------------------------------------
    print("Generating x7_summary.csv...", flush=True)
    summary_rows = []

    # Map topologies into family groups
    def topo_family(tp):
        if tp in SYN:
            return tp
        return "SNDlib"

    df["family"] = df["topo"].apply(topo_family)

    # Groupings: overall, per family, and per topo
    groupings = [
        ("Overall", df),
    ]
    for fam, gdf in df.groupby("family"):
        groupings.append((f"Family_{fam}", gdf))
    for tp, gdf in df.groupby("topo"):
        groupings.append((f"Topo_{tp}", gdf))

    for name, sub_df in groupings:
        n_req = len(sub_df)
        pct_refused = 100.0 * (sub_df.tmax_offered == -1).mean()
        pct_t0 = 100.0 * (sub_df.tmax_offered == 0).mean()
        pct_t1 = 100.0 * (sub_df.tmax_offered == 1).mean()
        pct_t2 = 100.0 * (sub_df.tmax_offered == 2).mean()
        pct_t3 = 100.0 * (sub_df.tmax_offered == 3).mean()
        pct_ge1 = 100.0 * (sub_df.tmax_offered >= 1).mean()
        pct_ge2 = 100.0 * (sub_df.tmax_offered >= 2).mean()
        pct_ge3 = 100.0 * (sub_df.tmax_offered >= 3).mean()

        # Limiter decomposition at t=1
        lim1_topo = 100.0 * (sub_df.limiter_t1 == "topology").mean()
        lim1_rate = 100.0 * (sub_df.limiter_t1 == "rate").mean()
        lim1_budg = 100.0 * (sub_df.limiter_t1 == "budget").mean()
        lim1_ref_tot = lim1_topo + lim1_rate + lim1_budg

        # Limiter decomposition at t=2
        lim2_topo = 100.0 * (sub_df.limiter_t2 == "topology").mean()
        lim2_rate = 100.0 * (sub_df.limiter_t2 == "rate").mean()
        lim2_budg = 100.0 * (sub_df.limiter_t2 == "budget").mean()
        lim2_ref_tot = lim2_topo + lim2_rate + lim2_budg

        summary_rows.append({
            "group": name,
            "requests": n_req,
            "pct_refused_t_neg1": round(pct_refused, 2),
            "pct_t_max_0": round(pct_t0, 2),
            "pct_t_max_1": round(pct_t1, 2),
            "pct_t_max_2": round(pct_t2, 2),
            "pct_t_max_3": round(pct_t3, 2),
            "pct_t_max_ge1": round(pct_ge1, 2),
            "pct_t_max_ge2": round(pct_ge2, 2),
            "pct_t_max_ge3": round(pct_ge3, 2),
            # Limiter at t=1
            "limiter_t1_topology_pct": round(lim1_topo, 2),
            "limiter_t1_rate_pct": round(lim1_rate, 2),
            "limiter_t1_budget_pct": round(lim1_budg, 2),
            "limiter_t1_total_refusal_pct": round(lim1_ref_tot, 2),
            # Limiter at t=2
            "limiter_t2_topology_pct": round(lim2_topo, 2),
            "limiter_t2_rate_pct": round(lim2_rate, 2),
            "limiter_t2_budget_pct": round(lim2_budg, 2),
            "limiter_t2_total_refusal_pct": round(lim2_ref_tot, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(X7_OUT, "x7_summary.csv"), index=False)

    # -------------------------------------------------------------
    # Source Data for Figure F8: Stacked Bar Chart per Topology Family
    # -------------------------------------------------------------
    f8_rows = []
    fam_list = ["ER", "BA", "WS", "grid", "ring", "SNDlib"]
    for fam in fam_list:
        sub = df[df.family == fam]
        n_req = len(sub)
        if n_req == 0:
            continue
        f8_rows.append({
            "family": fam,
            "requests": n_req,
            "refused_pct": round(100.0 * (sub.tmax_offered == -1).mean(), 2),
            "t0_only_pct": round(100.0 * (sub.tmax_offered == 0).mean(), 2),
            "t1_pct": round(100.0 * (sub.tmax_offered == 1).mean(), 2),
            "t_ge2_pct": round(100.0 * (sub.tmax_offered >= 2).mean(), 2),
        })
    df_f8 = pd.DataFrame(f8_rows)
    df_f8.to_csv(os.path.join(X7_OUT, "f8_source_data.csv"), index=False)

    # -------------------------------------------------------------
    # Summary 2: x7_fallback.csv (Fallback Policy Evaluation)
    # -------------------------------------------------------------
    print("Generating x7_fallback.csv...", flush=True)
    fb_rows = []

    # Evaluate requested t = 1 and requested t = 2
    for t_req in (1, 2):
        prefix = f"fb_req{t_req}"
        for name, sub_df in [("Overall", df)] + [(f"Family_{fam}", df[df.family == fam]) for fam in fam_list]:
            tot_req = len(sub_df)
            if tot_req == 0:
                continue
            # Partition requests into: served at requested t, served at t' < t, refused (t' = -1)
            served_full = sub_df[sub_df[f"{prefix}_offered_t"] == t_req]
            served_downgrade = sub_df[(sub_df[f"{prefix}_offered_t"] >= 0) & (sub_df[f"{prefix}_offered_t"] < t_req)]
            served_all = sub_df[sub_df[f"{prefix}_offered_t"] >= 0]
            refused = sub_df[sub_df[f"{prefix}_offered_t"] == -1]

            fb_rows.append({
                "requested_t": t_req,
                "group": name,
                "total_requests": tot_req,
                # Distribution of offered t'
                "share_served_full_t_pct": round(100.0 * len(served_full) / tot_req, 2),
                "share_served_downgrade_pct": round(100.0 * len(served_downgrade) / tot_req, 2),
                "share_refused_pct": round(100.0 * len(refused) / tot_req, 2),
                # If t_req = 2, break down offered t'=1 and t'=0
                "share_offered_t2_pct": round(100.0 * (sub_df[f"{prefix}_offered_t"] == 2).mean(), 2) if t_req == 2 else 0.0,
                "share_offered_t1_pct": round(100.0 * (sub_df[f"{prefix}_offered_t"] == 1).mean(), 2),
                "share_offered_t0_pct": round(100.0 * (sub_df[f"{prefix}_offered_t"] == 0).mean(), 2),
                # Mean rates
                "mean_rate_all_requests": round(sub_df[f"{prefix}_rate"].mean(), 4),
                "mean_rate_served_full": round(served_full[f"{prefix}_rate"].mean(), 4) if len(served_full) else 0.0,
                "mean_rate_served_downgrade": round(served_downgrade[f"{prefix}_rate"].mean(), 4) if len(served_downgrade) else 0.0,
                # Mean true-secure rates
                "mean_ts_rate_all_requests": round(sub_df[f"{prefix}_ts_rate"].mean(), 4),
                "mean_ts_rate_served_full": round(served_full[f"{prefix}_ts_rate"].mean(), 4) if len(served_full) else 0.0,
                "mean_ts_rate_served_downgrade": round(served_downgrade[f"{prefix}_ts_rate"].mean(), 4) if len(served_downgrade) else 0.0,
                # Exposure under worst-case t-relay coalition (on served requests)
                "exp_wc_served_full_pct": round(100.0 * served_full[f"{prefix}_exp_wc"].mean(), 2) if len(served_full) else 0.0,
                "exp_wc_served_downgrade_pct": round(100.0 * served_downgrade[f"{prefix}_exp_wc"].mean(), 2) if len(served_downgrade) else 0.0,
                "exp_wc_all_served_pct": round(100.0 * served_all[f"{prefix}_exp_wc"].mean(), 2) if len(served_all) else 0.0,
                # Exposure under independent compromise (p=0.05) on served requests
                "exp_p05_served_full_pct": round(100.0 * served_full[f"{prefix}_exp_p05"].mean(), 2) if len(served_full) else 0.0,
                "exp_p05_served_downgrade_pct": round(100.0 * served_downgrade[f"{prefix}_exp_p05"].mean(), 2) if len(served_downgrade) else 0.0,
                "exp_p05_all_served_pct": round(100.0 * served_all[f"{prefix}_exp_p05"].mean(), 2) if len(served_all) else 0.0,
            })

    df_fb = pd.DataFrame(fb_rows)
    df_fb.to_csv(os.path.join(X7_OUT, "x7_fallback.csv"), index=False)

    # -------------------------------------------------------------
    # Summary 3: X8 Confidence Intervals (x8_ci.csv)
    # -------------------------------------------------------------
    print("Generating x8_ci.csv (cluster-bootstrap over graphs)...", flush=True)
    ci_records = []

    # 1. Refusal rates at t = 0, 1, 2
    # Resample over the 10 graphs per cell across all cells
    # Dict of cell_idx -> dict(graph -> array)
    for t_val, metric_name, col_name in [
        (0, "Refusal_rate_t0 (B2)", "b2_deliv"),
        (1, "Refusal_rate_t1 (BMR-MP)", "mp_t1_deliv"),
        (2, "Refusal_rate_t2 (BMR-MP)", "mp_t2_deliv"),
    ]:
        # grand mean refusal
        refusal_arr = 1.0 - df[col_name].values
        mean_val = 100.0 * float(refusal_arr.mean())

        # Group by (cell_idx, graph)
        graph_means = {}
        for (c_idx, gi), g_sub in df.groupby(["cell_idx", "graph"]):
            graph_means[(c_idx, gi)] = 100.0 * (1.0 - g_sub[col_name].values)

        ci_lo, ci_hi = boot_cluster(graph_means, B=2000, seed=MASTER)
        ci_records.append({
            "metric_group": "Headline_Refusal_Rate",
            "quantity": metric_name,
            "t": t_val,
            "mean": round(mean_val, 2),
            "ci_lo": round(ci_lo, 2),
            "ci_hi": round(ci_hi, 2),
            "ci_str": f"{mean_val:.2f}% [{ci_lo:.2f}%, {ci_hi:.2f}%]",
        })

    # 2. Mean-rate cost (BMR-MP vs B2) at t = 1 and t = 2
    for t_val, metric_name, col_mp in [
        (1, "Mean_rate_cost_t1 (BMR-MP - B2)", "mp_t1_rate"),
        (2, "Mean_rate_cost_t2 (BMR-MP - B2)", "mp_t2_rate"),
    ]:
        diff_arr = df[col_mp].values - df["b2_rate"].values
        mean_diff = float(diff_arr.mean())

        graph_means = {}
        for (c_idx, gi), g_sub in df.groupby(["cell_idx", "graph"]):
            graph_means[(c_idx, gi)] = g_sub[col_mp].values - g_sub["b2_rate"].values

        ci_lo, ci_hi = boot_cluster(graph_means, B=2000, seed=MASTER)
        ci_records.append({
            "metric_group": "Headline_Mean_Rate_Cost",
            "quantity": metric_name,
            "t": t_val,
            "mean": round(mean_diff, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "ci_str": f"{mean_diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]",
        })

    # 3. Key exposure bootstrap from X2b
    x2b_ci_list = run_x8_exposure_bootstrap()
    for rec in x2b_ci_list:
        ci_records.append({
            "metric_group": rec["metric_group"],
            "quantity": f"Key_Exposure_{rec['router']}_p={rec['p']}_{rec['topo']}_t={rec['t']}",
            "t": rec["t"],
            "mean": rec["mean"],
            "ci_lo": rec["ci_lo"],
            "ci_hi": rec["ci_hi"],
            "ci_str": rec["ci_str"],
        })

    df_ci = pd.DataFrame(ci_records)
    df_ci.to_csv(os.path.join(X8_OUT, "x8_ci.csv"), index=False)

    # -------------------------------------------------------------
    # Supplement Figure Data: F3 Per-Topology Exposure
    # -------------------------------------------------------------
    if os.path.exists(os.path.join(ROOT_13NOV, "x2b_out", "x2b_summary.csv")):
        x2b_sum = pd.read_csv(os.path.join(ROOT_13NOV, "x2b_out", "x2b_summary.csv"))
        x2b_sum.to_csv(os.path.join(X8_OUT, "f3_per_topology_exposure.csv"), index=False)

    # -------------------------------------------------------------
    # Consistency Checks
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("CONSISTENCY CHECKS (against results_v2.md / developer guide):")
    print("=======================================================")
    pct_ge1 = 100.0 * (df.tmax_offered >= 1).mean()
    pct_ge2 = 100.0 * (df.tmax_offered >= 2).mean()
    pct_ref_t0 = 100.0 * (df.tmax_offered == -1).mean()
    pct_t0 = 100.0 * (df.tmax_offered == 0).mean()
    pct_t1 = 100.0 * (df.tmax_offered == 1).mean()
    pct_t_ge2 = 100.0 * (df.tmax_offered >= 2).mean()

    lim2_topo = 100.0 * (df.limiter_t2 == "topology").mean()
    lim2_rate = 100.0 * (df.limiter_t2 == "rate").mean()
    lim2_budg = 100.0 * (df.limiter_t2 == "budget").mean()

    lim1_topo = 100.0 * (df.limiter_t1 == "topology").mean()
    lim1_rate = 100.0 * (df.limiter_t1 == "rate").mean()
    lim1_budg = 100.0 * (df.limiter_t1 == "budget").mean()

    print(f"Share with t_max_offered >= 1: {pct_ge1:.2f}% (Expected: 71.7%)")
    print(f"Share with t_max_offered >= 2: {pct_ge2:.2f}% (Expected: 28.8%)")
    print(f"Share refused at t = 0 (B2):   {pct_ref_t0:.2f}% (Expected: 7.7%)")
    print(f"Share with t_max_offered == 0: {pct_t0:.2f}% (Expected: ~20.6%)")
    print(f"Share with t_max_offered == 1: {pct_t1:.2f}% (Expected: ~42.9%)")
    print(f"Share with t_max_offered >= 2: {pct_t_ge2:.2f}% (Expected: ~28.8%)")
    print(f"Limiter at t = 2: Topology={lim2_topo:.2f}%, Rate={lim2_rate:.2f}%, Budget={lim2_budg:.2f}% (Sum={lim2_topo+lim2_rate+lim2_budg:.2f}%, Exp: 58.3 / 6.2 / 6.7)")
    print(f"Limiter at t = 1: Topology={lim1_topo:.2f}%, Rate={lim1_rate:.2f}%, Budget={lim1_budg:.2f}% (Sum={lim1_topo+lim1_rate+lim1_budg:.2f}%, Exp: 3.9 / 10.8 / 13.6)")
    print("=======================================================\n")
    print(f"All runs completed in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
