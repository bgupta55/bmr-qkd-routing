"""
e31_alarm_mechanism.py - E3.1: per-trial, per-epoch route logging for BMR-live vs BMR+alarm (Table 8 cells).

Logging only. The RNG flow, topologies, pairs, attack sets, audit draws, router calls and the true-secure definition are
copied from campaign_R/run_EA2.py (which produced Table 8 / Table 12), so the aggregate true-secure rates must equal the
published ones. Both routers see the SAME certificate draw in every epoch.

Cells: ring30, grid30 (5x6), ER30  x  n_audit in {500, 10^4}  x  S_dagger in {2.62, 2.55, 2.48, 2.418}  = 24 cells,
100 repetitions, 80 epochs; attack set 1 (20 % of links) from epoch 25, set 2 (further 10 %) from epoch 60.
Alarm: link flagged iff S_hat < S0 - z_alpha*sigma_W(S0, n), S0 = 2.687, z_alpha = 4.7534.
Output: ../e31_out/e31_rows.csv.gz (one row per cell/rep/epoch), e31_meta.json
"""
import sys, os, json, time, math, itertools, platform, gzip, csv
import numpy as np
import networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_HONEST
from bmr.route import bmr_route
from q6_common import make_topology, attach_certs, evaluate, r_bmr, M_FLOOR, R_MIN

OUT = os.path.join(os.path.dirname(HERE), "e31_out")
os.makedirs(OUT, exist_ok=True)
Z_ALPHA = 4.7534
REPS = int(os.environ.get("E31_REPS", 100))
N_EP, E1, E2 = 80, 25, 60
F1, F2 = 0.20, 0.10
DELTA_BY_N = {500: 0.5, 10_000: 1e-3}
S1_LIST = [2.62, 2.55, 2.48, 2.418]
TOPOS = [("ring", 30), ("grid", 30), ("ER", 30)]


def sigma_w(S0, n):
    return math.sqrt(max(0.0, 4.0 - S0 ** 2 / 4.0) / n)


def alarm_flag(a, n, S0=S_HONEST):
    return a.get("S_hat", float("inf")) < S0 - Z_ALPHA * sigma_w(S0, n)


def r_bmr_alarm(G, s, d, Delta, S0=S_HONEST):
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        cert = a.get("cert")
        if cert is None:
            continue
        if not alarm_flag(a, cert.n, S0) and a.get("r", 0) >= R_MIN:
            H.add_edge(u, v, **a)
    try:
        res = bmr_route(H, s, d, R_MIN, Delta, 0, 10, m_min=M_FLOOR)
        return res[1] if res else None
    except Exception:
        return None


def has_attacked(path, att):
    return path is not None and any(frozenset((a, b)) in att for a, b in zip(path[:-1], path[1:]))


def classify(r):
    if r["same_route"]:
        return "a_same_route"
    if r["live_delivered"] and not r["alarm_delivered"]:
        return "b_alarm_drops_delivery"
    if r["live_has_attacked"] and r["alarm_delivered"] and not r["alarm_has_attacked"]:
        return "c_reroute_to_clean"
    if r["live_has_attacked"] and r["alarm_delivered"] and r["alarm_has_attacked"]:
        return "d_reroute_still_attacked"
    if r["live_delivered"] and not r["live_has_attacked"] and r["alarm_delivered"]:
        return "e2_leaves_clean_for_attacked" if r["alarm_has_attacked"] else "e1_reroute_clean_to_clean"
    return "f_other"        # live undelivered, alarm delivered


def run_cell(spec):
    (topo, n), na, S1, cell_seed = spec
    Delta = DELTA_BY_N[na]
    rng_cell = np.random.default_rng(cell_seed)
    rows = []
    for rep in range(REPS):
        for attempt in range(20):
            G0 = make_topology(topo, n, int(rng_cell.integers(1e9)))
            if nx.is_connected(G0):
                break
        dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a].get(b, 0) >= 4]
        if not pairs:
            pairs = [(a, b) for a in G0 for b in G0 if a < b and a != b]
        sv, dv = pairs[int(rng_cell.integers(len(pairs)))]
        edges = list(G0.edges())
        perm = rng_cell.permutation(len(edges))
        k1, k2 = max(1, round(F1 * len(edges))), max(1, round(F2 * len(edges)))
        set1 = [edges[i] for i in perm[:k1]]
        set2 = [edges[i] for i in perm[k1:k1 + k2]]
        sim = LinkSim(edges, na, delta=1e-6, ell=10_000, phi=0.99, seed=int(rng_cell.integers(1e9)))
        G_live = G0.copy()
        for ep in range(N_EP):
            if ep == E1:
                for e in set1: sim.set_true_S(e, S1)
            if ep == E2:
                for e in set2: sim.set_true_S(e, S1)
            attach_certs(G_live, sim, epoch=ep)
            p_live = r_bmr(G_live, sv, dv, Delta)
            p_al = r_bmr_alarm(G_live, sv, dv, Delta)
            _, _, rate_live, _, _, _, _ = evaluate(G_live, p_live, Delta)
            _, _, rate_al, _, _, _, _ = evaluate(G_live, p_al, Delta)
            att = set()
            if ep >= E1: att |= {frozenset(e) for e in set1}
            if ep >= E2: att |= {frozenset(e) for e in set2}
            # ground-truth diagnostic (NOT used by any router): best clean route under the same certificates
            Gc = G_live.copy()
            Gc.remove_edges_from([tuple(e) for e in att])
            res_c = bmr_route(Gc, sv, dv, R_MIN, Delta, 0, 10, m_min=M_FLOOR) if (sv in Gc and dv in Gc) else None
            clean_exists = int(res_c is not None)
            clean_rate = res_c[0] if res_c else 0.0
            fl_att = fl_hon = 0
            for u, v, a in G_live.edges(data=True):
                if alarm_flag(a, na):
                    if frozenset((u, v)) in att: fl_att += 1
                    else: fl_hon += 1
            r = dict(topo=topo, n_audit=na, S1=S1, rep=rep, epoch=ep, s=sv, d=dv,
                     live_delivered=int(p_live is not None), alarm_delivered=int(p_al is not None),
                     live_hops=(len(p_live) - 1 if p_live else 0), alarm_hops=(len(p_al) - 1 if p_al else 0),
                     live_has_attacked=int(has_attacked(p_live, att)), alarm_has_attacked=int(has_attacked(p_al, att)),
                     live_rate=rate_live, alarm_rate=rate_al, same_route=int(p_live == p_al),
                     n_attacked=len(att), n_flagged_attacked=fl_att, n_flagged_honest=fl_hon,
                     clean_exists=clean_exists, clean_rate=clean_rate)
            r["true_live"] = rate_live if (p_live is not None and not r["live_has_attacked"]) else 0.0
            r["true_alarm"] = rate_al if (p_al is not None and not r["alarm_has_attacked"]) else 0.0
            r["category"] = classify(r)
            r["live_path"] = "-".join(map(str, p_live)) if p_live else ""
            r["alarm_path"] = "-".join(map(str, p_al)) if p_al else ""
            rows.append(r)
    return rows


def main():
    t0 = time.time()
    rng = np.random.default_rng(77)
    specs = []
    for (topo, n), na, S1 in itertools.product(TOPOS, [500, 10_000], S1_LIST):
        specs.append(((topo, n), na, S1, int(rng.integers(1e9))))      # same order/derivation as run_EA2.py
    allrows = []
    with Pool(max(1, os.cpu_count() - 1)) as p:
        for i, rs in enumerate(p.imap(run_cell, specs)):
            allrows += rs
            print(f"cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
    keys = list(allrows[0].keys())
    with gzip.open(os.path.join(OUT, "e31_rows.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)
    json.dump(dict(reps=REPS, rows=len(allrows), seconds=round(time.time() - t0, 1), python=platform.python_version(),
                   machine=platform.platform(), seed_master=77), open(os.path.join(OUT, "e31_meta.json"), "w"), indent=1)
    print("done", len(allrows), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
