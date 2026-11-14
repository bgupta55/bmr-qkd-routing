"""
e34_wrong_baseline.py - E3.4 as specified (developer_guide sec. 4.3; revisionplan_v1 E3.4).
Wrong commissioned baseline: the commissioned value S0_e of the ATTACKED links is set to the degraded value S_dagger (the link was commissioned already
degraded), every other link keeps S0 = 2.687. The alarm flags link e iff S_hat_e < S0_e - z*sigma_W(S0_e, n_e), z = 4.7534.

Part A (as specified): the E3.1 pipeline, RNG flow and cells (rev/e31_alarm_mechanism.py: ring30/grid30/ER30 x n_audit {500, 1e4} x S_dagger {2.62, 2.55, 2.48, 2.418};
  100 reps x 80 epochs; two attack stages). Per trial and epoch three routes on the SAME certificate draw: live, alarm (correct baseline), blind (wrong baseline).
  Acceptance (guide): at S_dagger=2.62, n=1e4 blind == live (paired difference 0). Second metric: detection power = flagged attacked links / attacked links.
  Run for all 24 cells (the specified cell is one of them). live and alarm columns must reproduce e31_out exactly.

Part B (sensitivity): how much commissioning error does it take to lose the alarm's benefit?  Setting = e32_wide variant "2" (heterogeneous audit effort, attacked links drawn
  like the others), floor 0, 120 cells x 100 reps x 10 epochs. S0_e for attacked links = S_hon - lam*(S_hon - S_dagger), lam in {0, .25, .5, .75, 1};
  lam = 0 is the correct baseline, lam = 1 is the fully wrong (blind) baseline of Part A.
Output: ../e34_out/e34A_rows.csv.gz, e34A_cells.csv, e34B_rows.csv.gz, e34B_cells.csv, e34_meta.json
Env: E34_REPS (100) for Part A, E34B_REPS (100) for Part B, E34_PARTS ("AB").
"""
import sys, os, json, time, math, itertools, platform
os.environ["E32W_HETERO"] = "2"
import numpy as np, networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments")); sys.path.insert(0, os.path.join(HERE, "rev"))
from bmr.sim import LinkSim, S_HONEST
from bmr.route import bmr_route
from q6_common import make_topology, attach_certs, evaluate, r_bmr, M_FLOOR, R_MIN
import e32_wide as W                       # HetSim, graph_for, helpers (variant 2 selected above)
from e31_alarm_mechanism import has_attacked, classify, DELTA_BY_N, S1_LIST, TOPOS, N_EP, E1, E2, F1, F2

OUT = os.path.join(os.path.dirname(HERE), "e34_out"); os.makedirs(OUT, exist_ok=True)
Z = 4.7534
REPS_A = int(os.environ.get("E34_REPS", 100)); REPS_B = int(os.environ.get("E34B_REPS", 100))
MASTER = 20261113


def sigw(S0, n): return math.sqrt(max(0.0, 4.0 - S0 * S0 / 4.0) / n)


def flagged(a, n, S0e): return a["S_hat"] < S0e - Z * sigw(S0e, n)


def route_pl(G, s, d, Delta, base, m_min, n_of):
    """BMR after removing links flagged under per-link baselines base[frozenset(e)] (default S_HONEST)."""
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        if a.get("cert") is None or a["r"] < R_MIN: continue
        if flagged(a, n_of(a), base.get(frozenset((u, v)), S_HONEST)): continue
        H.add_edge(u, v, **a)
    if s not in H or d not in H: return None
    res = bmr_route(H, s, d, R_MIN, Delta, 0, 10, m_min=m_min)
    return res[1] if res else None


# ------------------------------------------------------------------ Part A
def cell_A(spec):
    (topo, n), na, S1, cell_seed = spec
    Delta = DELTA_BY_N[na]; rng_cell = np.random.default_rng(cell_seed); rows = []
    for rep in range(REPS_A):
        for _ in range(20):
            G0 = make_topology(topo, n, int(rng_cell.integers(1e9)))
            if nx.is_connected(G0): break
        dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a].get(b, 0) >= 4] or [(a, b) for a in G0 for b in G0 if a < b]
        sv, dv = pairs[int(rng_cell.integers(len(pairs)))]
        edges = list(G0.edges()); perm = rng_cell.permutation(len(edges))
        k1, k2 = max(1, round(F1 * len(edges))), max(1, round(F2 * len(edges)))
        set1 = [edges[i] for i in perm[:k1]]; set2 = [edges[i] for i in perm[k1:k1 + k2]]
        sim = LinkSim(edges, na, delta=1e-6, ell=10_000, phi=0.99, seed=int(rng_cell.integers(1e9)))
        G = G0.copy()
        for ep in range(N_EP):
            if ep == E1:
                for e in set1: sim.set_true_S(e, S1)
            if ep == E2:
                for e in set2: sim.set_true_S(e, S1)
            attach_certs(G, sim, epoch=ep)
            att = set()
            if ep >= E1: att |= {frozenset(e) for e in set1}
            if ep >= E2: att |= {frozenset(e) for e in set2}
            blind_base = {e: S1 for e in att}                           # attacked links commissioned in the degraded state
            nof = lambda a: a["cert"].n
            p_live = r_bmr(G, sv, dv, Delta)
            p_al = route_pl(G, sv, dv, Delta, {}, M_FLOOR, nof)
            p_bl = route_pl(G, sv, dv, Delta, blind_base, M_FLOOR, nof)
            rt = {}
            for nm, p in (("live", p_live), ("alarm", p_al), ("blind", p_bl)):
                _, _, rate, _, _, _, _ = evaluate(G, p, Delta); ha = has_attacked(p, att)
                rt[nm] = (rate, ha, p)
            fa_c = sum(1 for u, v, a in G.edges(data=True) if frozenset((u, v)) in att and flagged(a, na, S_HONEST))
            fa_b = sum(1 for u, v, a in G.edges(data=True) if frozenset((u, v)) in att and flagged(a, na, blind_base[frozenset((u, v))]))
            fh_b = sum(1 for u, v, a in G.edges(data=True) if frozenset((u, v)) not in att and flagged(a, na, S_HONEST))
            r = dict(topo=topo, n_audit=na, S1=S1, rep=rep, epoch=ep, n_attacked=len(att), flag_att_correct=fa_c, flag_att_blind=fa_b, flag_hon=fh_b,
                     live_delivered=int(p_live is not None), alarm_delivered=int(p_al is not None), blind_delivered=int(p_bl is not None),
                     live_has_attacked=int(rt["live"][1]), alarm_has_attacked=int(rt["alarm"][1]), blind_has_attacked=int(rt["blind"][1]),
                     same_blind_live=int(p_live == p_bl), same_alarm_live=int(p_live == p_al))
            for nm in ("live", "alarm", "blind"):
                r["true_" + nm] = rt[nm][0] if (rt[nm][2] is not None and not rt[nm][1]) else 0.0
            r["cat_alarm"] = classify(dict(same_route=int(p_live == p_al), live_delivered=r["live_delivered"], alarm_delivered=r["alarm_delivered"],
                                           live_has_attacked=r["live_has_attacked"], alarm_has_attacked=r["alarm_has_attacked"]))
            r["cat_blind"] = classify(dict(same_route=int(p_live == p_bl), live_delivered=r["live_delivered"], alarm_delivered=r["blind_delivered"],
                                           live_has_attacked=r["live_has_attacked"], alarm_has_attacked=r["blind_has_attacked"]))
            rows.append(r)
    return rows


# ------------------------------------------------------------------ Part B
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def cell_B(spec):
    topo, f, n, Sd, seed = spec
    rng = np.random.default_rng(seed); rows = []; G0 = None
    nof = lambda a: a["cert"].n
    for rep in range(REPS_B):
        if G0 is None or topo.startswith("ER"): G0 = W.graph_for(topo, rng)
        dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a].get(b, 0) >= 2]
        s, d = pairs[int(rng.integers(len(pairs)))]
        edges = list(G0.edges()); k = max(1, round(f * len(edges)))
        att_edges = [edges[i] for i in rng.permutation(len(edges))[:k]]; att = {frozenset(e) for e in att_edges}
        link_n = {LinkSim._key(e): int(W.N_CHOICES[int(rng.integers(len(W.N_CHOICES)))]) for e in edges}
        sim = W.HetSim(edges, link_n, seed=int(rng.integers(1e9)))
        for e in att_edges: sim.set_true_S(e, Sd)
        G = G0.copy()
        bases = {lam: {e: S_HONEST - lam * (S_HONEST - Sd) for e in att} for lam in LAMS}
        for ep in range(W.EPOCHS):
            attach_certs(G, sim, epoch=ep)
            r = dict(topo=topo, f=f, n=n, Sd=Sd, rep=rep, epoch=ep)
            pl = W.route(G, s, d, 0)
            r["true_live"] = 0.0 if (pl is None or W.has_att(pl, att)) else W.bott(G, pl); r["live_deliv"] = int(pl is not None)
            for lam in LAMS:
                p = route_pl(G, s, d, W.DELTA, bases[lam], 0, nof); ha = W.has_att(p, att)
                r[f"true_l{lam}"] = 0.0 if (p is None or ha) else W.bott(G, p)
                r[f"deliv_l{lam}"] = int(p is not None)
                r[f"flag_att_l{lam}"] = sum(1 for u, v, a in G.edges(data=True) if frozenset((u, v)) in att and flagged(a, nof(a), bases[lam][frozenset((u, v))]))
            r["n_att"] = len(att)
            rows.append(r)
    return rows


def boot_ci(x, B=2000, seed=11):
    rng = np.random.default_rng(seed); x = np.asarray(x); m = len(x)
    bs = x[rng.integers(0, m, (B, m))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    import pandas as pd
    t0 = time.time(); parts = os.environ.get("E34_PARTS", "AB")
    if "A" in parts:
        rng = np.random.default_rng(77)
        specs = [((topo, n), na, S1, int(rng.integers(1e9))) for (topo, n), na, S1 in itertools.product(TOPOS, [500, 10_000], S1_LIST)]
        rows = []
        with Pool(max(1, os.cpu_count() - 1)) as p:
            for i, rs in enumerate(p.imap(cell_A, specs)):
                rows += rs; print(f"A cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e34A_rows.csv.gz"), index=False)
        Wd = df[df.epoch.between(25, 34)]; cells = []
        for (topo, na, S1), g in Wd.groupby(["topo", "n_audit", "S1"]):
            db = (g.true_blind - g.true_live).groupby(g.rep).mean(); da = (g.true_alarm - g.true_live).groupby(g.rep).mean()
            cells.append(dict(topo=topo, n_audit=na, S_dag=S1, trials=len(g), true_live=g.true_live.mean(), true_alarm=g.true_alarm.mean(), true_blind=g.true_blind.mean(),
                              diff_alarm=da.mean(), diff_blind=db.mean(), blind_lo=boot_ci(db.values)[0], blind_hi=boot_ci(db.values)[1],
                              blind_same_route_pct=100 * g.same_blind_live.mean(), alarm_same_route_pct=100 * g.same_alarm_live.mean(),
                              power_correct=g.flag_att_correct.sum() / g.n_attacked.sum(), power_blind=g.flag_att_blind.sum() / g.n_attacked.sum(),
                              false_flags=int(g.flag_hon.sum()), blind_cat_nonA=int((g.cat_blind != "a_same_route").sum())))
        pd.DataFrame(cells).to_csv(os.path.join(OUT, "e34A_cells.csv"), index=False)
        print("A done", len(df), f"{time.time()-t0:.0f}s", flush=True)
    if "B" in parts:
        rng = np.random.default_rng(np.random.SeedSequence([MASTER, 34]))
        specs = [(tp, f, n, Sd, int(rng.integers(1e9))) for tp, f, n, Sd in itertools.product(W.TOPOS, W.FRACS, W.NS, W.SDAG)]
        rows = []
        with Pool(max(1, os.cpu_count() - 1)) as p:
            for i, rs in enumerate(p.imap(cell_B, specs)):
                rows += rs
                if (i + 1) % 10 == 0: print(f"B cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "e34B_rows.csv.gz"), index=False)
        cells = []
        for (tp, f, n, Sd), g in df.groupby(["topo", "f", "n", "Sd"]):
            c = dict(topo=tp, f=f, n=n, Sd=Sd, trials=len(g), true_live=g.true_live.mean())
            for lam in LAMS:
                dd = (g[f"true_l{lam}"] - g.true_live).groupby(g.rep).mean(); lo, hi = boot_ci(dd.values)
                c[f"diff_l{lam}"] = dd.mean(); c[f"lo_l{lam}"] = lo; c[f"hi_l{lam}"] = hi
                c[f"power_l{lam}"] = g[f"flag_att_l{lam}"].sum() / g.n_att.sum()
                c[f"deliv_l{lam}"] = g[f"deliv_l{lam}"].mean()
            cells.append(c)
        pd.DataFrame(cells).to_csv(os.path.join(OUT, "e34B_cells.csv"), index=False)
        print("B done", len(df), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(seed=MASTER, repsA=REPS_A, repsB=REPS_B, parts=parts, seconds=round(time.time() - t0, 1), python=platform.python_version(),
                   machine=platform.platform()), open(os.path.join(OUT, "e34_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
