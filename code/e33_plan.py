"""
e33_plan.py - RF3 (E3.3): the alarm question for MULTIPATH plans (t=1, two disjoint paths, key delivered by XOR sharing, P4).
Definition (stated in advance): a delivered plan is "plan-secure" iff at least ONE of its paths contains no attacked link (an attacked link leaks the
hop key of every path through it; the XOR-shared key stays secret while one share path is clean). plan true-secure rate = bottleneck of the plan if
plan-secure, else 0. Second metric, "all-paths-clean rate": bottleneck if NO path contains an attacked link.
Routers: BMR-MP-live (CP-SAT) vs BMR-MP + alarm (links with S_hat < S0 - z*sigma_W removed first). Same certificate draw per epoch.
Setting = heterogeneous audit effort, attacked links drawn like the others (mode "2" of e32_wide.py, the neutral variant), Delta = 1e-3, 10 epochs, 100 reps.
Topologies: ER4, ER6, grid5x6, polska, geant. f in {0.1,0.2,0.3}; the cell n is carried for parity with e32_wide (in the heterogeneous variant every link's
n is drawn from {1500,2500,5000,1e4,2e4}); S_dagger in {2.55, 2.62}. 60 cells.
Output ../e33_out/e33_cells.csv, e33_meta.json
"""
import sys, os, json, time, math, itertools, platform
os.environ["E32W_HETERO"] = "2"
import numpy as np, networkx as nx
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import e32_wide as W
from bmr.sim import LinkSim, S_HONEST
from bmr.bmr_mp import route_bmr_mp, plan_rate
from q6_common import attach_certs, R_MIN

OUT = os.path.join(os.path.dirname(HERE), "e33_out"); os.makedirs(OUT, exist_ok=True)
REPS, EPOCHS, DELTA, T = int(os.environ.get("E33_REPS", 100)), 10, 1e-3, 1
TOPOS = ["ER4", "ER6", "grid5x6", "polska", "geant"]


def alarm_graph(G, S0=S_HONEST):
    H = nx.Graph(); H.add_nodes_from(G.nodes())
    for u, v, a in G.edges(data=True):
        if a["cert"] is None or a["r"] < R_MIN: continue
        if a["S_hat"] < S0 - W.Z * W.sig_w(S0, a["cert"].n): continue
        H.add_edge(u, v, **a)
    return H


def path_att(p, att): return any(frozenset((a, b)) in att for a, b in zip(p[:-1], p[1:]))


def evaluate(G, plan, att):
    if plan is None: return 0, 0.0, 0.0
    ps = [path_att(p, att) for p in plan]; r = plan_rate(G, plan)
    return 1, (r if not all(ps) else 0.0), (r if not any(ps) else 0.0)


def run_cell(spec):
    topo, f, n, Sd, seed = spec
    rng = np.random.default_rng(seed); rows = []; G0 = None
    for rep in range(REPS):
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
        for ep in range(EPOCHS):
            attach_certs(G, sim, epoch=ep)
            pl = route_bmr_mp(G, s, d, T, DELTA, R_MIN, "ilp", ilp_time=10)["plan"]
            Ha = alarm_graph(G)
            pa = route_bmr_mp(Ha, s, d, T, DELTA, R_MIN, "ilp", ilp_time=10)["plan"] if (s in Ha and d in Ha) else None
            dl, tl, cl = evaluate(G, pl, att); da, ta, ca = evaluate(G, pa, att)
            rows.append(dict(rep=rep, live_deliv=dl, alarm_deliv=da, true_live=tl, true_alarm=ta, clean_live=cl, clean_alarm=ca,
                             live_has_att=int(pl is not None and any(path_att(p, att) for p in pl)),
                             live_planinsecure=int(pl is not None and all(path_att(p, att) for p in pl))))
    return spec, rows


def main():
    t0 = time.time(); rng = np.random.default_rng(np.random.SeedSequence([W.MASTER, 33, 3]))
    specs = [(tp, f, n, Sd, int(rng.integers(1e9))) for tp, f, n, Sd in itertools.product(TOPOS, [0.1, 0.2, 0.3], [10_000, 20_000], [2.55, 2.62])]
    import pandas as pd
    cells = []
    with Pool(max(1, os.cpu_count() - 1)) as p:
        for i, (spec, rows) in enumerate(p.imap(run_cell, specs)):
            g = pd.DataFrame(rows); tp, f, n, Sd, _ = spec
            rd = (g.true_alarm - g.true_live).groupby(g.rep).mean().values; rc = (g.clean_alarm - g.clean_live).groupby(g.rep).mean().values
            lo, hi = W.boot_ci(rd); clo, chi = W.boot_ci(rc)
            cells.append(dict(topo=tp, f=f, n=n, Sd=Sd, trials=len(g), live_deliv=g.live_deliv.mean(), alarm_deliv=g.alarm_deliv.mean(),
                              true_live=g.true_live.mean(), true_alarm=g.true_alarm.mean(), diff=rd.mean(), diff_lo=lo, diff_hi=hi,
                              clean_live=g.clean_live.mean(), clean_alarm=g.clean_alarm.mean(), clean_diff=rc.mean(), clean_lo=clo, clean_hi=chi,
                              live_plan_has_att_pct=100 * g.live_has_att.sum() / max(1, g.live_deliv.sum()),
                              live_plan_insecure_pct=100 * g.live_planinsecure.sum() / max(1, g.live_deliv.sum())))
            if (i + 1) % 10 == 0: print(f"cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(cells).to_csv(os.path.join(OUT, "e33_cells.csv"), index=False)
    json.dump(dict(reps=REPS, epochs=EPOCHS, delta=DELTA, t=T, seconds=round(time.time() - t0, 1), python=platform.python_version(), machine=platform.platform()),
              open(os.path.join(OUT, "e33_meta.json"), "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
