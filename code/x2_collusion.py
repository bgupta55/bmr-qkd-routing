"""
x2_collusion.py - X2: collusion exposure re-measured with the GF(2) verifier (X1), for single-path and multipath routers.

Topologies as in q4_collusion.py (ER30, BA30, ring12, hub12); 1500 trials each; same certificate draw (n_audit=1e4, Delta=1e-3, phi=0.99, no compromised links).
Per trial, one random (s,d) and one certificate set; every router sees the same instance.
Routers / delivery protocol analysed:
  SPF (hop-minimal)                       single path, delivered by P2 (trusted forward)   [also P0 reported]
  B1a-like rate-greedy (1/r weights)      single path, P2
  B2  BMR floor-off                       single path, P2
  BMR-single (floor on, t=1)              single path, P2  (and the paper's P0 chain)
  MP-SPF (Suurballe t+1 paths, hop-min)   t+1 disjoint paths, P4 (XOR sharing)
  BMR-MP (exact plan solver, t+1 paths)   t+1 disjoint paths, P4
t in {1,2}. A trial is exposed if SOME coalition of <= t relays (all subsets enumerated; ring/hub/ER/BA relays) can compute the key
(coalition of size t exactly is enough since exposure is monotone). A direct link is never exposed (no relay).
Also: "relay-only" worst-case P(min coalition <= t) via exact enumeration over all size-<=t subsets of the route's relays.
Output: x2_out/x2_rows.csv, x2_summary.csv
"""
import sys, os, math, csv, time, itertools, platform
import numpy as np, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import symbolic_exposure as X
from bmr.certificate import build_certificate
from bmr.route import bmr_route
from bmr.bmr_mp import route_bmr_mp
from q6_common import make_topology  # noqa (kept for parity)

OUT = os.path.join(os.path.dirname(HERE), "x2_out"); os.makedirs(OUT, exist_ok=True)
SEED = 20261113
TRIALS = int(os.environ.get("X2_TRIALS", 1500))
N_AUDIT, DELTA, R_MIN, PHI, ELL = 10_000, 1e-3, 0.01, 0.99, 10_000


def make_graph(topo, rng):
    if topo == "ER":
        while True:
            G = nx.erdos_renyi_graph(30, 4 / 29, seed=int(rng.integers(1e9)))
            if nx.is_connected(G): return G
    if topo == "BA":
        return nx.barabasi_albert_graph(30, 2, seed=int(rng.integers(1e9)))
    if topo == "ring":
        return nx.cycle_graph(12)
    if topo == "hub":
        G = nx.Graph(); G.add_edge("H1", "H2")
        for i in range(10): G.add_edge("H1", i); G.add_edge("H2", i)
        return nx.convert_node_labels_to_integers(G)


def certify(G, rng):
    for u, v in G.edges():
        S_hat = rng.normal(2.687, math.sqrt((4 - 2.687 ** 2 / 4) / N_AUDIT))
        c = build_certificate((u, v), float(S_hat), N_AUDIT, 1e-6, ELL, PHI, 0, 0)
        G[u][v]["r"] = c.r if c else 0.0
        G[u][v]["w"] = c.w if c else 2.0
        G[u][v]["T"] = 0
    return G


def exposed_any(inst, t):
    """exists a coalition of size <= t among the relays that computes the key (exhaustive; uses the checker, not the theory)."""
    W, key = inst.world, inst.key
    for k in range(1, t + 1):
        for C in itertools.combinations(inst.relays, k):
            if W.coalition_space(C).contains(key):
                return True
    return False


def suurballe_like(G, s, d, K):
    """K internally vertex-disjoint paths of minimum total hop count (min-cost flow, unit node capacities); None if fewer than K exist."""
    if G.has_edge(s, d):
        return [[s, d]]
    F = nx.DiGraph()
    for v in G.nodes():
        F.add_edge(("i", v), ("o", v), capacity=(K if v in (s, d) else 1), weight=0)
    for u, v in G.edges():
        for x, y in ((u, v), (v, u)):
            if y == s or x == d: continue
            F.add_edge(("o", x), ("i", y), capacity=1, weight=1)
    F.nodes[("o", s)]["demand"] = -K; F.nodes[("i", d)]["demand"] = K
    try:
        flow = nx.min_cost_flow(F)
    except nx.NetworkXUnfeasible:
        return None
    nxt = {}
    for (kind, x), dd in flow.items():
        if kind != "o": continue
        for (k2, y), f in dd.items():
            if f > 0 and k2 == "i" and y != x:
                nxt.setdefault(x, []).append(y)
    paths = []
    for _ in range(K):
        p = [s]
        while p[-1] != d:
            p.append(nxt[p[-1]].pop())
        paths.append(p)
    return paths


def routers_single(G, s, d):
    out = {}
    try: out["SPF"] = nx.shortest_path(G, s, d)
    except nx.NetworkXNoPath: out["SPF"] = None
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        if a["r"] >= R_MIN: H.add_edge(u, v, w=1.0 / (a["r"] + 1e-9))
    try: out["RateGreedy"] = nx.dijkstra_path(H, s, d, weight="w")
    except (nx.NetworkXNoPath, nx.NodeNotFound): out["RateGreedy"] = None
    r = bmr_route(G, s, d, R_MIN, DELTA, 0, 100, m_min=0); out["B2-BMR-NoFloor"] = r[1] if r else None
    out_t = {}
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    for topo in ["ER", "BA", "ring", "hub"]:
        for tr in range(TRIALS):
            G = certify(make_graph(topo, rng), rng)
            nodes = list(G.nodes())
            i, j = [int(x) for x in rng.choice(len(nodes), 2, replace=False)]
            s, d = nodes[i], nodes[j]
            base = routers_single(G, s, d)
            for t in (1, 2):
                single = dict(base)
                r = bmr_route(G, s, d, R_MIN, DELTA, 0, 100, m_min=2 * t + 1); single[f"BMR-single(t={t})"] = r[1] if r else None
                for rn, path in single.items():
                    if path is None:
                        rows.append(dict(topo=topo, trial=tr, t=t, router=rn, delivered=0, hops=0, n_paths=0, P0=-1, P2=-1)); continue
                    m = len(path) - 1
                    e0 = int(exposed_any(X.P0_xor_chain(path), t)) if m >= 1 else 0
                    e2 = int(exposed_any(X.P2_trusted_forward(path), t)) if m >= 1 else 0
                    rows.append(dict(topo=topo, trial=tr, t=t, router=rn, delivered=1, hops=m, n_paths=1, P0=e0, P2=e2))
                # multipath routers (delivered by P4)
                mp_spf = suurballe_like(G, s, d, t + 1)
                res = route_bmr_mp(G, s, d, t, DELTA, R_MIN, "ilp", ilp_time=10)
                for rn, plan in (("MP-SPF", mp_spf), ("BMR-MP", res["plan"])):
                    if plan is None:
                        rows.append(dict(topo=topo, trial=tr, t=t, router=rn, delivered=0, hops=0, n_paths=0, P4=-1, status=res["status"] if rn == "BMR-MP" else "ok")); continue
                    if len(plan) == 1 and len(plan[0]) == 2:
                        e4 = 0
                    else:
                        e4 = int(exposed_any(X.P4_xor_sharing(plan), t))
                    rows.append(dict(topo=topo, trial=tr, t=t, router=rn, delivered=1, hops=sum(len(p) - 1 for p in plan) / len(plan),
                                     n_paths=len(plan), P4=e4, status=res["status"] if rn == "BMR-MP" else "ok"))
        print(topo, f"{time.time()-t0:.0f}s", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "x2_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    out = []
    for (topo, t, rn), g in df.groupby(["topo", "t", "router"]):
        dl = g[g.delivered == 1]
        col = "P4" if rn in ("MP-SPF", "BMR-MP") else None
        rec = dict(topo=topo, t=t, router=rn, trials=len(g), delivered_pct=round(100 * g.delivered.mean(), 2),
                   mean_hops=round(dl.hops.mean(), 2) if len(dl) else float("nan"))
        if col:
            rec["exposed_pct_of_delivered"] = round(100 * dl[col].mean(), 2) if len(dl) else float("nan")
            rec["exposed_pct_of_all"] = round(100 * (dl[col].sum() / len(g)), 2)
        else:
            for c in ("P0", "P2"):
                rec[f"exposed_{c}_pct_of_delivered"] = round(100 * dl[c].mean(), 2) if len(dl) else float("nan")
        out.append(rec)
    o = pd.DataFrame(out); o.to_csv(os.path.join(OUT, "x2_summary.csv"), index=False)
    pd.set_option("display.width", 250); print(o.to_string())
    print("total", round(time.time() - t0), "s", platform.platform())


if __name__ == "__main__":
    main()
