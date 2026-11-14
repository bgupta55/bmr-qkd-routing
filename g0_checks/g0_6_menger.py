"""G0.6 - Menger cross-check.

5,000 random connected graphs (ER 2,000 / BA 1,500 / grid-with-chords 1,500; N = 8..12). Per graph: up to 8 random non-adjacent
ordered (s,d) pairs and up to 2 adjacent pairs. For non-adjacent pairs compute
  kappa    = nx.node_connectivity(G, s, d)
  maxflow  = max flow of an own vertex-split unit-capacity network
  nu       = number of internally vertex-disjoint s-d paths returned by nx.node_disjoint_paths
  cut_bf   = smallest relay set whose removal disconnects s,d (brute force over subsets; independent of any flow code)
  e_min    = smallest coalition size that exposes the key of P4 run on that maximum family of disjoint paths, found with the GF(2)
             knowledge-closure checker of code/symbolic_exposure.py (coalitions enumerated up to size nu, so e_min is not truncated)
Expected: kappa == maxflow == nu == cut_bf == e_min.
Adjacent pairs: P4 on the direct link alone, and on the direct link plus a maximum family of internally disjoint paths of G - {s,d edge}:
no coalition of size <= 3 exposes the key.
Seed 20261113.  Output: out/g0_6_menger.json
"""
import os, sys, json, time, itertools, random
from collections import Counter
from multiprocessing import Pool
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, "..", "..", "code"))
sys.path.insert(0, CODE)
import symbolic_exposure as X

SEED = 20261113
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def maxflow_vertex_split(G, s, d):
    D = nx.DiGraph()
    for v in G:
        D.add_edge((v, "in"), (v, "out"), capacity=(1 if v not in (s, d) else 10 ** 6))
    for u, v in G.edges():
        if {u, v} == {s, d}:
            continue
        D.add_edge((u, "out"), (v, "in"), capacity=10 ** 6)
        D.add_edge((v, "out"), (u, "in"), capacity=10 ** 6)
    return nx.maximum_flow_value(D, (s, "out"), (d, "in"))


def brute_cut(G, s, d, kmax):
    others = [v for v in G if v not in (s, d)]
    for k in range(0, kmax + 1):
        for C in itertools.combinations(others, k):
            H = G.copy()
            H.remove_nodes_from(C)
            if not nx.has_path(H, s, d):
                return k
    return None


def ren(p, s, d):
    return ["s" if x == s else "d" if x == d else "n%s" % x for x in p]


def make_graph(kind, gi):
    rng = random.Random(SEED * 7 + gi)
    n = rng.randint(8, 12)
    if kind == "ER":
        while True:
            G = nx.gnp_random_graph(n, rng.choice([0.25, 0.3, 0.4, 0.5, 0.6]), seed=rng.randrange(10 ** 9))
            if nx.is_connected(G):
                return G
    if kind == "BA":
        return nx.barabasi_albert_graph(n, rng.choice([1, 2, 3]), seed=rng.randrange(10 ** 9))
    r, c = rng.choice([(2, 4), (2, 5), (2, 6), (3, 3), (3, 4), (2, 3)])
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(r, c))
    nodes = list(G)
    for _ in range(rng.randint(0, 3)):
        a, b = rng.sample(nodes, 2)
        G.add_edge(a, b)
    return G


def job(args):
    kind, gi = args
    G = make_graph(kind, gi)
    rng = random.Random(SEED * 11 + gi)
    nodes = list(G)
    non_adj = [(a, b) for a in nodes for b in nodes if a != b and not G.has_edge(a, b)]
    adj = list(G.edges())
    rows = []
    for (s, d) in rng.sample(non_adj, min(8, len(non_adj))):
        kappa = nx.node_connectivity(G, s, d)
        mf = maxflow_vertex_split(G, s, d)
        paths = list(nx.node_disjoint_paths(G, s, d))
        nu = len(paths)
        cut = brute_cut(G, s, d, kappa)
        a = X.analyze(X.P4_xor_sharing([ren(p, s, d) for p in paths]), max_coal=max(1, nu))
        e_min = a["min_exposing_coalition_size"]
        e_min = -1 if e_min == "" else e_min
        rows.append(dict(kind=kind, graph=gi, adjacent=0, kappa=kappa, maxflow=mf, nu=nu, cut_bf=cut, e_min=e_min,
                         disagreements=a["disagreements"],
                         ok=int(kappa == mf == nu == cut == e_min and a["disagreements"] == 0)))
    for (s, d) in rng.sample(adj, min(2, len(adj))):
        H = G.copy()
        H.remove_edge(s, d)
        others = list(nx.node_disjoint_paths(H, s, d)) if nx.has_path(H, s, d) else []
        a1 = X.analyze(X.P4_xor_sharing([["s", "d"]]), max_coal=3)
        a2 = X.analyze(X.P4_xor_sharing([["s", "d"]] + [ren(p, s, d) for p in others]), max_coal=3)
        rows.append(dict(kind=kind, graph=gi, adjacent=1, extra_paths=len(others),
                         exposed_alone=a1["n_exposed"], exposed_with_others=a2["n_exposed"],
                         coalitions_checked=a1["n_coalitions_checked"] + a2["n_coalitions_checked"],
                         disagreements=a1["disagreements"] + a2["disagreements"],
                         ok=int(a1["n_exposed"] == 0 and a2["n_exposed"] == 0 and a1["disagreements"] + a2["disagreements"] == 0)))
    return rows


def main():
    t0 = time.time()
    jobs = [("ER", i) for i in range(2000)] + [("BA", 2000 + i) for i in range(1500)] + [("GRID", 3500 + i) for i in range(1500)]
    with Pool(os.cpu_count() - 1) as pool:
        res = pool.map(job, jobs, chunksize=20)
    rows = [r for rs in res for r in rs]
    na = [r for r in rows if not r["adjacent"]]
    ad = [r for r in rows if r["adjacent"]]
    kd = Counter(r["kappa"] for r in na)
    summary = dict(seed=SEED, graphs=len(jobs), nonadjacent_pairs=len(na),
                   nonadjacent_mismatches=sum(1 - r["ok"] for r in na),
                   kappa_ne_maxflow=sum(r["kappa"] != r["maxflow"] for r in na),
                   kappa_ne_nu=sum(r["kappa"] != r["nu"] for r in na),
                   kappa_ne_bruteforce_cut=sum(r["kappa"] != r["cut_bf"] for r in na),
                   kappa_ne_emin=sum(r["kappa"] != r["e_min"] for r in na),
                   checker_disagreements=sum(r["disagreements"] for r in rows),
                   kappa_histogram={int(k): v for k, v in sorted(kd.items())},
                   adjacent_pairs=len(ad), adjacent_mismatches=sum(1 - r["ok"] for r in ad),
                   adjacent_with_extra_paths=sum(1 for r in ad if r["extra_paths"] > 0),
                   adjacent_coalitions_checked=sum(r["coalitions_checked"] for r in ad),
                   by_kind={k: dict(pairs=sum(1 for r in na if r["kind"] == k),
                                    mismatches=sum(1 - r["ok"] for r in na if r["kind"] == k)) for k in ("ER", "BA", "GRID")},
                   seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(OUT, "g0_6_menger.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
