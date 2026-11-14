"""independence_check.py - are the graphs behind each reported cell independent draws?

Replays the graph generation of
  (a) experiments/q6_sweep.py  (Table 6): per cell, 10 graph seeds from a cell-specific generator, MASTER = 61;
  (b) run_pending_fast.py       (Table 7, RR1): a new graph for every trial, seeded by SeedSequence([20261012, trial, S_att*1000, Delta*1e6]).
For every cell it counts the repeated graphs (identical edge lists). Deterministic families (grid, ring, SNDlib files) repeat by construction and are
listed separately.  Expected for ER / BA / WS: 0 repeated graphs per cell.
Run:  python independence_check.py
"""
import sys, os, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import numpy as np
import networkx as nx
import q6_sweep as qs
from q6_common import make_topology


def h(edges):
    return hashlib.sha1(repr(sorted((min(a, b), max(a, b)) for a, b in edges)).encode()).hexdigest()


# ---------------------------------------------------------------- (a) Table 6
rng = np.random.default_rng(qs.MASTER)
specs = []
for topo in qs.SYN:
    for n in qs.SIZES:
        for na in (500, 10_000):
            for cond in ("clean", "compromised"):
                for mode in ("sealed", "public-XOR"):
                    specs.append((topo, n, na, cond, mode, int(rng.integers(1e9))))
for topo in qs.REAL:
    for na in (500, 10_000):
        for cond in ("clean", "compromised"):
            for mode in ("sealed", "public-XOR"):
                specs.append((topo, 0, na, cond, mode, int(rng.integers(1e9))))

rep_cells, det = [], 0
total_rand_cells = 0
seed_dups = len(specs) - len({s[5] for s in specs})
for topo, n, na, cond, mode, seed in specs:
    if topo in qs.REAL or topo in ("grid", "ring"):
        det += 1
        continue
    r = np.random.default_rng(seed); hs = []
    while len(hs) < qs.GRAPH_SEEDS:
        G = make_topology(topo, n, int(r.integers(1e9)))
        if nx.is_connected(G):
            hs.append(h(G.edges()))
    total_rand_cells += 1
    rep = len(hs) - len(set(hs))
    if rep:
        rep_cells.append((topo, n, na, cond, mode, rep))
print("Table 6 (q6_sweep.py, MASTER = 61)")
print(f"  cells: {len(specs)}; random-graph cells (ER/BA/WS): {total_rand_cells}; deterministic cells (grid, ring, SNDlib): {det}")
print(f"  repeated cell seeds: {seed_dups}")
print(f"  random-graph cells with a repeated graph inside the cell: {len(rep_cells)}")
for c in rep_cells:
    print("    ", c)
if rep_cells:
    by_topo = {}
    for c in rep_cells:
        by_topo[(c[0], c[1])] = by_topo.get((c[0], c[1]), 0) + 1
    print("  by (topology, n):", by_topo)

# ---------------------------------------------------------------- (b) Table 7 (RR1)
N_NODES, ER_P, N_TRIALS = 30, 4.0 / 29, 200


def er(n, p, g):
    iu, ju = np.triu_indices(n, 1)
    keep = g.random(len(iu)) < p
    return list(zip(iu[keep].tolist(), ju[keep].tolist()))


def ba(n, m, g):
    G = nx.barabasi_albert_graph(n, m, seed=int(g.integers(2**31)))
    return sorted((min(a, b), max(a, b)) for a, b in G.edges())


print("\nTable 7 (RR1, run_pending_fast.py): 200 trials per cell, new graph per trial")
tot_rep, allh = 0, {}
for topo in ("er", "ba"):
    for S in (2.418, 2.55, 2.62):
        for D in (1e-3, 3e-5):
            hs = []
            for trial in range(N_TRIALS):
                g = np.random.default_rng(np.random.SeedSequence([20261012, trial, int(S * 1000), int(D * 1e6)]))
                e = er(N_NODES, ER_P, g) if topo == "er" else ba(N_NODES, 2, g)
                hs.append(h(e))
            rep = len(hs) - len(set(hs)); tot_rep += rep
            for x in hs:
                allh[x] = allh.get(x, 0) + 1
            print(f"  {topo} S={S} D={D:g}: repeated graphs = {rep}")
print(f"  total repeats inside cells: {tot_rep}; graphs shared between the cells of one family: "
      f"{sum(v - 1 for v in allh.values())}")
