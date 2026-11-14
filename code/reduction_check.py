"""reduction_check.py - implementation check for the objective-only references B3, B6, B7 (blueprint L.11, Table 10b).

For random small graphs with link fidelities F_e and rates r_e (the attributes experiments/q6_common.py attaches), compare the router output with an
exhaustive search over all simple s-d paths:
  (i)   B6 (shortest path on -log F)            vs  exhaustive maximum of prod F
  (ii)  B7 (fewest hops with prod F >= F_TH)     vs  exhaustive minimum hop count among paths with prod F >= F_TH
  (iii) B3 (max-min rate, ties -> fewer hops)    vs  exhaustive maximum bottleneck rate (rates >= R_MIN)
Prints one table: check, instances, mismatches. It verifies the implementation of each reduction, not the behaviour of the original algorithms
in their own model.
Run:  python reduction_check.py
"""
import sys, os, math, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import networkx as nx
import q6_common as qc

rnd = random.Random(20261019)
TOL = 1e-12
cnt = {"B6 max prod F": [0, 0], "B7 min hops, prod F >= F_TH": [0, 0], "B3 max-min rate": [0, 0]}
skipped = 0


def paths(G, s, d):
    return list(nx.all_simple_paths(G, s, d))


for inst in range(3000):
    n = rnd.randint(4, 9)
    G = nx.gnp_random_graph(n, rnd.choice([0.35, 0.5, 0.7]), seed=rnd.randrange(10 ** 9))
    if not nx.is_connected(G):
        skipped += 1
        continue
    for u, v in G.edges():
        G[u][v]["F"] = rnd.uniform(0.80, 0.999)
        G[u][v]["r"] = rnd.choice([0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3])
    s, d = 0, n - 1
    P = paths(G, s, d)
    prod = lambda p: math.prod(G[a][b]["F"] for a, b in zip(p[:-1], p[1:]))
    bott = lambda p: min(G[a][b]["r"] for a, b in zip(p[:-1], p[1:]))
    # (i) B6
    p = qc.r_b6(G, s, d, 0.0)
    cnt["B6 max prod F"][0] += 1
    cnt["B6 max prod F"][1] += abs(prod(p) - max(prod(q) for q in P)) > 1e-9
    # (ii) B7
    ok = [q for q in P if prod(q) >= qc.F_TH]
    p = qc.r_b7(G, s, d, 0.0, cap=10 ** 6)
    cnt["B7 min hops, prod F >= F_TH"][0] += 1
    if not ok:
        cnt["B7 min hops, prod F >= F_TH"][1] += p is not None
    else:
        cnt["B7 min hops, prod F >= F_TH"][1] += (p is None) or (len(p) != min(len(q) for q in ok)) or prod(p) < qc.F_TH - 1e-12
    # (iii) B3 (links below R_MIN are unusable)
    use = [q for q in P if bott(q) >= qc.R_MIN]
    p = qc.r_b3(G, s, d, 0.0)
    cnt["B3 max-min rate"][0] += 1
    if not use:
        cnt["B3 max-min rate"][1] += p is not None
    else:
        best = max(bott(q) for q in use)
        cnt["B3 max-min rate"][1] += (p is None) or abs(bott(p) - best) > TOL

print(f"graphs: 3000 drawn, {skipped} disconnected and skipped")
print(f"{'check':34s} {'instances':>10s} {'mismatches':>11s}")
for k, (n, m) in cnt.items():
    print(f"{k:34s} {n:10d} {m:11d}")
sys.exit(1 if any(m for _, m in cnt.values()) else 0)
