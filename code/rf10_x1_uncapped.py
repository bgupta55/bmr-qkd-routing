"""
rf10_x1_uncapped.py - RF10: re-run the X1 (graph, s, d) cases whose disjoint-path family was capped at 1000 (43 cases, N=8 class),
now with ALL maximal families enumerated. Regenerates the same graphs and pairs as run_x1.py (same seeds), picks the 43 capped (gi, s, d)
from x1_out/x1_instances.csv, enumerates every maximal family of internally disjoint s-d paths, runs the GF(2) checker on each
(coalitions up to size 3) and reports instances, coalitions, disagreements with theory, Theorem-B violations.
Output: ../x1_out/rf10_uncapped.json
"""
import os, sys, re, json, random, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import pandas as pd, networkx as nx
import run_x1 as R
import symbolic_exposure as X


def main():
    t0 = time.time()
    d = pd.read_csv(os.path.join(os.path.dirname(HERE), "x1_out", "x1_instances.csv"), usecols=["protocol", "family", "fam_capped", "instance_id"])
    d = d[(d.protocol == "P4") & (d.fam_capped == 1)]
    cases = sorted({tuple(int(x) for x in re.match(r"g(\d+)_(\d+)_(\d+)_f", i).groups()) for i in d.instance_id})
    # regenerate N=8 graphs exactly as in run_x1.main (atlas graphs come first only in the job list, not in the RNG stream)
    rng = random.Random(R.seed_for("n8")); graphs = {}
    for i in range(5000):
        graphs[100000 + i] = list(R.random_connected(8, rng).edges())
    tot = dict(cases=len(cases), families=0, coalitions=0, exposed=0, disagreements=0, thmB_violations=0, exposed_min_size={}, max_family_count=0)
    for (gi, s, dd) in cases:
        G = nx.Graph(graphs[gi]); G.add_nodes_from(range(8))
        fams, capped = R.disjoint_families(G, s, dd, 10 ** 8)
        assert not capped
        tot["max_family_count"] = max(tot["max_family_count"], len(fams))
        for fam in fams:
            fam = [[("s" if x == s else "d" if x == dd else "n" + str(x)) for x in p] for p in fam]
            a = X.analyze(X.P4_xor_sharing(fam), 3)
            tot["families"] += 1; tot["coalitions"] += a["n_coalitions_checked"]; tot["exposed"] += a["n_exposed"]; tot["disagreements"] += a["disagreements"]
            n = a["n_paths"]
            tot["thmB_violations"] += (a["n_exposed_size1"] if n >= 2 else 0) + (a["n_exposed_size2"] if n >= 3 else 0) + (a["n_exposed_size3"] if n >= 4 else 0)
    tot["seconds"] = round(time.time() - t0, 1)
    json.dump(tot, open(os.path.join(os.path.dirname(HERE), "x1_out", "rf10_uncapped.json"), "w"), indent=1)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
