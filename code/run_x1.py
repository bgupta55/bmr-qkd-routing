"""run_x1.py - X1 experiment driver: runs the GF(2) knowledge-closure checker over all instance classes.

Usage:  python run_x1.py [--quick]
Outputs (in ../x1_out/): x1_instances.csv (one row per instance and protocol), x1_summary.csv, x1_lemmaA.csv, x1_meta.json
Master seed 20261113; per-instance seeds derived as hash((master, class, idx)) via numpy SeedSequence.
"""
import os, sys, csv, json, time, itertools, random, platform
from multiprocessing import Pool
import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import symbolic_exposure as X
from q6_common import make_topology, load_sndlib

OUT = os.path.join(os.path.dirname(HERE), "x1_out")
os.makedirs(OUT, exist_ok=True)
MASTER = 20261113
MAX_COAL = 3
FAM_CAP = 1000          # cap on maximal disjoint families per (graph, s, d) in the exhaustive class (logged)
QUICK = "--quick" in sys.argv


def seed_for(*parts):
    return int(np.random.SeedSequence([MASTER] + [(p if isinstance(p, int) else __import__('zlib').crc32(str(p).encode())) for p in parts]).generate_state(1)[0])


def rows_for(inst, family, N, iid, extra=None):
    a = X.analyze(inst, MAX_COAL)
    a.update(family=family, N=N, instance_id=iid, s=str(inst.s), d=str(inst.d),
             m_or_npaths=(len(inst.paths[0]) - 1) if inst.name != "P4" else len(inst.paths))
    if extra:
        a.update(extra)
    return a


# --------------------------------------------------------------------------- class 1: simple paths m = 1..13
def class1(_):
    out = []
    for m in range(1, 14):
        p = ["s"] + [f"v{i}" for i in range(1, m)] + ["d"]
        for f in (X.P0_xor_chain, X.P1_chain_with_return, X.P2_trusted_forward, X.P3_public_xor):
            out.append(rows_for(f(p), "path", m + 1, f"path_m{m}"))
    return out


# --------------------------------------------------------------------------- class 2: small graphs, maximal disjoint families
def disjoint_families(G, s, d, cap):
    paths = [p for p in nx.all_simple_paths(G, s, d)]
    rel = [frozenset(p[1:-1]) for p in paths]
    H = nx.Graph()
    H.add_nodes_from(range(len(paths)))
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if not (rel[i] & rel[j]):
                H.add_edge(i, j)
    fams = []
    capped = False
    for cl in nx.find_cliques(H):
        fams.append([paths[i] for i in cl])
        if len(fams) >= cap:
            capped = True
            break
    return fams, capped


def class2_job(args):
    gi, edges, nnodes, sample_pairs = args
    G = nx.Graph(edges)
    G.add_nodes_from(range(nnodes))
    rng = random.Random(seed_for("c2", gi))
    pairs = [(a, b) for a in G for b in G if a != b]
    if sample_pairs and len(pairs) > sample_pairs:
        pairs = rng.sample(pairs, sample_pairs)
    out = []
    for (s, d) in pairs:
        fams, capped = disjoint_families(G, s, d, FAM_CAP)
        if len(fams) > FAM_CAP:
            fams = rng.sample(fams, FAM_CAP)
        kappa = nx.node_connectivity(G, s, d) if not G.has_edge(s, d) else None
        for fi, fam in enumerate(fams):
            fam = [list(map(str, p)) for p in fam]
            fam = [[("s" if x == str(s) else "d" if x == str(d) else "n" + x) for x in p] for p in fam]
            inst = X.P4_xor_sharing(fam)
            r = rows_for(inst, "graph_N%d" % nnodes, nnodes, f"g{gi}_{s}_{d}_f{fi}", dict(fam_capped=int(capped)))
            out.append(r)
    return out


def atlas_graphs(upto=7):
    from networkx.generators.atlas import graph_atlas_g
    for gi, G in enumerate(graph_atlas_g()):
        n = G.number_of_nodes()
        if 3 <= n <= upto and nx.is_connected(G):
            yield gi, list(G.edges()), n


def random_connected(n8, rng):
    while True:
        G = nx.gnp_random_graph(8, rng.choice([0.3, 0.4, 0.5, 0.6]), seed=rng.randrange(10**9))
        if nx.is_connected(G):
            return G


# --------------------------------------------------------------------------- class 3: realistic families
SND = os.path.join(HERE, "data", "topologies")


def make_graph(fam, N, seed):
    if fam in ("abilene", "polska", "nobel-germany", "geant", "germany50"):
        return load_sndlib(os.path.join(SND, fam + ".txt"))
    if fam == "grid":
        return nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 6))
    if fam == "ring":
        return nx.cycle_graph(N)
    for _ in range(100):
        G = make_topology(fam, N, seed)
        if nx.is_connected(G):
            return G
        seed += 1
    raise RuntimeError


def class3_job(args):
    fam, N, gi, t = args
    seed = seed_for("c3", fam, N, gi)
    G = make_graph(fam, N, seed)
    rng = random.Random(seed ^ 0x5A5A)
    dist = dict(nx.all_pairs_shortest_path_length(G))
    pairs = [(a, b) for a in G for b in G if a != b and dist[a][b] >= 2]
    if not pairs:
        return []
    s, d = rng.choice(pairs)
    out = []
    paths = list(nx.node_disjoint_paths(G, s, d)) if not G.has_edge(s, d) else [[s, d]]
    n_max = len(paths)
    take = paths[:t + 1]
    ren = lambda p: [("s" if x == s else "d" if x == d else "n%s" % x) for x in p]
    extra = dict(t=t, n_max_disjoint=n_max, kappa=nx.node_connectivity(G, s, d))
    if len(take) >= t + 1:
        out.append(rows_for(X.P4_xor_sharing([ren(p) for p in take]), "%s" % fam, G.number_of_nodes(), f"{fam}{N}_g{gi}_t{t}", extra))
    else:   # fewer than t+1 disjoint paths: Lemma A says a coalition of size <= t separates; check the max family anyway
        out.append(rows_for(X.P4_xor_sharing([ren(p) for p in take]), "%s" % fam, G.number_of_nodes(), f"{fam}{N}_g{gi}_t{t}_short", extra))
    if t == 1:   # single-path trusted forwarding on the shortest path (t=0 router style)
        sp = nx.shortest_path(G, s, d)
        out.append(rows_for(X.P2_trusted_forward(ren(sp)), "%s" % fam, G.number_of_nodes(), f"{fam}{N}_g{gi}_sp", extra))
    return out


# --------------------------------------------------------------------------- Lemma A random linear protocols
def lemmaA_job(args):
    gi, edges, n = args
    G = nx.Graph(edges)
    rng = random.Random(seed_for("lemA", gi))
    nodes = list(G)
    allp = [(a, b) for a in nodes for b in nodes if a != b]
    pairs = allp if n <= 5 else rng.sample(allp, 4)
    out = []
    for (s, d) in pairs:
        ren = {x: ("s" if x == s else "d" if x == d else "n%s" % x) for x in nodes}
        ed = [(ren[a], ren[b]) for a, b in edges]
        nsep, nchk, nviol = X.lemmaA_check(ed, "s", "d", steps=25, seed=seed_for("lemA2", gi, s, d), max_sep=3)
        mchk, mbad = X.maximal_flow_check(ed, "s", "d", None)
        out.append(dict(graph=gi, N=n, s=s, d=d, n_separators=nsep, n_checks=nchk, n_violations=nviol,
                        maxflow_checks=mchk, maxflow_violations=mbad,
                        kappa=(nx.node_connectivity(G, s, d) if not G.has_edge(s, d) else -1)))
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["protocol"], r["family"] if r["family"].startswith(("path", "graph")) else "realistic")].append(r)
    out = []
    for (proto, fam), rs in sorted(g.items()):
        mins = [r["min_exposing_coalition_size"] for r in rs if r["min_exposing_coalition_size"] != ""]
        out.append(dict(protocol=proto, instance_class=fam, instances=len(rs),
                        coalitions_checked=sum(r["n_coalitions_checked"] for r in rs),
                        coalitions_exposed=sum(r["n_exposed"] for r in rs),
                        s_computes_key=sum(r["s_knows"] for r in rs), d_computes_key=sum(r["d_knows"] for r in rs),
                        external_eavesdropper_exposes=sum(r["external_exposed"] for r in rs),
                        any_single_relay_exposes=sum(r["any_single_relay_exposes"] for r in rs),
                        never_exposed=sum(1 for r in rs if r["min_exposing_coalition_size"] == ""),
                        min_expos_size_min=(min(mins) if mins else ""), min_expos_size_median=(float(np.median(mins)) if mins else ""),
                        thmB_violations=sum((r['n_exposed_size1'] if r['n_paths']>=2 else 0) + (r['n_exposed_size2'] if r['n_paths']>=3 else 0) + (r['n_exposed_size3'] if r['n_paths']>=4 else 0) for r in rs) if proto=='P4' else '',
                        disagreements=sum(r["disagreements"] for r in rs)))
    return out


def main():
    t0 = time.time()
    rows = []
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        rows += class1(None)
        print("class1 done", len(rows), flush=True)
        # class 2: all connected graphs N<=7 (atlas); N=8: 5000 sampled graphs
        jobs = [(gi, e, n, 0) for gi, e, n in atlas_graphs(7 if not QUICK else 5)]
        n_atlas = len(jobs)
        n8 = 5000 if not QUICK else 50
        rng = random.Random(seed_for("n8"))
        for i in range(n8):
            G = random_connected(8, rng)
            jobs.append((100000 + i, list(G.edges()), 8, 6))      # N=8: 6 random ordered pairs per graph
        res = pool.map(class2_job, jobs, chunksize=4)
        c2 = [r for rs in res for r in rs]
        rows += c2
        print("class2 done", len(c2), "atlas graphs", n_atlas, "N=8 graphs", n8, "capped families:",
              sum(r.get("fam_capped", 0) for r in c2), flush=True)
        # class 3
        jobs3 = []
        K = 1000 if not QUICK else 20
        for fam in ("ER", "BA", "WS"):
            for N in (20, 30):
                for gi in range(K):
                    for t in (1, 2):
                        jobs3.append((fam, N, gi, t))
        for fam, N in (("grid", 30), ("ring", 12), ("ring", 30)):
            for gi in range(K if fam != "grid" else K):
                for t in (1, 2):
                    jobs3.append((fam, N, gi, t))
        for fam in ("abilene", "polska", "nobel-germany", "geant", "germany50"):
            for gi in range(K):
                for t in (1, 2):
                    jobs3.append((fam, 0, gi, t))
        res = pool.map(class3_job, jobs3, chunksize=16)
        c3 = [r for rs in res for r in rs]
        rows += c3
        print("class3 done", len(c3), flush=True)
        # Lemma A random linear protocols on small connected graphs (N<=6)
        jl = [(gi, e, n) for gi, e, n in atlas_graphs(6 if not QUICK else 4)]
        resL = pool.map(lemmaA_job, jl, chunksize=4)
        lem = [r for rs in resL for r in rs]
    write_csv(os.path.join(OUT, "x1_instances.csv"), rows)
    summ = summarize(rows)
    write_csv(os.path.join(OUT, "x1_summary.csv"), summ)
    write_csv(os.path.join(OUT, "x1_lemmaA.csv"), lem)
    meta = dict(master_seed=MASTER, max_coalition=MAX_COAL, fam_cap=FAM_CAP, n_rows=len(rows), seconds=round(time.time() - t0, 1),
                python=platform.python_version(), machine=platform.platform(), networkx=nx.__version__, quick=QUICK,
                lemmaA_checks=sum(r["n_checks"] for r in lem), lemmaA_violations=sum(r["n_violations"] for r in lem),
                lemmaA_runs=len(lem), maxflow_checks=sum(r['maxflow_checks'] for r in lem), maxflow_violations=sum(r['maxflow_violations'] for r in lem), lemmaA_separators=sum(r["n_separators"] for r in lem),
                total_disagreements=sum(r["disagreements"] for r in rows), total_coalitions=sum(r["n_coalitions_checked"] for r in rows))
    json.dump(meta, open(os.path.join(OUT, "x1_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    for r in summ:
        print(r)


if __name__ == "__main__":
    main()
