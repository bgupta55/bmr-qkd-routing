"""
e32_wide.py - RF3: the wider detour-rich study the plan calls E3.2 (+ E3.4 wrong baseline), single-path routers.

Topologies (N=30 unless stated): ER with mean degree 4, 6, 8 (fresh graph per repetition); grid 5x6 and 7x7; the five SNDlib graphs
(abilene, polska, nobel-germany, geant, germany50; fixed graph, fresh attack set / pair / audits per repetition).
Attacked fraction f in {0.1, 0.2, 0.3}; n_audit in {1e4, 2e4}; S_dagger in {2.55, 2.62}; Delta = 1e-3.  => 10 x 3 x 2 x 2 = 120 cells.
Per cell: 100 repetitions x 10 epochs (attack on in all epochs, as in the Table 8 window); (s,d) = random pair at hop distance >= 2.
Both routers see the SAME certificate draw in every epoch. For each of two hop floors (0 = revised design, 3 = R1 router) the epoch is routed by
   live   : BMR on all certified links
   alarm  : BMR after removing links with S_hat < S0 - z*sigma_W(S0,n), S0 = 2.687, z = 4.7534 (alpha = 1e-6)
   wrong  : same, with the baseline S0' = S_dagger (E3.4: commissioned in the degraded state)
true-secure rate = certified bottleneck if the route contains no attacked link, else 0 (as in Table 8).
Diagnostics (ground truth, not used by any router): does a clean feasible route exist; its bottleneck; category a-f as in E3.1.
Output: ../e32w_out/e32w_cells.csv (one row per cell and floor), e32w_rows.csv.gz, e32w_meta.json. Statistics: bootstrap over repetitions (2000).
"""
import sys, os, json, time, math, itertools, platform, gzip, csv
import numpy as np, networkx as nx
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.sim import LinkSim, S_HONEST
from bmr.route import bmr_route
from q6_common import attach_certs, load_sndlib, R_MIN

HETMODE = os.environ.get("E32W_HETERO", "0")      # "0" homogeneous; "1" heterogeneous, attacked links keep the cell n; "2" heterogeneous, attacked links drawn like the others
HET = HETMODE in ("1", "2")
OUT = os.path.join(os.path.dirname(HERE), {"0": "e32w_out", "1": "e32w_het_out", "2": "e32w_het2_out"}[HETMODE]); os.makedirs(OUT, exist_ok=True)
N_CHOICES = [1500, 2500, 5000, 10_000, 20_000]   # HETERO: each link audited with n drawn uniformly from this set (fixed for the repetition)
MASTER = 20261113
Z = 4.7534
DELTA = 1e-3
REPS = int(os.environ.get("E32W_REPS", 100)); EPOCHS = 10
TOPOS = ["ER4", "ER6", "ER8", "grid5x6", "grid7x7", "abilene", "polska", "nobel-germany", "geant", "germany50"]
FRACS = [0.1, 0.2, 0.3]; NS = [10_000, 20_000]; SDAG = [2.55, 2.62]; FLOORS = [0, 3]
SND = os.path.join(HERE, "data", "topologies")


class HetSim(LinkSim):
    """LinkSim with a per-link audit size (heterogeneous audit effort)."""
    def __init__(self, links, link_n, seed=0):
        super().__init__(links, 10_000, seed=seed); self.link_n = link_n

    def observe(self, link):
        k = self._key(link); S = self.true_S[k]; n = self.link_n[k]
        s_hat = float(self.rng.normal(S, math.sqrt(max(0.0, 4.0 - S * S / 4.0) / n))); self.last_S_hat[k] = s_hat
        return s_hat

    def publish_certificate(self, link, epoch):
        from bmr.certificate import build_certificate
        k = self._key(link); s_hat = self.observe(k)
        cert = build_certificate(k, s_hat, self.link_n[k], self.delta, self.ell, self.phi, self.lam_ec, epoch)
        self.published[k] = cert; return cert


def sig_w(S0, n): return math.sqrt(max(0.0, 4.0 - S0 * S0 / 4.0) / n)


def graph_for(topo, rng):
    if topo.startswith("ER"):
        deg = int(topo[2:])
        while True:
            G = nx.erdos_renyi_graph(30, deg / 29, seed=int(rng.integers(1e9)))
            if nx.is_connected(G): return G
    if topo == "grid5x6": return nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 6))
    if topo == "grid7x7": return nx.convert_node_labels_to_integers(nx.grid_2d_graph(7, 7))
    G = load_sndlib(os.path.join(SND, topo + ".txt"))
    return G


def route(G, s, d, m_min, S0=None):
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        if S0 is not None and a["S_hat"] < S0 - Z * sig_w(S0, a["cert"].n if a.get("cert") is not None else 1e9):
            continue
        if a["cert"] is not None and a["r"] >= R_MIN: H.add_edge(u, v, **a)
    if s not in H or d not in H: return None
    res = bmr_route(H, s, d, R_MIN, DELTA, 0, 10, m_min=m_min)
    return res[1] if res else None


def has_att(path, att): return path is not None and any(frozenset((a, b)) in att for a, b in zip(path[:-1], path[1:]))
def bott(G, path): return min(G[a][b]["r"] for a, b in zip(path[:-1], path[1:])) if path else 0.0


def category(live, alarm, la, aa):
    if live == alarm: return "a_same_route"
    if live is not None and alarm is None: return "b_alarm_drops_delivery"
    if la and alarm is not None and not aa: return "c_reroute_to_clean"
    if la and alarm is not None and aa: return "d_reroute_still_attacked"
    if live is not None and not la and alarm is not None: return "e_leaves_clean_for_attacked" if aa else "e1_clean_to_clean"
    return "f_other"


def run_cell(spec):
    topo, f, n, Sd, seed = spec
    rng = np.random.default_rng(seed); rows = []
    G0 = None
    for rep in range(REPS):
        if G0 is None or topo.startswith("ER"): G0 = graph_for(topo, rng)
        dist = dict(nx.all_pairs_shortest_path_length(G0))
        pairs = [(a, b) for a in G0 for b in G0 if a < b and dist[a].get(b, 0) >= 2]
        s, d = pairs[int(rng.integers(len(pairs)))]
        edges = list(G0.edges()); k = max(1, round(f * len(edges)))
        att_edges = [edges[i] for i in rng.permutation(len(edges))[:k]]; att = {frozenset(e) for e in att_edges}
        if HET:
            link_n = {LinkSim._key(e): int(N_CHOICES[int(rng.integers(len(N_CHOICES)))]) for e in edges}
            if HETMODE == "1":
                for e in att_edges: link_n[LinkSim._key(e)] = n            # attacked links keep the cell's audit size
            sim = HetSim(edges, link_n, seed=int(rng.integers(1e9)))
        else:
            link_n = None
            sim = LinkSim(edges, n, delta=1e-6, ell=10_000, phi=0.99, seed=int(rng.integers(1e9)))
        for e in att_edges: sim.set_true_S(e, Sd)
        G = G0.copy()
        for ep in range(EPOCHS):
            attach_certs(G, sim, epoch=ep)
            Gc = G.copy(); Gc.remove_edges_from([tuple(e) for e in att])
            fl_att = fl_hon = 0
            for u, v, a in G.edges(data=True):
                if a["cert"] is not None and a["S_hat"] < S_HONEST - Z * sig_w(S_HONEST, a["cert"].n):
                    if frozenset((u, v)) in att: fl_att += 1
                    else: fl_hon += 1
            r = dict(topo=topo, f=f, n=n, Sd=Sd, rep=rep, epoch=ep, s=s, d=d, hops_sd=dist[s][d], n_att=len(att), n_flag_att=fl_att, n_flag_hon=fl_hon)
            for fl in FLOORS:
                pl, pa, pw = route(G, s, d, fl), route(G, s, d, fl, S_HONEST), route(G, s, d, fl, Sd)
                la, aa, wa = has_att(pl, att), has_att(pa, att), has_att(pw, att)
                pc = route(Gc, s, d, fl) if (s in Gc and d in Gc) else None
                x = f"_F{fl}"
                r.update({"live_deliv" + x: int(pl is not None), "alarm_deliv" + x: int(pa is not None), "wrong_deliv" + x: int(pw is not None),
                          "live_att" + x: int(la), "alarm_att" + x: int(aa), "wrong_att" + x: int(wa),
                          "true_live" + x: 0.0 if (pl is None or la) else bott(G, pl),
                          "true_alarm" + x: 0.0 if (pa is None or aa) else bott(G, pa),
                          "true_wrong" + x: 0.0 if (pw is None or wa) else bott(G, pw),
                          "clean_exists" + x: int(pc is not None), "clean_rate" + x: bott(G, pc) if pc else 0.0,
                          "cat" + x: category(pl, pa, la, aa)})
            rows.append(r)
    return rows


def boot_ci(x, B=2000, seed=11):
    rng = np.random.default_rng(seed); x = np.asarray(x); m = len(x)
    bs = x[rng.integers(0, m, (B, m))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    t0 = time.time(); rng = np.random.default_rng(MASTER)
    specs = [(tp, f, n, Sd, int(rng.integers(1e9))) for tp, f, n, Sd in itertools.product(TOPOS, FRACS, NS, SDAG)]
    rows = []
    with Pool(max(1, os.cpu_count() - 1)) as p:
        for i, rs in enumerate(p.imap(run_cell, specs)):
            rows += rs
            if (i + 1) % 10 == 0: print(f"cell {i+1}/{len(specs)} ({time.time()-t0:.0f}s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e32w_rows.csv.gz"), index=False)
    cells = []
    for (tp, f, n, Sd), g in df.groupby(["topo", "f", "n", "Sd"]):
        for fl in FLOORS:
            x = f"_F{fl}"
            rep_diff = (g["true_alarm" + x] - g["true_live" + x]).groupby(g.rep).mean()
            rep_wrong = (g["true_wrong" + x] - g["true_live" + x]).groupby(g.rep).mean()
            lo, hi = boot_ci(rep_diff.values); wlo, whi = boot_ci(rep_wrong.values)
            la = g["live_att" + x] == 1
            c = dict(topo=tp, f=f, n=n, Sd=Sd, floor=fl, trials=len(g), reps=g.rep.nunique(),
                     true_live=g["true_live" + x].mean(), true_alarm=g["true_alarm" + x].mean(), diff=rep_diff.mean(), diff_lo=lo, diff_hi=hi,
                     diff_wrong=rep_wrong.mean(), wrong_lo=wlo, wrong_hi=whi,
                     live_deliv=g["live_deliv" + x].mean(), alarm_deliv=g["alarm_deliv" + x].mean(),
                     live_att_of_deliv=(g["live_att" + x].sum() / max(1, g["live_deliv" + x].sum())),
                     alarm_att_of_deliv=(g["alarm_att" + x].sum() / max(1, g["alarm_deliv" + x].sum())),
                     live_att_clean_exists=(la & (g["clean_exists" + x] == 1)).sum(), live_att_total=int(la.sum()),
                     alarm_power=g.n_flag_att.sum() / g.n_att.sum(), false_flags=int(g.n_flag_hon.sum()))
            for cat, cnt in g["cat" + x].value_counts().items(): c["cat_" + cat] = int(cnt)
            cells.append(c)
    cd = pd.DataFrame(cells).fillna(0); cd.to_csv(os.path.join(OUT, "e32w_cells.csv"), index=False)
    json.dump(dict(seed=MASTER, reps=REPS, epochs=EPOCHS, cells=len(specs), seconds=round(time.time() - t0, 1), python=platform.python_version(),
                   machine=platform.platform()), open(os.path.join(OUT, "e32w_meta.json"), "w"), indent=1)
    print("done", len(df), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
