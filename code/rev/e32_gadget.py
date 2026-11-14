"""
e32_gadget.py - E3.2 (always-run gadget): can the alarm RAISE the true-secure rate when a clean alternative route exists?

Graph: s and d joined by two 3-link routes (3 = hop floor, so both routes are admissible).
  Route A (s-a1-a2-d): all links honest (n=n_A audits, S=S0) except ONE attacked link (true S = S_dagger, n=n_A audits).
  Route B (s-b1-b2-d): clean; one link audited with n_B samples to set its certified rate (the bottleneck b); the others n=10^4.
Both routers see the same certificates per epoch. BMR-live picks the route with the larger certified bottleneck;
BMR+alarm first removes links with S_hat < S0 - z*sigma_W (alpha=1e-6, z=4.7534), then does the same.
true-secure rate of a route = its certified bottleneck if no attacked link is on it, else 0.
Window: 10 epochs (as in Table 8b), 1000 trials per cell, each epoch draws fresh audits. Delta = 1e-3.
Extra column: alarm_wrongS0 = alarm with a baseline S0' set to the degraded value (E3.4-style "commissioned in a compromised state").
No parameter is tuned; cells are fixed in advance.
"""
import sys, os, math, json, csv, time, itertools, platform
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from bmr.certificate import build_certificate
from bmr.route import bmr_route
from bmr.sim import S_HONEST
from q6_common import M_FLOOR, R_MIN

OUT = os.path.join(os.path.dirname(HERE), "e32_out"); os.makedirs(OUT, exist_ok=True)
Z = 4.7534
DELTA = 1e-3
TRIALS, EPOCHS = 1000, 10
SEED = 20261113
NB_LIST = [1500, 2500, 5000, 10000]           # -> certified bottleneck b of route B ~ 0.20, 0.246, 0.298, 0.339
CELLS = list(itertools.product([10_000, 20_000], [2.62, 2.55, 2.48], NB_LIST))


def sig_w(S0, n): return math.sqrt(max(0.0, 4.0 - S0 * S0 / 4.0) / n)


def draw_link(rng, S_true, n):
    s_hat = float(rng.normal(S_true, sig_w(S_true, n)))
    c = build_certificate((0, 1), s_hat, n, 1e-6, 10_000, 0.99, 0, 0)
    return c, s_hat


def build(rng, nA, Sdag, nB):
    G = nx.Graph()
    # (u, v, S_true, n, attacked)
    spec = [("s", "a1", S_HONEST, nA, 0), ("a1", "a2", Sdag, nA, 1), ("a2", "d", S_HONEST, nA, 0),
            ("s", "b1", S_HONEST, 10_000, 0), ("b1", "b2", S_HONEST, nB, 0), ("b2", "d", S_HONEST, 10_000, 0)]
    for u, v, St, n, att in spec:
        c, sh = draw_link(rng, St, n)
        if c is None:
            continue
        G.add_edge(u, v, cert=c, S_hat=sh, r=c.r, w=c.w, eps=c.eps, T=0, att=att, n=n)
    return G


def route(G, S0=None):
    H = nx.Graph()
    for u, v, a in G.edges(data=True):
        if S0 is not None and a["S_hat"] < S0 - Z * sig_w(S0, a["n"]):
            continue
        if a["r"] >= R_MIN:
            H.add_edge(u, v, **a)
    if "s" not in H or "d" not in H:
        return None
    res = bmr_route(H, "s", "d", R_MIN, DELTA, 0, 10, m_min=M_FLOOR)
    return res[1] if res else None


def true_rate(G, path):
    if path is None: return 0.0
    es = [G[u][v] for u, v in zip(path[:-1], path[1:])]
    if any(e["att"] for e in es): return 0.0
    return min(e["r"] for e in es)


def main():
    t0 = time.time()
    master = np.random.default_rng(SEED)
    rows, cellrows = [], []
    for (nA, Sdag, nB) in CELLS:
        rng = np.random.default_rng(int(master.integers(1e9)))
        tl, ta, tw, d_ = [], [], [], []
        cnt = dict(same=0, alarm_to_clean=0, drop=0, other=0)
        flagged = 0
        b_nom = build_certificate((0, 1), S_HONEST, nB, 1e-6, 10_000, 0.99, 0, 0).r
        a_nom = build_certificate((0, 1), Sdag, nA, 1e-6, 10_000, 0.99, 0, 0).r
        for tr in range(TRIALS):
            sl = sa = sw = 0.0
            for ep in range(EPOCHS):
                G = build(rng, nA, Sdag, nB)
                pl, pa, pw = route(G), route(G, S_HONEST), route(G, Sdag)     # wrong-baseline S0' = S_dagger
                rl, ra, rw = true_rate(G, pl), true_rate(G, pa), true_rate(G, pw)
                sl += rl; sa += ra; sw += rw
                att_link = G["a1"]["a2"] if G.has_edge("a1", "a2") else None
                flagged += int(att_link is not None and att_link["S_hat"] < S_HONEST - Z * sig_w(S_HONEST, nA))
                if pl == pa: cnt["same"] += 1
                elif pl is not None and pa is not None and ra > rl: cnt["alarm_to_clean"] += 1
                elif pl is not None and pa is None: cnt["drop"] += 1
                else: cnt["other"] += 1
                rows.append((nA, Sdag, nB, tr, ep, "-".join(pl) if pl else "", "-".join(pa) if pa else "",
                             "-".join(pw) if pw else "", round(rl, 6), round(ra, 6), round(rw, 6)))
            tl.append(sl / EPOCHS); ta.append(sa / EPOCHS); tw.append(sw / EPOCHS); d_.append((sa - sl) / EPOCHS)
        d_ = np.array(d_); tl = np.array(tl); ta = np.array(ta); tw = np.array(tw)
        se = d_.std(ddof=1) / math.sqrt(TRIALS)
        cellrows.append(dict(n_A=nA, S_dagger=Sdag, A_cert_rate=round(a_nom, 4), n_B=nB, b_cert_rate=round(b_nom, 4),
                             pred_alarm_helps=int(b_nom < a_nom),
                             alarm_power=round(flagged / (TRIALS * EPOCHS), 4),
                             true_live=round(tl.mean(), 4), true_alarm=round(ta.mean(), 4),
                             diff=round(d_.mean(), 4), diff_lo=round(d_.mean() - 1.96 * se, 4), diff_hi=round(d_.mean() + 1.96 * se, 4),
                             true_alarm_wrongS0=round(tw.mean(), 4), diff_wrongS0=round((tw - tl).mean(), 4),
                             epochs_same=cnt["same"], epochs_alarm_to_clean=cnt["alarm_to_clean"],
                             epochs_alarm_drops=cnt["drop"], epochs_other=cnt["other"]))
        print(cellrows[-1], flush=True)
    with open(os.path.join(OUT, "e32_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cellrows[0].keys())); w.writeheader(); w.writerows(cellrows)
    with open(os.path.join(OUT, "e32_rows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("n_A S_dagger n_B trial epoch live_path alarm_path wrongS0_path true_live true_alarm true_wrongS0".split())
        w.writerows(rows)
    json.dump(dict(seed=SEED, trials=TRIALS, epochs=EPOCHS, delta=DELTA, seconds=round(time.time() - t0, 1),
                   python=platform.python_version(), machine=platform.platform()),
              open(os.path.join(OUT, "e32_meta.json"), "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
