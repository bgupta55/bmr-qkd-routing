"""G0.1 - Lemma A beyond linear protocols: exact enumeration on small graphs.

For each (topology, coalition) pair we draw 20,000 random non-linear interactive protocols (4 rounds, 1-bit public
messages, sender = random function of the transcript, message = random truth table of (sender, local state, transcript),
final 2-bit key of s and d = random function of (local state, full transcript)). For every protocol the full input space
(2 bits per node randomness, 2 bits per link key, all uniform and independent) is enumerated and the following are
computed EXACTLY from the counts:
  I(X_A ; X_B | V_C)   X_A = (R_A, internal keys of A), X_B = (R_B, internal keys of B), V_C = (local states of C, T)
  I(K_s ; K_d | V_C)
  p_agree = Pr[K_s = K_d],  E_max = E_v[max_k Pr[K_s = k | v]]   (p_agree <= E_max must hold when C separates)
Negative control: triangle s-r-d plus direct link (s,d), C = {r}; C does not separate, so the lemma does not apply.
Because s and d share K_sd there, the control splits the endpoint variables as X_A = (R_s, K_sd), X_B = R_d.

Seed 20261113 (SeedSequence([seed, 1, pair_index, protocol_index]) per protocol).  Output: out/g0_1_summary.json
"""
import json, os, sys, time, itertools
from multiprocessing import Pool
import numpy as np

SEED = 20261113
NPROTO = int(os.environ.get("G01_NPROTO", 20000))
ROUNDS = 4
TOL = 1e-12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

TOPOS = {
    "path3": dict(nodes=["s", "r", "d"], links=[("s", "r"), ("r", "d")], C=[["r"]]),
    "path4": dict(nodes=["s", "a", "b", "d"], links=[("s", "a"), ("a", "b"), ("b", "d")], C=[["a"], ["b"], ["a", "b"]]),
    "diamond": dict(nodes=["s", "a", "b", "d"], links=[("s", "a"), ("a", "d"), ("s", "b"), ("b", "d")], C=[["a", "b"]]),
    "triangle_control": dict(nodes=["s", "r", "d"], links=[("s", "r"), ("r", "d"), ("s", "d")], C=[["r"]]),
}


def entropy_ids(ids):
    _, c = np.unique(ids, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def cmi(a, b, v):
    """I(A;B|V) in bits from integer id arrays (exact counts, equal weights)."""
    nb = int(b.max()) + 1
    nv = int(v.max()) + 1
    av = a * nv + v
    bv = b * nv + v
    abv = (a * nb + b) * nv + v
    return entropy_ids(av) + entropy_ids(bv) - entropy_ids(abv) - entropy_ids(v)


def comb(arrs):
    """Combine list of small-int arrays (values 0..3) into one id array."""
    out = np.zeros_like(arrs[0], dtype=np.int64) if arrs else None
    for x in arrs:
        out = out * 4 + x
    return out


class Setup:
    def __init__(self, name):
        t = TOPOS[name]
        self.name = name
        self.nodes = t["nodes"]
        self.links = [tuple(e) for e in t["links"]]
        self.nn = len(self.nodes)
        self.vars = [("R", v) for v in self.nodes] + [("K", e) for e in self.links]
        nv = len(self.vars)
        self.N = 4 ** nv
        idx = np.arange(self.N, dtype=np.int64)
        self.val = {var: (idx // (4 ** i)) % 4 for i, var in enumerate(self.vars)}
        # local state index of every node over the whole input space
        self.lidx, self.nst = {}, {}
        for v in self.nodes:
            comp = [self.val[("R", v)]] + [self.val[("K", e)] for e in self.links if v in e]
            self.lidx[v] = comb(comp)
            self.nst[v] = 4 ** len(comp)

    def partition(self, C, mode):
        """Return (X_A id, X_B id, Z id) arrays."""
        C = set(C)
        H = self.links
        if mode == "separating":
            # A = nodes reachable from s in H - C
            A = {"s"}
            ch = True
            while ch:
                ch = False
                for (u, w) in H:
                    if u in C or w in C:
                        continue
                    if (u in A) != (w in A):
                        A |= {u, w}
                        ch = True
            assert "d" not in A, "C does not separate s and d"
            B = set(self.nodes) - C - A
            xa = [self.val[("R", v)] for v in sorted(A)] + [self.val[("K", e)] for e in H if e[0] in A and e[1] in A]
            xb = [self.val[("R", v)] for v in sorted(B)] + [self.val[("K", e)] for e in H if e[0] in B and e[1] in B]
            for e in H:   # every link: both ends in A, both in B, or an end in C
                assert (e[0] in A and e[1] in A) or (e[0] in B and e[1] in B) or e[0] in C or e[1] in C
            z = [self.val[("R", v)] for v in sorted(C)] + [self.val[("K", e)] for e in H if e[0] in C or e[1] in C]
            return comb(xa), comb(xb), comb(z)
        else:  # control: X_A = (R_s, K_sd), X_B = R_d, Z = C-variables (R_r, K_sr, K_rd)
            xa = [self.val[("R", "s")], self.val[("K", ("s", "d"))]]
            xb = [self.val[("R", "d")]]
            z = [self.val[("R", v)] for v in sorted(C)] + [self.val[("K", e)] for e in H if e[0] in C or e[1] in C]
            return comb(xa), comb(xb), comb(z)


_SETUPS = {}


def get_setup(name):
    if name not in _SETUPS:
        _SETUPS[name] = Setup(name)
    return _SETUPS[name]


def one_protocol(args):
    name, C, mode, pair_idx, k = args
    S = get_setup(name)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, pair_idx, k]))
    nn = S.nn
    # ---- random protocol: all tables drawn up-front (equivalent to lazy filling and cached per protocol)
    sender_tab = [rng.integers(0, nn, size=2 ** r) for r in range(ROUNDS)]
    msg_tab = [[rng.integers(0, 2, size=(S.nst[v], 2 ** r)) for v in S.nodes] for r in range(ROUNDS)]
    key_tab = {v: rng.integers(0, 4, size=(S.nst[v], 2 ** ROUNDS)) for v in ("s", "d")}
    # ---- run on the whole input space
    T = np.zeros(S.N, dtype=np.int64)
    for r in range(ROUNDS):
        snd = sender_tab[r][T]
        m = np.zeros(S.N, dtype=np.int64)
        for ui, v in enumerate(S.nodes):
            sel = snd == ui
            m[sel] = msg_tab[r][ui][S.lidx[v][sel], T[sel]]
        T = T + (m << r)          # transcript index: bit i = message i
    Ks = key_tab["s"][S.lidx["s"], T]
    Kd = key_tab["d"][S.lidx["d"], T]
    xa, xb, z = S.partition(C, mode)
    vid = z * (2 ** ROUNDS) + T
    I_AB = cmi(xa, xb, vid)
    I_KK = cmi(Ks, Kd, vid)
    # p_agree and E_max
    p_agree = float(np.mean(Ks == Kd))
    # agreement-maximising variant: K_d* = best response to K_s given (d's local state, transcript)
    grp = S.lidx["d"] * (2 ** ROUNDS) + T
    cnt = np.bincount(grp * 4 + Ks, minlength=S.nst["d"] * (2 ** ROUNDS) * 4).reshape(-1, 4)
    Kd_opt = cnt.argmax(1)[grp]
    I_KK_opt = cmi(Ks, Kd_opt, vid)
    p_agree_opt = float(np.mean(Ks == Kd_opt))
    pair = vid * 4 + Ks
    u, c = np.unique(pair, return_counts=True)
    v_of = u // 4
    # max over k of count(v,k)
    order = np.argsort(v_of, kind="stable")
    v_sorted, c_sorted = v_of[order], c[order]
    starts = np.r_[0, np.flatnonzero(np.diff(v_sorted)) + 1]
    E_max = float(np.maximum.reduceat(c_sorted, starts).sum() / S.N)
    return I_AB, I_KK, p_agree, E_max, I_KK_opt, p_agree_opt


def run_pair(pool, name, C, mode, pair_idx):
    res = pool.map(one_protocol, [(name, C, mode, pair_idx, k) for k in range(NPROTO)], chunksize=200)
    a = np.array(res)
    I_AB, I_KK, pa, em, I_KKo, pao = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    d = dict(topology=name, coalition=C, mode=mode, n_protocols=NPROTO, input_points=get_setup(name).N,
             max_abs_I_AB_given_VC=float(np.abs(I_AB).max()), max_abs_I_KsKd_given_VC=float(np.abs(I_KK).max()),
             n_I_AB_above_tol=int((np.abs(I_AB) > TOL).sum()), n_I_KsKd_above_tol=int((np.abs(I_KK) > TOL).sum()),
             n_p_agree_gt_Emax=int((pa > em + TOL).sum()),
             max_p_agree_minus_Emax=float((pa - em).max()),
             mean_p_agree=float(pa.mean()), mean_E_max=float(em.mean()),
             n_I_AB_above_1e6=int((I_AB > 1e-6).sum()), frac_I_AB_above_1e6=float((I_AB > 1e-6).mean()),
             n_I_KsKd_above_1e6=int((I_KK > 1e-6).sum()), max_I_AB=float(I_AB.max()), max_I_KsKd=float(I_KK.max()),
             n_nontrivial_protocols_p_agree_gt_0_5=int((pa > 0.5).sum()),
             opt_max_abs_I_KsKd=float(np.abs(I_KKo).max()), opt_n_I_KsKd_above_tol=int((np.abs(I_KKo) > TOL).sum()),
             opt_n_I_KsKd_above_1e6=int((I_KKo > 1e-6).sum()),
             opt_n_p_agree_gt_Emax=int((pao > em + TOL).sum()), opt_max_p_agree_minus_Emax=float((pao - em).max()),
             opt_mean_p_agree=float(pao.mean()), opt_max_p_agree=float(pao.max()),
             opt_min_slack_Emax_minus_p_agree=float((em - pao).min()),
             n_protocols_with_I_AB_exactly_zero=int((np.abs(I_AB) <= TOL).sum()))
    return d


def main():
    t0 = time.time()
    cases = []
    for name in ("path3", "path4", "diamond"):
        for C in TOPOS[name]["C"]:
            cases.append((name, C, "separating"))
    cases.append(("triangle_control", ["r"], "control"))
    out = []
    with Pool(os.cpu_count() - 1) as pool:
        for i, (name, C, mode) in enumerate(cases):
            t1 = time.time()
            d = run_pair(pool, name, C, mode, i)
            d["seconds"] = round(time.time() - t1, 1)
            out.append(d)
            print(json.dumps(d), flush=True)
    sep = [d for d in out if d["mode"] == "separating"]
    ctrl = [d for d in out if d["mode"] == "control"][0]
    summary = dict(
        seed=SEED, rounds=ROUNDS, n_protocols_per_pair=NPROTO, tolerance=TOL,
        n_separating_pairs=len(sep),
        total_separating_protocols=sum(d["n_protocols"] for d in sep),
        violations_I_AB=sum(d["n_I_AB_above_tol"] for d in sep),
        violations_I_KsKd=sum(d["n_I_KsKd_above_tol"] for d in sep),
        violations_p_agree_le_Emax=sum(d["n_p_agree_gt_Emax"] for d in sep),
        opt_violations_I_KsKd=sum(d["opt_n_I_KsKd_above_tol"] for d in sep),
        opt_violations_p_agree_le_Emax=sum(d["opt_n_p_agree_gt_Emax"] for d in sep),
        opt_min_slack_Emax_minus_p_agree=min(d["opt_min_slack_Emax_minus_p_agree"] for d in sep),
        control_opt_fraction_p_agree_gt_Emax=ctrl["opt_n_p_agree_gt_Emax"] / ctrl["n_protocols"],
        control_opt_max_p_agree=ctrl["opt_max_p_agree"],
        max_abs_I_AB_over_all_separating=max(d["max_abs_I_AB_given_VC"] for d in sep),
        max_abs_I_KsKd_over_all_separating=max(d["max_abs_I_KsKd_given_VC"] for d in sep),
        control_fraction_I_AB_gt_1e6=ctrl["frac_I_AB_above_1e6"],
        control_fraction_I_KsKd_gt_1e6=ctrl["n_I_KsKd_above_1e6"] / ctrl["n_protocols"],
        control_fraction_p_agree_gt_Emax=ctrl["n_p_agree_gt_Emax"] / ctrl["n_protocols"],
        per_pair=out, seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(OUT, "g0_1_summary.json"), "w"), indent=1)
    print("TOTAL", {k: v for k, v in summary.items() if k != "per_pair"})


if __name__ == "__main__":
    main()
