"""G0.2 - the constant in Theorem 1' (factor 1 vs factor 2), numerical search.

Pads P_j in {0..N-1}, N = 2^L, side information E_j in {0..q-1}, one joint distribution D_j(p, e) per link; D = tensor product.
eps_j = d(D_j, pi (x) D_j|E).  Twirl T D(p, e) = N^-1 sum_k D(p xor k*1, e)  (pad axes only, never the side-information axes).
Quantities:
  R1 = d(D, TD) / sum_j eps_j                 (the ratio asked for in developer_guide_G0.md; factor-1 claim is R1 <= 1)
  R2 = d(D, TD) / d(D, pi (x) D_E)            (ratio against the true free-marginal distance of the joint pad vector)
Search = multi-start (1+1) evolution strategy on the entries of the D_j with decreasing step size (hill climb) to MAXIMISE the ratio.
Starts: perturbations of the uniform pad distribution of scale 0.01 / 0.1 / 1 plus random Dirichlet starts.
Also: (i) numerical check of the identity d(rho_{KCE}, pi_K (x) rho_{CE}) = d(D, TD) against a direct computation of rho_{KCE};
      (ii) the non-product control (one joint D over (p1, p2), no product structure; q = 1) for L = 1, 2, 3, maximising R2;
      (iii) the explicit non-product construction attaining 2(1 - 2^-L).
Seed 20261113.  Output: out/g0_2_constant.json
"""
import json, os, sys, time, itertools
from multiprocessing import Pool
import numpy as np

SEED = 20261113
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
STARTS = int(os.environ.get("G02_STARTS", 400))
ITERS = int(os.environ.get("G02_ITERS", 4000))


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


class Geo:
    """Precomputed permutations for the twirl over diagonal shifts."""
    def __init__(self, L, m):
        self.N, self.m = 2 ** L, m
        N = self.N
        grids = np.indices((N,) * m).reshape(m, -1)               # (m, N^m)
        flat = np.ravel_multi_index(grids, (N,) * m)
        self.perm = [np.ravel_multi_index(grids ^ k, (N,) * m) for k in range(N)]
        assert (flat == np.arange(N ** m)).all()

    def twirl(self, D):                                           # D: (N^m, Q) with pad index flattened
        return sum(D[p] for p in self.perm) / self.N


def joint_product(Ds, N, q):
    m = len(Ds)
    ops = []
    for j, Dj in enumerate(Ds):
        ops += [Dj, [j, m + j]]
    D = np.einsum(*ops, list(range(2 * m)))
    return D.reshape(N ** m, q ** m)


def eps_link(Dj):
    N = Dj.shape[0]
    pe = Dj.sum(0, keepdims=True)
    return tv(Dj, np.ones_like(Dj) * pe / N)


def ratios_product(Ds, geo, q):
    D = joint_product(Ds, geo.N, q)
    dist = tv(D, geo.twirl(D))
    s = sum(eps_link(Dj) for Dj in Ds)
    pe = D.sum(0, keepdims=True)
    d_free = tv(D, np.ones_like(D) * pe / D.shape[0])
    return dist, s, d_free


def normalise(x):
    x = np.maximum(x, 1e-12)
    return x / x.sum()


def search_product(args):
    L, m, q, idx = args
    geo = Geo(L, m)
    N = 2 ** L
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, L, m, q, idx]))
    scale0 = [0.01, 0.1, 1.0, 5.0][idx % 4]
    if idx % 5 == 4:
        Ds = [normalise(rng.dirichlet(np.ones(N * q) * 0.3)).reshape(N, q) for _ in range(m)]
    else:
        Ds = [normalise((1 + scale0 * rng.standard_normal((N, q))) / (N * q) * 1.0).reshape(N, q) for _ in range(m)]
        Ds = [d / d.sum() for d in Ds]

    def score(Ds, which):
        dist, s, dfree = ratios_product(Ds, geo, q)
        den = s if which == 1 else dfree
        return dist / den if den > 1e-9 else 0.0

    best = {1: score(Ds, 1), 2: score(Ds, 2)}
    out = {}
    for which in (1, 2):
        cur = [d.copy() for d in Ds]
        cs = score(cur, which)
        step = 0.3
        for it in range(ITERS):
            j = int(rng.integers(m))
            cand = [d.copy() for d in cur]
            noise = rng.standard_normal((N, q)) * step * (rng.random((N, q)) < 0.5)
            cand[j] = normalise(cand[j] * np.exp(noise)).reshape(N, q)
            cand[j] = cand[j] / cand[j].sum()
            sc = score(cand, which)
            if sc > cs:
                cur, cs = cand, sc
            if it % 500 == 499:
                step *= 0.6
        out[which] = (cs, cur)
    return dict(L=L, m=m, q=q, idx=idx, start_scale=scale0, R1=out[1][0], R2=out[2][0],
                D1=[d.tolist() for d in out[1][1]], D2=[d.tolist() for d in out[2][1]])


def search_nonproduct(args):
    L, idx = args
    m = 2
    geo = Geo(L, m)
    N = 2 ** L
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 22, L, idx]))
    cur = normalise(rng.dirichlet(np.ones(N ** m) * [0.3, 1.0, 3.0][idx % 3])).reshape(-1, 1)
    U = np.ones_like(cur) / cur.shape[0]

    def sc(D):
        d0 = tv(D, U)
        return tv(D, geo.twirl(D)) / d0 if d0 > 1e-9 else 0.0
    cs = sc(cur)
    step = 0.5
    for it in range(ITERS * 2):
        cand = cur * np.exp(rng.standard_normal(cur.shape) * step * (rng.random(cur.shape) < 0.3))
        cand = normalise(cand.ravel()).reshape(-1, 1)
        s = sc(cand)
        if s > cs:
            cur, cs = cand, s
        if it % 1000 == 999:
            step *= 0.7
    return dict(L=L, idx=idx, R2=cs, D=cur.ravel().tolist())


def identity_check(n_inst=40):
    """d(rho_{KCE}, pi_K (x) rho_{CE}) computed directly vs d(D, TD)."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 20]))
    worst = 0.0
    for i in range(n_inst):
        L, m, q = [(1, 2, 2), (2, 2, 2), (2, 2, 1), (1, 3, 2), (2, 3, 1)][i % 5]
        N = 2 ** L
        geo = Geo(L, m)
        if i % 2 == 0:       # product
            Ds = [normalise(rng.dirichlet(np.ones(N * q) * 0.7)).reshape(N, q) for _ in range(m)]
            D = joint_product(Ds, N, q)
        else:                # non-product
            D = normalise(rng.dirichlet(np.ones(N ** m * q ** m) * 0.5)).reshape(N ** m, q ** m)
        lhs = tv(D, geo.twirl(D))
        # direct: rho(k, c1..cm, e) = N^-1 D(c xor k, e)
        grids = np.indices((N,) * m).reshape(m, -1)
        rho = np.zeros((N, N ** m, q ** m))
        for k in range(N):
            rho[k] = D[np.ravel_multi_index(grids ^ k, (N,) * m)] / N
        rho_ce = rho.sum(0, keepdims=True)
        rhs = tv(rho, np.ones((N, 1, 1)) / N * rho_ce)
        worst = max(worst, abs(lhs - rhs))
    return worst


def explicit_construction(L):
    """D = uniform on pad pairs, +eps at (0,0), -eps at (0,1); the two points lie in different diagonal orbits (p1 xor p2 differs)."""
    N = 2 ** L
    geo = Geo(L, 2)
    eps = 1.0 / (2 * N * N)
    D = (np.ones((N, N)) / (N * N))
    D[0, 0] += eps
    D[0, 1] -= eps
    D = D.reshape(-1, 1)
    U = np.ones_like(D) / D.shape[0]
    return tv(D, geo.twirl(D)) / tv(D, U), 2 * (1 - 1 / N)


def main():
    t0 = time.time()
    settings = [(1, 2, 2), (1, 3, 2), (2, 2, 1), (2, 2, 2), (2, 3, 1), (2, 3, 2), (3, 2, 1), (3, 2, 2), (3, 3, 1)]
    jobs = [(L, m, q, i) for (L, m, q) in settings for i in range(STARTS)]
    with Pool(os.cpu_count() - 1) as pool:
        res = pool.map(search_product, jobs, chunksize=4)
        npjobs = [(L, i) for L in (1, 2, 3) for i in range(max(40, STARTS // 5))]
        npres = pool.map(search_nonproduct, npjobs, chunksize=2)
    per = []
    for (L, m, q) in settings:
        rs = [r for r in res if (r["L"], r["m"], r["q"]) == (L, m, q)]
        b1 = max(rs, key=lambda r: r["R1"])
        b2 = max(rs, key=lambda r: r["R2"])
        per.append(dict(L=L, m=m, q=q, starts=len(rs), best_R1=b1["R1"], best_R2=b2["R2"],
                        n_starts_R1_above_1=sum(r["R1"] > 1 + 1e-9 for r in rs),
                        n_starts_R1_above_0_999=sum(r["R1"] > 0.999 for r in rs),
                        best_R1_instance=b1["D1"], best_R2_instance=b2["D2"], best_R1_start_scale=b1["start_scale"]))
    npsum = []
    for L in (1, 2, 3):
        rs = [r for r in npres if r["L"] == L]
        b = max(rs, key=lambda r: r["R2"])
        npsum.append(dict(L=L, starts=len(rs), best_R2=b["R2"], theory_2_1_minus_2_pow_minus_L=2 * (1 - 2.0 ** -L), best_instance=b["D"]))
    cons = [dict(L=L, ratio=explicit_construction(L)[0], theory=explicit_construction(L)[1]) for L in (1, 2, 3)]
    idc = identity_check()
    summary = dict(seed=SEED, starts_per_setting=STARTS, iterations_per_start=ITERS, product_pads=per, nonproduct=npsum,
                   explicit_nonproduct_construction=cons, identity_max_abs_error=idc,
                   max_R1_over_all_product_settings=max(p["best_R1"] for p in per),
                   max_R2_over_all_product_settings=max(p["best_R2"] for p in per),
                   seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(OUT, "g0_2_constant.json"), "w"), indent=1)
    for p in per:
        print({k: p[k] for k in ("L", "m", "q", "starts", "best_R1", "best_R2", "n_starts_R1_above_1")})
    for n in npsum:
        print({k: n[k] for k in ("L", "starts", "best_R2", "theory_2_1_minus_2_pow_minus_L")})
    print("explicit", cons, "identity err", idc, "secs", summary["seconds"])


if __name__ == "__main__":
    main()
