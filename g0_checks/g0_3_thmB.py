"""G0.3 - Theorem B (and Theorem 1' for one path) end to end, exact enumeration with biased pads and side information.

Topologies (every path internally vertex-disjoint):
  a_2x2           2 paths x 2 hops          relays a1, b1
  b_3x2           3 paths x 2 hops          relays a1, b1, c1
  c_direct+1x2    direct link + 1 path x 2 hops   relay b1
  d_2x3           2 paths x 3 hops          (extra)
  e_1x3           1 path x 3 hops           (extra; Theorem 1' with a coalition)
L = 1 bit.  Link e has pad bit P_e with Pr[P_e=0] = 1/2 + beta_e, beta_e in {0, .02, .05, .1} x random sign; optional ("leak" mode)
side-information bit E_e = P_e xor Bernoulli(gamma_e), gamma_e in {.5,.45,.4,.3}; eps_e = d(D_e, pi (x) D_e|E) computed exactly.
Protocol (Algorithm 4 without MACs): s draws K and Y_1..Y_{n-1}; Y_n = K xor sum Y; on path i hop j the ciphertext Y_i xor P_ij is public;
the relays of path i learn Y_i; d outputs the xor of the Y_i.
Coalition C = any subset of relays, including the empty set (external adversary only), all sizes (also larger than t).
View = (all ciphertexts, pads of links with an end in C, shares Y_i of paths with a relay in C, all side-information bits E_e).
U = unhit paths.  real = law of (K, view).  ideal_i = same world with the pads of path i replaced by uniform bits independent of E
(E_e keeps its real marginal); the other paths keep their biased pads (they are adversary information).
dist_free(i) = d(real, pi_K (x) view-marginal of ideal_i) must be <= S_i = sum_{e in P_i} eps_e (delta = 0);
dist_actual = d(real, pi_K (x) real view marginal) must be <= 2 min_i S_i.   U empty: K must be a function of the view.
Seed 20261113.  Output: out/g0_3_thmB.json
"""
import json, os, itertools, time
import numpy as np

SEED = 20261113
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
DRAWS = int(os.environ.get("G03_DRAWS", 300))
BETAS = [0.0, 0.02, 0.05, 0.1]
GAMMAS = [0.5, 0.45, 0.4, 0.3]

TOPOS = {
    "a_2x2": [2, 2],
    "b_3x2": [2, 2, 2],
    "c_direct+1x2": [1, 2],
    "d_2x3": [3, 3],
    "e_1x3": [3],
}


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def build(hops):
    """link (i,j) joins node j-1 and node j of path i (node 0 = s, node hops[i] = d); relay (i,r), 1 <= r < hops[i]."""
    links = [(i, j) for i, h in enumerate(hops) for j in range(1, h + 1)]
    relays = [(i, r) for i, h in enumerate(hops) for r in range(1, h)]
    return links, relays


def link_touches(i, j, h, C):
    return any(e in C for e in ([(i, j - 1)] if j - 1 >= 1 else []) + ([(i, j)] if j < h else []))


def world(hops, beta, gamma, leak, uniform_paths=()):
    n = len(hops)
    links, relays = build(hops)
    nl = len(links)
    nv = 1 + (n - 1) + nl + (nl if leak else 0)
    idx = np.indices((2,) * nv).reshape(nv, -1)
    prob = np.full(idx.shape[1], 1.0 / 2 ** n)       # K, Y_1..Y_{n-1} uniform
    K = idx[0]
    Y = [idx[1 + i] for i in range(n - 1)]
    Yn = K.copy()
    for y in Y:
        Yn = Yn ^ y
    Y = Y + [Yn]
    P, E = {}, {}
    for li, (i, j) in enumerate(links):
        p = idx[1 + (n - 1) + li]
        P[(i, j)] = p
        b, g = beta[(i, j)], gamma[(i, j)]
        if leak:
            e = idx[1 + (n - 1) + nl + li]
            E[(i, j)] = e
        if i in uniform_paths:
            prob = prob * 0.5
            if leak:
                pe0 = (0.5 + b) * (1 - g) + (0.5 - b) * g       # real marginal of E
                prob = prob * np.where(e == 0, pe0, 1 - pe0)
        else:
            prob = prob * np.where(p == 0, 0.5 + b, 0.5 - b)
            if leak:
                prob = prob * np.where(e == p, 1 - g, g)
    return dict(n=n, links=links, relays=relays, prob=prob, K=K, Y=Y, P=P, E=E, hops=hops)


def view_of(W, C, leak):
    n, hops = W["n"], W["hops"]
    comps = [W["Y"][i] ^ W["P"][(i, j)] for (i, j) in W["links"]]
    comps += [W["P"][(i, j)] for (i, j) in W["links"] if link_touches(i, j, hops[i], C)]
    comps += [W["Y"][i] for i in range(n) if any(r in C for r in W["relays"] if r[0] == i)]
    if leak:
        comps += [W["E"][l] for l in W["links"]]
    v = np.zeros_like(W["K"])
    for c in comps:
        v = v * 2 + c
    return v


def dist_to_product(K, view, prob, marg_prob=None, marg_view=None):
    nvw = int(view.max()) + 1
    if marg_view is not None:
        nvw = max(nvw, int(marg_view.max()) + 1)
    joint = np.zeros((2, nvw))
    np.add.at(joint, (K, view), prob)
    if marg_prob is None:
        m = joint.sum(0)
    else:
        m = np.zeros(nvw)
        np.add.at(m, marg_view, marg_prob)
    return 0.5 * float(np.abs(joint - 0.5 * m[None, :]).sum())


def eps_of(b, g, leak):
    if not leak:
        return abs(b)
    pp = np.array([0.5 + b, 0.5 - b])
    D = np.zeros((2, 2))
    for p in (0, 1):
        D[p, p] = pp[p] * (1 - g)
        D[p, 1 - p] = pp[p] * g
    pe = D.sum(0, keepdims=True)
    return tv(D, np.ones((2, 2)) / 2 * pe)


def run_case(hops, beta, gamma, leak):
    n = len(hops)
    links, relays = build(hops)
    eps = {l: eps_of(beta[l], gamma[l], leak) for l in links}
    S = [sum(eps[l] for l in links if l[0] == i) for i in range(n)]
    real = world(hops, beta, gamma, leak)
    ideal = {i: world(hops, beta, gamma, leak, uniform_paths=(i,)) for i in range(n)}
    rec = []
    for r in range(0, len(relays) + 1):
        for C in itertools.combinations(relays, r):
            C = set(C)
            U = [i for i in range(n) if not any(x in C for x in relays if x[0] == i)]
            vr = view_of(real, C, leak)
            if not U:
                nvw = int(vr.max()) + 1
                joint = np.zeros((2, nvw))
                np.add.at(joint, (real["K"], vr), real["prob"])
                fn = bool(((joint > 1e-15).sum(0) <= 1).all())
                rec.append(dict(U_empty=True, key_is_function_of_view=fn, dist=dist_to_product(real["K"], vr, real["prob"])))
                continue
            dist_actual = dist_to_product(real["K"], vr, real["prob"])
            fr = {}
            for i in U:
                vi = view_of(ideal[i], C, leak)
                fr[i] = dist_to_product(real["K"], vr, real["prob"], marg_prob=ideal[i]["prob"], marg_view=vi)
            ratios = [fr[i] / S[i] for i in U if S[i] > 1e-12]
            minS = min(S[i] for i in U)
            rec.append(dict(U_empty=False, minS=minS, dist_actual=dist_actual,
                            max_ratio_free=(max(ratios) if ratios else 0.0),
                            max_viol_free=max(fr[i] - S[i] for i in U),
                            n_zero_S_with_nonzero_dist=int(sum(1 for i in U if S[i] <= 1e-12 and fr[i] > 1e-12)),
                            actual_over_minS=(dist_actual / minS if minS > 1e-12 else None),
                            actual_le_2minS=bool(dist_actual <= 2 * minS + 1e-12)))
    return rec


def main():
    t0 = time.time()
    summary = {}
    for ti, (name, hops) in enumerate(TOPOS.items()):
        for leak in (False, True):
            links, relays = build(hops)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, ti, int(leak)]))
            cases = []
            for _ in range(DRAWS):
                beta = {l: float(rng.choice(BETAS)) * float(rng.choice([-1, 1])) for l in links}
                gamma = {l: float(rng.choice(GAMMAS)) for l in links}
                cases.append((beta, gamma))
            cases.append(({l: 0.1 for l in links}, {l: 0.3 for l in links}))
            cases.append(({l: 0.0 for l in links}, {l: 0.5 for l in links}))
            for i in range(len(hops)):
                cases.append(({l: (0.1 if l[0] == i else 0.0) for l in links}, {l: (0.3 if l[0] == i else 0.5) for l in links}))
            agg = dict(cases=len(cases), coalition_instances=0, violations_free=0, max_ratio_free=0.0, max_actual_over_minS=0.0,
                       violations_actual_2minS=0, U_empty_instances=0, U_empty_key_function_failures=0,
                       U_empty_dist_min=1.0, U_empty_dist_max=0.0, zero_S_nonzero_dist=0, tight_cases_ratio_ge_0_999=0,
                       max_viol_free=-1.0)
            for beta, gamma in cases:
                for rc in run_case(hops, beta, gamma, leak):
                    agg["coalition_instances"] += 1
                    if rc["U_empty"]:
                        agg["U_empty_instances"] += 1
                        agg["U_empty_key_function_failures"] += int(not rc["key_is_function_of_view"])
                        agg["U_empty_dist_min"] = min(agg["U_empty_dist_min"], rc["dist"])
                        agg["U_empty_dist_max"] = max(agg["U_empty_dist_max"], rc["dist"])
                    else:
                        agg["violations_free"] += int(rc["max_viol_free"] > 1e-12)
                        agg["max_viol_free"] = max(agg["max_viol_free"], rc["max_viol_free"])
                        agg["max_ratio_free"] = max(agg["max_ratio_free"], rc["max_ratio_free"])
                        if rc["actual_over_minS"] is not None:
                            agg["max_actual_over_minS"] = max(agg["max_actual_over_minS"], rc["actual_over_minS"])
                        agg["violations_actual_2minS"] += int(not rc["actual_le_2minS"])
                        agg["zero_S_nonzero_dist"] += rc["n_zero_S_with_nonzero_dist"]
                        agg["tight_cases_ratio_ge_0_999"] += int(rc["max_ratio_free"] >= 0.999)
            summary[f"{name}|leak={leak}"] = agg
            print(name, "leak" if leak else "noleak", agg, flush=True)
    tot = dict(
        configs=len(summary),
        cases=sum(a["cases"] for a in summary.values()),
        coalition_instances=sum(a["coalition_instances"] for a in summary.values()),
        violations_free=sum(a["violations_free"] for a in summary.values()),
        violations_actual_2minS=sum(a["violations_actual_2minS"] for a in summary.values()),
        U_empty_instances=sum(a["U_empty_instances"] for a in summary.values()),
        U_empty_key_function_failures=sum(a["U_empty_key_function_failures"] for a in summary.values()),
        max_ratio_free=max(a["max_ratio_free"] for a in summary.values()),
        max_actual_over_minS=max(a["max_actual_over_minS"] for a in summary.values()),
        zero_S_nonzero_dist=sum(a["zero_S_nonzero_dist"] for a in summary.values()))
    out = dict(seed=SEED, draws_per_config=DRAWS, total=tot, per_config=summary, seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(OUT, "g0_3_thmB.json"), "w"), indent=1)
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
