"""G0.4 - confirmation tag: Wegman-Carter (WC) tag leaks nothing about K; a public-seed hash (PH) tag leaks.

GF(2^b), b = 3 (modulus x^3 + x + 1).  K uniform; r1, r2 uniform and independent, forwarded to d under the pads, unknown to the adversary.
WC tag  t = r1*K xor r2   (published).   PH tag  t = r*K with r public (uniform, published).
Exact enumeration of all (K, r1, r2) (512 points) and (K, r) (64 points):
  I(K; t) for WC, I(K; t, r) for PH, and I(K; t | r=r0) for each r0;
  forgery: the adversary sees (K', ...) and must output a pair (K*, t*) with K* != K accepted by d (d accepts iff t* = r1*K* xor r2).
  Worst case over adversary strategies: given the public transcript (t) -- the adversary does NOT know r1, r2 -- the best success
  probability is max over (K*, t*) of Pr[t* = r1 K* xor r2 | public view]; computed exactly by enumeration for every K (honest key) and
  every adversary choice. Also with the adversary knowing K (worst case: K leaked by other means), since forgery must hold even then.
Also generic b in {2,3,4}: I(K;t) and forgery probability, to show 2^-b.
Seed not needed (exhaustive).  Output: out/g0_4_tag.json
"""
import json, os, itertools, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
MODS = {2: 0b111, 3: 0b1011, 4: 0b10011}


def gmul(a, b, bits):
    mod = MODS[bits]
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> bits:
            a ^= mod
    return r


def H(counts):
    c = np.asarray([x for x in counts if x > 0], float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def mi_from_pairs(pairs):
    """I(A;B) for equiprobable list of (a,b) pairs."""
    from collections import Counter
    ca, cb, cab = Counter(), Counter(), Counter()
    for a, b in pairs:
        ca[a] += 1
        cb[b] += 1
        cab[(a, b)] += 1
    return H(ca.values()) + H(cb.values()) - H(cab.values())


def analyse(bits):
    N = 2 ** bits
    out = {}
    # WC
    pairs = [(K, gmul(r1, K, bits) ^ r2) for K in range(N) for r1 in range(N) for r2 in range(N)]
    out["WC_I_K_tag_bits"] = mi_from_pairs(pairs)
    # PH, adversary sees r and t
    pairs = [(K, (r, gmul(r, K, bits))) for K in range(N) for r in range(N)]
    out["PH_I_K_tag_and_seed_bits"] = mi_from_pairs(pairs)
    # PH with a FIXED public seed r != 0 : the tag determines K (min and max over the N-1 nonzero seeds)
    per_r = [mi_from_pairs([(K, gmul(r, K, bits)) for K in range(N)]) for r in range(1, N)]
    out["PH_I_K_tag_given_fixed_r_nonzero_bits_min"] = min(per_r)
    out["PH_I_K_tag_given_fixed_r_nonzero_bits_max"] = max(per_r)
    # PH, r = 0 : tag is constant 0
    out["PH_I_K_tag_given_r_zero_bits"] = mi_from_pairs([(K, 0) for K in range(N)])
    out["PH_avg_I_over_r_bits"] = float(np.mean([mi_from_pairs([(K, gmul(r, K, bits)) for K in range(N)]) for r in range(N)]))
    out["H_K_bits"] = bits
    # WC forgery. Public view = tag t of the honest key K. d accepts (K*, t*) iff t* = r1 K* xor r2.
    # Adversary knows K (worst case), sees t = r1 K xor r2, wants K* != K, t*.  Pr over (r1, r2) consistent with t.
    best = 0.0
    best_unconditional = 0.0
    for K in range(N):
        for t in range(N):
            consistent = [(r1, r2) for r1 in range(N) for r2 in range(N) if gmul(r1, K, bits) ^ r2 == t]
            for Ks in range(N):
                if Ks == K:
                    continue
                cnt = {}
                for r1, r2 in consistent:
                    ts = gmul(r1, Ks, bits) ^ r2
                    cnt[ts] = cnt.get(ts, 0) + 1
                best = max(best, max(cnt.values()) / len(consistent))
    out["WC_best_forgery_probability_given_K_and_tag"] = best
    out["WC_forgery_expected_2_pow_minus_b"] = 2.0 ** -bits
    # PH forgery (adversary sees public r and t): trivially succeeds with K* = anything with r*K* = t, i.e. recovers K
    ph_best = 0.0
    for K in range(N):
        for r in range(1, N):
            t = gmul(r, K, bits)
            consistent = [k for k in range(N) if gmul(r, k, bits) == t]
            ph_best = max(ph_best, 1.0 / len(consistent))     # prob of guessing K from (r, t)
    out["PH_prob_recover_K_from_tag_r_nonzero"] = ph_best
    return out


def main():
    res = {b: analyse(b) for b in (2, 3, 4)}
    json.dump(dict(per_b=res), open(os.path.join(OUT, "g0_4_tag.json"), "w"), indent=1)
    for b, r in res.items():
        print(b, r)


if __name__ == "__main__":
    main()
