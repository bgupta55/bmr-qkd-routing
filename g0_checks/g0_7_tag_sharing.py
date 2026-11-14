"""G0.7 (H1 review): end-to-end Wegman-Carter tag t = h_{r1}(K) xor r2 of plan 8.3 under XOR sharing.

Model: hop-key, passive coalition, information-theoretic. Payload (K, r1, r2) is delivered
over n = 2 disjoint paths. The coalition C sits on path 1, so it knows share 1 of every
payload item; the honest path 2 carries share 2 under pads C cannot see.

Case A (r1, r2 XOR-shared like K): C knows only the share-1 values; K, r1, r2 are uniform to C.
Case B (r1, r2 forwarded in the clear on every path, K shared): C knows r1, r2 exactly.
Public: tag t.  We compute I(K; t | C's knowledge) exactly by enumeration.

h_{r1}(K) = A(r1) K over GF(2), A a b x L matrix filled by r1 (L*b bits): a universal hash.
"""
import itertools, json, math, os
import numpy as np

L, b = 3, 2          # key bits, tag bits
RB = L * b           # r1 bits


def hmul(r1, K):
    out = 0
    for row in range(b):
        bits = (r1 >> (row * L)) & ((1 << L) - 1)
        out |= (bin(bits & K).count("1") & 1) << row
    return out


def H(p):
    p = np.array([x for x in p if x > 0])
    return float(-(p * np.log2(p)).sum())


def mi_K_t(cond_known_r):
    """I(K;t | known) where 'known' is r1,r2 (case B) or nothing (case A: r1,r2 uniform hidden)."""
    nK, n1, n2 = 1 << L, 1 << RB, 1 << b
    if cond_known_r:           # case B: r1, r2 fixed and known -> average over (r1,r2) of I(K;t)
        tot = 0.0
        for r1 in range(n1):
            for r2 in range(n2):
                joint = np.zeros((nK, n2))
                for K in range(nK):
                    joint[K, hmul(r1, K) ^ r2] += 1 / nK
                pK = joint.sum(1); pt = joint.sum(0)
                tot += (H(pK) + H(pt) - H(joint.ravel())) / (n1 * n2)
        return tot
    joint = np.zeros((nK, n2))   # case A: r1, r2 hidden uniform
    for K in range(nK):
        for r1 in range(n1):
            for r2 in range(n2):
                joint[K, hmul(r1, K) ^ r2] += 1 / (nK * n1 * n2)
    pK = joint.sum(1); pt = joint.sum(0)
    return H(pK) + H(pt) - H(joint.ravel())


res = {
    "L": L, "b": b,
    "caseA_shared_r1r2_I_K_t_bits": mi_K_t(False),
    "caseB_forwarded_r1r2_known_to_C_avg_I_K_t_bits": mi_K_t(True),
}
print(json.dumps(res, indent=2))
os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "out", "g0_7_tag_sharing.json"), "w"), indent=2)


# Case A': C sits on path 1 and sees share-1 of K, r1, r2 (uniform, independent) plus public tag t.
# I(K; t, Y1K, Y1r1, Y1r2) must be 0.  L2, b1 keeps the enumeration small.
def caseA_prime(L2=2, b2=1):
    global L, b, RB
    oldL, oldb, oldRB = L, b, RB
    L, b, RB = L2, b2, L2 * b2
    nK, n1, n2 = 1 << L, 1 << RB, 1 << b
    joint = {}
    for K in range(nK):
        for r1 in range(n1):
            for r2 in range(n2):
                t = hmul(r1, K) ^ r2
                for yk in range(nK):
                    for y1 in range(n1):
                        for y2 in range(n2):
                            key = (K, (t, yk, y1, y2))
                            joint[key] = joint.get(key, 0) + 1
    tot = sum(joint.values())
    pj = np.array(list(joint.values())) / tot
    pK, pV = {}, {}
    for (k, v), c in joint.items():
        pK[k] = pK.get(k, 0) + c / tot; pV[v] = pV.get(v, 0) + c / tot
    mi = H(list(pK.values())) + H(list(pV.values())) - H(pj)
    L, b, RB = oldL, oldb, oldRB
    return mi

res["caseAprime_C_sees_share1_of_K_r1_r2_plus_tag_I_bits"] = caseA_prime()
print(res["caseAprime_C_sees_share1_of_K_r1_r2_plus_tag_I_bits"])
json.dump(res, open(os.path.join(os.path.dirname(__file__), "out", "g0_7_tag_sharing.json"), "w"), indent=2)
