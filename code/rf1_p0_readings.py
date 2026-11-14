"""
rf1_p0_readings.py - RF1: is the X1 finding on the paper's XOR chain an artefact of ONE reading of the text?

Manuscript wording (main.tex, sec. 5.1 and A3/A4): "each relay XORs the hop key it received with the hop key it generates for the next link and
forwards the result over a confidential channel; relay j encrypts its partial XOR with a one-time pad drawn from fresh bits of the hop key that
it shares with the next relay". The text allows three readings of what is XORed/forwarded:

  R-ACC   accumulating chain (= P0 of X1): v_j sends P_j = P_{j-1} xor K_{j+1} to v_{j+1} under a pad of link j+1; d ends with K = K_1^...^K_m.
  R-LOC-HOP  local pair: v_i computes z_i = K_i xor K_{i+1} and sends it TOWARD d hop by hop (each relay re-encrypts under the next hop's pad);
             d recovers K_1..K_{m-1} from z_i and K_m.   key K = K_1^...^K_m
  R-LOC-IDEAL  local pair, but each v_i has a private confidential channel to d (an ideal channel that the hop keys do NOT provide); this is
             the only reading under which the paper's corollary (min exposing coalition = vertex cover of the links) is the answer.
  R-ACC-REV  accumulating chain run from d towards s (mirror of R-ACC); key = XOR of all hop keys, ends at s.

For each reading and m = 1..13 the GF(2) checker (symbolic_exposure.World) computes who can compute K, whether a SINGLE relay can, and the minimum exposing
coalition over ALL relay subsets of size <= 7; it is compared with the paper's closed form  ceil(m/2) (vertex cover of m links by relays, m>=2).
Output: ../rf1_out/rf1_readings.csv, rf1_summary.txt
"""
import os, sys, itertools, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symbolic_exposure as X

OUT = os.path.join(os.path.dirname(HERE), "rf1_out"); os.makedirs(OUT, exist_ok=True)


def path_of(m):
    return ["s"] + [f"v{i}" for i in range(1, m)] + ["d"]


def keys(W, p):
    m = len(p) - 1
    return [None] + [W.link(p[j - 1], p[j]) for j in range(1, m + 1)]


def acc(p):
    return X.P0_xor_chain(p)


def acc_rev(p):
    # run the accumulation from d towards s: d sends K_m to v_{m-1}; v_j sends Q_j = K_m^...^K_{j+1}... to v_{j-1}
    W = X._build([p]); m = len(p) - 1; K = keys(W, p)
    if m == 1:
        return X.Instance(W, K[1], p[0], p[-1], [p], "R-ACC-REV")
    Q = K[m]
    for j in range(m - 1, 0, -1):          # relay v_j holds Q (accumulated from the d side), XORs K_j, sends to v_{j-1}
        Q = Q ^ K[j]
        W.send(p[j], p[j - 1], Q)
    return X.Instance(W, Q, p[0], p[-1], [p], "R-ACC-REV")


def loc_hop(p):
    W = X._build([p]); m = len(p) - 1; K = keys(W, p)
    key = 0
    for j in range(1, m + 1): key ^= K[j]
    for i in range(1, m):                    # z_i from v_i, forwarded v_i -> v_{i+1} -> ... -> d
        z = K[i] ^ K[i + 1]
        for a in range(i, m):
            W.send(p[a], p[a + 1], z)
    return X.Instance(W, key, p[0], p[-1], [p], "R-LOC-HOP")


def loc_ideal(p):
    W = X._build([p]); m = len(p) - 1; K = keys(W, p)
    key = 0
    for j in range(1, m + 1): key ^= K[j]
    for i in range(1, m):
        z = K[i] ^ K[i + 1]
        pad = W.new_var(); W.grant(p[i], pad); W.grant(p[-1], pad)      # private v_i <-> d channel (not derivable from hop keys)
        W.publish(z ^ pad)
    return X.Instance(W, key, p[0], p[-1], [p], "R-LOC-IDEAL")


READINGS = [("R-ACC", acc), ("R-ACC-REV", acc_rev), ("R-LOC-HOP", loc_hop), ("R-LOC-IDEAL", loc_ideal)]


def analyse(inst, maxc=7):
    W, key = inst.world, inst.key
    rel = inst.relays
    first = {}
    single = [v for v in rel if W.coalition_space([v]).contains(key)]
    mn = None
    for k in range(1, min(maxc, len(rel)) + 1):
        for C in itertools.combinations(rel, k):
            if W.coalition_space(C).contains(key):
                mn = k; break
        if mn: break
    return dict(s_knows=int(W.knows(inst.s, key)), d_knows=int(W.knows(inst.d, key)), external=int(W.external_exposed(key)),
                single_relays=";".join(single), n_single=len(single), min_coalition=("" if mn is None else mn))


def main():
    rows = []
    for name, f in READINGS:
        for m in range(1, 14):
            inst = f(path_of(m)); a = analyse(inst)
            cover = (m + 1) // 2 if m >= 2 else ""            # paper's Corollary: min coalition = vertex cover of the m links by relays
            a.update(reading=name, m=m, paper_corollary_min=cover,
                     matches_corollary=("" if m == 1 else int(a["min_coalition"] == cover)))
            rows.append(a)
    with open(os.path.join(OUT, "rf1_readings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = []
    for name, _ in READINGS:
        rs = [r for r in rows if r["reading"] == name]
        multi = [r for r in rs if r["m"] >= 2]
        lines.append(f"{name:12s} m>=2: single relay exposes key in {sum(r['n_single']>0 for r in multi)}/{len(multi)} paths | "
                     f"min coalition == ceil(m/2) in {sum(r['matches_corollary']==1 for r in multi)}/{len(multi)} | "
                     f"s knows key {sum(r['s_knows'] for r in rs)}/13 | d knows key {sum(r['d_knows'] for r in rs)}/13 | external {sum(r['external'] for r in rs)}/13")
        for r in rs:
            if r["m"] in (2, 3, 4, 5, 7):
                lines.append(f"    m={r['m']:2d}  single-relay exposers: {r['single_relays'] or '-':24s} min coalition: {r['min_coalition']}  (corollary {r['paper_corollary_min']})  s/d know: {r['s_knows']}/{r['d_knows']}")
    open(os.path.join(OUT, "rf1_summary.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
