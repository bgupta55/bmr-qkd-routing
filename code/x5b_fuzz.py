"""
x5b_fuzz.py - RF8: differential / property fuzz of the plan-ledger verifier (X5 follow-up).

X5's T13-T15 were built by me to hit the checks I wrote. Here the ORACLE is independent of the verifier code: each trial draws a random subset of
12 elementary defects (each present with probability 1/4, so ~3 per trial and many multi-defect combinations); the oracle says
   accept  iff  the defect set is empty,
and the verifier (bmr.verify_plan.verify_plan) is run on the ledger built with exactly those defects. Defects:
  D1 shared relay between two paths          D2 wrong number of paths (t, or t+2)        D3 one path over the per-path budget
  D4 stale certificate (age > W)             D5 certificate signed by a non-attestor      D6 rate below R on one link
  D7 session bound to another (s,d)          D8 replayed nonce                            D9 direct link entry inside a multi-path plan
  D10 loop in one path                       D11 discontinuity (entries swapped)          D12 DEGRADED record in the DB after T_i
Also fixed-seed sanity: the defect-free draw is always accepted (benign control, ~ 1/32 of trials at p=1/4... reported separately with p=0 for 2000 trials).
For each defect the number of trials in which it was present and the verdict was ACCEPT is reported (must be 0), plus verdict==reject for defect-free trials (must be 0).
Rejection reasons are tallied to show the checks that actually fire first.  Seed 20261113.  Output ../x5b_out/x5b_results.json
"""
import sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from bmr import verify as V
from bmr import verify_plan as VP

OUT = os.path.join(os.path.dirname(HERE), "x5b_out"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261113)
NOW, R, DELTA, W = 10, 0.05, 1e-2, 8
NODES = list(range(60)); S_, D_ = 0, 59
reg = V.KeyRegistry(NODES); forger = V.KeyRegistry(NODES)
DEF = ["D1_shared_relay", "D2_wrong_count", "D3_over_budget_one_path", "D4_stale", "D5_non_attestor", "D6_rate_below_R", "D7_other_sd",
       "D8_replayed_nonce", "D9_direct_in_multipath", "D10_loop", "D11_discontinuity", "D12_db_degraded"]


def entry(link, T=None, r=None, eps=1e-3, rg=None):
    T = NOW - int(rng.integers(0, W)) if T is None else T
    r = 0.11 + float(rng.uniform(0, .1)) if r is None else r
    return V.make_cert_entry_v2(rg or reg, link, r, eps, 1e-6, T)


def build(defects, t, p_new):
    """-> paths(list of entry lists), session args, db, s, d, expected nonce"""
    K = t + 1
    if "D2_wrong_count" in defects:
        K = t if t >= 1 and rng.random() < .5 else t + 2
    nid = 1; seqs = []
    for k in range(K):
        m = int(rng.integers(2, 5))
        seqs.append([S_] + list(range(nid, nid + m - 1)) + [D_]); nid += m - 1
    if "D1_shared_relay" in defects and K >= 2:
        j = int(rng.integers(len(seqs[0]) - 2)) + 1                       # a relay of path 0
        k = 1 + int(rng.integers(K - 1)); pos = int(rng.integers(1, len(seqs[k]) - 1)) if len(seqs[k]) > 2 else None
        if pos is None: seqs[k] = [S_, seqs[0][j], D_]
        else: seqs[k][pos] = seqs[0][j]
    paths = []
    bad_k = int(rng.integers(len(seqs)))
    for k, q in enumerate(seqs):
        links = list(zip(q[:-1], q[1:])); ents = []
        for i, l in enumerate(links):
            T = None; r = None; eps = 1e-3 / max(1, len(links)) * 0.9; rg = None
            if k == bad_k and i == 0:
                if "D4_stale" in defects: T = NOW - W - 1 - int(rng.integers(0, 5))
                if "D6_rate_below_R" in defects: r = R / 2
                if "D5_non_attestor" in defects: rg = forger
            if k == bad_k and "D3_over_budget_one_path" in defects: eps = DELTA / len(links) * 1.5
            ents.append(entry(l, T, r, eps, rg))
        paths.append(ents)
    if "D9_direct_in_multipath" in defects and len(paths) >= 2:
        paths[-1] = [entry((S_, D_))]
        # a direct entry also repeats nothing; relays of that path removed
    if "D10_loop" in defects:
        k = int(rng.integers(len(paths))); a = seqs[k][1] if len(seqs[k]) > 2 else 58
        loop = [entry((S_, a)), entry((a, 57)), entry((57, a)), entry((a, D_))]; paths[k] = loop
    if "D11_discontinuity" in defects:
        k = int(rng.integers(len(paths)))
        if len(paths[k]) >= 2: paths[k][0], paths[k][1] = paths[k][1], paths[k][0]
        else: paths[k] = paths[k] + [entry((56, 55))]
    db = {(e["link"], e["T"]): (e["r"], e["eps"], e["delta"]) for p in paths for e in p}
    if "D12_db_degraded" in defects:
        k = int(rng.integers(len(paths))); e = paths[k][int(rng.integers(len(paths[k])))]
        if e["T"] < NOW: db[(e["link"], e["T"] + 1)] = (0.0, 0.0, 0.0)
        else:
            e2 = V.make_cert_entry_v2(reg, e["link"], e["r"], e["eps"], e["delta"], NOW - 1); idx = paths[k].index(e); paths[k][idx] = e2
            db = {(x["link"], x["T"]): (x["r"], x["eps"], x["delta"]) for p in paths for x in p}; db[(e2["link"], NOW)] = (0.0, 0.0, 0.0)
    nonce = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
    s_b, d_b = (S_, D_) if "D7_other_sd" not in defects else (S_, 58)
    sig_nonce = nonce if "D8_replayed_nonce" not in defects else bytes(rng.integers(0, 256, 16, dtype=np.uint8))
    ss = VP.bind_plan(reg, paths, sig_nonce, s_b, d_b, NOW)
    return paths, ss, db, nonce


def main():
    t0 = time.time(); N = 40_000
    stats = {d: dict(present=0, accepted=0) for d in DEF}; clean = dict(n=0, accepted=0); reasons = {}
    benign_ok = 0
    for trial in range(N):
        p = 0.0 if trial < 2000 else 0.25
        defects = {d for d in DEF if rng.random() < p}
        t = int(rng.integers(1, 3))
        paths, ss, db, nonce = build(defects, t, p)
        ok, why = VP.verify_plan(paths, ss, reg, S_, D_, R, DELTA, W, t, NOW, db, nonce)
        if not defects:
            clean["n"] += 1; clean["accepted"] += int(ok)
            if not ok: reasons.setdefault("FALSE_REJECT:" + why, 0); reasons["FALSE_REJECT:" + why] += 1
        else:
            for d in defects:
                stats[d]["present"] += 1; stats[d]["accepted"] += int(ok)
            if ok:
                reasons.setdefault("FALSE_ACCEPT:" + "+".join(sorted(defects)), 0); reasons["FALSE_ACCEPT:" + "+".join(sorted(defects))] += 1
            else: reasons[why] = reasons.get(why, 0) + 1
    res = dict(trials=N, defect_free_trials=clean["n"], defect_free_accepted=clean["accepted"], per_defect=stats, reasons=reasons,
               false_accepts=sum(v for k, v in reasons.items() if k.startswith("FALSE_ACCEPT")), seconds=round(time.time() - t0, 1), seed=20261113)
    json.dump(res, open(os.path.join(OUT, "x5b_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
