"""
q11_plan.py - X5: ledger tamper tests for PLANS (t+1 disjoint paths). T0-T12 are the q11_alg3_v2 tests re-expressed on a 2-path plan
(tamper applied to one randomly chosen path of the plan); T13-T16 are new:
  T13 two paths share a relay -> reject     T14 plan with only t paths -> reject
  T15 one path over budget (others fine) -> reject    T16 plan with all constraints met -> accept (benign control)
2,000 trials each, rng seed 20261113. Also ledger bytes and verification time (1,000 runs, mean +- SD) for plans with K in {2,3}, m in {2,4,8} per path.
Outputs: x5_out/x5_results.json, x5_table.txt
"""
import sys, os, json, time, platform, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from bmr import verify as V
from bmr import verify_plan as VP

OUT = os.path.join(os.path.dirname(HERE), "x5_out"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261113)
NT = 2000
NOW, R, DELTA, W, T_COLL = 10, 0.05, 1e-2, 8, 1
NODES = list(range(16))
reg = V.KeyRegistry(NODES)
forger = V.KeyRegistry(NODES)
S_, D_ = 0, 4
SEQS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 4]]


def mk_path(seq, now=NOW, reg_=None):
    reg_ = reg_ or reg
    return [V.make_cert_entry_v2(reg_, (a, b), 0.11 + float(rng.uniform(0, .1)), 1e-7, 1e-6, now - int(rng.integers(0, W)))
            for a, b in zip(seq[:-1], seq[1:])]


def mk_plan(seqs=SEQS):
    return [mk_path(q) for q in seqs]


def db_of(plan):
    return {(e["link"], e["T"]): (e["r"], e["eps"], e["delta"]) for p in plan for e in p}


def sess(plan, nonce, s=S_, d=D_, rg=None):
    return VP.bind_plan(rg or reg, plan, nonce, s, d, NOW)


def run(tamper):
    acc = 0; reasons = {}
    for _ in range(NT):
        nonce = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        plan = mk_plan(); db = db_of(plan)
        plan2, ss, db2, s, d, exp = tamper(plan, db, nonce)
        ok, why = VP.verify_plan(plan2, ss, reg, s, d, R, DELTA, W, T_COLL, NOW, db2, exp)
        acc += ok; reasons[why] = reasons.get(why, 0) + 1
    return acc / NT, reasons


def pick(plan):
    return int(rng.integers(len(plan)))


def cp(plan): return [[dict(e) for e in p] for p in plan]


def benign(plan, db, nonce): return plan, sess(plan, nonce), db, S_, D_, nonce
def T1(plan, db, nonce):
    p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k]))); p[k][i]["r"] += .5
    return p, sess(plan, nonce), db, S_, D_, nonce
def T1b(plan, db, nonce):
    p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k]))); p[k][i]["r"] += .5
    return p, sess(p, nonce, rg=forger), db, S_, D_, nonce
def T2(plan, db, nonce):
    p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k])))
    e = p[k][i]; p[k][i] = V.make_cert_entry_v2(reg, e["link"], e["r"], e["eps"], e["delta"], NOW - 100)
    return p, sess(p, nonce), db, S_, D_, nonce
def T3(plan, db, nonce):
    p = cp(plan); k = pick(p); p[k] = [p[k][0], p[k][-1]]
    return p, sess(p, nonce), db, S_, D_, nonce
def T4(plan, db, nonce):
    p = cp(plan); k = pick(p)
    p[k] = [p[k][0], V.make_cert_entry_v2(reg, (1, 0) if k == 0 else (5, 0), .2, 1e-7, 1e-6, NOW - 1)] + p[k]
    return p, sess(p, nonce), db, S_, D_, nonce
def T5(plan, db, nonce):
    p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k]))); p[k][i]["eps"] /= 1000
    return p, sess(plan, nonce), db, S_, D_, nonce
def T6(plan, db, nonce):
    p = cp(plan); k = pick(p); p[k][0], p[k][1] = p[k][1], p[k][0]
    return p, sess(p, nonce), db, S_, D_, nonce
def T7(plan, db, nonce):
    p = cp(plan); k = pick(p)
    p[k][1] = V.make_cert_entry_v2(reg, p[k][1]["link"], p[k][1]["r"], 1e-7, 1e-6, NOW - 2)
    d2 = db_of(p); d2[(p[k][1]["link"], NOW - 1)] = (0.0, 0.0, 0.0)
    return p, sess(p, nonce), d2, S_, D_, nonce
def T7b(plan, db, nonce):
    p = cp(plan); k = pick(p)
    p[k][0] = V.make_cert_entry_v2(reg, p[k][0]["link"], p[k][0]["r"], 1e-7, 1e-6, NOW - 4)
    d2 = db_of(p); d2[(p[k][0]["link"], NOW - 2)] = (0.0, 0.0, 0.0)
    return p, sess(p, nonce), d2, S_, D_, nonce
def T8(plan, db, nonce):
    p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k]))); e = p[k][i]
    p[k][i] = V.make_cert_entry_v2(forger, e["link"], e["r"] + .1, e["eps"], e["delta"], e["T"])
    return p, sess(p, nonce), db, S_, D_, nonce
def T9(plan, db, nonce):
    old = bytes(rng.integers(0, 256, 16, dtype=np.uint8)); return plan, sess(plan, old), db, S_, D_, nonce
def T10(plan, db, nonce): return plan, sess(plan, nonce, d=3), db, S_, D_, nonce
def T10b(plan, db, nonce):
    other = mk_plan([[8, 9, 10, 11, 12], [8, 13, 14, 15, 12]])
    return other, VP.bind_plan(reg, other, nonce, 8, 12, NOW), db, S_, D_, nonce
def T11(plan, db, nonce):
    alt = mk_plan(); p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k]))); p[k][i] = alt[k][i]
    return p, sess(plan, nonce), db, S_, D_, nonce
def T11b(plan, db, nonce):
    alt = mk_plan([[0, 8, 9, 10, 4], [0, 11, 12, 13, 4]])
    return alt, sess(plan, nonce), db, S_, D_, nonce
def T12(plan, db, nonce):
    p = cp(plan); k = pick(p); i = int(rng.integers(len(p[k]))); e = p[k][i]
    p[k][i] = V.make_cert_entry_v2(reg, e["link"], e["r"] + 0.3, e["eps"], e["delta"], e["T"])
    return p, sess(p, nonce), {}, S_, D_, nonce
def T12db(plan, db, nonce):
    p, s_, _, a, b, c = T12(plan, db, nonce); return p, s_, db, a, b, c
def T13(plan, db, nonce):                    # two paths share relay 2: second path 0-5-2-7-4 (all certificates valid, session valid)
    p = [plan[0], mk_path([0, 5, 2, 7, 4])]
    return p, sess(p, nonce), db_of(p), S_, D_, nonce
def T14(plan, db, nonce):                    # only t = 1 path
    p = [plan[pick(plan)]]
    return p, sess(p, nonce), db_of(p), S_, D_, nonce
def T15(plan, db, nonce):                    # one path over budget, all certificates validly signed and session valid
    p = cp(plan); k = pick(p)
    p[k] = [V.make_cert_entry_v2(reg, e["link"], e["r"], 6e-3, 1e-6, e["T"]) for e in p[k]]      # 4 hops * 6e-3 > 1e-2
    return p, sess(p, nonce), db_of(p), S_, D_, nonce
def T16(plan, db, nonce): return plan, sess(plan, nonce), db, S_, D_, nonce

tests = [("T0 benign", benign, "accept"), ("T1 inflate r (plan digest)", T1, "reject"), ("T1b inflate r + re-bind with forger key", T1b, "reject"),
         ("T2 stale certificate", T2, "reject"), ("T3 drop a hop", T3, "reject"), ("T4 loop", T4, "reject"), ("T5 lower eps", T5, "reject"),
         ("T6 reorder", T6, "reject"), ("T7 DEGRADED in (T_i,now] (late)", T7, "reject"), ("T7b DEGRADED in (T_i,now] (early)", T7b, "reject"),
         ("T8 re-signed by non-attestor", T8, "reject"), ("T9 replay old plan+session", T9, "reject"),
         ("T10 session bound to another (s,d')", T10, "reject"), ("T10b plan of another request", T10b, "reject"),
         ("T11 entry swapped for another valid cert", T11, "reject"), ("T11b substitute other valid plan", T11b, "reject"),
         ("T12 lying attestor, ledger-only", T12, "ACCEPT (documented limit)"), ("T12-DB lying attestor, DB has audit record", T12db, "reject (DB cross-check)"),
         ("T13 two paths share a relay", T13, "reject"), ("T14 plan with only t paths", T14, "reject"),
         ("T15 one path over budget (others fine)", T15, "reject"), ("T16 plan with all constraints met", T16, "accept")]


def wilson_upper(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + h) / den


rows, lines = [], []
hdr = f"{'test':46s} {'expected':26s} accepted/{NT}  Wilson95-upper  reasons"
print(hdr); lines.append(hdr)
for name, fn, exp in tests:
    a, why = run(fn)
    k = int(round(a * NT))
    ok = (a > .99) if exp.startswith("accept") or exp.startswith("ACCEPT") else (a < .01)
    rows.append(dict(test=name, expected=exp, accepted=k, trials=NT, wilson95_upper=wilson_upper(k, NT), reasons=why, as_expected=bool(ok)))
    s = f"{name:46s} {exp:26s} {k:5d}        {100*wilson_upper(k, NT):6.3f}%      {why}"
    print(s); lines.append(s)
assert all(r["as_expected"] for r in rows), "unexpected result"


def ledger_bytes(plan, sess_):
    n = 0
    for p in plan:
        for e in p:
            n += len(cert_json(e))
    return n + len(sess_["sig"]) + len(sess_["nonce"]) + 8


def cert_json(e):
    return V.cert_message(e["link"], e["r"], e["eps"], e["delta"], e["T"]) + e["sig"]


def timing(use_ed, reps=1000):
    rg = V.KeyRegistry(list(range(40)), use_ed25519=use_ed); out = {}; D = 39
    for K in (2, 3):
        for m in (2, 4, 8):
            seqs = []
            nid = 1
            for k in range(K):
                seqs.append([0] + list(range(nid, nid + m - 1)) + [D]); nid += m - 1
            ts = []
            for _ in range(reps):
                plan = [[V.make_cert_entry_v2(rg, (a, b), .2, 1e-7, 1e-6, NOW) for a, b in zip(q[:-1], q[1:])] for q in seqs]
                nonce = b"n" * 16; ss = VP.bind_plan(rg, plan, nonce, 0, D, NOW); db = db_of(plan)
                t0 = time.perf_counter()
                ok, why = VP.verify_plan(plan, ss, rg, 0, D, R, 1.0, W, K - 1, NOW, db, nonce)
                ts.append((time.perf_counter() - t0) * 1e6); assert ok, why
            ts = np.array(ts[50:])
            out[f"K{K}_m{m}"] = dict(paths=K, hops_per_path=m, entries=K * m, bytes=ledger_bytes(plan, ss), mean_us=float(ts.mean()), sd_us=float(ts.std(ddof=1)))
    return out

tm_ed, tm_hm = timing(True), timing(False)
lines.append("\nTiming/size per plan (µs, 1000 runs, drop 50 warm-up); " + platform.platform() + " Python " + platform.python_version())
for k in tm_ed:
    e, h = tm_ed[k], tm_hm[k]
    s = f"  {k:7s} entries={e['entries']:2d} bytes={e['bytes']:5d} | Ed25519 {e['mean_us']:.1f}±{e['sd_us']:.1f} | HMAC {h['mean_us']:.1f}±{h['sd_us']:.1f}"
    print(s); lines.append(s)
json.dump(dict(tests=rows, ed25519=tm_ed, hmac=tm_hm, seed=20261113, machine=platform.platform(), python=platform.python_version()),
          open(os.path.join(OUT, "x5_results.json"), "w"), indent=1, default=str)
open(os.path.join(OUT, "x5_table.txt"), "w").write("\n".join(lines))
