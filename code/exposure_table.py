"""exposure_table.py - Table 12 (degradation exposure and true-secure rate) from the per-epoch store.

Source: campaign_R/out/out_E_A2/RR4_dynamic_alarm.json (run_EA2.py; 24 cells x 100 repetitions x 80 epochs).
Exposure   = fraction of DELIVERED routes that cross an attacked link (simulator ground truth).
Absolute   = exposure x delivered fraction.
Window     = epochs 25..34 (the first ten epochs after the attack starts at epoch 25).
Run:  python exposure_table.py            (prints the table; nothing is typed by hand)
"""
import json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "campaign_R", "out", "out_E_A2", "RR4_dynamic_alarm.json")
W0, W1 = 25, 35            # python slice -> epochs 25..34
KEYS = (("spf", "SPF"), ("bmr", "BMR"), ("bma", "BMR+alarm"))


def load():
    return json.load(open(SRC))["rows"]


def window_mean(row, key, what):
    v = row[f"{key}_{what}_by_epoch"][W0:W1]
    return sum(v) / len(v)


def table():
    """-> {(n_audit, S1): {key: {'exp':, 'del':, 'abs':}}} averaged over the three topologies."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in load():
        for k, _ in KEYS:
            acc[(r["n_audit"], r["S1"])][k]["exp"].append(window_mean(r, k, "exposure"))
            acc[(r["n_audit"], r["S1"])][k]["del"].append(window_mean(r, k, "delivered"))
            acc[(r["n_audit"], r["S1"])][k]["abs"].append(window_mean(r, k, "exposure_abs"))
    out = {}
    for cell, d in acc.items():
        out[cell] = {k: {w: sum(v) / len(v) for w, v in d[k].items()} for k in d}
    return out


def true_secure():
    """post-1 window (epochs 25..59) true-secure and compliant rates, mean over topologies."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in load():
        for k, _ in KEYS:
            acc[(r["n_audit"], r["S1"])][k + "_ts"].append(r[f"{k}_true_secure_post1"])
            acc[(r["n_audit"], r["S1"])][k + "_cr"].append(r[f"{k}_compliant_rate_post1"])
    return {c: {k: sum(v) / len(v) for k, v in d.items()} for c, d in acc.items()}


if __name__ == "__main__":
    T = table()
    print("Table 12 - exposure of delivered routes, epochs 25-34, mean over ring/grid/ER (100 repetitions per cell)")
    print(f"{'n':>6} {'S_dag':>6} | {'SPF':>6} {'BMR':>6} {'BMR+al':>7} | delivered BMR -> alarm | absolute BMR -> alarm")
    for (n, s), d in sorted(T.items(), key=lambda x: (x[0][0], -x[0][1])):
        print(f"{n:>6} {s:>6} | {d['spf']['exp']:6.3f} {d['bmr']['exp']:6.3f} {d['bma']['exp']:7.3f} | "
              f"{d['bmr']['del']:.3f} -> {d['bma']['del']:.3f}        | {d['bmr']['abs']:.3f} -> {d['bma']['abs']:.3f}")
    print("\nBy topology at S_dag = 2.62, n = 10^4 (SPF / BMR / BMR+alarm exposure):")
    for r in load():
        if r["n_audit"] == 10000 and r["S1"] == 2.62:
            print(f"  {r['topo']:5s} " + " / ".join(f"{window_mean(r, k, 'exposure'):.3f}" for k, _ in KEYS))
    TS = true_secure()
    print("\nTrue-secure rate and compliant rate, post-1 window (epochs 25-59), mean over topologies:")
    print(f"{'n':>6} {'S_dag':>6} | true-secure SPF / BMR / BMR+alarm | compliant BMR / BMR+alarm")
    for (n, s), d in sorted(TS.items(), key=lambda x: (x[0][0], -x[0][1])):
        print(f"{n:>6} {s:>6} | {d['spf_ts']:.4f} / {d['bmr_ts']:.4f} / {d['bma_ts']:.4f} | {d['bmr_cr']:.4f} / {d['bma_cr']:.4f}")
    same = all(abs(d["bmr_ts"] - d["bma_ts"]) < 1e-12 for d in TS.values())
    print("\ntrue-secure rate identical with and without the alarm in every (n, S_dag) cell:", same)
