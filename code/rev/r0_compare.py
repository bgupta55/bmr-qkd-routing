"""R0: compare fresh reruns against the stored (as-submitted) result files. Prints a markdown table."""
import json, os, math
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(HERE, "_orig")
ld = lambda p: json.load(open(p))
rows = []


def add(name, old, new):
    rows.append((name, old, new, abs(old - new)))


def flat(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, path + "/" + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, path + "/" + str(i))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def full_diff(a, b, skip=()):
    fa, fb = dict(flat(a)), dict(flat(b))
    n = mx = 0
    nz = 0
    for k, v in fa.items():
        if any(s in k for s in skip):
            continue
        if k in fb:
            n += 1
            d = abs(v - fb[k])
            mx = max(mx, d)
            nz += d > 0
    return n, nz, mx


# 1. q6_sweep (Tables 15/16/18/19, cells)
a, b = ld(f"{O}/results/q6_sweep.json"), ld(f"{HERE}/data/results/q6_sweep.json")
print("q6_sweep numeric fields compared / differing / max abs diff:", full_diff(a["rows"], b["rows"]), full_diff(a["paired_stats"], b["paired_stats"]))


def agg(d, router, key, sel=lambda r: True):
    rs = [r for r in d["rows"] if r["router"] == router and sel(r)]
    return sum(r[key] for r in rs) / len(rs)


for nm, key, router in [("Table 16 BMR full compliance % (200 cells)", "compl_pct", "BMR"),
                        ("Table 15 B2 budget-only rate", "mean_rate_budget_only", "B2-BMR-NoFloor"),
                        ("BMR budget-only rate", "mean_rate_budget_only", "BMR"),
                        ("BMR refusal % (200 cells)", "refusal_pct", "BMR"),
                        ("SPF full compliance %", "compl_pct", "B0-SPF")]:
    add(nm, agg(a, router, key), agg(b, router, key))
add("Table 15 BMR-B2 budget-only (-0.0028)", agg(a, "BMR", "mean_rate_budget_only") - agg(a, "B2-BMR-NoFloor", "mean_rate_budget_only"),
    agg(b, "BMR", "mean_rate_budget_only") - agg(b, "B2-BMR-NoFloor", "mean_rate_budget_only"))

# 2. Table 8 / 9 / 12 (dynamic + alarm, E-A2 per-epoch store)
a2 = ld(f"{O}/out_E_A2/RR4_dynamic_alarm.json")["rows"]
b2 = ld(f"{HERE}/out/R0_EA2/RR4_dynamic_alarm.json")["rows"]
print("E-A2 (Tables 8/12) numeric fields compared / differing / max abs diff:", full_diff(a2, b2))


def ts(rows_, key, n, S):
    v = [r[f"{key}_true_secure_post1"] for r in rows_ if r["n_audit"] == n and r["S1"] == S]
    return sum(v) / len(v)


add("Table 8 true-secure BMR (S=2.62,n=1e4)", ts(a2, "bmr", 10000, 2.62), ts(b2, "bmr", 10000, 2.62))
add("Table 8 true-secure BMR+alarm (S=2.62,n=1e4)", ts(a2, "bma", 10000, 2.62), ts(b2, "bma", 10000, 2.62))
for S in (2.55, 2.48):
    add(f"Table 8 true-secure BMR+alarm (S={S},n=1e4)", ts(a2, "bma", 10000, S), ts(b2, "bma", 10000, S))

# 3. RR1 decomposition (Table 14), RR2 (ring flagship), RR5 (Table 20 style cross-validation)
A, B = ld(f"{O}/rr_results/RR1_decomposition_R_family.json"), ld(f"{HERE}/rr_results/RR1_decomposition_R_family.json")
print("RR1 numeric fields compared / differing / max abs diff:", full_diff(A, B))
A, B = ld(f"{O}/rr_results/RR2_ring_flagship_R_family.json"), ld(f"{HERE}/rr_results/RR2_ring_flagship_R_family.json")
print("RR2:", full_diff(A, B))
A, B = ld(f"{O}/rr_results/RR5_cross_validation.json"), ld(f"{HERE}/rr_results/RR5_cross_validation.json")
print("RR5 (excluding timings):", full_diff(A, B, skip=("_ms",)))
A, B = ld(f"{O}/rr_results/RR4_dynamic_alarm.json"), ld(f"{HERE}/rr_results/RR4_dynamic_alarm.json")
print("RR4 (50-rep file, run_pending_fast does not rewrite it; skipped)")

print("\n| Quantity | Manuscript / stored | Rerun | Abs. diff |\n|---|---|---|---|")
for n, o, w, d in rows:
    print(f"| {n} | {o:.6f} | {w:.6f} | {d:.1e} |")
