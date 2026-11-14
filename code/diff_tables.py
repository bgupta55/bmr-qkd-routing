"""diff_tables.py - old value, new value and |difference| for every result that the m = 1 patch could move.

old = campaign_R/original_prepatch/   (files as deposited before the patch)
new = the current files in data/results/ and rr_results/
Files not listed here were rerun with the patched router and are identical in every non-timing field (see campaign_R/out/rerun_logs/).
Run:  python diff_tables.py
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "campaign_R", "original_prepatch")
ld = lambda *p: json.load(open(os.path.join(*p)))
rows_out = []


def rec(table, label, old, new):
    rows_out.append((table, label, old, new))


# ---- Table 7 (RR1)
o = ld(OLD, "rr_results", "RR1_decomposition_R_family.json"); n = ld(HERE, "rr_results", "RR1_decomposition_R_family.json")
for k in o:
    for f in ("n_feasible", "total_BMR_SPF", "ord1_budget", "ord1_validity", "objective"):
        rec("Table 7 / RR1", f"{k} {f}", o[k][f], n[k][f])
    rec("Table 7 / RR1", f"{k} CI low", o[k]["BMR_SPF_ci95"][0], n[k]["BMR_SPF_ci95"][0])
    rec("Table 7 / RR1", f"{k} CI high", o[k]["BMR_SPF_ci95"][1], n[k]["BMR_SPF_ci95"][1])

# ---- RR2, RR4 (no change expected)
for fn in ("RR2_ring_flagship_R_family.json", "RR4_dynamic_alarm.json"):
    o = ld(OLD, "rr_results", fn); n = ld(HERE, "rr_results", fn)
    if fn.startswith("RR2"):
        for k in o:
            for f, v in o[k].items():
                if isinstance(v, (int, float)):
                    rec("Table 8 / RR2", f"{k} {f}", v, n[k][f])
    else:
        for i, (ro, rn) in enumerate(zip(o["rows"], n["rows"])):
            for f, v in ro.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rec("Table 8 / RR4", f"{ro['topo']} n={ro['n_audit']} S={ro['S1']} {f}", v, rn[f])

# ---- RR5 and q9 (Table 10)
o = ld(OLD, "rr_results", "RR5_cross_validation.json"); n = ld(HERE, "rr_results", "RR5_cross_validation.json")
for k, v in o.items():
    if isinstance(v, (int, float)):
        rec("Table 10b / RR5", k, v, n[k])
o = ld(OLD, "q9_solver_validation.json"); n = ld(HERE, "data", "results", "q9_solver_validation.json")
for k in ("total", "feasible", "mismatches"):
    rec("Table 10 / q9", k, o["validation"][k], n["validation"][k])

# ---- q4
o = ld(OLD, "q4_collusion.json"); n = ld(HERE, "data", "results", "q4_collusion.json")
for ro, rn in zip(o["rows"], n["rows"]):
    for f in ("worst_case_pct", "mean_hops", "mean_rate"):
        rec("q4 collusion", f"{ro['topology']} {ro['router']} {f}", ro[f], rn[f])

# ---- Table 6 headline (q6_sweep) and phi = 0.9 variant
import statistics
for fn, loc_old in (("q6_sweep_phi09.json", os.path.join(OLD, "data_results", "q6_sweep_phi09.json")),):
    o = ld(loc_old); n = ld(HERE, "data", "results", fn)
    for ro, rn in zip(o["rows"], n["rows"]):
        for f in ("mean_rate", "compl_pct", "refusal_pct"):
            rec("q6 phi=0.9", f"{ro['topo']} {ro['n']} {ro['n_audit']} {ro['cond']} {ro['mode']} {ro['router']} {f}", ro[f], rn[f])

# ---- q15 Delta regime
o = ld(OLD, "data_results", "q15_delta_regime.json")["rows"]; n = ld(HERE, "data", "results", "q15_delta_regime.json")["rows"]
for ro, rn in zip(o, n):
    for f in ("feasible_pct", "mean_rate"):
        rec("q15 Delta regime", f"n={ro['n_audit']} phi={ro['phi']} D={ro['Delta']:g} fcomp={ro['fcomp']} {f}", ro[f], rn[f])

# ---- q9b sparse timing (feasibility only; timings are machine dependent)
o = ld(OLD, "data_results", "q9b_sparse_timing.json"); n = ld(HERE, "data", "results", "q9b_sparse_timing.json")
for ro, rn in zip(o, n):
    rec("q9b sparse", f"mean_deg={ro['mean_deg']} n={ro['n']} t={ro['t']} feasible_pct", ro["feasible_pct"], rn["feasible_pct"])

# ---- print
import sys
verbose = "--all" in sys.argv
by = {}
for t, l, a, b in rows_out:
    d = abs(a - b)
    by.setdefault(t, []).append((l, a, b, d))
print(f"{'table':22s} {'values':>7s} {'changed':>8s} {'max |diff|':>12s}")
for t, v in by.items():
    ch = [x for x in v if x[3] > 1e-12]
    print(f"{t:22s} {len(v):7d} {len(ch):8d} {max(x[3] for x in v):12.6g}")
print()
for t, v in by.items():
    ch = [x for x in v if x[3] > 1e-12]
    if not ch:
        continue
    print(f"--- {t}: changed values (old -> new, |diff|)" + ("" if verbose or len(ch) <= 30 else f" - first 30 of {len(ch)}; use --all"))
    for l, a, b, d in (ch if verbose else ch[:30]):
        print(f"  {l:70s} {a:12.6g} -> {b:12.6g}  {d:10.6g}")
