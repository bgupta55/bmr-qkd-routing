"""make_macros.py - write results_macros.tex (blueprint L.8) from the result files. No number is typed here.

Step 1  runs experiments/generate_macros.py (Q4, Q5, Q6, Q8, Q9, Q9b, Q11, Q15 macros).
Step 2  renames the router macros to the fixed v14 names (Qpathstyle -> FidelityMax, Qcaststyle -> HopMinF, Rategreedy -> RateGreedy).
Step 3  appends: Table 6 gaps and floor price, collusion ranges, n_min and eta ratio, Table 7 (RR1), alarm cost and exposure (Table 12),
        Lagrangian count, cells, sensitivity (q15_delta_regime, 56 rows), drift (q15_drift_robustness, 40 rows), tamper tests (q11),
        hardware day means and fez paired test, RR2 false alarms and detection epochs, solver-validation counts.
Run:  python make_macros.py
"""
import json, math, os, re, subprocess, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "data", "results")
ld = lambda *p: json.load(open(os.path.join(*p)))
OUT = os.path.join(HERE, "results_macros.tex")

subprocess.run([sys.executable, os.path.join(HERE, "experiments", "generate_macros.py")], check=True, cwd=HERE)
txt = open(OUT).read()
for a, b in (("Qpathstyle", "FidelityMax"), ("Qcaststyle", "HopMinF"), ("Rategreedy", "RateGreedy"),
             ("Qpath-style", "fidelity-max"), ("Qcast-style", "hop-min-F")):
    txt = txt.replace(a, b)
lines = txt.rstrip("\n").split("\n")

DIG = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def cname(raw):
    return "".join(c for c in "".join(DIG.get(c, c) if c.isdigit() else c for c in raw) if c.isalpha())


def mc(name, value, comment="", fmt=None):
    s = format(value, fmt) if fmt else (f"{value:.4g}" if isinstance(value, float) else str(value))
    lines.append(rf"\newcommand{{\{cname(name)}}}{{{s}}} % {comment}")


def sec(title):
    lines.extend(["", f"% === {title} ==="])


def dtok(x):
    m, e = f"{x:.0e}".split("e")
    return f"{DIG[m]}EMinus{DIG[str(abs(int(e)))]}"


# ------------------------------------------------------------------ Table 6 prices
sec("TABLE 6: budget-only gaps, floor price (q6_sweep.json)")
rows = ld(RES, "q6_sweep.json")["rows"]
bo = {}
for r in rows:
    bo.setdefault(r["router"], []).append(r["mean_rate_budget_only"])
m = {k: sum(v) / len(v) for k, v in bo.items()}
mc("qSixRateGapBudgetOnlyOne", m["BMR"] - m["B1a-Plambda(0.01)"], "BMR minus B1a, budget-only mean rate", "+.4f")
mc("qSixRateGapBudgetOnlyTwo", m["BMR"] - m["B2-BMR-NoFloor"], "BMR minus B2 (floor-less BMR), budget-only mean rate", "+.4f")
mc("qSixRateGapBudgetOnlyThree", m["BMR"] - m["B3-Zhang-style"], "BMR minus B3, budget-only mean rate", "+.4f")
mc("qSixFloorCostPct", 100 * (m["B2-BMR-NoFloor"] - m["BMR"]) / m["B2-BMR-NoFloor"], "relative budget-only rate price of the hop floor, %", ".1f")
cells = {(r["topo"], r["n"], r["n_audit"], r["cond"], r["mode"]) for r in rows}
mc("cellsN", len(cells), "cells of the Table 6 sweep (cells_table.py)")
mc("cellsTrials", sorted({r["trials"] for r in rows})[0], "trials per cell")

# ------------------------------------------------------------------ collusion (q4)
sec("Q4 COLLUSION (worst-case exposure %, t = 1)")
q4 = ld(RES, "q4_collusion.json")["rows"]
sel = lambda name: [r["worst_case_pct"] for r in q4 if r["router"] == name]
mc("collusionNoFloorLow", min(sel("BMR-NoFloor")), "BMR-NoFloor, min over topologies", ".1f")
mc("collusionNoFloorHigh", max(sel("BMR-NoFloor")), "BMR-NoFloor, max over topologies", ".1f")
mc("collusionSPFLow", min(sel("SPF")), "SPF, min over topologies", ".1f")
mc("collusionSPFHigh", max(sel("SPF")), "SPF, max over topologies", ".1f")
mc("collusionBMRMax", max(sel("BMR(t=1)")), "BMR(t=1), max over topologies", ".1f")

# ------------------------------------------------------------------ bounds
sec("BOUNDS (bmr/bounds.py)")
from bmr.bounds import n_min
mc("nMinTight", n_min(2.687, 1e-6, "tight"), "n_min, eta_tight, S = 2.687, delta = 1e-6")
mc("nMinUnion", n_min(2.687, 1e-6, "union"), "n_min, eta_union, S = 2.687, delta = 1e-6")
mc("etaRatio", math.sqrt(32 * math.log(8 / 1e-6)) / math.sqrt(8 * math.log(1 / 1e-6)), "eta_union / eta_tight at delta = 1e-6", ".2f")

# ------------------------------------------------------------------ Table 7 (RR1)
sec("TABLE 7 / RR1 (rr_results/RR1_decomposition_R_family.json)")
rr1 = ld(HERE, "rr_results", "RR1_decomposition_R_family.json")
for fam, tag in (("er", "ER"), ("ba", "BA")):
    t = [v["total_BMR_SPF"] for v in rr1.values() if v["topo"] == fam]
    mc(f"gapMin{tag}", min(t), f"min BMR-SPF gap, {tag}", ".3f")
    mc(f"gapMax{tag}", max(t), f"max BMR-SPF gap, {tag}", ".3f")
allv = list(rr1.values())
mc("gapCellsN", len(allv), "number of cells")
mc("gapMinCILower", min(v["BMR_SPF_ci95"][0] for v in allv), "smallest lower 95% CI bound over the cells", ".4f")
mc("gapAllPositive", sum(v["BMR_SPF_ci95"][0] > 0 for v in allv), "cells whose 95% CI excludes zero")
mc("gapBudgetShareLowPct", 100 * min(v["ord1_budget"] / v["total_BMR_SPF"] for v in allv), "budget share of the gap, min %", ".0f")
mc("gapBudgetShareHighPct", 100 * max(v["ord1_budget"] / v["total_BMR_SPF"] for v in allv), "budget share of the gap, max %", ".0f")
mc("gapObjectiveShareLowPct", 100 * min(v["objective"] / v["total_BMR_SPF"] for v in allv), "objective share of the gap, min %", ".0f")
mc("gapObjectiveShareHighPct", 100 * max(v["objective"] / v["total_BMR_SPF"] for v in allv), "objective share of the gap, max %", ".0f")
mc("gapMinAll", min(v["total_BMR_SPF"] for v in allv), "smallest gap, all cells", ".3f")
mc("gapMaxAll", max(v["total_BMR_SPF"] for v in allv), "largest gap, all cells", ".3f")
a_all = [v["ord1_validity"] for v in allv]
mc("alarmTermAllLow", min(a_all), "alarm term (D3 - D2), all 12 cells, min", "+.4f")
mc("alarmTermAllHigh", max(a_all), "alarm term (D3 - D2), all 12 cells, max", "+.4f")
a3 = [v["ord1_validity"] for k, v in rr1.items() if k.endswith("D3e-05")]
mc("alarmTermLow", min(a3), "alarm term at Delta = 3e-5, 6 cells, min", "+.4f")
mc("alarmTermHigh", max(a3), "alarm term at Delta = 3e-5, 6 cells, max", "+.4f")
mc("alarmTermZeroCells", sum(abs(x) < 1e-12 for x in a3), "cells at Delta = 3e-5 where the alarm term is exactly zero")

# ------------------------------------------------------------------ Table 8 / RR2
sec("RR2 (rr_results/RR2_ring_flagship_R_family.json)")
rr2 = ld(HERE, "rr_results", "RR2_ring_flagship_R_family.json")
mc("falseAlarmsObs", sum(v["fa_pre_attack"] + v["fa_post_attack_other_links"] for v in rr2.values()), "false alarms, all six cells")
mc("falseAlarmsExp", sum(v["expected_fa_at_alpha_1e-6"] for v in rr2.values()), "expected false alarms at alpha = 1e-6", ".2f")
mc("falseAlarmDecisions", sum(v["clean_decisions"] for v in rr2.values()), "clean decisions, all six cells")
mc("firstDetEpochObsOne", rr2["ring8_Satt2.62_D1e-03"]["alarm_mean_detect_epoch"], "S = 2.62, Delta = 1e-3", ".2f")
mc("firstDetEpochObsTwo", rr2["ring8_Satt2.62_D3e-05"]["alarm_mean_detect_epoch"], "S = 2.62, Delta = 3e-5", ".2f")
mc("gatePassMin", min(v["gate_pass_rate"] for v in rr2.values()), "smallest gate pass rate over the six cells", ".3f")

# ------------------------------------------------------------------ Table 12 (E-A)
sec("TABLE 12 / E-A (campaign_R/out/out_E_A2/RR4_dynamic_alarm.json; epochs 25-34, mean of ring/grid/ER)")
import exposure_table as ex
T = ex.table(); TS = ex.true_secure()
c = T[(10000, 2.62)]
mc("expoSPF", c["spf"]["exp"], "SPF exposure, S = 2.62, n = 1e4", ".3f")
mc("expoNoAlarm", c["bmr"]["exp"], "BMR exposure, S = 2.62, n = 1e4", ".3f")
mc("expoAlarm", c["bma"]["exp"], "BMR + alarm exposure, S = 2.62, n = 1e4", ".3f")
mc("expoDelta", c["bmr"]["exp"] - c["bma"]["exp"], "BMR minus BMR + alarm exposure", ".3f")
mc("expoDeliveredAlarm", c["bma"]["del"], "delivered fraction with alarm", ".3f")
mc("expoAlarmLowS", max(T[(10000, s)]["bma"]["exp"] for s in (2.55, 2.48)), "BMR + alarm exposure at S = 2.55 and 2.48, n = 1e4, max", ".3f")
mc("expoNoAlarmLowS", min(T[(10000, s)]["bmr"]["exp"] for s in (2.55, 2.48)), "BMR exposure at S = 2.55 and 2.48, n = 1e4, min", ".3f")
mc("expoFiveHundredEqual", int(all(abs(T[(500, s)]["bmr"]["exp"] - T[(500, s)]["bma"]["exp"]) < 1e-12 for s in (2.62, 2.55, 2.48, 2.418))), "1 if the alarm changes nothing at n = 500")
t = TS[(10000, 2.62)]
mc("alarmCost", t["bmr_cr"] - t["bma_cr"], "compliant-rate cost of the alarm, S = 2.62, n = 1e4, post-1 window", ".4f")
mc("trueSecureBMR", t["bmr_ts"], "true-secure rate BMR, S = 2.62, n = 1e4, post-1", ".4f")
mc("trueSecureAlarm", t["bma_ts"], "true-secure rate BMR + alarm, S = 2.62, n = 1e4, post-1", ".4f")

# ------------------------------------------------------------------ Lagrangian heuristic (q14)
sec("Q14 LAGRANGIAN HEURISTIC (q14_twobudget_v2.json): a miss = a feasible instance for which the heuristic finds no feasible route")
q14 = ld(RES, "q14_twobudget_v2.json")
feas = [round(r["feasible"] * r["inst"]) for r in q14["random"]]
miss = [round((r["lagr_miss_rate"] or 0.0) * f) for r, f in zip(q14["random"], feas)]
mc("lagN", sum(feas), "feasible random instances (exact DP)")
mc("lagMiss", sum(miss), "of these, instances where the Lagrangian heuristic finds no feasible route")
dm = q14["diamond"]
mc("lagDiamondN", len(dm), "diamond-chain instances (tight budgets, a feasible route exists)")
mc("lagDiamondMiss", sum(1 for x in dm if x["dp_feasible"] and not x["lagr_feasible"]), "diamond instances where the heuristic finds no feasible route")

# ------------------------------------------------------------------ sensitivity (q15_delta_regime)
sec("SENSITIVITY (q15_delta_regime.json, 56 rows): feasible % by n, phi, Delta, f_comp")
seen = {}
for r in ld(RES, "q15_delta_regime.json")["rows"]:
    base = f"sensFeasN{r['n_audit']}Phi{int(round(r['phi'] * 100))}D{dtok(r['Delta'])}F{int(round(r['fcomp'] * 10))}"
    if base in seen:   # the baseline cell is in both the Delta sweep and the f_comp sweep; the two runs must agree to 0.05
        assert abs(seen[base] - r["feasible_pct"]) < 0.05, base
        continue
    seen[base] = r["feasible_pct"]
    mc(base, r["feasible_pct"], f"n={r['n_audit']} phi={r['phi']} Delta={r['Delta']:g} f_comp={r['fcomp']}: feasible %", ".1f")

# ------------------------------------------------------------------ drift (q15_drift_robustness)
sec("DRIFT ROBUSTNESS (q15_drift_robustness.json, 40 rows)")
dr = ld(RES, "q15_drift_robustness.json")["rows"]
brk = [r for r in dr if r["honest_pass"] < 0.99]
ok = [r for r in dr if r["honest_pass"] >= 0.99]
mc("driftRowsN", len(dr), "rows (10 variants x n x phi)")
mc("driftBreakN", len(brk), "rows with honest pass rate below 0.99")
mc("driftBreakFAMin", min(r["fa"] for r in brk) * 100 if brk else 0.0, "false-alarm rate of the breaking rows, min %", ".1f")
mc("driftBreakFAMax", max(r["fa"] for r in brk) * 100 if brk else 0.0, "false-alarm rate of the breaking rows, max %", ".1f")
mc("driftBreakHonestPassMin", min(r["honest_pass"] for r in brk) if brk else 1.0, "honest pass rate of the breaking rows, min", ".3f")
mc("driftOtherFAMax", max(r["fa"] for r in ok) * 100, "false-alarm rate, all other rows, max %", ".1f")
mc("driftOtherHonestPassMin", min(r["honest_pass"] for r in ok), "honest pass rate, all other rows, min", ".4f")
mc("driftViolMax", max(r["viol"] for r in ok) * 100, "budget violation when accepted, all other rows, max %", ".1f")
mc("driftDetectedMin", min(r["detected"] for r in dr), "detection rate, all rows, min", ".3f")

# ------------------------------------------------------------------ tamper tests (q11)
sec("TAMPER TESTS (q11_alg3_v2.json)")
q11 = ld(RES, "q11_alg3_v2.json")
N = q11["n_per_test"]
tests = q11["tests"]
rej = [x for x in tests if x["expected"].startswith("reject")]
mc("tamperTrials", N, "trials per test")
mc("tamperTestsReject", len(rej), "tests expected to reject (T1 ... T11b and T12-DB)")
mc("tamperAcceptT0", round([x for x in tests if x["test"].startswith("T0")][0]["accept_rate"] * N), "benign certificates accepted")
mc("tamperRejectMin", min(round((1 - x["accept_rate"]) * N) for x in rej), "smallest number of rejections among the reject tests")
mc("tamperAcceptedMax", max(round(x["accept_rate"] * N) for x in rej), "most acceptances among the reject tests")
mc("tamperAcceptTwelve", round([x for x in tests if x["test"] == "T12 lying attestor, ledger-only"][0]["accept_rate"] * N), "T12 (documented limit) accepted")
z = 1.959963984540054
p0 = 0.0
wil = (p0 + z * z / (2 * N) + z * math.sqrt(p0 * (1 - p0) / N + z * z / (4 * N * N))) / (1 + z * z / N)
mc("tamperWilsonUpperPct", 100 * wil, "Wilson 95% upper bound for 0 acceptances in N trials, %", ".2f")

# ------------------------------------------------------------------ solver validation
sec("SOLVER VALIDATION (q9 + RR5)")
q9 = ld(RES, "q9_solver_validation.json")["validation"]
r5 = ld(HERE, "rr_results", "RR5_cross_validation.json")
mc("yenInstances", q9["total"] + r5["total_instances"], "q9 instances + RR5 instances")
mc("yenMismatches", q9["mismatches"] + r5["mismatches"], "mismatches q9 + RR5")
mc("rrFiveInstances", r5["total_instances"], "RR5 instances")
mc("rrFiveInfeasibleBoth", r5["infeasible_both"], "RR5: infeasible for both solvers")
mc("rrFiveMatches", r5["matches"], "RR5: both feasible with equal value")

# ------------------------------------------------------------------ hardware
sec("HARDWARE (q8_hw_repeats*.json raw counts; hardware_fez/P24_combined5.json)")
from scipy import stats


def S_of(rc):
    E = [(lambda c: (c["00"] + c["11"] - c["01"] - c["10"]) / sum(c.values()))(rc[f"0.0|{a}|{b}"]) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
    return E[0] + E[1] + E[2] - E[3]


d1 = [S_of(r["raw_counts"]) for r in ld(RES, "q8_hw_repeats.json")["runs"]]
d2 = [S_of(r["raw_counts"]) for r in ld(RES, "q8_hw_repeats_14oct.json")["runs"]]
mc("hwDayOneN", len(d1), "runs, 13 Oct repeats")
mc("hwDayTwoN", len(d2), "runs, 14 Oct repeats")
mc("hwDayOneMean", statistics.mean(d1), "mean S-hat(0), 13 Oct repeats", ".3f")
mc("hwDayTwoMean", statistics.mean(d2), "mean S-hat(0), 14 Oct repeats", ".3f")
mc("hwDayDiff", statistics.mean(d2) - statistics.mean(d1), "day 2 minus day 1", ".4f")
mc("hwWelchT", stats.ttest_ind(d2, d1, equal_var=False).statistic, "Welch t, day 2 vs day 1", ".2f")
p24 = ld(RES, "hardware_fez", "P24_combined5.json")
mc("fezPairedDiff", p24["paired_diff_mean"], "ibm_fez, degraded minus intended-sign, paired mean", ".3f")
mc("fezPairedSE", p24["paired_diff_sem"], "standard error of the paired mean", ".4f")
mc("fezPairedT", p24["t"], "paired t", ".1f")

open(OUT, "w").write("\n".join(lines) + "\n")
n_new = sum(1 for l in lines if l.startswith(r"\newcommand"))
print(f"Written {len(lines)} lines, {n_new} macros, to {OUT}")
