"""check_counts.py - which hardware runs have raw counts, and the text of the Table 11 footnote.

Reads data/results/q8_hardware_reconcile.json (S0 table, 9 runs), q8_hw_refetched.json, q8_hw_repeats*.json,
hardware_fez/*.json and the job-metadata files. Prints one line per job and the footnote that follows from them.
Run:  python check_counts.py
"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "data", "results")
ld = lambda *p: json.load(open(os.path.join(RES, *p)))


def S_from_counts(c4):
    """c4: four count dicts for (a,b) = 00, 01, 10, 11 at phi = 0; S = E00 + E01 + E10 - E11."""
    E = [(c["00"] + c["11"] - c["01"] - c["10"]) / sum(c.values()) for c in c4]
    return E[0] + E[1] + E[2] - E[3]


# ---- raw counts available per marrakesh job id (phi = 0 settings)
raw = {}
for jid, v in ld("q8_hw_refetched.json").items():
    if v.get("backend") == "ibm_marrakesh" and v.get("raw_counts_by_pub_index"):
        c = v["raw_counts_by_pub_index"][:4]
        if all(c):
            raw[jid] = ("q8_hw_refetched.json", S_from_counts(c), v.get("created_utc"))
for f in ("q8_hw_repeats.json", "q8_hw_repeats_14oct.json"):
    d = ld(f)
    for r in d["runs"]:
        rc = r["raw_counts"]
        c = [rc[f"0.0|{a}|{b}"] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
        raw[r["job_id"]] = (f, S_from_counts(c), r["utc"])

rec = ld("q8_hardware_reconcile.json")
refetched = ld("q8_hw_refetched.json")
runs = [(r["run"], r["job"], r["S0"]) for r in rec["S0_table"]]
# the 12 Oct run (job db659...) enters generate_macros.py as the tenth run
J12 = "db659bklf4us73c4qju0"
runs.append(("marrakesh 12 Oct (job in q8_hw_refetched.json)", J12, raw[J12][1] if J12 in raw else None))

print(f"{'#':>2} {'job':24s} {'S0 (stored)':>12s} {'raw counts?':>12s} {'S0 from counts':>15s}  source")
have = []
for i, (name, job, s0) in enumerate(runs, 1):
    r = raw.get(job)
    print(f"{i:>2} {job:24s} {s0:12.4f} {'yes' if r else 'NO':>12s} {(f'{r[1]:.4f}' if r else '-'):>15s}  {r[0] if r else name}")
    have.append(bool(r))
no_raw = [i for i, h in enumerate(have, 1) if not h]
print("\nruns without raw counts:", no_raw or "none")
bad = [(job, s0, raw[job][1]) for _, job, s0 in runs if job in raw and abs(raw[job][1] - s0) > 5e-4]
print("runs where S0 recomputed from raw counts differs from the stored value by > 5e-4:", bad or "none")

S = [s for _, _, s in runs]
print(f"\nall {len(S)} runs: mean {statistics.mean(S):.4f}, SD {statistics.stdev(S):.4f}")
drop = lambda idx: [s for i, s in enumerate(S, 1) if i not in idx]
if no_raw:
    sub = drop(no_raw)
    print(f"runs {no_raw} without raw counts, excluded: n = {len(sub)}, mean {statistics.mean(sub):.4f}, SD {statistics.stdev(sub):.4f}")

# ---- fez data sets
print("\nfez / old data sets (job ids, counts present in file):")
x13 = ld("hardware_fez", "06_ibm_chsh_results.json")
n_c = sum(1 for g in x13["hardware_results"].values() for k, v in g.items() if isinstance(v, dict) and "counts" in v)
print(f"  X13 native + degraded (06_ibm_chsh_results.json): {len(x13['job_ids'])} job ids, {n_c} settings with counts")
p24 = ld("hardware_fez", "P24_combined5.json")
jobs = sorted({(r["cond"], r["rep"], r["job"]) for r in p24["rows"]})
print(f"  P24 (P24_combined5.json): {len(jobs)} (cond, rep, job) triples, all rows with counts: {all(r.get('counts') for r in p24['rows'])}")

print("\nqubits read from the files / job metadata:")
for f in ("q8_hw_repeats.json", "q8_hw_repeats_14oct.json"):
    d = ld(f)
    print(f"  {f}: backend {d['backend']}, qubit_pair {d['qubit_pair']}")
for f in ("x13_fez_job_metadata.json", os.path.join("hardware_fez", "p24_fez_job_metadata.json")):
    d = ld(f)
    print(f"  {f}: used_qubits {sorted({json.dumps(v.get('used_qubits')) for v in d.values()})}, backends {sorted({v.get('backend') for v in d.values()})}")

# ---- footnote
print("\nTable 11 footnote generated from the above:")
if not no_raw:
    print('  "Raw counts are retained for all ten runs; the job identifiers are listed in the deposit (data/results/q8_*.json)."')
else:
    lst = " and ".join(str(i) for i in no_raw)
    print(f'  "Raw counts for run(s) {lst} were not retained; their S-hat values are taken from archived results."')
