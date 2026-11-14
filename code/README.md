# Bell-Metered Routing (BMR): Code and Data Deposit

**Paper:** *Bell-Metered Routing (BMR): Security-Budget-Constrained Path Selection for QKD over Optical Networks*  
**Journal:** Optical Switching and Networking (Elsevier) — major revision, October 2026  
**Authors:** [authors] · **Proof reader for theorems:** Sanjeev Rana

---

## Overview

BMR is a routing framework for quantum key distribution (QKD) networks. Each link runs a live CHSH Bell audit, generating a finite-key certificate `(r, ε, δ, γ, T)`. A route is a *simple path* whose summed failure bound Σ(ε+δ) stays within a budget Δ and whose length satisfies a collusion floor (m = 1, or m ≥ 2t+1). A calibrated alarm detects link degradation independently of the certificate. The destination verifies a signed, session-bound ledger.

This deposit contains all code and data needed to reproduce every number in the paper. One command regenerates all results:

```bash
python run_all.py
```

---

## Repository Structure

```
.
├── bmr/                        # Core library
│   ├── bounds.py               # η_tight / η_union radii; h_pironio entropy function
│   ├── certificate.py          # Algorithm 1: build_certificate() from audit counts
│   ├── route.py                # Algorithm 2: bmr_route() (Yen); brute_force_max_rate()
│   ├── sim.py                  # LinkSim: binomial CHSH sampling
│   └── verify.py               # Ledger verification (Algorithm 3)
│
├── experiments/                # Numbered experiments (each self-contained)
│   ├── e1_coverage.py          # η_tight / η_union coverage test (Table 4)
│   ├── e2_table.py             # Reference-value table (Table 4, §8.1)
│   ├── e3_certificate_tests.py # Unit + integration tests (Algorithm 1)
│   ├── e4_paths.py             # Walk counter-examples (§8.3)
│   ├── e5_runtime.py           # Solver timing sweep (Table 11, §8.10)
│   ├── e6_hardware.py          # IBM Quantum φ-sweep (§8.13)  ← credentials via env
│   ├── e6v2_hardware_fixed.py  # IBM Quantum 10-run repeat + fez opposite-sign test
│   ├── e10_ledger.py           # Ledger overhead (Table 10, §8.11)
│   ├── q4_collusion.py         # Collusion exposure (§8.3)
│   ├── q5_evidence_source.py   # Bell vs QBER evidence (§8.4)
│   ├── q6_common.py            # Router definitions used by sweep + dynamic
│   ├── q6_sweep.py             # 200-cell sweep (Table 6, §8.5)
│   ├── q6_dynamic.py           # Dynamic compromise study (Table 8, §8.2)
│   ├── q6_dynamic_latency.py   # Alarm latency detail (§8.2)
│   ├── q6_sweep_degenerate.py  # Degenerate topologies (ring, star, line)
│   ├── q7_e0_audit.py          # Audit design window (§4.2)
│   ├── q8_hw_repeats.py        # Hardware repeat runs (§8.13)
│   ├── q8_hw_repeats_14oct.py  # 14 Oct session, 10 runs, raw counts
│   ├── q8_hardware_reconcile.py# Session drift analysis
│   ├── q9_solver_validation.py # 36,000-instance Yen vs brute-force (§8.10)
│   ├── q9b_sparse_timing.py    # Timing on sparse graphs (§8.10)
│   ├── q11_alg3_v2.py          # Ledger attack table T0–T12 (§8.11)
│   ├── q13_scope_numbers.py    # Scope / certificate byte layout (§3.2)
│   ├── q14_twobudget_v2.py     # Two-budget solver study (§8.10, supplement)
│   ├── q15_delta_regime.py     # Regime map / Δ-feasibility (Fig. 9, §8.6)
│   ├── q15_drift_robustness.py # Drift robustness (§8.12)
│   └── generate_macros.py      # Base macros (called by make_macros.py)
│
├── rr_results/                 # (R)-family re-runs (new, October 2026)
│   ├── RR1_decomposition_R_family.json   # Decomposition ER+BA N=30 (Table 7)
│   ├── RR2_ring_flagship_R_family.json   # Ring flagship 500 trials (Fig. 5/6)
│   ├── RR4_dynamic_alarm.json            # Dynamic study + BMR+alarm column (Table 8)
│   └── RR5_cross_validation.json         # Yen vs brute-force 4,968 instances (§8.10)
│
├── data/results/               # All pre-computed result JSON files
│   ├── q6_sweep.json           # 200-cell sweep (main results)
│   ├── q6_dynamic.json         # Dynamic compromise (100 reps)
│   ├── q9_solver_validation.json # 36,000 solver instances
│   ├── q8_hw_repeats_14oct.json  # IBM Quantum runs 5–9, raw counts + job IDs
│   └── ...                     # (see SHA256SUMS.txt for full list)
│
├── tests/                      # Unit and integration tests
├── run_all.py                  # Reproduces every result in sequence
├── run_pending_fast.py         # (R)-family RR1+RR2+RR5 re-runs (200 trials)
├── run_rr4_alarm.py            # (R)-family RR4 dynamic + alarm column
├── results_macros.tex          # LaTeX macros — every number in the paper
└── SHA256SUMS.txt              # SHA-256 checksums of all source and result files
```

---

## Key Parameters

| Symbol | Value | Role |
|--------|-------|------|
| `n` | 10,000 | audit rounds per setting (4n total) |
| `δ` | 10⁻⁶ | per-link soundness |
| `φ` | 0.99 | extraction fraction (sealed) |
| `ℓ` | 10,000 | raw rounds per key block |
| `S₀` | 2.687 | honest link baseline |
| `η_tight` | √(8 ln(1/δ)/n) | finite-sample radius (primary) |
| `R_MIN` | 0.01 | minimum acceptable rate |
| `t` | 1 | collusion floor (m = 1 or hop ≥ 3) |
| `Δ` | 10⁻³ | security budget (main) |
| `α` | 10⁻⁶ | alarm false-alarm rate |

---

## Quick Start

```bash
# 1. Clone and install
git clone <this-repo-url>
cd bmr-qkd-routing
pip install numpy scipy networkx

# 2. Reproduce the reference-value table (Table 4)
python experiments/e2_table.py

# 3. Run the main sweep (Table 6, ~5 min)
python experiments/q6_sweep.py

# 4. Run (R)-family decomposition (Table 7, ~2 min)
python run_pending_fast.py

# 5. Regenerate all LaTeX macros
python make_macros.py
```

**IBM Quantum hardware scripts** (`e6_hardware.py`, `e6v2_hardware_fixed.py`, `q8_hw_repeats_14oct.py`) require a valid IBM Quantum account. Set the environment variable before running — **never paste the token into code**:

```bash
export IBM_QUANTUM_TOKEN="your-token-here"
python experiments/e6v2_hardware_fixed.py
```

---

## Results Summary

| Contribution | Key number | Source |
|---|---|---|
| Minimum audit rounds | **n_min = 235** (η_tight), 1,078 (η_union) | `e2_table.py` |
| BMR compliance | **92.8%** vs ≤78.1% (all competitors) | `q6_sweep.json` |
| Collusion exposure | **0.0%** (BMR) vs 10.6–69.8% (others) | `q4_collusion.json` |
| Rate cost of floor | **−0.0028** vs floor-free (2.2%) | `q6_sweep.json` |
| Solver validation | **0 / 40,968 mismatches** (Yen vs brute-force) | `q9_solver_validation.json` + `RR5_cross_validation.json` |
| Alarm detection | **500/500** at S†=2.55, mean first-detection epoch **26.5** at S†=2.62 | `RR2_ring_flagship_R_family.json` |
| IBM Quantum plausibility | Ŝ = **2.704 ± 0.066** (ten runs); fez paired diff **−0.189 ± 0.004** | `q8_hw_repeats*.json`, `q8_hw_refetched.json`, `hardware_fez/P24_combined5.json` |

---

## Experiment Families

### (R) Family — derived ε (main text)
Uses η_tight, ε derived from each audit, n=10⁴, φ=0.99, Δ∈{10⁻³, 3×10⁻⁵}. Every trial draws an independent graph. Scripts: `q6_sweep.py`, `run_pending_fast.py`, `run_rr4_alarm.py`.

### (D) Family — declared class ε (Appendix D)
Uses ε from a fixed class table, φ=1.0, Δ=9×10⁻⁶, 50 graphs × 5 draws. Results in `data/results/` prefixed `R*` and `L*`. BMR−SPF = +0.2127 [+0.194, +0.232] under these settings.

---

## Artefact → k-rule map

Every result file below was produced with `certificate.py` (k = ⌊φH⌋). This was verified on 19 Oct 2026 by regenerating the files with the released code (`campaign_R/` in the revision folder): `q6_sweep.json` (2,000 rows), `RR2`, `RR4` and `q4_collusion` (exposure values) reproduce exactly; `RR1` reproduces exactly with the pre-patch code.

| File | k rule | Basis |
|---|---|---|
| `RR1_decomposition_R_family.json` | k = ⌊φH⌋ | regenerated with the m = 1 router and SPF scored with the hop floor waived (`run_pending_fast.py`) |
| `RR2_ring_flagship_R_family.json` | k = ⌊φH⌋ | regenerated, identical to the earlier file |
| `RR4_dynamic_alarm.json` | k = ⌊φH⌋ | regenerated, identical to the earlier file |
| `q6_sweep.json` | k = ⌊φH⌋ | regenerated, identical (no request has hop distance 1) |
| `q9_solver_validation.json` | k = ⌊φH⌋ | solver comparison independent of k |
| `RR5_cross_validation.json` | not applicable | r and w assigned directly; `build_certificate()` not called |
| `q11_alg3_v2.json` | k = ⌊φH⌋ | regenerated, identical |
| Table 4 / `cert_check.py` | k = ⌊φH⌋ | reference implementation |

`check_k.py` returns "no stored k" for every result file (the files store aggregates, not k), so it cannot decide the k rule; the regeneration test above does.

`certificate.py` implements Algorithm 1 with k = ⌊φH⌋. In result files and plotting code the objective-only references B6 and B7 of the paper appear under the internal labels `B6-Qpath-style` and `B7-Qcast-style`; they do not implement Q-PATH, Q-LEAP or Q-CAST.

## Direct link (m = 1)

`bmr_route` admits a route of hop count m when `m == 1` or `m >= m_min` (m_min = 2t+1). A direct link has no relay and cannot be broken by a relay coalition (Corollary 1: a route is exposed iff 2 ≤ m ≤ 2t). `tests/test_direct_link.py` checks this. `tests/test_cor1.py` checks Corollary 1 by brute force for m ≤ 13, t ≤ 5.

## Tests

```bash
pip install pytest
python -m pytest tests -q          # 13 tests: Corollary 1, eta, certificate vectors, direct link, integration
python cells_table.py              # factor counts and per-router aggregates of the Table 6 sweep
```

## Helper scripts (each prints a plain-text table from the stored files; no number is typed in)

| Script | What it prints or writes |
|---|---|
| `make_macros.py` | Writes `results_macros.tex`: runs `experiments/generate_macros.py`, applies the v14 macro names, adds the Table 7, Table 12, Lagrangian, sensitivity, drift, tamper, hardware and solver-count macros |
| `exposure_table.py` | Table 12: exposure of delivered routes and true-secure rate, from `campaign_R/out/out_E_A2/RR4_dynamic_alarm.json` |
| `check_counts.py` | Job IDs, raw-count availability, qubits, and the Table 11 footnote |
| `independence_check.py` | Repeated graphs per cell for Table 6 (ER/BA/WS) and Table 7 |
| `reduction_check.py` | Table 10b: B3, B6, B7 against exhaustive search (instances, mismatches) |
| `diff_tables.py` | Old value, new value and \|difference\| for every number the m = 1 patch could move (`--all` lists all) |
| `check_text.py <file.tex>` | Abstract length, highlight length, red-line grep, numeric literals not produced by a macro, undefined macros |
| `check_k.py` | k-rule check of the stored files |

## Hardware data

| Data set | File | Backend | Physical qubits |
|---|---|---|---|
| Ten-run repeat, runs 2–4 | `data/results/q8_hw_repeats.json` | ibm_marrakesh | [6, 5] |
| Runs 5–9 | `data/results/q8_hw_repeats_14oct.json` (five runs) | ibm_marrakesh | [6, 5] |
| Runs 1 and 10 | `data/results/q8_hw_refetched.json` (raw counts) | ibm_marrakesh | [6, 5] |
| Intended-sign pairs (P24) | `data/results/hardware_fez/P24_combined5.json` | ibm_fez | [0, 1] (`p24_fez_job_metadata.json`) |
| Same-sign control (X13) | `data/results/hardware_fez/X13_ibm_hardware_results.json` | ibm_fez | [0, 1] (`x13_fez_job_metadata.json`, read from the job records with `experiments/fetch_x13_job_metadata.py`) |

## Security Notice

- **No credentials are stored in this repository.**
- All IBM Quantum scripts load tokens exclusively from environment variables.
- Job IDs in `q8_hw_repeats_14oct.json` are read-only references to public IBM job records; they contain no authentication material.
- Before reusing any script that previously contained `IBM_API_KEY`, verify the token has been rotated in the IBM Quantum dashboard.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{bmr2026,
  title   = {Bell-Metered Routing ({BMR}): Security-Budget-Constrained
             Path Selection for {QKD} over Optical Networks},
  journal = {Optical Switching and Networking},
  year    = {2026},
  note    = {Major revision submitted October 2026}
}
```

---

## License

Code: MIT. Data: CC-BY 4.0. See `LICENSE` (to be added).

---

*Proof-reading of Corollary 1, Theorems 3a–d, Theorem 4, Propositions 1–3 and Lemma 1: Sanjeev Rana.*
