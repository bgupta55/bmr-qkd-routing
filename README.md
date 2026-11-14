# Bell-Metered Routing (BMR) for QKD Networks

Code, data and figures for **Bell-Metered Routing (BMR)**: security-budget-constrained path
selection for quantum key distribution (QKD) networks, where every link runs a live CHSH Bell audit
and produces a finite-key certificate that the router uses as a hard security budget.

> **Status.** Research code accompanying manuscripts under review (Journal of Network and Computer
> Applications; Optical Switching and Networking). Add the DOI / citation here once published.

## Idea in one paragraph

Each link continuously audits itself with a CHSH test. From the audit counts a finite-sample
certificate `(r, ε, δ, γ, T)` is derived: a secret-key rate `r` and a failure bound `ε + δ`. BMR routes a
key only over a **simple path** whose summed failure bound stays within a budget **Δ** and whose
length satisfies a **collusion floor** (a direct link, or `m ≥ 2t+1` hops, so `t` colluding relays
cannot reconstruct the key). A calibrated alarm detects link degradation independently of the
certificate, and the destination verifies a signed, session-bound ledger. The revision adds
**BMR-MP**: `t+1` internally disjoint paths with per-path budgets, key delivered by XOR sharing, and a
GF(2) knowledge-closure verifier that checks exposure to passive coalitions exactly.

## Repository layout

```
.
├── v1_jnca_simulation/   First version: Monte-Carlo simulation framework (JNCA manuscript)
├── v2_osn_revision/      Latest version: library, tests, revision experiments, results, figures (OSN)
├── SHA256SUMS.txt        Checksums of every file in the repository
├── requirements.txt
├── LICENSE               MIT (code); results and figures CC-BY 4.0
└── README.md
```

| Folder | Use it for |
|---|---|
| [`v2_osn_revision/`](v2_osn_revision/README.md) | **Start here.** Latest library (`code/bmr/`), 13 tests, every experiment script, output of every experiment, IBM Quantum raw counts, proofs appendix |
| [`v1_jnca_simulation/`](v1_jnca_simulation/README.md) | Earlier, self-contained simulation package and its results and figures |

## Quick start

```bash
git clone https://github.com/<your-user>/bmr-qkd-routing.git
cd bmr-qkd-routing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Latest version: tests, then regenerate the base-paper results
cd v2_osn_revision/code
python -m pytest tests -q            # 13 passed
python run_all.py --fast             # classical experiments, no quantum hardware needed

# Headline multipath (BMR-MP) results
python tier3_sweep.py                # -> ../tier3_out/
python run_x7_x8.py                  # -> ../x7_out/, ../x8_out/
```

Python 3.10+ is enough (results were produced with Python 3.14, NumPy 2.5, NetworkX 3.6,
pandas 3.0, OR-Tools 9.15, Qiskit 2.5).

## Headline results (v2)

Tier-3 sweep, 200 cells, 1,000 graph clusters per collusion level; 95 % cluster-bootstrap CIs.

| Collusion bound | Single-path (B2) refusal | BMR-MP refusal | BMR-MP delivery | Rate cost (BMR-MP − B2) |
|---|---|---|---|---|
| t = 0 | 7.7 % [6.6, 8.9] | – | – | – |
| t = 1 | – | 28.3 % [26.3, 30.3] | 71.7 % | −0.051 [−0.054, −0.048] |
| t = 2 | – | 71.2 % [69.3, 73.3] | 28.8 % | −0.139 [−0.146, −0.132] |

Other key numbers (base paper, see `v2_osn_revision/code/README.md` for sources):

| Item | Value |
|---|---|
| Minimum audit rounds | n_min = 235 (η_tight) vs 1,078 (η_union) |
| Solver validation | 0 mismatches over 40,968 Yen vs brute-force instances; 20,000 BMR-MP vs brute-force/CP-SAT instances |
| Alarm detection | 500/500 at S† = 2.55 |
| IBM Quantum CHSH plausibility | Ŝ = 2.704 ± 0.066 (ten runs, `ibm_marrakesh`) |

## Reproducibility

- Seeds are fixed and logged (revision master seed `20261113`); machine and package versions are in
  `v2_osn_revision/logs/env_stamp.txt`.
- `v2_osn_revision/code/_orig/` holds the results of the first deposit and
  `v2_osn_revision/code/campaign_R/` the pre-patch/patched regeneration logs, so any change between
  versions can be diffed (`code/diff_tables.py`, `code/rev/r0_compare.py`).
- Large per-trial tables are gzip-compressed (`*.csv.gz`); `pandas.read_csv("file.csv.gz")` reads them directly.
- `SHA256SUMS.txt` verifies integrity: `shasum -a 256 -c SHA256SUMS.txt`.

## IBM Quantum hardware (optional)

The stored raw counts and job IDs in `v2_osn_revision/code/data/results/` are enough to regenerate
all hardware tables without any quantum access. To re-run on hardware you need **your own** IBM Quantum
account. **No credentials are stored in this repository.** Scripts read them from the environment
or from your local `~/.qiskit/qiskit-ibm.json`:

```bash
export IBM_QUANTUM_TOKEN="<your token>"     # never put a token in code or commit it
python v2_osn_revision/code/experiments/e6v2_hardware_fixed.py
```

Job IDs in the result files are references to IBM job records and carry no authentication material.

## Notes on naming

In stored results the objective-only baselines B6 and B7 appear as `B6-Qpath-style` and
`B7-Qcast-style`; they imitate the objective of Q-PATH / Q-CAST and are **not** implementations of
those protocols. The campaign scripts in `code/campaign_R/` ran from a temporary directory and keep their
original `/tmp/campR` paths (see its README).

## Citation

```bibtex
@misc{bmr_qkd_routing,
  title        = {Bell-Metered Routing ({BMR}): Security-Budget-Constrained Path Selection for {QKD} Networks},
  author       = Bhanwar Gupta,
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Code and data accompanying manuscripts under review}
}
```

## Licence

Code: MIT. Results, logs and figures: CC-BY 4.0. See [`LICENSE`](LICENSE).
