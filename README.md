# GRIT: CDF-Gated Thresholding — code and results

Code, stored results and table/figure generators for

> V. Lakmini, M. Premarathna, D. Vimalajeewa, F. Ruggeri and B. Vidakovic,
> *GRIT: CDF-Gated Thresholding, Its Structural Limits, and Risk-Based Gate Design* (2026).

The repository contains the code and saved numerical results for GRIT.
The table generators reproduce the printed numerical entries in all eight
simulation and numerical-study tables. The figures are regenerated from the
gate formulas, numerical integration and saved Bayes-risk results.
The data are synthetic; the reproduction commands require no external dataset.

## Quick start

From the repository root, in Bash (Linux, macOS, WSL or Git Bash):

```bash
python -m venv .venv
source .venv/bin/activate       # Git Bash with Windows Python: source .venv/Scripts/activate
python -m pip install -r requirements-repro.txt
PYTHON=python bash run_all.sh --tables    # regenerate eight tables from saved results
PYTHON=python bash run_all.sh             # also regenerate the three PDFs and run the oracle check
```

The reference environment uses Python 3.12, with library versions listed in
`requirements-repro.txt`. The more flexible `requirements.txt` lists minimum
versions. On a system where the environment's interpreter is named `python3`,
`./run_all.sh` works without setting `PYTHON`.

To compare the generated table rows with your copy of the manuscript:

```bash
python checks/verify_tables.py "/path/to/GRIT07.tex"
```

The checker compares row order and displayed values, including paired results
and standard errors, and ignores LaTeX boldface. A mismatch exits with a nonzero
status and prints both rows.

To rerun the experiments from scratch with the paper's fixed seeds, use a fresh
clone or copy and run `PYTHON=python bash run_all.sh --full`. It takes several
hours and replaces the saved results in that copy as each stage completes.
Set `JOBS` to change the wavelet worker count (default: 2). Floating-point
results and figure rendering can vary with numerical-library versions and
platforms. The saved results are the reference for the printed tables.

## What was computed, and where

| Paper item | What it is | Script(s) | Stored output |
|---|---|---|---|
| Figure 1 (`gates_rules.pdf`) | canonical gates and rules at λ = 2 | `figures/make_figs.py` | `figures/Figs/` |
| Figure 2 (`oracle_frontier.pdf`) | oracle gates (isotonic projection) and oracle risk R*(λ)/r_B, ε = 0.05 | `figures/make_figs.py` (uses `grit/bayes.py`) | `figures/Figs/` |
| Figure 3 (`risk_curves.pdf`) | frequentist risk curves of rules tuned for the Laplace slab | `figures/make_figs.py` | `figures/Figs/` |
| Table `tab:bayes` (Section 6.1) | Bayes risks computed by numerical integration for tuned soft, hard, garrote, firm, SCAD, OP₁ and OP rules under four slabs × three sparsities; posterior median; oracle risk | `bayes_risk/bayes_study.py`, `bayes_risk/op1_study.py`; table body `bayes_risk/make_tables.py` | `bayes_risk/bayes_study.json`, `tab_bayes_body.tex` |
| Table `tab:seq` (Section 6.2) | sparse sequences, n = 1000, 200 replications: SURE-tuned families, oracle on grid, EB posterior median | `sequences/mc_seq.py 200`; table body `bayes_risk/make_tables.py` | `sequences/mc_seq.json`, `tab_seq_body.tex` |
| Tables `tab:qn`, `tab:wav`, `tab:paired`, `tab:wav512`, `tab:wav2048` (Section 6.3, Appendix B) | wavelet denoising: 4 signals × N ∈ {512, 1024, 2048} × root SNR ∈ {3, 5, 7}, 1000 replications per setting, 15 rules on common realizations | `wavelets/study.py`, `wavelets/study2.py` (jobs in `wavelets/jobs.txt`); aggregation `wavelets/analyze2.py` | `wavelets/res/`, `wavelets/res3/`, `wavelets/tables/`, `wavelets/analysis2.txt` |
| Numbers quoted in Section 6.3 text (Q_N, paired win counts, SURE/oracle ratios, selected tail exponents) | | `wavelets/analyze2.py` | `wavelets/analysis2.txt` |
| Table `tab:reg` (Section 6.4) | penalized regression, n = 100, p = 200, 200 replications, coordinate descent, validation-chosen threshold | `regression/regsim.py 0.5`, `regression/regsim.py 0.8`; table and paired summaries `regression/make_table.py` | `regression/res_0.5.npz`, `res_0.8.npz`, `tab_reg_body.tex`, logs |
| Text after Theorem 3 (oracle gates below the uniform gate on 0.10 < u < 0.31; at most 6% excess shrinkage; Gaussian-slab gate approaching 0.9) | | `checks/oracle_check.py` | printed |

### Shared modules (`grit/`)
- `gates.py` — canonical gates: soft, hard, power garrote, firm/MCP, SCAD, odds-power (OP).
- `bayes.py` — sparse-mixture priors, marginal and posterior mean on a grid, the risk identity, and the oracle gate as a clipped isotonic projection.
- `sim_core.py` — SURE selection over rule families and the EB posterior median used by the sequence experiment.

### Wavelet study details (`wavelets/`)
- `engine.py`: periodized orthonormal DWT. The filters are Symmlet-4 for Doppler and HeaviSine, db3 for Bumps (time-reversed, MATLAB `wavedec` orientation) and Haar for Blocks, all built by spectral factorization. Test signals are sampled at t = i/N.
- `rules.py`: gates, the vectorized SURE grid, and the empirical Bayes posterior median (Laplace a = 0.5, marginal-maximum-likelihood weight bounded below by the universal-threshold weight).
- `study.py` runs 10 rules: universal hard, universal soft, SureShrink, SURE-soft, SCAD, OP b = 3, OP with b tuned by level, garrote with b tuned, firm with a tuned, and EB.
- `study2.py` runs 5 variants on the same noise draws (same seeds): OP b = 2, OP b = 5, OP with a common tuned b, garrote b = 2 and firm a = 3.
- Design: j₀ = 3, MAD noise estimate from the finest level, SURE over 50 thresholds on [0, √(2 log n_j)].
- Each run writes one `.npz` per setting with per-replication losses.

### Timings (2 cores)
| Step | Time |
|---|---|
| Wavelet study, `study.py`, all 36 settings | about 25 min |
| Wavelet study, `study2.py`, all 36 settings | about 20 min |
| Regression, per correlation level | about 25 min |
| Sparse sequences, 200 replications | minutes |
| `checks/oracle_check.py` | about 1.5 min |
| Bayes-risk study | not re-timed for this release |

## Archive
`archive/` keeps material that is not used by the current paper but documents its development:
- `grit02_pilots/` — earlier pilot scripts and outputs: a wavelet pilot and the first regression pilot, both superseded by `wavelets/` and `regression/`.
- `legacy_mathematica/` — Mathematica notebooks for the gate families of the first draft (gate gallery).
- `legacy_matlab/` — MATLAB simulations of the first draft and their summary CSVs.

## License
MIT (see `LICENSE`).
