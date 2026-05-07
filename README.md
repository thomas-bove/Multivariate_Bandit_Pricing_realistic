# Anonymous NeurIPS submission — code

This repository implements the proposed method referred to in the code as
`DMS-X0-BPE`*(Tensorized BPE in the paper) and a set of baselines/ablations:

- `GP-UCB-Iso-Mat12`* — isotropic Matérn-1/2 GP-UCB ablation.
- `BPE-Iso-Mat12`* — BPE with the isotropic kernel ablation.
- `HGP-UCB-CPP`* — heteroscedastic GP-UCB baseline for complementary product pricing.
- `BZ-ETC`, `Kleinberg`*, `SPSA`, `Univariate` — additional baselines.

> *\* The names above are the **internal identifiers** used in code and in
> `results.csv`. Plot legends and titles use different display labels — see
> [Algorithm names: code vs plots](#algorithm-names-code-vs-plots) for the full
> mapping.*

The synthetic complementary-pricing environment is implemented from scratch in
`environments/`; no external dataset is required.

> **Anonymity note.** This README intentionally omits author names, affiliations,
> and links. Algorithm and method names follow the published literature they cite
> internally.

---

## Repository layout (relevant files only)

```
pricing_realistic/        # main package: config, kernels, runs, plots, agents
  agents/                 # bandit policies (DMS-X0-BPE, GP-UCB variants, baselines)
  config.py               # CFG (default experiment), EXP_CFG (BPE/GP knobs)
  main_entry.py           # main() and main_single_agent() experiment drivers
  runs.py                 # per-(T, seed) run loops
  plots.py                # regret / heatmap PDFs
environments/             # synthetic complementary-pricing environment
hgp_ucb_cpp/              # HGP-UCB-CPP baseline implementation
agent_runs/               # CLI scripts: one algorithm per file
comparison.py             # full multi-baseline run (all algorithms)
experiment_realistic.py   # equivalent legacy CLI to comparison.py
plot_regret_from_csv.py   # regenerate regret plots from an existing results.csv
estimate_rkhs_norm.py     # diagnostic: RKHS norm under the product Matérn-1/2 kernel
requirements.txt
```

Other top-level directories (`results_*`, `...`) are leftover experiment outputs
and can be ignored.

---

## Installation

Python ≥ 3.9 is recommended. Verify the interpreter version first:

```bash
python --version    # or: python3 --version
```

> If your system only ships `python3` (e.g. a stock macOS without a `python`
> alias), substitute `python` with `python3` in every command that follows.

From the project root (the directory containing `comparison.py` and
`requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists only libraries actually imported by the relevant code:

```
numpy
pandas
scipy
matplotlib
ortools
```

`ortools` is used only by the `HGP-UCB-CPP` baseline
(`hgp_ucb_cpp/integer_optimizers.py`). If you do not need that baseline, the
remaining algorithms run with `numpy`, `pandas`, `scipy`, `matplotlib` alone.

The repository has no install step (no `setup.py` / `pyproject.toml`); it is run
directly from the project root.

---

## Running the main experiment

All experiments are driven by `pricing_realistic.main_entry`. The default
environment configuration is set in `pricing_realistic/config.py`
(`paired_config(n_products=6, n_actions=5, n_baskets=10, mc_ep=2000,
dependency_strength=3.0)`). To change the problem scale, edit `CFG` there.

### Single-policy runs (cheaper)

One CLI per algorithm under `agent_runs/`, each producing its own results
directory:

```bash
python agent_runs/run_dmsx0_bpe.py          # proposed method
python agent_runs/run_gp_ucb_iso_mat12.py
python agent_runs/run_bpe_iso_mat12.py
python agent_runs/run_hgp_ucb_cpp.py
python agent_runs/run_bz_etc.py
python agent_runs/run_kleinberg.py
python agent_runs/run_spsa.py
python agent_runs/run_univariate.py
```

### Long full multi-baseline run (`T=1000`, 3 seeds, all algorithms)

Reproduces the headline figures: every algorithm at a single horizon, three
seeds, full PDFs. Uses the project defaults from `pricing_realistic/config.py`
(`n_products=6`, `n_actions=5`, |X°| = 5⁶ = 15625). No code changes required.

```bash
EXP_SEQ_T1000=1 EXP_SEQ_T=1000 EXP_N_SEEDS=3 EXP_N_WORKERS=3 \
  EXP_OUT=results_T1000_s3_d6 \
  python comparison.py
```

Output (`results_T1000_s3_d6/`): `results.csv` (≈7 MB, 27 000 rows) plus the
five regret/heatmap PDFs (`regret_pair_solo`, `regret_pairs_combined`,
`simple_regret`, `best_simple_regret`, `heatmaps`).

**Indicative wall-time (Apple silicon, 3 workers running 3 seeds in
parallel):** ~28 min. **Peak RAM:** ~1 GB per worker, ~3 GB total at
`EXP_N_WORKERS=3`. To change horizon or seed count, adjust `EXP_SEQ_T` and
`EXP_N_SEEDS`. Increasing `EXP_N_WORKERS` beyond `EXP_N_SEEDS` has no effect:
parallelism is per-seed (each worker runs all algorithms sequentially).

There is no checkpointing or resume — if a run is interrupted, restart it
from scratch.

### Single-policy variant

If you only need one algorithm (cheaper than the full comparison), the same
env vars work with the per-algorithm scripts in `agent_runs/`:

```bash
EXP_SEQ_T1000=1 EXP_SEQ_T=1000 EXP_N_SEEDS=3 \
  EXP_OUT=results_T1000_s3_dmsx0_bpe \
  python agent_runs/run_dmsx0_bpe.py
```

For these scripts `EXP_N_WORKERS` defaults to `1` (single-policy runs are
already light); set it explicitly if you want parallel seeds.

### Replot from an existing CSV

```bash
python plot_regret_from_csv.py
```

(Edit `CSV_PATH` and `OUT_DIR` at the top of the script.)

### Cleanup

When you are done, deactivate the virtualenv (and optionally remove it):

```bash
deactivate
rm -rf .venv          # optional, only if you no longer need the environment
```

The result directories (`results_*`) are kept on disk; remove the ones you do
not need with `rm -rf results_<name>`.

---

## Inputs and outputs

**Inputs.** None. The environment is synthetic and is built deterministically
in-process from `CFG` (see `environments/demand_generator.py`).

**Outputs.** Each run writes to the directory given by `EXP_OUT`:

- `results.csv` — per-round records (algorithm, seed, `T`, `t`, played action,
  revenue, instantaneous and cumulative regret, optional BPE recommendation
  diagnostics).
- `regret_pair_solo.pdf`, `regret_pairs_combined.pdf` — cumulative pseudo-regret
  on played actions (the metric used in the theorem).
- `simple_regret.pdf`, `best_simple_regret.pdf` — instantaneous / best simple
  regret (the source can be switched via `EXP_PLOT_SIMPLE_REGRET_SOURCE` ∈
  `{played, recommendation, both}`).
- `heatmaps.pdf` — diagnostic heatmaps over the action grid.

---

## Algorithm names: code vs plots

Internal algorithm identifiers (used in the agent modules, run dispatch, the
`algorithm` column of `results.csv`, and the `COLORS` / `MARKERS` dictionaries)
are kept stable for code-level traceability and to avoid touching agent files.
Plot legends and titles use a separate display label, defined in
`pricing_realistic/plots.py` (`_DISPLAY_NAMES`, `_HIDE_FROM_PLOTS`, `_disp(...)`).
Editing those structures is the only thing required to rename or hide a series
in figures — agents, runs, and the CSV schema stay untouched.

| Internal name (code, CSV) | Display label (plots) |
| --- | --- |
| `DMS-X0-BPE`        | **Tensorized BPE**  |
| `BPE-Iso-Mat12`     | Isotropic BPE       |
| `GP-UCB-Iso-Mat12`  | Isotropic-GPUCB     |
| `HGP-UCB-CPP`       | CPP (Mussi)         |
| `Kleinberg`         | CAB1                |
| `BZ-ETC`            | BZ-ETC              |
| `SPSA`              | SPSA                |
| `Univariate`        | Univariate          |

Practical consequences:

- When inspecting figures, use the display labels above.
- When filtering `results.csv` (e.g. `df[df.algorithm == "DMS-X0-BPE"]`) or
  reading agent code, use the internal names — they are not affected by the
  display mapping.
- The console summary printed by `_print_summary` reports internal names by
  design (it is a textual log, not a plot).
- To rename or hide more series, extend `_DISPLAY_NAMES` / `_HIDE_FROM_PLOTS`
  in `pricing_realistic/plots.py`; nothing else needs to change.

---

## Reproducibility

Settings already present in the code:

- **Environment determinism.** The synthetic environment is built with
  `make_default_env(cfg=CFG, seed=0)` everywhere; the demand draw is fixed by
  `EnvConfig.demand_seed=0`.
- **Algorithm seeds.** Tasks iterate over `seed ∈ range(N_SEEDS)`; each
  `(T, seed)` tuple is reproducible. `N_SEEDS` defaults to `3` (single-horizon
  mode) or `3` / `10` (multi-horizon `FAST_MODE` on/off in `main_entry.py`), and
  can be overridden by `EXP_N_SEEDS`.
- **Default experiment configuration** (`pricing_realistic/config.py`):
  `paired_config(n_products=6, n_actions=5, n_baskets=10, mc_ep=2000,
  dependency_strength=3.0)`, giving a full uniform ε-cover `|X°| = 5^6 = 15625`.
- **Bandit hyperparameters.** Held in `EXP_CFG = BanditExperimentConfig()` in
  the same file: `dms_rkhs_norm=1.0`, `dms_elimination_delta=0.05`,
  `dms_bpe_noise_var=0.005`, `dms_kernel_L=1.0`, `dms_bpe_noise_R=0.05`, etc.
- **Exploration constants.** `c_beta=0.5` (1-D GP-UCB only), `delta=0.1`,
  `B_rkhs=EXP_CFG.dms_rkhs_norm` (set in `main_entry.py`).

### Useful environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `EXP_OUT` | Output directory for results / PDFs | `results_realistic` |
| `EXP_SEQ_T1000`, `EXP_SEQ_T` | Single-horizon mode and its `T` | unset, `1000` |
| `EXP_N_SEEDS`, `EXP_N_WORKERS` | Number of seeds / parallel workers | `3`, `min(cpu_count(), N_SEEDS)` |
| `EXP_PLOT_SIMPLE_REGRET_SOURCE` | `played` / `recommendation` / `both` | `played` |
| `EXP_SPSA_A_PAR` | SPSA gain `a_par` | `1e-6` |
| `DMS_FINAL_COMMIT` | Append a final-commit row to `results.csv` (deployment diagnostic, not the theorem regret) | unset |