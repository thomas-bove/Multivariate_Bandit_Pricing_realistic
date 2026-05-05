"""
Quick replot: load results_d6_full_v2/results.csv and regenerate the
two regret PDFs (regret_pair_solo.pdf, regret_pairs_combined.pdf)
without re-running the experiment.

Usage:
    python plot_regret_from_csv.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from environments.demand_generator import make_default_env
from pricing_realistic.config import CFG, ALL_ACTIONS_REAL
from pricing_realistic.plots import _add_best_simple_regret, _plot_regret_per_pair

CSV_PATH = "results_d6_full_v2/results.csv"
OUT_DIR  = "results_pair_plot_test"

os.makedirs(OUT_DIR, exist_ok=True)

print("Building env …")
env, demands = make_default_env(cfg=CFG, seed=0)
all_vals        = np.array([env.compute_givenaction_value(a) for a in ALL_ACTIONS_REAL])
opt_action_real = ALL_ACTIONS_REAL[int(np.argmax(all_vals))]
print(f"  opt_action_real = {opt_action_real}")

print(f"Loading {CSV_PATH} …")
df = pd.read_csv(CSV_PATH)
df = _add_best_simple_regret(df)
n_seeds = df["seed"].nunique()
print(f"  {len(df):,} rows  |  {df['algorithm'].nunique()} algorithms  |  {n_seeds} seeds")

print("Plotting …")
_plot_regret_per_pair(df, env, opt_action_real, OUT_DIR, n_seeds)
print("Done.")
