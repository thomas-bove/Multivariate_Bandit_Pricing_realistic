#!/usr/bin/env python3
"""
Legacy import path and CLI for the complementary-pricing experiments.

The implementation is split under ``pricing_realistic/`` (agents, kernels, runs, plots).
Use ``comparison.py`` for the full multi-baseline run (all agents, like the original monolith).
Use ``agent_runs/run_*.py`` for **one policy only** — ``run_one_single_algorithm`` (lower CPU than ``run_one``).

See package docstrings and ``pricing_realistic.main_entry`` for environment variables.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

_mpl_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.makedirs(_mpl_cfg, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg)

from environments.demand_generator import (
    EnvConfig,
    default_config,
    make_default_env,
    paired_config,
    univariate_action,
)

from pricing_realistic import (
        ALL_ACTIONS_NORM,
    ALL_ACTIONS_REAL,
    ALL_ALGS,
    ALPHA,
    BanditExperimentConfig,
    BZ_ETC,
    CFG,
    COLORS,
    DMSGPUCB,
    DMSX0BPE,
    EXP_CFG,
    effective_gp_noise_variance,
    GRAPH_DICT,
    HGP_UCB_CPP_Wrapper,
    ISO_MATERN_GP_UCB_ALG,
    IsotropicMatern12GPUCB,
    JOINT_CONT_GP_UCB_ALG,
    KERNEL_L,
    KleinbergUCB,
    MARKERS,
    MARGIN_VALS,
    N_ACTIONS,
    N_BASKETS,
        N_PRODUCTS,
    NOISE_VAR,
    NOISE_VAR_MARG,
        PROPOSED_JOINT_ALG,
    REWARD_SCALE,
    SPSAPricing,
    UnivariateBaseline,
    best_independent_product_surrogate,
    li_22_bpe_batch_sizes,
    normalize_joint_reward,
    played_action_csv_keys,
    validate_full_cartesian_x0,
    main,
    main_single_agent,
    recommendation_csv_keys,
    run_bpe_x0_only_trace,
    run_dependency_strength_ablation,
    run_one,
    run_one_single_algorithm,
    run_one_two_joint,
    unnormalize_joint_reward,
    _matern12_isotropic,
    _matern12_product,
    _run_task,
    compute_joint_reward,
    compute_marginal_rewards,
    joint_reward_from_sales,
    mean_joint_reward_replicated,
    norm_to_real,
    snap_to_grid,
)


if __name__ == "__main__":
    main()

