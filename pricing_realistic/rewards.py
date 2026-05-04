"""Reward helpers and grid mapping."""
from __future__ import annotations

import numpy as np
from environments.complementary import ComplementaryPricingEnvironment

from .config import (
    N_PRODUCTS,
    N_ACTIONS,
    MARGIN_VALS,
    ALL_ACTIONS_REAL,
)

def norm_to_real(action_norm: np.ndarray) -> np.ndarray:
    idx = np.round(action_norm * (N_ACTIONS - 1)).astype(int)
    return MARGIN_VALS[idx]


def snap_to_grid(action_norm: np.ndarray) -> np.ndarray:
    """Snap a continuous [0,1]^d point to the nearest normalised grid point."""
    idx = np.round(action_norm * (N_ACTIONS - 1)).astype(int)
    return idx.astype(float) / (N_ACTIONS - 1)


def compute_joint_reward(env: ComplementaryPricingEnvironment,
                         margins: np.ndarray) -> float:
    """
    Draw N_BASKETS customers; return SCALAR joint profit.

    profit = sum_i  margin_i * fraction_sold_i   (alpha=0)
    """
    sales_mx = env.step(margins)                        # (N_PRODUCTS, N_BASKETS)
    return float(np.sum(margins * np.mean(sales_mx, axis=1)))


def joint_reward_from_sales(margins: np.ndarray, sales_mx: np.ndarray) -> float:
    """Scalar profit from a sales matrix already drawn from env.step."""
    return float(np.sum(margins * np.mean(sales_mx, axis=1)))


def mean_joint_reward_replicated(
    env: ComplementaryPricingEnvironment,
    margins: np.ndarray,
    n_rep: int,
) -> float:
    """
    Mean profit over ``n_rep`` i.i.d. basket batches at the same price vector.

    Reduces feedback variance ~1/n_rep without changing the arm set.
    """
    if n_rep <= 1:
        sales_mx = env.step(margins)
        return joint_reward_from_sales(margins, sales_mx)
    tot = 0.0
    for _ in range(int(n_rep)):
        sales_mx = env.step(margins)
        tot += joint_reward_from_sales(margins, sales_mx)
    return tot / float(n_rep)


def compute_marginal_rewards(env: ComplementaryPricingEnvironment,
                              margins: np.ndarray) -> np.ndarray:
    """
    Draw N_BASKETS customers; return per-product marginal profits.

    Used ONLY by UnivariateBaseline.  revenue_i = margin_i * fraction_sold_i.
    """
    sales_mx = env.step(margins)                        # (N_PRODUCTS, N_BASKETS)
    return margins * np.mean(sales_mx, axis=1)          # (N_PRODUCTS,)

