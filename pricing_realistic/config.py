import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Matplotlib reads MPLCONFIGDIR before its config/cache; without a writable path
# it warns (and can be slow on first run).  Use a repo-local cache.
_mpl_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.makedirs(_mpl_cfg, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg)

import time
import itertools
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.optimize import minimize as _sp_minimize
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from environments.complementary import ComplementaryPricingEnvironment
from environments.demand_generator import (
    EnvConfig, default_config, paired_config,
    make_default_env, univariate_action,
)
from hgp_ucb_cpp.catalog import CatalogPricingAgent


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Plot style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex"    : False,
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.labelsize" : 12,
    "axes.titlesize" : 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 300,
    "savefig.bbox"   : "tight",
    "axes.grid"      : True,
    "grid.alpha"     : 0.3,
    "grid.linestyle" : "--",
})

# Paper method: BPE Alg. 1 (Li–Scarlett) on full ε-cover X° with k_Π + elimination (not Smolyak).
PROPOSED_JOINT_ALG = "DMS-X0-BPE"

# Competitor: same k_Π + Li batch schedule + continuous GP-UCB + snap (no finite-arm elimination).
JOINT_CONT_GP_UCB_ALG = "Joint-DMS-GP-UCB"

# Isotropic Matérn-1/2 GP-UCB in ℝ^d (same UCB/β/γ machinery, single exponential of ‖Δ‖₂).
ISO_MATERN_GP_UCB_ALG = "GP-UCB-Iso-Mat12"

ALL_ALGS = [
    PROPOSED_JOINT_ALG,
    JOINT_CONT_GP_UCB_ALG,
    ISO_MATERN_GP_UCB_ALG,
    "HGP-UCB-CPP",
    "BZ-ETC",
    "SPSA",
    "Kleinberg",
    "Univariate",
]
COLORS = {
    PROPOSED_JOINT_ALG : "#1f77b4",
    JOINT_CONT_GP_UCB_ALG : "#17becf",
    ISO_MATERN_GP_UCB_ALG : "#7f7f7f",
    "HGP-UCB-CPP"      : "#9467bd",
    "BZ-ETC"           : "#e377c2",
    "SPSA"             : "#8c564b",
    "Kleinberg"        : "#2ca02c",
    "Univariate"      : "#d62728",
}
MARKERS = {
    PROPOSED_JOINT_ALG : "o",
    JOINT_CONT_GP_UCB_ALG : "v",
    ISO_MATERN_GP_UCB_ALG : "s",
    "HGP-UCB-CPP"      : "P",
    "BZ-ETC"           : "X",
    "SPSA"             : "*",
    "Kleinberg"        : "^",
    "Univariate"      : "D",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Experiment configuration
#
# Change CFG to scale the experiment.  Examples:
#   CFG = paired_config(n_products=6)
#   CFG = paired_config(n_products=4, n_baskets=50)
#   CFG = paired_config(n_products=8, n_actions=7, n_baskets=20)
# ─────────────────────────────────────────────────────────────────────────────

# ``n_actions=5``, ``n_products=4`` ⇒ ``5^4 = 625`` vertices = **full** uniform ε-net on margins.
# **Not** Smolyak.  ``mc_ep * n_baskets = 2000*10`` ⇒ 20_000 MC/cell.
CFG = paired_config(
    n_products=6,
    n_actions=5,
    n_baskets=10,
    mc_ep=2000,
    dependency_strength=3.0,
)

N_PRODUCTS     = CFG.n_products
N_ACTIONS      = CFG.n_actions
MARGIN_VALS    = CFG.margin_vals
GRAPH_DICT     = CFG.graph_dict
ALPHA          = CFG.alpha
N_BASKETS      = CFG.n_baskets
NOISE_VAR      = CFG.noise_var
NOISE_VAR_MARG = CFG.noise_var_marg

# Derived action spaces — ``ALL_ACTIONS_NORM`` is the discrete **X°** (full Cartesian ε-cover).
MARGIN_NORM = np.linspace(0.0, 1.0, N_ACTIONS)

ALL_ACTIONS_NORM = np.array(
    list(itertools.product(*([MARGIN_NORM] * N_PRODUCTS)))
)   # (N_ACTIONS^N_PRODUCTS, N_PRODUCTS) — full grid, not sparse

ALL_ACTIONS_REAL = np.array(
    list(itertools.product(*([MARGIN_VALS] * N_PRODUCTS)))
)   # (N_ACTIONS^N_PRODUCTS, N_PRODUCTS)


def validate_full_cartesian_x0(
    actions: np.ndarray,
    n_products: int,
    n_actions: int,
    atol: float = 1e-12,
) -> None:
    """
    Verify ``actions`` is exactly the full normalized Cartesian price grid (finite cover ``X°``).

    **DMS-X0-BPE** (NeurIPS / Li–Scarlett BPE) experiments must use this full grid — **not** a
    subset, not Smolyak-sparse nodes, not arbitrary continuous margins without snapping.

    Raises ``ValueError`` with an explicit message if the check fails.
    """
    A = np.asarray(actions, dtype=float)
    if A.ndim != 2:
        raise ValueError(
            f"DMS-X0-BPE must run on full X°: expected actions.ndim == 2, got {A.ndim}"
        )
    d = int(n_products)
    ka = int(n_actions)
    n_exp = int(ka ** d)
    if A.shape != (n_exp, d):
        raise ValueError(
            "DMS-X0-BPE proposed method must run on full normalized X° = Cartesian grid; "
            f"expected shape ({n_exp}, {d}), got {A.shape}"
        )
    if not np.all(np.isfinite(A)):
        raise ValueError("DMS-X0-BPE must run on full X°: non-finite entries in ``actions``")
    if np.any(A < -atol) or np.any(A > 1.0 + atol):
        raise ValueError(
            "DMS-X0-BPE must run on the **normalized** grid in [0, 1]^d; "
            "got values outside [0, 1] (pass ALL_ACTIONS_NORM, not raw margin arrays)."
        )

    grid = np.linspace(0.0, 1.0, ka)
    for j in range(d):
        col = np.sort(np.unique(np.round(A[:, j], 12)))
        if len(col) != ka or not np.allclose(col, grid, atol=max(atol, 1e-10)):
            raise ValueError(
                f"DMS-X0-BPE proposed method must run on full normalized X° = Cartesian grid; "
                f"column {j} is not exactly np.linspace(0, 1, n_actions={ka}) (unique count={len(col)})."
            )
    expected = np.array(list(itertools.product(grid, repeat=d)), dtype=float)

    def _lexsort_rows(M: np.ndarray) -> np.ndarray:
        if M.size == 0:
            return M
        return M[np.lexsort(M.T[::-1])]

    As = _lexsort_rows(A)
    Es = _lexsort_rows(expected)
    if not np.allclose(As, Es, atol=max(atol, 1e-10)):
        # Duplicate-row detection (distinct failure mode)
        if len(np.unique(np.round(A, 12), axis=0)) != len(A):
            raise ValueError(
                "DMS-X0-BPE must run on full X°: duplicate rows in ``actions`` "
                "(expected exactly one vertex per grid cell)."
            )
        raise ValueError(
            "DMS-X0-BPE must run on full X°: rows are not exactly the Cartesian product of "
            f"np.linspace(0, 1, n_actions={ka}) along each of d={d} coordinates."
        )


# Scale for normalizing scalar joint rewards before GP / BPE (profit scale vs RKHS-like noise).
REWARD_SCALE = float(N_PRODUCTS * float(np.max(MARGIN_VALS)))


def normalize_joint_reward(r: float) -> float:
    """Map joint profit to ~O(1) scale for GP likelihood (``dms_bpe_noise_var``, ``B_rkhs``)."""
    if REWARD_SCALE <= 1e-18:
        return float(r)
    return float(r) / REWARD_SCALE


def unnormalize_joint_reward(r: float) -> float:
    """Inverse of ``normalize_joint_reward``."""
    if REWARD_SCALE <= 1e-18:
        return float(r)
    return float(r) * REWARD_SCALE


def best_independent_product_surrogate(
    env: ComplementaryPricingEnvironment,
) -> float:
    """
    Naive **product-of-base-marginals** benchmark on the same discrete grid as the bandits:
    ``max_p (∑_i p_i) · ∏_i D_i^{base}(p_i)`` using only ``demands[..., 0]`` (ignores leader–follower
    coupling).  Not the Univariate bandit policy; a closed-form scalar for reporting.
    """
    best = float("-inf")
    n = int(env.n_products)
    for margins in ALL_ACTIONS_REAL:
        idxs = [
            int(np.argmin(np.abs(env.margins[i, :] - margins[i])))
            for i in range(n)
        ]
        probs = np.array([float(env.demands[i, idxs[i], 0]) for i in range(n)])
        val = float(np.sum(margins)) * float(np.prod(probs))
        if val > best:
            best = val
    return best


@dataclass
class BanditExperimentConfig:
    """
    Tunable knobs for **``DMSX0BPE``** (paper BPE) and for GP-UCB agents that reuse ``dms_*``
    noise / kernel length; HGP uses ``hgp_kernel_L``.

    dms_rkhs_norm
        RKHS norm bound ``Ψ`` (``B_rkhs``) in Li–Scarlett elimination ``√β`` (see ``DMSX0BPE``).
    dms_bpe_noise_R
        Optional sub-Gaussian scale ``R`` for ``√β``.  If ``None``, use theorem calibration
        ``λ = R²`` (equivalently ``noise_factor = 1`` in ``√β = Ψ + noise_factor·√(2 log(…))``).
    dms_bpe_use_global_history
        If ``True``, BPE uses **global** GP history for σ exploration and elimination (engineering
        ablation).  If ``False`` (**default**, paper-aligned), σ uses **batch-only** locations with
        fantasy ``y = prior_mean``; elimination UCB/LCB uses **only the completed batch**.
    dms_elimination_delta
        ``δ`` in the elimination log-term.
    dms_bpe_noise_var
        GP noise ``λ`` (must upper-bound variance of the scalar reward fed to the BPE GP).
    dms_kernel_L
        Length-scale ``L`` in ``k_Π`` and in the isotropic baseline; use **1.0** for the textbook
        product kernel ``∏_i exp(-|x_i-x'_i|)``.
    joint_reward_mc_replications
        If R>1, scalar-reward bandits average R i.i.d. ``env.step`` calls per round.
    hgp_kernel_L
        RBF length-scale for HGP-UCB-CPP.
    dms_gp_noise_use_auto
        If True, GP / BPE ``noise_var`` = ``1 / (4 · N_BASKETS · R_mc)`` on the **normalized**
        reward scale (``R_mc`` = ``joint_reward_mc_replications``).  Otherwise use
        ``dms_bpe_noise_var``.
    dms_bpe_beta_use_active_count
        If True (legacy), elimination ``√β`` uses ``|X_active|`` in the log-term; default **False**
        uses full ``|X°|`` (uniform confidence over the ε-cover).
    dms_bpe_prior_mean
        GP prior mean on **normalized** rewards for ``DMSX0BPE`` (default ``0.0``).
    oracle_feedback_for_debug
        If True, ``run_one`` feeds ``DMSX0BPE`` **noiseless** ``env.compute_givenaction_value`` at
        the played price (diagnostic only; **not** for published experiments).
    """

    dms_rkhs_norm               : float = 1.3   # just above empirical ‖J‖_H ≈ 1.286 (subsample)
    dms_elimination_delta       : float = 0.50  # more aggressive than 0.30 → smaller log term
    dms_bpe_noise_var           : float = 0.005
    dms_kernel_L                : float = 1.0
    joint_reward_mc_replications: int = 1
    hgp_kernel_L                : float = 1.0
    dms_gp_noise_use_auto       : bool = False
    dms_bpe_beta_use_active_count: bool = True  # β shrinks with active set → cascading elimination
    dms_bpe_prior_mean          : float = 0.24  # empirical mean J_norm=0.245 (H2 confirmed)
    oracle_feedback_for_debug : bool = False
    dms_bpe_noise_R           : Optional[float] = 0.05  # noise_factor=0.707 → √β≈4.5 (was 6.2)
    dms_bpe_use_global_history: bool = True
    dms_bpe_within_batch_ucb_c: float = 0.5    # UCB-within-batch: exploits good arms immediately


EXP_CFG = BanditExperimentConfig()


def effective_gp_noise_variance(cfg: BanditExperimentConfig) -> float:
    """GP observation noise λ on the **normalized** reward scale."""
    if cfg.dms_gp_noise_use_auto:
        R_mc = max(1, int(cfg.joint_reward_mc_replications))
        return 1.0 / (4.0 * float(N_BASKETS) * float(R_mc))
    return float(cfg.dms_bpe_noise_var)


def recommendation_csv_keys() -> Tuple[str, ...]:
    """
    Column names for **diagnostic / deployment** recommendation metrics (not theorem cumulative regret).

    For **DMS-X0-BPE**, filled from ``recommend()``; baselines mirror played actions.
    """
    return (
        "recommendation_revenue",
        "recommendation_simple_regret",
        *(f"recommendation_a{i}" for i in range(N_PRODUCTS)),
    )


def played_action_csv_keys() -> Tuple[str, ...]:
    """Theorem / paper metrics on **played** pulls (aliases of ``revenue`` / ``simple_regret`` / ``regret_cumulative``)."""
    return ("played_revenue", "played_simple_regret", "played_regret_cumulative")

