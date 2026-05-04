"""
Demand generator for ComplementaryPricingEnvironment.

Generates demands[n_products, n_actions, 2] using a "kinked-linear" model
that guarantees by construction:

  1. Leader standalone optimal at the middle margin (margins_vals[n_actions//2]).
     D_peak ∈ [0.65, 0.78] ensures Revenue(0.5) = 0.5·D_peak ≥ 0.325 > 0.300 =
     Revenue(0.3), so the leader ALWAYS prefers p=0.5 in isolation.
     Large slope_lo ∈ [2.0, 4.0] clips D_L(0.3) to 1.0: maximises the
     follower cross-selling gain when the joint optimizer lowers p_L.

  2. Follower base demand ≈ 0 (near-zero everywhere, clipped to [0.001, 0.02]).
     The follower is almost never sold without the leader.

  3. Follower enhanced demand peaks at margins_vals[-2] = 0.7, with
     D_enh_peak ∈ [0.85, 0.97]: very high sales rate when the leader is sold.

WHY this creates a structural gap (~15–25%):
  Univariate:    leader → 0.5, follower → 0.7  (misses cross-effect)
  Joint optimal: leader → 0.3 (D_L jumps to 1.0, follower revenue ×1.4)
  The joint optimizer gains ~0.3·D_enh(0.7)·ΔD_L per pair, which the
  Univariate agent cannot discover because it observes only marginal profits.
"""

from dataclasses import dataclass

import numpy as np
from environments.complementary import ComplementaryPricingEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
#
# Use EnvConfig to pass all parameters in one object.
# Factory functions:
#   default_config()                        — 4 products, 5 margins, 2 pairs
#   paired_config(n_products, n_actions)    — arbitrary scale, auto-paired graph
#
# To scale an experiment, change one line in the experiment script:
#   CFG = default_config()
#   CFG = paired_config(n_products=6)
#   CFG = paired_config(n_products=8, n_actions=7, n_baskets=20)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """All environment parameters in one object. Fully parameterised."""
    n_products  : int
    n_actions   : int
    margin_vals : np.ndarray   # shape (n_actions,)
    graph_dict  : dict         # leader → list[followers]
    alpha       : float = 0.0  # 0 = pure profit, 1 = pure revenue
    n_baskets   : int   = 10   # customers per step()
    demand_seed : int   = 0    # fixed demand draw; algorithm seeds vary
    mc_ep       : int   = 2000 # Monte Carlo episodes for action-value table
    dependency_strength: float = 1.0  # leader–follower coupling (≥0; >1 amplifies lift before clip)
    exact_oracle: bool = True  # closed-form action_values; step() remains stochastic

    @property
    def noise_var(self) -> float:
        # Var[Σ_i p_i·Bernoulli(D_i)] / n_baskets, using p~0.5, D~0.7 typical.
        # Factor 0.12 ≈ 0.5²·0.7·0.3·(2 products per pair), scales with n_products.
        return self.n_products * 0.12 / self.n_baskets

    @property
    def noise_var_marg(self) -> float:
        # Per-product marginal Var[p·Bernoulli(D)] / n_baskets.
        return 0.12 / self.n_baskets


def default_config() -> EnvConfig:
    """
    Default experiment: 4 products, 5 margin levels, 2 leader-follower pairs.
    Margins {0.1, 0.3, 0.5, 0.7, 0.9}, alpha=0 (pure profit).
    """
    return EnvConfig(
        n_products  = 4,
        n_actions   = 5,
        margin_vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        graph_dict  = {0: [1], 2: [3]},
    )


def paired_config(
    n_products : int,
    n_actions  : int   = 5,
    margin_lo  : float = 0.1,
    margin_hi  : float = 0.9,
    **kwargs,
) -> EnvConfig:
    """
    Build an EnvConfig for n_products arranged in n_products/2 independent
    leader-follower pairs with a uniform margin grid.

    graph_dict = {0:[1], 2:[3], …, n-2:[n-1]}

    Extra keyword arguments (alpha, n_baskets, demand_seed, mc_ep) are
    forwarded to EnvConfig, so you can write:
        paired_config(n_products=6, n_baskets=20, demand_seed=42)
    """
    if n_products % 2 != 0:
        raise ValueError(f"n_products must be even for paired layout, got {n_products}")
    return EnvConfig(
        n_products  = n_products,
        n_actions   = n_actions,
        margin_vals = np.linspace(margin_lo, margin_hi, n_actions),
        graph_dict  = {2 * k: [2 * k + 1] for k in range(n_products // 2)},
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kinked-linear demand model
# ─────────────────────────────────────────────────────────────────────────────

def _kinked(margins_vals: np.ndarray, p_peak: float,
            D_peak: float, slope_lo: float, slope_hi: float,
            D_min: float = 0.02, D_max: float = 1.0) -> np.ndarray:
    """
    D(p) = D_peak + slope_lo*(p_peak - p)  for p <  p_peak
    D(p) = D_peak - slope_hi*(p - p_peak)  for p >= p_peak
    clipped to [D_min, D_max].
    """
    d = np.where(
        margins_vals <= p_peak,
        D_peak + slope_lo * (p_peak - margins_vals),
        D_peak - slope_hi * (margins_vals - p_peak),
    )
    return np.clip(d, D_min, D_max)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_demands(
    graph_dict   : dict,
    n_products   : int,
    n_actions    : int,
    margins_vals : np.ndarray,
    seed         : int = 0,
) -> np.ndarray:
    """
    Generate a (n_products, n_actions, 2) demand tensor.

    Parameters
    ----------
    graph_dict   : leader → list[followers] mapping
    n_products   : total number of products
    n_actions    : number of discrete margin levels
    margins_vals : 1-D array of margin values, shape (n_actions,)
    seed         : random seed for reproducibility

    Returns
    -------
    demands : ndarray, shape (n_products, n_actions, 2)
    """
    rng      = np.random.default_rng(seed)
    demands  = np.zeros((n_products, n_actions, 2))
    leaders  = list(graph_dict.keys())
    followers = [f for flist in graph_dict.values() for f in flist]

    # Standalone optimal price for leaders (middle of the grid)
    p_opt_leader = margins_vals[n_actions // 2]   # e.g., 0.5 for 5 actions

    # Peak price for follower enhanced demand (second-to-last, e.g., 0.7)
    p_peak_enh = margins_vals[-2]

    # ── leaders ──────────────────────────────────────────────────────────────
    for i in leaders:
        # D_peak ∈ [0.65, 0.78]: Revenue(0.5)=0.5·D_peak ≥ 0.325 > 0.300=Revenue(0.3)
        # so leader always prefers p=0.5 standalone (Univariate convergence).
        # slope_lo ∈ [2.0, 4.0]: D_L(0.3) = D_peak + slope_lo·0.2 ≥ 1.05 → clips to 1.0
        # → ΔD_L = 1.0 − D_peak ∈ [0.22, 0.35], strong follower boost at joint optimum.
        D_at_opt = rng.uniform(0.65, 0.78)
        slope_lo = rng.uniform(2.0, 4.0)
        slope_hi = rng.uniform(3.0, 5.0)
        d = _kinked(margins_vals, p_opt_leader, D_at_opt, slope_lo, slope_hi,
                    D_min=0.02, D_max=1.0)
        demands[i, :, 0] = d
        demands[i, :, 1] = d   # enhancement unused for leaders

    # ── followers ────────────────────────────────────────────────────────────
    for i in followers:
        # Base: near-zero everywhere (follower almost never bought without leader)
        base_at_lo = rng.uniform(0.005, 0.015)
        base_at_hi = rng.uniform(0.002, 0.008)
        d_base = np.linspace(base_at_lo, base_at_hi, n_actions)
        d_base += rng.uniform(-0.002, 0.002, n_actions)
        demands[i, :, 0] = np.clip(d_base, 0.001, 0.02)

        # Enhanced: kinked, peaks at p_peak_enh; high D_enh_peak → strong boost
        D_enh_peak   = rng.uniform(0.85, 0.97)
        slope_enh_lo = rng.uniform(0.20, 0.60)
        slope_enh_hi = rng.uniform(1.00, 2.50)
        d_enh = _kinked(margins_vals, p_peak_enh, D_enh_peak, slope_enh_lo, slope_enh_hi,
                        D_min=0.15, D_max=0.99)
        demands[i, :, 1] = d_enh

    return demands


def build_env(
    graph_dict   : dict,
    n_products   : int,
    n_actions    : int,
    margins_vals : np.ndarray,
    alpha        : float = 0.0,
    n_baskets    : int   = 10,
    mc_ep        : int   = 2000,
    demand_seed  : int   = 0,
    env_seed     : int   = 0,
    dependency_strength: float = 1.0,
    exact_oracle : bool = True,
) -> tuple:
    """
    Generate demands and build a ComplementaryPricingEnvironment.

    Returns
    -------
    (env, demands)
    """
    demands = generate_demands(
        graph_dict=graph_dict,
        n_products=n_products,
        n_actions=n_actions,
        margins_vals=margins_vals,
        seed=demand_seed,
    )
    margins_env = np.tile(margins_vals, (n_products, 1))
    env = ComplementaryPricingEnvironment(
        n_products=n_products,
        n_actions=n_actions,
        margins=margins_env,
        demands=demands,
        n_baskets=n_baskets,
        alpha=alpha,
        graph_dict=graph_dict,
        mc_ep=mc_ep,
        seed=env_seed,
        dependency_strength=dependency_strength,
        exact_oracle=exact_oracle,
    )
    return env, demands


def make_default_env(cfg: EnvConfig = None, seed: int = 0) -> tuple:
    """
    Build a ComplementaryPricingEnvironment from an EnvConfig.

    Parameters
    ----------
    cfg  : EnvConfig or None.  If None, uses default_config().
    seed : env RNG seed (the demand draw is fixed by cfg.demand_seed).

    Returns
    -------
    (env, demands)
    """
    if cfg is None:
        cfg = default_config()
    return build_env(
        graph_dict   = cfg.graph_dict,
        n_products   = cfg.n_products,
        n_actions    = cfg.n_actions,
        margins_vals = cfg.margin_vals,
        alpha        = cfg.alpha,
        n_baskets    = cfg.n_baskets,
        mc_ep        = cfg.mc_ep,
        demand_seed  = cfg.demand_seed,
        env_seed     = seed,
        dependency_strength=cfg.dependency_strength,
        exact_oracle=cfg.exact_oracle,
    )


def univariate_action(
    demands      : np.ndarray,
    graph_dict   : dict,
    n_products   : int,
    n_actions    : int,
    margins_vals : np.ndarray,
    alpha        : float = 0.0,
    dependency_strength: float = 1.0,
) -> np.ndarray:
    """
    Univariate (coordinate-wise) reference action.

    Leaders: argmax (alpha+p)*D_base  (standalone).
    Followers: argmax (alpha+p)*mixed  given the leader at its standalone optimum, where
    ``mixed = D_L * enh_eff + (1-D_L) * base`` and
    ``enh_eff = clip(base + dependency_strength * (enh - base), 0, 1)`` (same as ``step`` / exact oracle).
    """
    leaders = list(graph_dict.keys())
    f2l     = {f: l for l, fs in graph_dict.items() for f in fs}
    action  = np.zeros(n_products)

    for i in leaders:
        rev = [(alpha + margins_vals[j]) * demands[i, j, 0] for j in range(n_actions)]
        action[i] = margins_vals[np.argmax(rev)]

    for i in range(n_products):
        if i in leaders:
            continue
        ldr     = f2l[i]
        ldr_idx = int(np.where(margins_vals == action[ldr])[0][0])
        D_L     = demands[ldr, ldr_idx, 0]
        base    = demands[i, :, 0]
        enh     = demands[i, :, 1]
        enh_eff = np.clip(
            base + float(dependency_strength) * (enh - base),
            0.0,
            1.0,
        )
        mixed   = D_L * enh_eff + (1.0 - D_L) * base
        rev     = [(alpha + margins_vals[j]) * mixed[j] for j in range(n_actions)]
        action[i] = margins_vals[np.argmax(rev)]

    return action
