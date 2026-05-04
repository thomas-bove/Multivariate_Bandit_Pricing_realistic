"""Univariate per-product DMSGPUCB baseline."""
import numpy as np

from ..kernels import KERNEL_L
from ..rewards import snap_to_grid
from .dmsgp_ucb import DMSGPUCB

class UnivariateBaseline:
    """
    N_PRODUCTS independent 1-D DMS GP-UCB agents, one per product.
    Each observes only its own marginal profit: margin_i * fraction_sold_i.
    """

    def __init__(self, d: int, noise_var: float, c_beta: float = 1.0,
                 delta: float = 0.1, kernel_L: float = KERNEL_L,
                 B_rkhs: float = 1.0, seed: int = 0):
        self.d      = d
        self.agents = [
            DMSGPUCB(
                1, noise_var, c_beta=c_beta, delta=delta,
                kernel_L=kernel_L, B_rkhs=B_rkhs,
                seed=seed + i,
                use_batches=False,
                batch_base=1,
                batch_ratio=2.0,
                n_restarts=20,
                normalize_rewards=False,
            )
            for i in range(d)
        ]

    def pull(self) -> np.ndarray:
        out = []
        for ag in self.agents:
            x = ag.pull(snap_fn=snap_to_grid)
            out.append(float(x[0]))
        return np.array(out)

    def update(self, revenues: np.ndarray):
        """revenues[i] = margin_i * fraction_sold_i (from compute_marginal_rewards)."""
        for ag, r in zip(self.agents, revenues):
            ag.update(float(r))

    def reset(self, seed: int = 0):
        for i, ag in enumerate(self.agents):
            ag.reset(seed=seed + i)

