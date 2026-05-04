"""SPSA pricing baseline — bandit wrapper around the user's ``SimpleSPSA`` logic (Python 3).

Ported from the provided ``SimpleSPSA`` class: same class-level exponents, gain formulas,
Bernoulli ``delta``, gradient accumulation averaged over ``ens_size``, and the per-dimension
``ak`` halving loop when an update would leave the box.  **Maximization** of revenue uses
``theta + this_ak * ghat`` (sign flip vs the pasted minimizer).

Not ported: ``pdb``, ``matplotlib`` reporting, ``function_tolerance`` / ``param_tolerance``
branches (no full loss oracle in the bandit API).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..rewards import snap_to_grid


class SPSAPricing:
    """Simultaneous perturbation stochastic approximation (SimpleSPSA-style)."""

    # Constants from the provided SimpleSPSA
    alpha = 0.602
    gamma = 0.101

    def __init__(
        self,
        d: int,
        T: int,
        a_par: float = 1e-6,
        noise_var: float = 0.01,
        ens_size: int = 2,
        x0: Optional[np.ndarray] = None,
        seed: int = 0,
        min_vals: Optional[np.ndarray] = None,
        max_vals: Optional[np.ndarray] = None,
    ):
        self._d = int(d)
        self._T = int(T)
        self.a_par = float(a_par)
        self.c_par = float(noise_var)
        self.ens_size = int(ens_size)
        self._rng = np.random.default_rng(seed)
        self._min_vals = (
            np.zeros(self._d, dtype=float) if min_vals is None else np.asarray(min_vals, dtype=float)
        )
        self._max_vals = (
            np.ones(self._d, dtype=float) if max_vals is None else np.asarray(max_vals, dtype=float)
        )
        self.big_a_par = max(float(self._T), 1) / 10.0

        self.x = 0.5 * np.ones(self._d) if x0 is None else np.asarray(x0, dtype=float).copy()

        # Outer iteration index n_iter in minimise() — fixed ck, ak for one full gradient step
        self._n_iter = 0
        # Inner ensemble + plus/minus half-step
        self._ens_j = 0
        self._need_plus = True
        self._delta: Optional[np.ndarray] = None
        self._ghat_acc = np.zeros(self._d, dtype=float)
        self._r_plus: Optional[float] = None
        self._last_action: np.ndarray = snap_to_grid(self.x)

    def _ak_scalar(self) -> float:
        return self.a_par / (self._n_iter + 1.0 + self.big_a_par) ** self.alpha

    def _ck_scalar(self) -> float:
        return self.c_par / (self._n_iter + 1.0) ** self.gamma

    def _perturb_plus(self, theta: np.ndarray, ck: float, delta: np.ndarray) -> np.ndarray:
        theta_plus = theta + ck * delta
        return np.minimum(theta_plus, self._max_vals)

    def _perturb_minus(self, theta: np.ndarray, ck: float, delta: np.ndarray) -> np.ndarray:
        theta_minus = theta - ck * delta
        return np.maximum(theta_minus, self._min_vals)

    def _apply_gradient_ascent(self, ghat: np.ndarray) -> None:
        """Analogue of ``theta = theta - this_ak*ghat`` with maximization and box line search."""
        ak = self._ak_scalar()
        this_ak = np.ones(self._d, dtype=float) * ak
        not_all_pass = True
        while not_all_pass:
            cand = self.x + this_ak * ghat
            out_of_bounds = np.where(
                np.logical_or(cand > self._max_vals, cand < self._min_vals)
            )[0]
            theta_new = self.x + this_ak * ghat
            if len(out_of_bounds) == 0:
                self.x = np.clip(theta_new, self._min_vals, self._max_vals)
                not_all_pass = False
            else:
                this_ak[out_of_bounds] = this_ak[out_of_bounds] / 2.0
                if float(np.max(this_ak)) < 1e-30:
                    self.x = np.clip(theta_new, self._min_vals, self._max_vals)
                    not_all_pass = False

        # Post-step nudge from pasted code (slightly shrink from boundary hits)
        i_max = np.where(self.x >= self._max_vals)[0]
        i_min = np.where(self.x <= self._min_vals)[0]
        if len(i_max) > 0:
            self.x[i_max] = self._max_vals[i_max] * 0.9
        if len(i_min) > 0:
            self.x[i_min] = self._min_vals[i_min] * 1.1
        self.x = np.clip(self.x, self._min_vals, self._max_vals)

    def pull(self) -> np.ndarray:
        ck = self._ck_scalar()
        if self._need_plus:
            self._delta = self._rng.integers(0, 2, size=self._d, dtype=np.int8) * 2 - 1
            x_pert = self._perturb_plus(self.x, ck, self._delta.astype(float))
        else:
            assert self._delta is not None
            x_pert = self._perturb_minus(self.x, ck, self._delta.astype(float))
        self._last_action = snap_to_grid(x_pert)
        return self._last_action.copy()

    def update(self, reward: float) -> None:
        ck = self._ck_scalar()
        if self._need_plus:
            self._r_plus = float(reward)
            self._need_plus = False
            return
        assert self._r_plus is not None and self._delta is not None
        dlt = self._delta.astype(float)
        self._ghat_acc += (self._r_plus - float(reward)) / (2.0 * ck * dlt)
        self._need_plus = True
        self._delta = None
        self._ens_j += 1
        if self._ens_j >= self.ens_size:
            ghat = self._ghat_acc / float(self.ens_size)
            self._apply_gradient_ascent(ghat)
            self._n_iter += 1
            self._ens_j = 0
            self._ghat_acc = np.zeros(self._d, dtype=float)

    def reset(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.x = 0.5 * np.ones(self._d)
        self._n_iter = 0
        self._ens_j = 0
        self._need_plus = True
        self._delta = None
        self._ghat_acc = np.zeros(self._d, dtype=float)
        self._r_plus = None
        self._last_action = snap_to_grid(self.x)
