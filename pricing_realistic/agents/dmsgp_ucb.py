"""DMSGPUCB + IsotropicMatern12GPUCB (competitor / iso baseline)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize as _sp_minimize

from ..config import normalize_joint_reward
from ..kernels import KERNEL_L, _matern12_product, _matern12_isotropic

class DMSGPUCB:
    _JITTER       = 1e-6
    _HALLUC_NOISE = 1e-8

    def __init__(
        self,
        d          : int,
        noise_var  : float,
        c_beta     : float = 1.0,
        delta      : float = 0.1,
        kernel_L   : float = KERNEL_L,
        B_rkhs     : float = 1.0,
        seed       : int   = 0,
        use_batches: bool  = True,
        batch_base : int   = 2,
        batch_ratio: float = 2.0,
        n_restarts : int   = 20,
        li_scarlett_batch_sizes: Optional[np.ndarray] = None,
        normalize_rewards: bool = True,
    ):
        self._d          = d
        self.noise_var   = noise_var
        self.c_beta      = c_beta
        self.delta       = delta
        self.kernel_L    = kernel_L
        self.B_rkhs      = B_rkhs
        self.n_restarts  = n_restarts
        self.normalize_rewards = bool(normalize_rewards)
        self._rng        = np.random.default_rng(seed)
        self._kfn        = lambda X, Y: _matern12_product(X, Y, kernel_L)

        self.use_batches      = use_batches
        self.batch_base       = batch_base
        self.batch_ratio      = batch_ratio
        self._batch_idx       = 0
        self._buf             = []

        if li_scarlett_batch_sizes is not None:
            if not use_batches:
                raise ValueError("li_scarlett_batch_sizes requires use_batches=True")
            self._li_sched = np.asarray(li_scarlett_batch_sizes, dtype=int).ravel()
            if self._li_sched.size == 0:
                raise ValueError("li_scarlett_batch_sizes must be non-empty")
            self._li_m_total = int(self._li_sched.size)
            self._li_seg = 0
            self._cur_batch_len = int(self._li_sched[0])
            self._batch_remaining = self._cur_batch_len
        else:
            self._li_sched = None
            self._li_m_total = 0
            self._li_seg = 0
            self._cur_batch_len = int(batch_base)
            self._batch_remaining = int(batch_base)

        self._t      = 0
        self._X      = None   # (n, d) observed points in [0,1]^d
        self._y      = None   # (n,)   averaged rewards
        self._N      = None   # (n,)   observation counts
        self._Kinv   = None   # (n, n)
        self._last_x = None   # (d,)   last pulled point

        self._hX    = None    # (n_h, d) hallucinated points
        self._hy    = None    # (n_h,)   hallucinated posterior means
        self._wKinv = None    # working Kinv (real + hallucinated)

        self._gamma_cache = 0.0
        self._gamma_dirty = True

    def _current_batch_size(self) -> int:
        if self._li_sched is not None:
            return max(int(self._cur_batch_len), 1)
        return int(np.ceil(self.batch_base * (self.batch_ratio ** self._batch_idx)))

    def _compute_gamma(self) -> float:
        if not self._gamma_dirty:
            return self._gamma_cache
        if self._X is None or len(self._X) == 0:
            self._gamma_cache = 0.0
        else:
            K         = self._kfn(self._X, self._X)
            inv_noise = self._N / self.noise_var
            M         = K * inv_noise[None, :] + np.eye(len(self._X))
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                _, logdet = np.linalg.slogdet(M)
            ld = float(logdet)
            self._gamma_cache = 0.5 * max(ld if np.isfinite(ld) else 0.0, 0.0)
        self._gamma_dirty = False
        return self._gamma_cache

    def _beta(self) -> float:
        gamma = self._compute_gamma()
        if self.use_batches:
            B = max(self._current_batch_size(), 1)
            if self._li_sched is not None:
                # Li & Scarlett (2022) style width: batch length B, M_tot = #batches (cf. BPE §3).
                ratio = max(float(self._li_m_total) * float(B) / float(self.delta), np.e)
                base = gamma + 1.0 + np.log(ratio)
            else:
                base = gamma + 1.0 + np.log(B / self.delta)
        else:
            base = gamma + 1.0 + np.log(1.0 / self.delta)
        return self.B_rkhs + np.sqrt(self.noise_var) * self.c_beta * np.sqrt(
            2.0 * base
        )

    def _build_kinv(self, X: np.ndarray, N: np.ndarray) -> np.ndarray:
        n = len(N)
        K = self._kfn(X, X) + np.diag(self.noise_var / N + self._JITTER)
        try:
            L = np.linalg.cholesky(K)
            return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        except np.linalg.LinAlgError:
            return np.linalg.solve(K + np.eye(n) * 1e-4, np.eye(n))

    def _rebuild_working_kinv(self) -> None:
        if self._X is None:
            self._wKinv = None
            return
        if self._hX is None or len(self._hX) == 0:
            self._wKinv = self._Kinv
            return
        X_all = np.vstack([self._X, self._hX])
        n_h   = len(self._hX)
        diag  = np.concatenate([
            self.noise_var / self._N + self._JITTER,
            np.full(n_h, self._HALLUC_NOISE + self._JITTER),
        ])
        K     = self._kfn(X_all, X_all) + np.diag(diag)
        n_tot = len(X_all)
        try:
            L = np.linalg.cholesky(K)
            self._wKinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n_tot)))
        except np.linalg.LinAlgError:
            self._wKinv = np.linalg.solve(K + np.eye(n_tot) * 1e-4, np.eye(n_tot))

    def _optimise_ucb(self, X_data: np.ndarray, y_data: np.ndarray,
                      Kinv: np.ndarray, beta: float) -> np.ndarray:
        bounds = [(0.0, 1.0)] * self._d

        def neg_ucb(x):
            xq  = x.reshape(1, -1)
            kx  = self._kfn(xq, X_data).ravel()
            mu  = float(kx @ Kinv @ y_data)
            var = max(1.0 - float(kx @ Kinv @ kx), 0.0)
            return -(mu + beta * np.sqrt(var))

        starts   = self._rng.uniform(0.0, 1.0, (self.n_restarts, self._d))
        best_x   = starts[0]
        best_val = np.inf
        for x0 in starts:
            res = _sp_minimize(neg_ucb, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 200, "ftol": 1e-9})
            if res.fun < best_val:
                best_val = res.fun
                best_x   = res.x
        return np.clip(best_x, 0.0, 1.0)

    def _gp_add(self, x: np.ndarray, r: float):
        if self._X is None:
            self._X = x[None, :]
            self._y = np.array([r])
            self._N = np.array([1.0])
        else:
            dists = np.max(np.abs(self._X - x), axis=1)
            m     = np.where(dists < 1e-10)[0]
            if len(m):
                i = m[0]; n0 = self._N[i]
                self._y[i] = (self._y[i] * n0 + r) / (n0 + 1)
                self._N[i] = n0 + 1
            else:
                self._X = np.vstack([self._X, x[None, :]])
                self._y = np.append(self._y, r)
                self._N = np.append(self._N, 1.0)

    def pull(self, snap_fn=None) -> np.ndarray:
        self._t += 1

        if self._X is None or self._Kinv is None:
            x = self._rng.uniform(0.0, 1.0, self._d)
        else:
            beta = self._beta()
            if (self.use_batches and self._wKinv is not None
                    and self._hX is not None and len(self._hX) > 0):
                X_all = np.vstack([self._X, self._hX])
                y_all = np.concatenate([self._y, self._hy])
                Kq    = self._wKinv
            else:
                X_all, y_all, Kq = self._X, self._y, self._Kinv
            x = self._optimise_ucb(X_all, y_all, Kq, beta)

        if snap_fn is not None:
            x = snap_fn(x)

        self._last_x = x

        if (self.use_batches and self._X is not None
                and self._Kinv is not None
                and np.all(np.isfinite(self._Kinv))):
            kx     = self._kfn(x[None, :], self._X).ravel()
            mu_sel = float(kx @ self._Kinv @ self._y)
            if self._hX is None:
                self._hX = x[None, :]
                self._hy = np.array([mu_sel])
            else:
                self._hX = np.vstack([self._hX, x[None, :]])
                self._hy = np.append(self._hy, mu_sel)
            self._rebuild_working_kinv()

        return x

    def update(self, reward: float):
        """
        Append the last played point and ``reward``.

        Pass rewards in **original / profit scale** (same units as ``env``); when
        ``normalize_rewards`` is ``True`` (joint / isotropic GP-UCB), ``DMSGPUCB`` maps them with
        ``normalize_joint_reward`` before the GP.  Do **not** pass pre-normalized rewards unless
        ``normalize_rewards=False`` (e.g. Univariate marginal agents).
        """
        self._buf.append((self._last_x.copy(), reward))
        self._batch_remaining -= 1

        if self.use_batches and self._batch_remaining > 0:
            return

        for x, r in self._buf:
            r_gp = (
                normalize_joint_reward(float(r))
                if self.normalize_rewards
                else float(r)
            )
            self._gp_add(x, r_gp)

        if self._X is not None:
            self._Kinv = self._build_kinv(self._X, self._N)

        self._gamma_dirty = True
        self._buf.clear()
        self._hX = self._hy = None
        self._wKinv = None
        if self._li_sched is not None:
            self._li_seg += 1
            if self._li_seg < len(self._li_sched):
                self._cur_batch_len = int(self._li_sched[self._li_seg])
                self._batch_remaining = self._cur_batch_len
            else:
                self._cur_batch_len = 1
                self._batch_remaining = 1
        else:
            self._batch_idx += 1
            self._batch_remaining = self._current_batch_size()

    def reset(self, seed: int = 0):
        self._rng             = np.random.default_rng(seed)
        self._t               = 0
        self._X = self._y = self._N = self._Kinv = None
        self._last_x          = None
        self._buf.clear()
        self._hX = self._hy = None
        self._wKinv = None
        self._gamma_cache = 0.0
        self._gamma_dirty = True
        if self._li_sched is not None:
            self._li_seg = 0
            self._cur_batch_len = int(self._li_sched[0])
            self._batch_remaining = self._cur_batch_len
        else:
            self._batch_idx = 0
            self._batch_remaining = self.batch_base


class IsotropicMatern12GPUCB(DMSGPUCB):
    """
    GP-UCB on ``[0,1]^d`` with **isotropic** Matérn-1/2 covariance ``exp(-‖x-x'‖_2 / L)`` — same
    update rules, ``γ_t``, ``β_t``, Li–Scarlett batch schedule, and hallucination path as
    ``DMSGPUCB``, but **no** product-kernel / dominating-mixed-smoothness structure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        L = float(self.kernel_L)
        self._kfn = lambda X, Y, L=L: _matern12_isotropic(X, Y, L)

