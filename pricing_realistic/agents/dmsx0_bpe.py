"""Li–Scarlett BPE on full X° (DMSX0BPE)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import normalize_joint_reward
from ..kernels import KERNEL_L, _matern12_product

def li_22_bpe_batch_sizes(T: int) -> np.ndarray:
    """PRE-COMPUTE: N_0=1, N_i=⌈√(T·N_{i-1})⌉, clip last batch so ∑_i N_i = T (Li & Scarlett 2022)."""
    if T <= 0:
        return np.array([], dtype=int)
    batch_size = [1]
    while True:
        N_i = int(np.ceil(np.sqrt(T * batch_size[-1])))
        if sum(batch_size) + N_i - 1 >= T:
            batch_size.append(T + 1 - sum(batch_size))
            break
        batch_size.append(N_i)
    return np.asarray(batch_size[1:], dtype=int)


class DMSX0BPE:
    """
    **Paper method:** BPE = Algorithm 1 of Li & Scarlett (2022) on the **full** discrete cover
    ``X°`` = ``ALL_ACTIONS_NORM`` (**not** Smolyak).  GP = **product Matérn-1/2** ``k_Π``, Li batch
    lengths, **active set**, UCB/LCB elimination.  The confidence width ``√β`` follows
    Theorem 4.3 of the NeurIPS 2026 submission (leaner than the original Li–Scarlett version:
    no ``M_total`` factor in the log-term).

    Within each Li–Scarlett batch, ``pull`` explores by maximum posterior σ on a **query set** that
    excludes arms already pulled in the current batch until every active arm has been selected
    (then duplicates are allowed), matching set-style batch accumulation without wasted repeats.

    Pass **raw** scalar rewards (original profit scale) into ``update``; they are mapped with
    ``normalize_joint_reward`` for the GP — do **not** pass pre-normalized values.

    Default ``bpe_use_global_history=True`` (Li–Scarlett / paper-consistent): elimination and
    the GP posterior for UCB/LCB are built on **all data** collected up to the current batch,
    which is what gives the T^{3/4} rate of Theorem 4.3.  Set ``bpe_use_global_history=False``
    to switch to the experimental batch-local variant (σ and elimination use only the current
    batch's rewards); global ``_all_train_*`` is still updated after each batch for
    ``recommend_idx`` / diagnostics.
    """

    _JITTER = 1e-6

    def __init__(
        self,
        actions   : np.ndarray,
        T         : int,
        noise_var : float,
        delta     : float = 0.1,
        kernel_L  : float = KERNEL_L,
        B_rkhs    : float = 1.0,   # illustrative default; override with C_d·d·2^{d/2+2}·√L_2 (Thm 3.2/3.4)
        seed      : int = 0,
        prior_mean: float = 0.0,
        bpe_beta_use_active_count: bool = False,
        noise_R   : Optional[float] = None,
        bpe_use_global_history: bool = True,
        compute_full_posterior_for_debug: bool = False,
    ):
        self.actions    = np.asarray(actions, dtype=float)
        self.T          = int(T)
        self.noise_var = float(noise_var)
        self.delta      = float(delta)
        self.kernel_L   = float(kernel_L)
        self.B_rkhs     = float(B_rkhs)
        self.prior_mean = float(prior_mean)
        self.compute_full_posterior_for_debug = bool(compute_full_posterior_for_debug)
        self._bpe_beta_use_active_count = bool(bpe_beta_use_active_count)
        self._bpe_use_global_history = bool(bpe_use_global_history)
        lam_reg = max(float(self.noise_var), 1e-18)
        if noise_R is None:
            self._noise_factor = 1.0
        else:
            self._noise_factor = float(noise_R) / float(np.sqrt(lam_reg))
        self._rng       = np.random.default_rng(seed)
        self._kfn       = lambda X, Y: _matern12_product(X, Y, kernel_L)

        self._batch_sizes = li_22_bpe_batch_sizes(self.T)
        self._M_total     = max(len(self._batch_sizes), 1)
        self._n_arms      = len(self.actions)

        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._active_mask: np.ndarray = np.ones(self._n_arms, dtype=bool)
        self._all_train_idx: list[int] = []
        self._all_rewards: list[float] = []
        self._batch_idx = 0
        self._t_done = 0
        self._cur_batch_len = 0
        self._pos_in_batch = 0
        self._batch_intr_idx: list[int] = []
        self._batch_intr_rewards: list[float] = []
        self._pending_arm_idx: int = 0
        self._last_x: Optional[np.ndarray] = None
        self._committed_idx: Optional[int] = None
        self._mu_post: Optional[np.ndarray] = None
        self._last_active_idx: Optional[np.ndarray] = None
        self._last_active_mu: Optional[np.ndarray] = None
        self._last_active_sig: Optional[np.ndarray] = None
        self._start_new_batch()

    def _sqrt_beta_elimination(self, n_active: Optional[int] = None) -> float:
        """
        Theorem 4.3 confidence width (paper-faithful): ``√β = Ψ + (R/√λ)·√(2 log(|X°|/δ))``.

        Matches the Λ term in Theorem 4.3 of the NeurIPS 2026 submission — no M_total factor
        (leaner than the original Li–Scarlett union-bound version).
        If ``noise_R`` was omitted at construction, ``R/√λ = 1`` (``λ = R²`` on the reward scale).
        ``|X|_eff = |X°|`` unless legacy ``bpe_beta_use_active_count`` passes ``n_active``.
        """
        if self._bpe_beta_use_active_count and n_active is not None:
            n_eff = int(n_active)
        else:
            n_eff = int(self._n_arms)
        ratio = max(float(n_eff) / float(self.delta), np.e)
        log_m = float(np.log(ratio))
        return float(
            self.B_rkhs
            + float(self._noise_factor) * float(np.sqrt(2.0 * log_m))
        )

    def _gp_mu_sigma_vectors(
        self,
        train_idx: list[int],
        y        : np.ndarray,
        query_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean / std at ``query_idx`` (``k_Π``, λ); fixed prior mean ``prior_mean`` on y."""
        y = np.asarray(y, dtype=float)
        m = len(train_idx)
        nq = len(query_idx)
        lam = self.noise_var + self._JITTER
        if m == 0:
            return np.full(nq, self.prior_mean), np.ones(nq)

        y_work = y - self.prior_mean
        X_tr = self.actions[train_idx]
        K_mm = self._kfn(X_tr, X_tr) + lam * np.eye(m)
        X_q = self.actions[np.asarray(query_idx, dtype=int)]
        K_q = self._kfn(X_tr, X_q)

        try:
            Lch = np.linalg.cholesky(K_mm)
            alpha = np.linalg.solve(Lch.T, np.linalg.solve(Lch, y_work))
            tmp = np.linalg.solve(Lch, K_q)
            S = np.linalg.solve(Lch.T, tmp)
        except np.linalg.LinAlgError:
            Kinv = np.linalg.solve(K_mm + np.eye(m) * 1e-4, np.eye(m))
            alpha = Kinv @ y_work
            S = Kinv @ K_q

        mu = (K_q.T @ alpha).ravel() + self.prior_mean
        # Assumes k(x,x) = 1 (holds for the product Matérn-1/2 with signal_variance=1,
        # i.e. _matern12_product as currently implemented). If _matern12_product is ever
        # extended with a signal variance ≠ 1, replace the 1.0 with the diagonal of the
        # prior kernel evaluated at X_q: k_prior = np.diag(self._kfn(X_q, X_q)).
        var = np.maximum(1.0 - np.sum(K_q * S, axis=0), 0.0)
        return mu, np.sqrt(var)

    def _full_posterior_means(self) -> np.ndarray:
        """Posterior mean at every arm in X° (expensive; only when ``compute_full_posterior_for_debug``)."""
        if not self._all_train_idx:
            return np.full(self._n_arms, self.prior_mean, dtype=float)
        mu, _ = self._gp_mu_sigma_vectors(
            self._all_train_idx,
            np.asarray(self._all_rewards, dtype=float),
            np.arange(self._n_arms, dtype=int),
        )
        return mu

    def _repair_empty_active(self) -> None:
        if int(self._active_mask.sum()) > 0:
            return
        if self._last_active_idx is not None and self._last_active_mu is not None:
            lai = self._last_active_idx
            lam = self._last_active_mu
            best = int(lai[int(np.argmax(lam))])
        elif self._all_train_idx:
            idx_a = np.asarray(self._all_train_idx, dtype=int)
            y_a = np.asarray(self._all_rewards, dtype=float)
            uniq = np.unique(idx_a)
            means = np.array([float(y_a[idx_a == u].mean()) for u in uniq], dtype=float)
            best = int(uniq[int(np.argmax(means))])
        else:
            best = 0
        self._active_mask[:] = False
        self._active_mask[best] = True

    def _arm_max_sigma_active(self) -> int:
        """
        Max posterior σ among actives **eligible for this batch step**.

        Within a Li–Scarlett batch we prefer not to repeat an arm until every currently active
        arm has been pulled at least once in this batch (matches Algorithm~1 set accumulation:
        ``S_i ← S_i ∪ {x_t}`` without redundant duplicates while unexplored actives remain).

        * ``bpe_use_global_history=False`` (default): train only on **current batch** locations,
          with fantasy ``y = prior_mean`` everywhere (batch-local exploration, Li–Scarlett).
        * ``bpe_use_global_history=True``: global history ∪ batch locations; fantasy ``y`` on
          the still-uncommitted batch rows only.

        Posterior σ is evaluated only on the **query** set (available ∪ fallback to all active);
        elimination still uses all actives (see ``_eliminate``).
        """
        self._repair_empty_active()
        active = np.where(self._active_mask)[0].astype(int)
        already = {int(i) for i in self._batch_intr_idx}
        available = np.array(
            [int(a) for a in active if int(a) not in already],
            dtype=int,
        )
        query_set = available if len(available) > 0 else active

        if self._bpe_use_global_history:
            train_idx = list(self._all_train_idx) + list(self._batch_intr_idx)
            if len(train_idx) == 0:
                return int(self._rng.choice(query_set))
            y_parts = list(self._all_rewards) + [self.prior_mean] * len(self._batch_intr_idx)
            y = np.asarray(y_parts, dtype=float)
        else:
            train_idx = list(self._batch_intr_idx)
            if len(train_idx) == 0:
                return int(self._rng.choice(query_set))
            y = np.full(len(train_idx), self.prior_mean, dtype=float)

        _mu, sig = self._gp_mu_sigma_vectors(train_idx, y, query_set.astype(int))
        smax = float(np.max(sig))
        cands = query_set[sig >= smax - 1e-12]
        return int(self._rng.choice(cands))

    def _eliminate(self) -> None:
        if self._bpe_use_global_history:
            if not self._all_train_idx:
                return
            train_idx = list(self._all_train_idx)
            y = np.asarray(self._all_rewards, dtype=float)
        else:
            if not self._batch_intr_idx:
                return
            train_idx = list(self._batch_intr_idx)
            y = np.asarray(self._batch_intr_rewards, dtype=float)

        self._repair_empty_active()
        active = np.where(self._active_mask)[0]
        if len(active) == 0:
            return

        n_active = int(len(active))
        sqrt_beta = (
            self._sqrt_beta_elimination(n_active)
            if self._bpe_beta_use_active_count
            else self._sqrt_beta_elimination()
        )

        mu, sig = self._gp_mu_sigma_vectors(train_idx, y, active.astype(int))
        self._last_active_idx = active.astype(int).copy()
        self._last_active_mu = mu.copy()
        self._last_active_sig = sig.copy()

        ucbs = mu + sqrt_beta * sig
        lcbs = mu - sqrt_beta * sig
        max_lcb = float(lcbs.max())
        keep = ucbs >= max_lcb
        self._active_mask[active] = keep

        if not np.any(keep):
            best = int(active[np.argmax(mu)])
            self._active_mask[:] = False
            self._active_mask[best] = True

        if int(self._active_mask.sum()) == 1:
            self._committed_idx = int(np.where(self._active_mask)[0][0])

    def recommend_idx(self) -> int:
        """
        Best arm **recommendation** from completed-batch GP state only (not necessarily the
        exploratory pull).  Prefer cached elimination posterior on actives; optional debug
        full-grid ``_mu_post``; else GP mean on ``_all_train_idx`` at **current** actives only.
        With no global history, returns ``active[0]`` (deterministic; does not advance ``_rng``).
        """
        if self._committed_idx is not None:
            return int(self._committed_idx)
        self._repair_empty_active()
        active = np.where(self._active_mask)[0].astype(int)

        if self._last_active_idx is not None and self._last_active_mu is not None:
            lai = self._last_active_idx
            lam = self._last_active_mu
            mask_keep = self._active_mask[lai]
            if np.any(mask_keep):
                sub_idx = lai[mask_keep]
                sub_mu = lam[mask_keep]
                j = int(np.argmax(sub_mu))
                return int(sub_idx[j])

        if self._mu_post is not None:
            sub = self._mu_post[active]
            j = int(np.argmax(sub))
            return int(active[j])
        if not self._all_train_idx:
            return int(active[0])
        y = np.asarray(self._all_rewards, dtype=float)
        mu, _ = self._gp_mu_sigma_vectors(self._all_train_idx, y, active)
        j = int(np.argmax(mu))
        return int(active[j])

    def recommend(self) -> np.ndarray:
        """Normalised action vector on ``X°`` for ``recommend_idx()``."""
        return self.actions[self.recommend_idx()].copy()

    def _start_new_batch(self) -> None:
        if self._committed_idx is not None:
            self._cur_batch_len = 0
            return
        remaining = self.T - self._t_done
        if remaining <= 0:
            self._cur_batch_len = 0
            return
        if self._batch_idx >= len(self._batch_sizes):
            N_sched = remaining
        else:
            N_sched = int(self._batch_sizes[self._batch_idx])
        N_i = int(min(N_sched, remaining))
        self._cur_batch_len = max(N_i, 1)
        self._pos_in_batch = 0
        self._batch_intr_idx.clear()
        self._batch_intr_rewards.clear()

    def pull(self, snap_fn=None) -> np.ndarray:
        if self._committed_idx is not None:
            x = self.actions[self._committed_idx].copy()
            if snap_fn is not None:
                x = snap_fn(x)
            self._last_x = x
            self._pending_arm_idx = int(self._committed_idx)
            return x

        if self._cur_batch_len <= 0:
            self._repair_empty_active()
            idx = int(np.where(self._active_mask)[0][0])
            self._pending_arm_idx = idx
            x = self.actions[idx].copy()
        else:
            arm = self._arm_max_sigma_active()
            self._pending_arm_idx = int(arm)
            x = self.actions[arm].copy()

        if snap_fn is not None:
            x = snap_fn(x)
        self._last_x = x
        return x

    def update(self, reward: float) -> None:
        """
        Record ``reward`` for the last ``pull``.

        ``reward`` must be in **original profit scale** (same as ``env``); it is normalized
        internally for the GP.  Do **not** pass values already divided by ``REWARD_SCALE``.
        """
        if self._committed_idx is not None:
            self._all_train_idx.append(self._committed_idx)
            self._all_rewards.append(float(normalize_joint_reward(float(reward))))
            self._t_done += 1
            return

        self._batch_intr_idx.append(int(self._pending_arm_idx))
        self._batch_intr_rewards.append(float(normalize_joint_reward(float(reward))))
        self._pos_in_batch += 1
        self._t_done += 1

        if self._pos_in_batch >= self._cur_batch_len:
            if self._bpe_use_global_history:
                self._all_train_idx.extend(self._batch_intr_idx)
                self._all_rewards.extend(self._batch_intr_rewards)
                self._batch_intr_idx.clear()
                self._batch_intr_rewards.clear()
                self._eliminate()
            else:
                self._eliminate()
                self._all_train_idx.extend(self._batch_intr_idx)
                self._all_rewards.extend(self._batch_intr_rewards)
                self._batch_intr_idx.clear()
                self._batch_intr_rewards.clear()
            self._batch_idx += 1
            self._pos_in_batch = 0
            if self._committed_idx is None:
                self._start_new_batch()
            if self.compute_full_posterior_for_debug and self._all_train_idx:
                self._mu_post = self._full_posterior_means()
            else:
                self._mu_post = None

