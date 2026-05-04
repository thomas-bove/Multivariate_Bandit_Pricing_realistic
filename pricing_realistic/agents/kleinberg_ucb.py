"""UCB1 on the finite grid ``X°`` (misleading historical label ``KleinbergUCB``).


"""
from __future__ import annotations

import numpy as np


class KleinbergUCB:
    def __init__(self, actions: np.ndarray, seed: int = 0, alpha: float = 1.0):
        self.actions = actions
        self.n_actions = len(actions)
        self._alpha = float(alpha)
        self._counts = np.zeros(self.n_actions)
        self._means = np.zeros(self.n_actions)
        self._t = 0
        self._rng = np.random.default_rng(seed)
        self._last_idx = 0

    def pull(self) -> np.ndarray:
        self._t += 1
        untried = np.where(self._counts == 0)[0]
        if len(untried):
            idx = int(self._rng.choice(untried))
        else:
            ucb = self._means + self._alpha * np.sqrt(
                2.0 * np.log(self._t) / self._counts
            )
            idx = int(np.argmax(ucb))
        self._last_idx = idx
        return self.actions[idx]

    def update(self, reward: float) -> None:
        i = self._last_idx
        n = self._counts[i]
        self._means[i] = (self._means[i] * n + reward) / (n + 1)
        self._counts[i] += 1

    def reset(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._counts[:] = 0
        self._means[:] = 0
        self._t = 0
