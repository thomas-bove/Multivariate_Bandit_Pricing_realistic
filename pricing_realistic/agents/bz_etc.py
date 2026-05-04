"""BZ-ETC pricing."""
import numpy as np

class BZ_ETC:
    """Explore-Then-Commit pricing (Besbes & Zeevi 2012, OR 60(6)).

    Phase 1 (t < tau): cycle round-robin through candidate_actions.
    Phase 2 (t >= tau): commit forever to argmax empirical mean.

    tau = max(k, int(tau_factor * T^{2/3}))  where k = |candidate_actions|.

    Reference: Besbes, O. & Zeevi, A. (2012). "Blind network revenue
    management." Operations Research, 60(6), 1537–1550.  Without capacity
    constraints the LP collapses to argmax, so this is ETC on the sub-grid.
    """

    def __init__(self, actions: np.ndarray, T: int,
                 candidate_actions: np.ndarray = None,
                 tau_factor: float = 1.0, seed: int = 0):
        self._cands  = candidate_actions if candidate_actions is not None else actions
        self.k       = len(self._cands)
        self.tau     = max(self.k, int(tau_factor * T ** (2.0 / 3.0)))
        self._sums   = np.zeros(self.k)
        self._counts = np.zeros(self.k)
        self._t      = 0
        self._best   = None

    def pull(self) -> np.ndarray:
        if self._t < self.tau:
            return self._cands[self._t % self.k]
        if self._best is None:
            self._best = int(np.argmax(self._sums / np.maximum(self._counts, 1)))
        return self._cands[self._best]

    def update(self, reward: float) -> None:
        if self._t < self.tau:
            idx = self._t % self.k
            self._sums[idx]   += reward
            self._counts[idx] += 1
        self._t += 1

    def reset(self, seed: int = 0) -> None:
        self._sums[:]   = 0
        self._counts[:] = 0
        self._t         = 0
        self._best      = None

