import numpy as np
from hgp_ucb_cpp.gaussianprocess import (
    HeteroscedasticGaussianProcessRegressorRBF,
    GaussianProcessRegressorRBF
)


class IGPUCB:
    def __init__(self, n_actions, action_dim, actions, kernel_L, sigma_sq, B, delta, het=True):
        if not (actions.ndim == 2 and actions.shape == (n_actions, action_dim)):
            raise ValueError(f"Error in action dimension. Expected ({n_actions}, {action_dim}), got {actions.shape}")
        if not kernel_L > 0:
            raise ValueError("kernel_L must be positive")

        self.n_actions = n_actions
        self.action_dim = action_dim
        self.actions = actions
        self.kernel_L = kernel_L
        self.sigma_sq = sigma_sq
        self.B = B
        self.delta = delta
        self.het = het

        self.reset()

    def pull(self):
        if self.no_samples:
            choice = np.random.choice(self.n_actions)
            self.last_action = self.actions[choice, :]
        else:
            self.last_action = self.actions[np.argmax(self.get_optimistic_estimates()), :]
        return self.last_action

    def get_optimistic_estimates(self, return_raw_info=False):
        if self.no_samples:
            mu = np.zeros((self.n_actions))
            sigmasq = np.ones((self.n_actions))
            ucbs = np.ones((self.n_actions))
        else:
            mu, sigmasq = self.regressor.compute(self.actions)
            beta = self._get_beta()
            ucbs = mu + beta * np.sqrt(sigmasq)

        if return_raw_info:
            return ucbs.ravel(), mu.ravel(), sigmasq.ravel()
        else:
            return ucbs.ravel()

    def update(self, reward, sample_weight=1):
        if self.last_action is None:
            raise ValueError("No action has been selected yet. Call pull() before update().")
        if sample_weight != 1 and not self.het:
            raise NotImplementedError("Batch/weighted updates are not implemented for the standard (non-heteroscedastic) GP.")
        if sample_weight == 0:
            raise ValueError("sample_weight cannot be 0")

        self.regressor.add_sample(self.last_action.reshape(1, self.action_dim),
                                  np.array(reward).reshape(1, 1),
                                  sample_weight=sample_weight)

        self.no_samples = False

    def update_complete(self, action, reward, sample_weight=1):
        if sample_weight != 1 and not self.het:
            raise NotImplementedError("Batch/weighted updates are not implemented for the standard (non-heteroscedastic) GP.")

        self.regressor.add_sample(np.array(action).reshape(1, self.action_dim),
                                  np.array(reward).reshape(1, 1),
                                  sample_weight=sample_weight)

        self.no_samples = False

    def _get_beta(self):
        return self.B + np.sqrt(self.sigma_sq) * np.sqrt(2 * (
            self.regressor.get_info_gain() + 1 + np.log(1 / self.delta)))

    def reset(self):
        self.no_samples = True
        self.last_action = None

        if self.het:
            self.regressor = HeteroscedasticGaussianProcessRegressorRBF(
                self.kernel_L, self.sigma_sq,
                input_dim=self.action_dim, one_sample_mod=True)
        else:
            self.regressor = GaussianProcessRegressorRBF(
                self.kernel_L, self.sigma_sq,
                input_dim=self.action_dim, keep_info_gain_estimate=True)
