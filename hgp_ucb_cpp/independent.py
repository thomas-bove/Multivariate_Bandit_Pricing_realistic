import numpy as np
from hgp_ucb_cpp.igpucb import IGPUCB


class OptIndepPricingAgent:
    def __init__(self, actions, kernel_L, horizon, alpha):
        if not (isinstance(actions, np.ndarray) and actions.ndim == 2 and
                actions.shape[0] > 1 and actions.shape[1] == 1):
            raise ValueError("actions must be a 2D array with one column and more than one action.")
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")

        self.actions = actions
        self.action_dim = self.actions.shape[1]
        self.n_actions = self.actions.shape[0]
        self.alpha = alpha
        self.bandit_agent = IGPUCB(n_actions=self.n_actions,
                                   action_dim=1,
                                   actions=actions,
                                   kernel_L=kernel_L,
                                   sigma_sq=0.25,
                                   B=1,
                                   delta=1 / horizon,
                                   het=True)
        self.last_action = None

    def pull(self, return_val=False):
        demand_ucbs = self.bandit_agent.get_optimistic_estimates()
        obj_ucbs = (self.alpha + self.actions.ravel()) * demand_ucbs

        choice_idx = np.argmax(obj_ucbs)
        self.last_action = self.actions[choice_idx, :]

        la = float(np.asarray(self.last_action, dtype=float).reshape(-1)[0])
        if return_val:
            return la, float(np.max(obj_ucbs))
        return la

    def update(self, sales, impressions):
        if impressions > 0:
            self.bandit_agent.update_complete(self.last_action,
                                              float(sales / impressions),
                                              sample_weight=int(impressions))

    def update_complete(self, action, sales, impressions):
        if impressions > 0:
            self.bandit_agent.update_complete(action,
                                              float(sales / impressions),
                                              sample_weight=int(impressions))

    def get_expected_obj_max(self):
        _, mu, _ = self.bandit_agent.get_optimistic_estimates(return_raw_info=True)
        obj = (self.alpha + self.actions.ravel()) * mu
        return np.max(obj)
