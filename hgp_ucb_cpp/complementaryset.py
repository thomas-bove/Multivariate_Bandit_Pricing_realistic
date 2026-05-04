import numpy as np
from hgp_ucb_cpp.igpucb import IGPUCB

LEAD = "lead"
NOLEAD = "no_lead"


class OptComplementarySetPricingAgent:
    def __init__(self, n_actions, n_followers, margins_leader,
                 margins_followers, kernel_L, horizon, alpha):
        if not isinstance(margins_leader, np.ndarray) or margins_leader.shape != (n_actions,):
            raise ValueError(f"Shape of margins_leader ({margins_leader.shape}) not coherent with n_actions ({n_actions}). Expected ({n_actions},).")

        if not isinstance(margins_followers, np.ndarray) or margins_followers.shape != (n_followers, n_actions):
            raise ValueError(f"Shape of margins_followers ({margins_followers.shape}) not coherent. Expected ({n_followers}, {n_actions}).")

        if not (alpha >= 0 and alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], but got {alpha}")

        if n_followers not in [1, 2]:
            raise NotImplementedError(f"Agent currently only supports 1 or 2 followers, but got {n_followers}.")

        self.n_followers = n_followers
        self.n_actions = n_actions
        self.margins_leader = margins_leader
        self.margins_followers = margins_followers
        self.kernel_L = kernel_L
        self.horizon = horizon
        self.alpha = alpha

        self.leader_bandit_agent = IGPUCB(self.n_actions, 1, self.margins_leader.reshape(self.n_actions, 1),
                                          kernel_L, 0.25, 1, 1 / horizon, het=True)

        self.followers_bandit_agent = []
        for i in range(self.n_followers):
            follower_margins = self.margins_followers[i, :].reshape(self.n_actions, 1)
            self.followers_bandit_agent.append({
                LEAD: IGPUCB(self.n_actions, 1, follower_margins,
                             kernel_L, 0.25, 1, 1 / horizon, het=True),
                NOLEAD: IGPUCB(self.n_actions, 1, follower_margins,
                               kernel_L, 0.25, 1, 1 / horizon, het=True)
            })

        self.last_action = None

    def pull(self, return_val=False):
        demand_ucbs_leader, demand_mu_leader, _ = self.leader_bandit_agent.get_optimistic_estimates(return_raw_info=True)
        obj_ucbs_leader = (self.alpha + self.margins_leader.ravel()) * demand_ucbs_leader

        demand_ucbs_followers = {LEAD: -1 * np.ones((self.n_followers, self.n_actions)),
                                 NOLEAD: -1 * np.ones((self.n_followers, self.n_actions))}

        for i in range(self.n_followers):
            demand_ucbs_followers[LEAD][i, :] = self.followers_bandit_agent[i][LEAD].get_optimistic_estimates()
            demand_ucbs_followers[NOLEAD][i, :] = self.followers_bandit_agent[i][NOLEAD].get_optimistic_estimates()

        if self.n_followers == 1:
            opt_mx = -1 * np.ones((self.n_actions, self.n_actions))

            for leader_act_i, _ in enumerate(list(self.margins_leader.ravel())):
                for follower0_act_i, follower0_act in enumerate(list(self.margins_followers[0, :])):
                    demand_composite = demand_mu_leader[leader_act_i] * demand_ucbs_followers[LEAD][0, follower0_act_i] + \
                        (1 - demand_mu_leader[leader_act_i]) * demand_ucbs_followers[NOLEAD][0, follower0_act_i]

                    follower_obj_ucb = (self.alpha + follower0_act) * demand_composite
                    opt_mx[leader_act_i, follower0_act_i] = obj_ucbs_leader[leader_act_i] + follower_obj_ucb

            opt_action_idx_leader, opt_action_idx_follower0 = np.unravel_index(np.argmax(opt_mx), opt_mx.shape)

            self.last_action = np.array([self.margins_leader[opt_action_idx_leader],
                                         self.margins_followers[0, opt_action_idx_follower0]])

        if self.n_followers == 2:
            opt_mx = -1 * np.ones((self.n_actions, self.n_actions, self.n_actions))

            for leader_act_i, _ in enumerate(list(self.margins_leader.ravel())):
                for follower0_act_i, follower0_act in enumerate(list(self.margins_followers[0, :])):
                    for follower1_act_i, follower1_act in enumerate(list(self.margins_followers[1, :])):
                        demand_composite0 = demand_mu_leader[leader_act_i] * demand_ucbs_followers[LEAD][0, follower0_act_i] + \
                            (1 - demand_mu_leader[leader_act_i]) * demand_ucbs_followers[NOLEAD][0, follower0_act_i]

                        demand_composite1 = demand_mu_leader[leader_act_i] * demand_ucbs_followers[LEAD][1, follower1_act_i] + \
                            (1 - demand_mu_leader[leader_act_i]) * demand_ucbs_followers[NOLEAD][1, follower1_act_i]

                        follower0_obj_ucb = (self.alpha + follower0_act) * demand_composite0
                        follower1_obj_ucb = (self.alpha + follower1_act) * demand_composite1

                        opt_mx[leader_act_i, follower0_act_i, follower1_act_i] = obj_ucbs_leader[leader_act_i] + \
                            follower0_obj_ucb + follower1_obj_ucb

            opt_action_idx_leader, opt_action_idx_follower0, opt_action_idx_follower1 = np.unravel_index(np.argmax(opt_mx), opt_mx.shape)

            self.last_action = np.array([self.margins_leader[opt_action_idx_leader],
                                         self.margins_followers[0, opt_action_idx_follower0],
                                         self.margins_followers[1, opt_action_idx_follower1]])

        if return_val:
            return self.last_action, np.max(opt_mx)
        else:
            return self.last_action

    def update(self, sales_leader_vect, sales_followers_mx):
        if self.last_action is None:
            raise RuntimeError("update() called before pull(). Must pull an action first.")

        if not (isinstance(sales_leader_vect, np.ndarray) and sales_leader_vect.ndim == 1):
            raise ValueError("sales_leader_vect shape error. Expected 1D array.")
        if not (isinstance(sales_followers_mx, np.ndarray) and sales_followers_mx.ndim == 2 and sales_followers_mx.shape[0] == self.n_followers):
            raise ValueError(f"sales_followers_mx shape error. Expected 2D array with shape ({self.n_followers}, n_samples).")
        if sales_leader_vect.shape[0] != sales_followers_mx.shape[1]:
            raise ValueError("Mismatch in number of samples between leader sales vector and follower sales matrix.")

        n_samples = sales_leader_vect.shape[0]
        n_leader_sales = np.sum(sales_leader_vect)
        n_no_leader_sales = n_samples - n_leader_sales

        observed_leader_demand = float(np.sum(sales_leader_vect) / n_samples)
        self.leader_bandit_agent.update_complete(self.last_action[0], observed_leader_demand, sample_weight=n_samples)

        mask_leader_sales = (sales_leader_vect == 1)

        for fl_i in range(self.n_followers):
            follower_sales_all = sales_followers_mx[fl_i, :]
            sales_with_leader = np.sum(follower_sales_all[mask_leader_sales])
            sales_no_leader = np.sum(follower_sales_all[np.logical_not(mask_leader_sales)])

            follower_price = self.last_action[1 + fl_i]

            if n_leader_sales > 0:
                observed_demand_with_lead = float(sales_with_leader / n_leader_sales)
                self.followers_bandit_agent[fl_i][LEAD].update_complete(follower_price,
                                                                        observed_demand_with_lead,
                                                                        sample_weight=int(n_leader_sales))

            if n_no_leader_sales > 0:
                observed_demand_no_lead = float(sales_no_leader / n_no_leader_sales)
                self.followers_bandit_agent[fl_i][NOLEAD].update_complete(follower_price,
                                                                           observed_demand_no_lead,
                                                                           sample_weight=int(n_no_leader_sales))

    def compute_best_expected_leader(self):
        _, demand_mu_leader, _ = self.leader_bandit_agent.get_optimistic_estimates(return_raw_info=True)
        return (self.alpha + self.margins_leader.ravel()) * demand_mu_leader

    def compute_best_expected_overall(self):
        _, demand_mu_leader, _ = self.leader_bandit_agent.get_optimistic_estimates(return_raw_info=True)
        obj_mu_leader = (self.alpha + self.margins_leader.ravel()) * demand_mu_leader

        demand_mu_followers = {LEAD: -1 * np.ones((self.n_followers, self.n_actions)),
                               NOLEAD: -1 * np.ones((self.n_followers, self.n_actions))}

        for i in range(self.n_followers):
            _, val, _ = self.followers_bandit_agent[i][LEAD].get_optimistic_estimates(return_raw_info=True)
            demand_mu_followers[LEAD][i, :] = val
            _, val_nolead, _ = self.followers_bandit_agent[i][NOLEAD].get_optimistic_estimates(return_raw_info=True)
            demand_mu_followers[NOLEAD][i, :] = val_nolead

        if self.n_followers == 1:
            opt_mx = -1 * np.ones((self.n_actions, self.n_actions))

            for leader_act_i, _ in enumerate(list(self.margins_leader.ravel())):
                for follower0_act_i, follower0_act in enumerate(list(self.margins_followers[0, :])):
                    demand_composite = demand_mu_leader[leader_act_i] * demand_mu_followers[LEAD][0, follower0_act_i] + \
                        (1 - demand_mu_leader[leader_act_i]) * demand_mu_followers[NOLEAD][0, follower0_act_i]

                    follower_obj_ucb = (self.alpha + follower0_act) * demand_composite
                    opt_mx[leader_act_i, follower0_act_i] = obj_mu_leader[leader_act_i] + follower_obj_ucb

        if self.n_followers == 2:
            opt_mx = -1 * np.ones((self.n_actions, self.n_actions, self.n_actions))

            for leader_act_i, _ in enumerate(list(self.margins_leader.ravel())):
                for follower0_act_i, follower0_act in enumerate(list(self.margins_followers[0, :])):
                    for follower1_act_i, follower1_act in enumerate(list(self.margins_followers[1, :])):
                        demand_composite0 = demand_mu_leader[leader_act_i] * demand_mu_followers[LEAD][0, follower0_act_i] + \
                            (1 - demand_mu_leader[leader_act_i]) * demand_mu_followers[NOLEAD][0, follower0_act_i]

                        demand_composite1 = demand_mu_leader[leader_act_i] * demand_mu_followers[LEAD][1, follower1_act_i] + \
                            (1 - demand_mu_leader[leader_act_i]) * demand_mu_followers[NOLEAD][1, follower1_act_i]

                        follower0_obj_ucb = (self.alpha + follower0_act) * demand_composite0
                        follower1_obj_ucb = (self.alpha + follower1_act) * demand_composite1

                        opt_mx[leader_act_i, follower0_act_i, follower1_act_i] = obj_mu_leader[leader_act_i] + \
                            follower0_obj_ucb + follower1_obj_ucb

        return np.max(opt_mx)
