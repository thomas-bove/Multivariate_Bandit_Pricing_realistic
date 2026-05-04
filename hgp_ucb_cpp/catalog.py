import numpy as np
from hgp_ucb_cpp.independent import OptIndepPricingAgent
from hgp_ucb_cpp.complementaryset import OptComplementarySetPricingAgent
from hgp_ucb_cpp.integer_optimizers import complementary_products_solver

LEAD = "lead"
NOLEAD = "no_lead"


class CatalogPricingAgent:
    def __init__(self, n_products, n_actions, margins, alpha, kernel_L,
                 horizon, graph_dict=None, recompute_graph_every=1):
        if not (isinstance(margins, np.ndarray) and margins.shape == (n_products, n_actions)):
            raise ValueError(f"Shape of margins ({margins.shape}) not coherent. Expected ({n_products}, {n_actions}).")
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], but got {alpha}")

        self.margins = margins
        self.n_products = n_products
        self.n_actions = n_actions
        self.kernel_L = kernel_L
        self.horizon = horizon
        self.alpha = alpha
        self.recompute_graph_every = recompute_graph_every
        self.iteration = 0

        self.margins_to_idx_lst = []
        for prod in range(self.n_products):
            self.margins_to_idx_lst.append({self.margins[prod, idx]: idx for idx in range(self.n_actions)})

        if graph_dict is not None:
            self.known_graph = True
            self.graph_dict = graph_dict

            self.agents_dict = {}
            for leader in self.graph_dict.keys():
                ld_margins = self.margins[leader, :].copy()
                if len(self.graph_dict[leader]) == 0:
                    self.agents_dict[leader] = OptIndepPricingAgent(
                        actions=ld_margins.reshape(self.n_actions, 1),
                        kernel_L=self.kernel_L,
                        horizon=self.horizon,
                        alpha=self.alpha
                    )
                else:
                    followers_lst = self.graph_dict[leader]
                    fl_margins = self.margins[followers_lst, :].copy()
                    self.agents_dict[leader] = OptComplementarySetPricingAgent(
                        self.n_actions, len(followers_lst), ld_margins, fl_margins,
                        self.kernel_L, self.horizon, self.alpha
                    )

        else:
            self.known_graph = False

            self.demands_indep = np.zeros((self.n_products, self.n_actions))
            self.demands_withleader = np.zeros((self.n_products, self.n_products, self.n_actions))
            self.demands_noleader = np.zeros((self.n_products, self.n_products, self.n_actions))

            self.demands_indep_weights = self.demands_indep.copy()
            self.demands_withleader_weights = self.demands_withleader.copy()
            self.demands_noleader_weights = self.demands_noleader.copy()

    def pull(self):
        if not self.known_graph and self.iteration % self.recompute_graph_every == 0:
            self.compute_graph_create_agents()

        self.iteration += 1

        self.last_action = -1 * np.ones((self.n_products))

        for leader in self.graph_dict.keys():
            if len(self.graph_dict[leader]) == 0:
                self.last_action[leader] = self.agents_dict[leader].pull()
            else:
                followers_lst = self.graph_dict[leader]
                actions = self.agents_dict[leader].pull()
                self.last_action[leader] = actions[0]
                for i in range(len(followers_lst)):
                    self.last_action[followers_lst[i]] = actions[i + 1]

        return self.last_action

    def update(self, sales):
        for leader in self.graph_dict.keys():
            if len(self.graph_dict[leader]) == 0:
                leader_sales_vect = sales[leader, :]
                self.agents_dict[leader].update(np.sum(leader_sales_vect), leader_sales_vect.shape[0])
            else:
                followers_lst = self.graph_dict[leader]

                leader_sales_vect = sales[leader, :].ravel()
                follower_sales_mx = sales[followers_lst, :]

                self.agents_dict[leader].update(leader_sales_vect, follower_sales_mx)

        if not self.known_graph:
            self.update_demand_tables(sales, self.last_action)

    def update_demand_tables(self, sales, action_vect):
        if not action_vect.shape == (self.n_products,):
            raise ValueError(f"action_vect shape mismatch. Expected ({self.n_products},), got {action_vect.shape}.")

        if not (sales.ndim == 2 and sales.shape[0] == self.n_products):
            raise ValueError(f"Sales matrix shape mismatch. Expected ndim=2 and shape[0]={self.n_products}, got ndim={sales.ndim} and shape={sales.shape}.")

        n_samples = sales.shape[1]

        action_idxs = -1 * np.ones((self.n_products), dtype=int)
        for prod_i in range(self.n_products):
            action_idxs[prod_i] = self.margins_to_idx_lst[prod_i][action_vect[prod_i]]

        demands_current_iter = np.sum(sales, axis=1) / n_samples

        if not demands_current_iter.shape == (self.n_products,):
            raise ValueError(f"Internal error: demands_current_iter shape mismatch. Expected ({self.n_products},), got {demands_current_iter.shape}")

        sales_bool = (sales == 1)

        for leader in range(self.n_products):
            leader_act_idx = action_idxs[leader]

            self.demands_indep[leader, leader_act_idx] = (
                (self.demands_indep[leader, leader_act_idx] * self.demands_indep_weights[leader, leader_act_idx]) +
                (demands_current_iter[leader] * n_samples)
            ) / (self.demands_indep_weights[leader, leader_act_idx] + n_samples)
            self.demands_indep_weights[leader, leader_act_idx] += n_samples

            mask_leader = sales_bool[leader, :]

            n_samples_withleader = np.sum(mask_leader)
            n_samples_noleader = n_samples - n_samples_withleader

            for follower in range(self.n_products):
                if follower == leader:
                    continue

                follower_act_idx = action_idxs[follower]

                if n_samples_withleader > 0:
                    self.demands_withleader[leader, follower, follower_act_idx] = (
                        (self.demands_withleader[leader, follower, follower_act_idx] * self.demands_withleader_weights[leader, follower, follower_act_idx]) +
                        np.sum(sales[follower, mask_leader])
                    ) / (self.demands_withleader_weights[leader, follower, follower_act_idx] + n_samples_withleader)

                    self.demands_withleader_weights[leader, follower, follower_act_idx] += n_samples_withleader

                if n_samples_noleader > 0:
                    self.demands_noleader[leader, follower, follower_act_idx] = (
                        (self.demands_noleader[leader, follower, follower_act_idx] * self.demands_noleader_weights[leader, follower, follower_act_idx]) +
                        np.sum(sales[follower, np.logical_not(mask_leader)])
                    ) / (self.demands_noleader_weights[leader, follower, follower_act_idx] + n_samples_noleader)

                    self.demands_noleader_weights[leader, follower, follower_act_idx] += n_samples_noleader

    def create_leader(self, leader):
        ag = OptIndepPricingAgent(actions=self.margins[leader, :].copy().reshape(self.n_actions, 1),
                                  kernel_L=self.kernel_L, horizon=self.horizon, alpha=self.alpha)

        for act_i in range(self.n_actions):
            if self.demands_indep_weights[leader, act_i] > 0:
                ag.update_complete(ag.actions[act_i],
                                   self.demands_indep[leader, act_i] * self.demands_indep_weights[leader, act_i],
                                   self.demands_indep_weights[leader, act_i])

        return ag

    def create_complementaryset(self, leader, followers_lst):
        if not len(followers_lst) > 0:
            raise ValueError("followers not provided. followers_lst cannot be empty when creating a complementary set.")

        ag = OptComplementarySetPricingAgent(self.n_actions, len(followers_lst),
            self.margins[leader, :].copy().reshape(self.n_actions, ),
            self.margins[followers_lst, :].copy(),
            self.kernel_L, self.horizon, self.alpha)

        for act_i in range(self.n_actions):
            if self.demands_indep_weights[leader, act_i] > 0:
                ag.leader_bandit_agent.update_complete(
                    ag.margins_leader[act_i],
                    self.demands_indep[leader, act_i],
                    sample_weight=self.demands_indep_weights[leader, act_i])

            for fl_i, fl in enumerate(followers_lst):
                if self.demands_withleader_weights[leader, fl, act_i] > 0:
                    ag.followers_bandit_agent[fl_i][LEAD].update_complete(
                        ag.margins_followers[fl_i, act_i],
                        self.demands_withleader[leader, fl, act_i],
                        sample_weight=self.demands_withleader_weights[leader, fl, act_i])

                if self.demands_noleader_weights[leader, fl, act_i] > 0:
                    ag.followers_bandit_agent[fl_i][NOLEAD].update_complete(
                        ag.margins_followers[fl_i, act_i],
                        self.demands_noleader[leader, fl, act_i],
                        sample_weight=self.demands_noleader_weights[leader, fl, act_i])

        return ag

    def compute_graph_create_agents(self):
        if self.iteration == 0:
            self.graph_dict = {i: [] for i in range(self.n_products)}
            self.agents_dict = {}

            for leader in self.graph_dict.keys():
                self.agents_dict[leader] = OptIndepPricingAgent(
                    actions=self.margins[leader, :].copy().reshape(self.n_actions, 1),
                    kernel_L=self.kernel_L, horizon=self.horizon, alpha=self.alpha)

        else:
            value_mx = np.zeros((self.n_products, self.n_products))

            for leader in range(self.n_products):
                for follower in range(self.n_products):
                    if leader != follower:
                        aux_agent = self.create_complementaryset(leader, [follower])
                        _, value = aux_agent.pull(return_val=True)
                        value_mx[leader, follower] = value
                    else:
                        aux_agent = self.create_leader(leader)
                        _, value = aux_agent.pull(return_val=True)
                        value_mx[leader, leader] = value

            for leader in range(self.n_products):
                for follower in range(self.n_products):
                    if follower != leader:
                        value_mx[leader, follower] = value_mx[leader, follower] - value_mx[leader, leader]

            X_mx, _ = complementary_products_solver(value_mx)

            for lead in range(self.n_products):
                followers_count = np.sum(X_mx[lead, :]) - 1
                for fl in list(np.linspace(self.n_products - 1, 0, self.n_products, dtype=int)):
                    if followers_count > 2:
                        if X_mx[lead, fl] == 1 and fl != lead:
                            X_mx[lead, fl] = 0
                            X_mx[fl, fl] = 1
                            followers_count -= 1

            self.graph_dict = {}
            self.agents_dict = {}

            for leader in range(self.n_products):
                if X_mx[leader, leader] == 1:
                    followers_lst = list(np.where(X_mx[leader, :] == 1)[0])
                    followers_lst.remove(leader)

                    self.graph_dict[leader] = followers_lst

                    if len(followers_lst) == 0:
                        self.agents_dict[leader] = self.create_leader(leader)
                    else:
                        self.agents_dict[leader] = self.create_complementaryset(leader, followers_lst)
