import numpy as np


class ComplementaryPricingEnvironment:
    """
    Simulates a retail environment with complementary product pricing.
    This environment models customer purchasing behavior where "leader" products
    (e.g., game consoles) can increase the demand for "follower" products
    (e.g., games). The relationships are defined by a graph.
    The environment is used to simulate sales based on a given pricing strategy
    (a set of margins) and to compute the expected total value (a mix of
    revenue and profit) for different pricing combinations.
    """


    def __init__(
        self,
        n_products,
        n_actions,
        margins,
        demands,
        n_baskets,
        alpha,
        graph_dict,
        mc_ep=1000,
        seed=0,
        dependency_strength: float = 1.0,
        exact_oracle: bool = True,
    ):
        """
        Initializes the pricing environment.

        Args:
            n_products (int): The total number of products.
            n_actions (int): The number of available pricing actions for each product.
            margins (np.ndarray): 2D array of shape (n_products, n_actions) of
                                  possible margin values.
            demands (np.ndarray): 3D array of shape (n_products, n_actions, 2).
                                  demands[i, j, 0] is the base demand for product i
                                  at margin j.
                                  demands[i, j, 1] is the enhanced demand if its
                                  leader is purchased.
            n_baskets (int): Default number of customer baskets for `step()`.
            alpha (float): Balances profit (0) vs. revenue (1). Must be in [0, 1].
            graph_dict (dict): Defines complementary relationships.
                               Keys: leader product indices (int).
                               Values: lists of follower indices (list[int]).
            mc_ep (int, optional): Monte Carlo episodes for `compute_values()` when
                                   ``exact_oracle=False``. Ignored when ``exact_oracle=True``.
                                   Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            dependency_strength (float, optional): Non-negative. When the leader is sold,
                follower purchase probability is
                ``clip(d_base + s * (d_enh - d_base), 0, 1)``. Values s>1 amplify the
                leader–follower lift before clipping (strong coupling); s=1 is the
                baseline “full enhanced” blend.
            exact_oracle (bool, optional): If True (default), precompute ``action_values``
                with **closed-form expectations** (deterministic oracle for regret /
                ``compute_givenaction_value``). If False, use Monte Carlo ``compute_values()``
                as before. ``step()`` is always stochastic regardless.
        """
        if float(dependency_strength) < 0.0:
            raise ValueError(f"dependency_strength must be >= 0, got {dependency_strength}")

        if not demands.shape == (n_products, n_actions, 2):
            raise ValueError(f"Shape of the demand not coherent. Expected {(n_products, n_actions, 2)}, got {demands.shape}")
        if not margins.shape == (n_products, n_actions):
            raise ValueError(f"Shape of the margins not coherent. Expected {(n_products, n_actions)}, got {margins.shape}")
        if not ((demands <= 1).all() and (demands >= 0).all()):
            raise ValueError("Error in demand values: all demands must be probabilities in [0, 1]")
        if not (alpha >= 0 and alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not n_baskets >= 1:
            raise ValueError(f"n_baskets must be positive, got {n_baskets}")
        self.exact_oracle = bool(exact_oracle)
        if not self.exact_oracle and not mc_ep >= 10:
            raise ValueError(f"mc_ep too low for reliable estimates (e.g., >= 10), got {mc_ep}")

        self.n_products = n_products
        self.n_actions = n_actions
        self.n_baskets = n_baskets
        self.demands = demands
        self.margins = margins
        self.alpha = alpha # 1 revenue, 0 profit
        self.graph_dict = graph_dict # elements for 0 to num_products-1
        self.mc_ep = mc_ep
        self.margins_to_idx_lst = []
        
        for prod in range(n_products):
            self.margins_to_idx_lst.append({self.margins[prod, idx]: idx for idx in range(0, self.n_actions)})

        self.dependency_strength = float(dependency_strength)

        self.leaders_lst = list(self.graph_dict.keys())
        self.followers_lst = list(self.graph_dict.values())
        aux = self.followers_lst.copy()
        self.followers_lst = np.array([x for sublist in self.followers_lst for x in sublist])
        
        aux.append(self.leaders_lst)
        aux = np.array([x for sublist in aux for x in sublist])
        if not (np.issubdtype(aux.dtype, np.integer) and
                np.all(aux >= 0) and
                np.all(aux <= self.n_products - 1) and
                len(aux) == len(list(set(aux)))):
            raise ValueError("Error in graph_dict: All product indices must be unique integers between 0 and n_products - 1")
        
        self.follower_to_leader_dict = {value: key for key, values in self.graph_dict.items() for value in values}

        for leader in self.leaders_lst:
            followers = self.graph_dict[leader]
            if not isinstance(followers, list):
                raise TypeError(f"All graph_dict values must be of type list, but leader {leader} has type {type(followers)}")
            if not len(followers) <= 2:
                raise NotImplementedError(f"Leaders with >2 followers not implemented yet in this env (leader {leader} has {len(followers)})")

        self.reset(seed)

        if self.exact_oracle:
            self.compute_values_exact()
        else:
            self.compute_values()
            self.reset(seed)

    def _margin_idx(self, product: int, margin: float) -> int:
        """Discrete margin index; tolerant to tiny float drift vs ``self.margins``."""
        row = self.margins[int(product), :]
        idx = int(np.argmin(np.abs(row - float(margin))))
        err = float(abs(row[idx] - float(margin)))
        tol = 1e-9 + 1e-12 * max(1.0, abs(float(margin)))
        if err > tol:
            raise ValueError(
                f"margin {margin!r} for product {product} not on env grid "
                f"(closest {row[idx]!r}, |Δ|={err:.3e})"
            )
        return idx

    def compute_givenaction_value_exact(self, margins: np.ndarray) -> float:
        """
        Exact expected objective (same definition MC estimates in ``compute_values``).

        Leader contribution: ``(alpha + m_L) * D_L``.
        Follower: unconditional sale probability
        ``(1 - D_L) * D_base + D_L * clip(D_base + s*(D_enh - D_base), 0, 1)`` per basket.
        """
        if not margins.ndim == 1:
            raise ValueError(f"The action (margins) must be 1-dimensional, but got {margins.ndim} dimensions")
        if not margins.shape[0] == self.n_products:
            raise ValueError(
                f"The action (margins) must be of dimension n_products ({self.n_products}), but got {margins.shape[0]}"
            )
        s = float(self.dependency_strength)
        total = 0.0
        for leader in self.leaders_lst:
            followers = self.graph_dict[leader]
            li = self._margin_idx(leader, margins[leader])
            ml = float(margins[leader])
            D_L = float(self.demands[leader, li, 0])

            if len(followers) == 0:
                total += (self.alpha + ml) * D_L
            elif len(followers) == 1:
                f = followers[0]
                fi = self._margin_idx(f, margins[f])
                mf = float(margins[f])
                base = float(self.demands[f, fi, 0])
                enh = float(self.demands[f, fi, 1])
                p_eff = float(np.clip(base + s * (enh - base), 0.0, 1.0))
                p_F = (1.0 - D_L) * base + D_L * p_eff
                total += (self.alpha + ml) * D_L + (self.alpha + mf) * p_F
            elif len(followers) == 2:
                f0, f1 = followers[0], followers[1]
                i0 = self._margin_idx(f0, margins[f0])
                i1 = self._margin_idx(f1, margins[f1])
                m0 = float(margins[f0])
                m1 = float(margins[f1])
                b0 = float(self.demands[f0, i0, 0])
                e0 = float(self.demands[f0, i0, 1])
                b1 = float(self.demands[f1, i1, 0])
                e1 = float(self.demands[f1, i1, 1])
                pe0 = float(np.clip(b0 + s * (e0 - b0), 0.0, 1.0))
                pe1 = float(np.clip(b1 + s * (e1 - b1), 0.0, 1.0))
                p0 = (1.0 - D_L) * b0 + D_L * pe0
                p1 = (1.0 - D_L) * b1 + D_L * pe1
                total += (self.alpha + ml) * D_L + (self.alpha + m0) * p0 + (self.alpha + m1) * p1
        return float(total)

    def compute_values_exact(self) -> None:
        """Populate ``action_values`` with deterministic expectations (same shapes as ``compute_values``)."""
        self.action_values = {}
        s = float(self.dependency_strength)
        for leader in self.leaders_lst:
            followers = self.graph_dict[leader]

            if len(followers) == 0:
                vals = np.zeros((self.n_actions,))
                for leader_margin_i in range(self.n_actions):
                    leader_margin = float(self.margins[leader, leader_margin_i])
                    D_L = float(self.demands[leader, leader_margin_i, 0])
                    vals[leader_margin_i] = (self.alpha + leader_margin) * D_L
                self.action_values[leader] = vals

            elif len(followers) == 1:
                vals = np.zeros((self.n_actions, self.n_actions))
                f = followers[0]
                for leader_margin_i in range(self.n_actions):
                    leader_margin = float(self.margins[leader, leader_margin_i])
                    D_L = float(self.demands[leader, leader_margin_i, 0])
                    for follower_margin_i in range(self.n_actions):
                        follower_margin = float(self.margins[f, follower_margin_i])
                        base = float(self.demands[f, follower_margin_i, 0])
                        enh = float(self.demands[f, follower_margin_i, 1])
                        p_eff = float(np.clip(base + s * (enh - base), 0.0, 1.0))
                        p_F = (1.0 - D_L) * base + D_L * p_eff
                        vals[leader_margin_i, follower_margin_i] = (
                            (self.alpha + leader_margin) * D_L
                            + (self.alpha + follower_margin) * p_F
                        )
                self.action_values[leader] = vals

            elif len(followers) == 2:
                vals = np.zeros((self.n_actions, self.n_actions, self.n_actions))
                f0, f1 = followers[0], followers[1]
                for leader_margin_i in range(self.n_actions):
                    leader_margin = float(self.margins[leader, leader_margin_i])
                    D_L = float(self.demands[leader, leader_margin_i, 0])
                    for follower0_margin_i in range(self.n_actions):
                        follower0_margin = float(self.margins[f0, follower0_margin_i])
                        b0 = float(self.demands[f0, follower0_margin_i, 0])
                        e0 = float(self.demands[f0, follower0_margin_i, 1])
                        pe0 = float(np.clip(b0 + s * (e0 - b0), 0.0, 1.0))
                        p0 = (1.0 - D_L) * b0 + D_L * pe0
                        for follower1_margin_i in range(self.n_actions):
                            follower1_margin = float(self.margins[f1, follower1_margin_i])
                            b1 = float(self.demands[f1, follower1_margin_i, 0])
                            e1 = float(self.demands[f1, follower1_margin_i, 1])
                            pe1 = float(np.clip(b1 + s * (e1 - b1), 0.0, 1.0))
                            p1 = (1.0 - D_L) * b1 + D_L * pe1
                            vals[leader_margin_i, follower0_margin_i, follower1_margin_i] = (
                                (self.alpha + leader_margin) * D_L
                                + (self.alpha + follower0_margin) * p0
                                + (self.alpha + follower1_margin) * p1
                            )
                self.action_values[leader] = vals

    def step(self, margins, override_n_baskets=None):
        """
        Simulates sales for a given set of margins.

        Generates sales results for a number of customer baskets based on the
        chosen margins for all products. Follower product demand is enhanced
        if its corresponding leader is sold in the same basket.

        Args:
            margins (np.ndarray): 1D array of shape (n_products,).
                                  Specifies the *margin value* (not index)
                                  for each product.
            override_n_baskets (int, optional): If provided, simulates this
                                                number of baskets instead of
                                                `self.n_baskets`.

        Returns:
            np.ndarray: 2D int array of shape (n_products, n_bsk).
                        `sales_mx[i, j] = 1` if product `i` was sold in
                        basket `j`, and 0 otherwise.
        """
        
        if not margins.ndim == 1:
            raise ValueError(f"The action (margins) must be 1-dimensional, but got {margins.ndim} dimensions")
        if not margins.shape[0] == self.n_products:
            raise ValueError(f"The action (margins) must be of dimension n_products ({self.n_products}), but got {margins.shape[0]}")
        
        if override_n_baskets is not None:
            n_bsk = override_n_baskets
        else:
            n_bsk = self.n_baskets
        
        sales_mx = np.ones((self.n_products, n_bsk), dtype=int)
        
        for leader in self.leaders_lst:

            li = self._margin_idx(leader, margins[leader])
            demand = self.demands[leader, li, 0]
            sales_mx[leader, :] = np.random.uniform(0, 1, (n_bsk)) < demand
        
        for follower in self.followers_lst:
            
            corr_leader = self.follower_to_leader_dict[follower]
            
            fi = self._margin_idx(follower, margins[follower])
            demand = self.demands[follower, fi, 0]
            enhancement_demand = self.demands[follower, fi, 1]
            
            mask_leader_sales = sales_mx[corr_leader, :] == 1

            sales_demand = np.random.uniform(0, 1, (n_bsk - np.sum(mask_leader_sales))) < demand
            p_eff = np.clip(
                demand
                + self.dependency_strength * (enhancement_demand - demand),
                0.0,
                1.0,
            )
            sales_enhancement_demand = np.random.uniform(0, 1, (np.sum(mask_leader_sales))) < p_eff

            sales_mx[follower, ~mask_leader_sales] = sales_demand
            sales_mx[follower, mask_leader_sales] = sales_enhancement_demand

        return sales_mx


    def compute_values(self):
        """
        Pre-computes expected values for all margin combinations.

        Iterates through each leader and its followers, simulating all possible
        margin combinations for that subgraph using Monte Carlo.
        The total number of samples for each combination is
        `self.mc_ep * self.n_baskets`.

        This method populates the `self.action_values` dictionary.
        The keys are leader indices, and the values are N-dimensional arrays
        (where N = 1 + num_followers) containing the expected objective value
        for each margin combination.
        """
        
        self.action_values = {}
        n_samples = int(self.mc_ep * self.n_baskets)
        
        for leader in self.leaders_lst:
            
            followers = self.graph_dict[leader]
            
            if len(followers) == 0:
                vals = -1 * np.ones((self.n_actions, ))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    margins_action = self.margins[:, 0].copy()
                    margins_action[leader] = leader_margin
                    sales_mx = self.step(margins_action, override_n_baskets=n_samples)
                    empirical_demand = np.sum(sales_mx[leader, :]) / n_samples
                    vals[leader_margin_i] = (self.alpha + leader_margin) * empirical_demand

            if len(followers) == 1:
                vals = -1 * np.ones((self.n_actions, self.n_actions))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    for follower_margin_i, follower_margin in enumerate(self.margins[followers[0], :]):
                        margins_action = self.margins[:, 0].copy() 
                        margins_action[leader] = leader_margin
                        for follower in followers:
                            margins_action[follower] = follower_margin
                        sales_mx = self.step(margins_action, override_n_baskets=n_samples)
                        mc_sales = np.sum(sales_mx, axis=1) / n_samples
                        obj_fun = (self.alpha + leader_margin) * mc_sales[leader]
                        for follower in followers:
                            obj_fun += (self.alpha + follower_margin) * mc_sales[follower]
                        vals[leader_margin_i, follower_margin_i] = obj_fun
            
            if len(followers) == 2:
                vals = -1 * np.ones((self.n_actions, self.n_actions, self.n_actions))
                for leader_margin_i, leader_margin in enumerate(self.margins[leader, :]):
                    for follower0_margin_i, follower0_margin in enumerate(self.margins[followers[0], :]):
                        for follower1_margin_i, follower1_margin in enumerate(self.margins[followers[1], :]):
                            margins_action = self.margins[:, 0].copy() 
                            margins_action[leader] = leader_margin
                            margins_action[followers[0]] = follower0_margin
                            margins_action[followers[1]] = follower1_margin
                            sales_mx = self.step(margins_action, override_n_baskets=n_samples)
                            mc_sales = np.sum(sales_mx, axis=1) / n_samples
                            obj_fun = (self.alpha + leader_margin) * mc_sales[leader]
                            obj_fun += (self.alpha + follower0_margin) * mc_sales[followers[0]]
                            obj_fun += (self.alpha + follower1_margin) * mc_sales[followers[1]]  
                            vals[leader_margin_i, follower0_margin_i, follower1_margin_i] = obj_fun
            
            self.action_values[leader] = vals
    
    
    def compute_givenaction_value(self, margins):
        """
        Calculates the total expected value for a specific action.

        Uses the pre-computed `self.action_values` to look up the expected
        value for the given margin combination for each leader-follower
        subgraph and sums them up.

        Args:
            margins (np.ndarray): 1D array of shape (n_products,).
                                  Specifies the *margin value* for each product.

        Returns:
            float: The total expected value for the given set of margins.
        """
        
        if not margins.ndim == 1:
            raise ValueError(f"The action (margins) must be 1-dimensional, but got {margins.ndim} dimensions")
        if not margins.shape[0] == self.n_products:
            raise ValueError(f"The action (margins) must be of dimension n_products ({self.n_products}), but got {margins.shape[0]}")
        
        value = 0
        
        for leader in self.leaders_lst:
            
            followers = self.graph_dict[leader]
            li = self._margin_idx(leader, margins[leader])
            if len(followers) == 0:
                value += self.action_values[leader][li]
            elif len(followers) == 1:
                fi0 = self._margin_idx(followers[0], margins[followers[0]])
                value += self.action_values[leader][li, fi0]
            elif len(followers) == 2:
                fi0 = self._margin_idx(followers[0], margins[followers[0]])
                fi1 = self._margin_idx(followers[1], margins[followers[1]])
                value += self.action_values[leader][li, fi0, fi1]
        
        return value


    def compute_best_action_value(self):
        """
        Calculates the maximum possible expected value achievable.

        Finds the maximum value within each leader's `action_values` matrix
        (representing the optimal policy for that subgraph) and sums these
        maximums.

        Returns:
            float: The theoretical maximum expected value.
        """

        value = 0

        for leader in self.leaders_lst:
            value += np.max(self.action_values[leader])

        return value


    def reset(self, seed=0):
        """
        Resets the environment's random number generator.

        Args:
            seed (int, optional): The seed for np.random.seed(). Defaults to 0.
        """
        
        np.random.seed(seed)
