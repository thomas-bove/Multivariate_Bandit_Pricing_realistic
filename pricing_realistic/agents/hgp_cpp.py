"""HGP-UCB-CPP wrapper."""
from __future__ import annotations

from typing import Optional

import numpy as np

from hgp_ucb_cpp.catalog import CatalogPricingAgent

from ..config import GRAPH_DICT, MARGIN_VALS, N_ACTIONS, N_PRODUCTS

class HGP_UCB_CPP_Wrapper:
    """Thin adapter around CatalogPricingAgent (Mussi & Restelli 2025).

    By default (**known_graph=False**) the agent is in the paper’s *unknown-graph*
    regime: ``graph_dict=None``, demand tables are filled from sales, and the
    leader–follower structure is inferred periodically via
    ``compute_graph_create_agents`` (MILP ``complementary_products_solver``).  This
    matches HGP-UCB-CPP without an oracle graph — harder than supplying
    ``GRAPH_DICT`` at construction time, but the same algorithmic family.

    Set ``known_graph=True`` only for an oracle baseline (true graph wired in).

    Interface contract
    ------------------
    pull()             → normalized action in [0,1]^d  (like other agents)
    update(sales_mx)   → raw (n_products, n_baskets) sales matrix
    reset(seed)        → resets inner agent; NOTE: HGP-UCB-CPP uses numpy's
                         global random state internally — no seed is forwarded.

    Caveat: the vendored IGPUCB component uses np.random (global) for
    tie-breaking among untouched arms, so per-run reproducibility at the
    algorithm level is not guaranteed.  Outer-loop seeds fix the environment
    and other RNGs; variance across runs averages out over N_SEEDS.
    """

    def __init__(
        self,
        T             : int,
        alpha         : float,
        kernel_L      : float,
        known_graph   : bool = False,
        graph_reperiod : Optional[int] = None,
    ):
        self._T             = T
        self._alpha         = alpha
        self._kL            = kernel_L
        self._known_graph   = known_graph
        # Unknown-graph: avoid MILP every round on very sparse stats (still same code path).
        self._regraph_every = (
            1 if known_graph else (graph_reperiod if graph_reperiod is not None else max(8, T // 25))
        )
        self._inner = self._make_inner()
        self._last_real: np.ndarray = np.zeros(N_PRODUCTS)

    def _make_inner(self) -> CatalogPricingAgent:
        return CatalogPricingAgent(
            n_products=N_PRODUCTS,
            n_actions=N_ACTIONS,
            margins=np.tile(MARGIN_VALS, (N_PRODUCTS, 1)),
            alpha=self._alpha,
            kernel_L=self._kL,
            horizon=self._T,
            graph_dict=GRAPH_DICT if self._known_graph else None,
            recompute_graph_every=self._regraph_every,
        )

    def pull(self) -> np.ndarray:
        real = self._inner.pull()          # real margins in MARGIN_VALS
        self._last_real = real
        idx = np.array([
            int(np.where(np.isclose(MARGIN_VALS, m))[0][0]) for m in real
        ])
        return idx.astype(float) / (N_ACTIONS - 1)

    def update(self, sales_mx: np.ndarray) -> None:
        self._inner.update(sales_mx)

    def reset(self, seed: int = 0) -> None:
        self._inner = self._make_inner()
        self._last_real = np.zeros(N_PRODUCTS)

