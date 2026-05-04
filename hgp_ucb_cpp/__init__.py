"""HGP-UCB-CPP: Heteroscedastic GP-UCB for Complementary Product Pricing.

Implementation of Mussi & Restelli (2025, arXiv:2511.22291).
In known-graph mode the IP-based graph identification phase is skipped;
the complementary structure is passed directly via graph_dict.
"""
from hgp_ucb_cpp.catalog import CatalogPricingAgent
