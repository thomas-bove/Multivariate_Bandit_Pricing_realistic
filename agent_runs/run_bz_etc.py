#!/usr/bin/env python3
"""CLI: only **BZ-ETC**."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("EXP_OUT", "results_agent_bz_etc")
os.environ.setdefault("EXP_N_WORKERS", "1")

from pricing_realistic.main_entry import main_single_agent

if __name__ == "__main__":
    main_single_agent("BZ-ETC")
