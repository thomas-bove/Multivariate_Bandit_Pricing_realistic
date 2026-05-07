#!/usr/bin/env python3
"""CLI: only **SPSA**.

In ``EXP_SMOKE=1`` mode, ``runs`` uses a higher ``a_par`` (visible motion on the grid);
override with ``EXP_SPSA_A_PAR=0.05`` (or other) if needed.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("EXP_OUT", "results_agent_spsa")
os.environ.setdefault("EXP_N_WORKERS", "1")

from pricing_realistic.main_entry import main_single_agent

if __name__ == "__main__":
    main_single_agent("SPSA")
