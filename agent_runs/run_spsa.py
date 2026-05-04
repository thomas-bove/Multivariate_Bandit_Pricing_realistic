#!/usr/bin/env python3
"""CLI: solo **SPSA**.

In modalità ``EXP_SMOKE=1``, ``runs`` usa un ``a_par`` più alto (movimento visibile sul grid);
sovrascrivi con ``EXP_SPSA_A_PAR=0.05`` (o altro) se serve.
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
