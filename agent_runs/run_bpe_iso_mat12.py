#!/usr/bin/env python3
"""CLI: **BPE-Iso-Mat12** — BPE on full ``ALL_ACTIONS_NORM`` with isotropic Matérn-1/2 kernel (ablation).

Example::

    EXP_SEQ_T1000=1 EXP_N_SEEDS=3 EXP_OUT=results_bpe_iso \\
        ./.venv_ci/bin/python agent_runs/run_bpe_iso_mat12.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("EXP_OUT", "results_agent_bpe_iso_mat12")
os.environ.setdefault("EXP_N_WORKERS", "1")

from pricing_realistic.config import ISO_MATERN_BPE_ALG
from pricing_realistic.main_entry import main_single_agent

if __name__ == "__main__":
    main_single_agent(ISO_MATERN_BPE_ALG)
