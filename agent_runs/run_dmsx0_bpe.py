#!/usr/bin/env python3
"""CLI: **DMS-X0-BPE** only — proposed Li–Scarlett BPE on full ``ALL_ACTIONS_NORM`` (not Smolyak).

Calls ``main_single_agent(PROPOSED_JOINT_ALG)``. On startup, ``main_single_agent`` prints:

* full ``X°`` validation,
* ``|X°|``, ``T``, seeds,
* Li–Scarlett batch sizes,
* within-batch duplicate-avoidance policy,
* CSV/plot note separating theorem regret (played) vs recommendation diagnostics.

Example::

    EXP_SEQ_T1000=1 EXP_SEQ_T=1000 EXP_N_SEEDS=3 EXP_OUT=results_dmsx0_bpe_T1000_s3 \\
        ./.venv_ci/bin/python agent_runs/run_dmsx0_bpe.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("EXP_OUT", "results_agent_dmsx0_bpe")
os.environ.setdefault("EXP_N_WORKERS", "1")

from pricing_realistic.config import PROPOSED_JOINT_ALG
from pricing_realistic.main_entry import main_single_agent

if __name__ == "__main__":
    main_single_agent(PROPOSED_JOINT_ALG)
