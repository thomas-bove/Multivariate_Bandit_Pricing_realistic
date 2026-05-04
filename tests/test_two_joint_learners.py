"""
``run_one_two_joint``: paper **DMS-X0-BPE** (full ``X°`` cover + elimination) vs baseline
**Joint-DMS-GP-UCB** (continuous acquisition + snap).

A **fast** sanity check runs by default.  The long full-grid stress test
(``T=7000``) runs only with ``RUN_SLOW_BPE_TESTS=1``.

Optional stress (both learners, long ``T``)::

    EXP_FULL_TWO_JOINT=1 python -m unittest tests.test_two_joint_learners.TestTwoJointLearners.test_full_horizon_two_joint_slow -v

Reference BPE code (Li & Scarlett 2022): https://github.com/lizihan97/BPE
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

import experiment_realistic as er
from environments.demand_generator import make_default_env


class TestTwoJointLearners(unittest.TestCase):
    def test_run_one_two_joint_shape_and_algorithms(self) -> None:
        env, _ = make_default_env(cfg=er.CFG, seed=0)
        cfg = replace(
            er.EXP_CFG,
            joint_reward_mc_replications=8,
            dms_bpe_noise_var=0.04,
        )
        T = 120
        rows, diag = er.run_one_two_joint(
            T=T, env=env, seed=1, c_beta=0.5, delta=0.1, B_rkhs=1.0, exp_cfg=cfg,
        )
        self.assertEqual(len(rows), 2 * T)
        algs = {r["algorithm"] for r in rows}
        self.assertEqual(algs, {er.PROPOSED_JOINT_ALG, er.JOINT_CONT_GP_UCB_ALG})
        self.assertEqual(diag["n_arms"], len(er.ALL_ACTIONS_NORM))
        self.assertEqual(len(diag["bpe_survivors_per_round"]), T)
        self.assertEqual(len(diag["bpe_opt_still_active_per_round"]), T)
        self.assertEqual(len(diag["bpe_recommendation_simple_regret_per_round"]), T)
        self.assertIn("bpe_final_survivors", diag)
        self.assertIn("bpe_committed_idx", diag)
        self.assertIn("opt_grid_idx", diag)
        self.assertIsInstance(diag["opt_grid_idx"], int)
        r_prop = next(r for r in rows if r["algorithm"] == er.PROPOSED_JOINT_ALG and r["t"] == 1)
        r_joint = next(r for r in rows if r["algorithm"] == er.JOINT_CONT_GP_UCB_ALG and r["t"] == 1)
        self.assertTrue(np.isfinite(r_prop["bpe_opt_still_active"]))
        self.assertTrue(np.isnan(r_joint["bpe_opt_still_active"]))

    def test_bpe_x0_trace_fast_full_grid(self) -> None:
        """Cheap default check: full ``|X°|`` env + short ``T`` (no long MC loop)."""
        env, _ = make_default_env(cfg=er.CFG, seed=0)
        cfg = replace(er.EXP_CFG, joint_reward_mc_replications=2, dms_bpe_noise_var=0.08)
        T = 35
        diag = er.run_bpe_x0_only_trace(T=T, env=env, seed=3, B_rkhs=1.0, exp_cfg=cfg)
        self.assertEqual(diag["n_arms"], 5**6)
        self.assertEqual(len(diag["bpe_survivors_per_round"]), T)
        self.assertTrue(np.all(np.isfinite(diag["bpe_survivors_per_round"])))

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_BPE_TESTS", "").lower() in ("1", "true", "yes"),
        "set RUN_SLOW_BPE_TESTS=1 for full-grid T=7000 BPE elimination stress (~hours)",
    )
    def test_bpe_eliminates_on_full_grid_bpe_only(self) -> None:
        """
        Full ``X°`` (5^6); BPE only (same object as inside ``run_one_two_joint``).
        Averaged rewards + λ, B tuned so confidence bands separate arms after
        several Li–Scarlett batches.
        """
        env, _ = make_default_env(cfg=er.CFG, seed=0)
        cfg = replace(
            er.EXP_CFG,
            joint_reward_mc_replications=28,
            dms_bpe_noise_var=0.017,
            dms_elimination_delta=0.12,
            dms_kernel_L=1.0,
        )
        T = 7_000
        seed = 11
        diag = er.run_bpe_x0_only_trace(
            T=T, env=env, seed=seed, B_rkhs=0.86, exp_cfg=cfg,
        )
        n = diag["n_arms"]
        trace = diag["bpe_survivors_per_round"]
        self.assertEqual(n, 5**6)
        self.assertLess(
            min(int(x) for x in trace),
            n,
            msg=(
                f"BPE never shrank active set (min={min(trace)}, n={n}). "
                "Raise T or joint_reward_mc_replications, lower dms_bpe_noise_var / B_rkhs; "
                "see https://github.com/lizihan97/BPE"
            ),
        )

    @unittest.skipUnless(
        os.environ.get("EXP_FULL_TWO_JOINT", "").lower() in ("1", "true", "yes"),
        "set EXP_FULL_TWO_JOINT=1 for slow both-learners run (~2× BPE-only cost)",
    )
    def test_full_horizon_two_joint_slow(self) -> None:
        env, _ = make_default_env(cfg=er.CFG, seed=0)
        cfg = replace(
            er.EXP_CFG,
            joint_reward_mc_replications=35,
            dms_bpe_noise_var=0.018,
            dms_elimination_delta=0.12,
            dms_kernel_L=1.0,
        )
        T = 14_000
        rows, diag = er.run_one_two_joint(
            T=T,
            env=env,
            seed=11,
            c_beta=0.5,
            delta=0.1,
            B_rkhs=0.88,
            exp_cfg=cfg,
        )
        self.assertEqual(len(rows), 2 * T)
        trace = diag["bpe_survivors_per_round"]
        self.assertLess(min(int(x) for x in trace), diag["n_arms"])


if __name__ == "__main__":
    unittest.main()
