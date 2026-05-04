"""
Regression / preflight tests for ``experiment_realistic`` large-scale runs.

Run from repo root::

    python -m unittest discover -s tests -v

Optional slow checks (Pool + subprocess)::

    python -m unittest tests.test_experiment_scale.ExperimentScaleTests.test_pool_two_tasks -v
    python -m unittest tests.test_experiment_scale.ExperimentScaleTests.test_smoke_main_subprocess -v

Full-grid long BPE stress (hours): ``RUN_SLOW_BPE_TESTS=1`` — see ``tests.test_two_joint_learners``.
"""

from __future__ import annotations

import itertools

import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# Repo root on sys.path (``experiment_realistic`` also adjusts path on import)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_realistic as er  # noqa: E402


class ValidateFullCartesianX0Tests(unittest.TestCase):
    def test_passes_on_all_actions_norm(self) -> None:
        er.validate_full_cartesian_x0(er.ALL_ACTIONS_NORM, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_random_subset(self) -> None:
        sub = er.ALL_ACTIONS_NORM[: max(3, len(er.ALL_ACTIONS_NORM) // 2)]
        with self.assertRaises(ValueError):
            er.validate_full_cartesian_x0(sub, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_duplicate_rows(self) -> None:
        bad = er.ALL_ACTIONS_NORM.copy()
        bad[-1] = bad[0].copy()
        with self.assertRaises(ValueError):
            er.validate_full_cartesian_x0(bad, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            er.validate_full_cartesian_x0(er.ALL_ACTIONS_NORM[:100], er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_real_margins_not_normalized_grid(self) -> None:
        with self.assertRaises(ValueError):
            er.validate_full_cartesian_x0(er.ALL_ACTIONS_REAL, er.N_PRODUCTS, er.N_ACTIONS)


class ExperimentScaleTests(unittest.TestCase):
    """Sanity checks before EXP_SEQ_T1000 / full FAST_MODE runs."""

    def test_li_bpe_schedule_nonempty_and_covers_horizon(self) -> None:
        for T in (1, 50, 200, 1000, 10000):
            sizes = er.li_22_bpe_batch_sizes(T)
            if T <= 0:
                self.assertEqual(len(sizes), 0)
                continue
            self.assertGreater(len(sizes), 0, msg=f"T={T}")
            self.assertEqual(
                int(sizes.sum()),
                T,
                msg=f"BPE batch lengths must sum to T={T}, got {sizes.tolist()} sum={sizes.sum()}",
            )

    def test_dmsx0bpe_batched_posterior_matches_singletons(self) -> None:
        """One Cholesky GP posterior must agree with per-arm scalar path (Alg.~1 GP block)."""
        rng = np.random.default_rng(1)
        actions = rng.uniform(0.0, 1.0, size=(30, 3))
        dms = er.DMSX0BPE(
            actions=actions,
            T=200,
            noise_var=0.04,
            delta=0.1,
            kernel_L=0.7,
            B_rkhs=1.0,
            seed=2,
        )
        train_idx = [2, 7, 11, 18]
        y = np.array([0.3, -0.1, 0.8, 0.0], dtype=float)
        active = np.array([0, 1, 3, 5, 9, 12, 19, 22], dtype=int)
        mu_b, sig_b = dms._gp_mu_sigma_vectors(train_idx, y, active)
        for j, a in enumerate(active):
            mj, sj = dms._gp_mu_sigma_vectors(train_idx, y, np.array([a]))
            self.assertAlmostEqual(float(mu_b[j]), float(mj[0]), places=10)
            self.assertAlmostEqual(float(sig_b[j]), float(sj[0]), places=10)

    def test_dmsx0bpe_sqrt_beta_li_scarlett_provisional(self) -> None:
        """Legacy ``|X_active|``: √β = Ψ + √(2 log(|X_active|·M_total/δ)) with default ``noise_R=None``."""
        actions = np.array([[0.0], [1.0]], dtype=float)
        dms = er.DMSX0BPE(
            actions=actions,
            T=100,
            noise_var=0.25,
            delta=0.1,
            kernel_L=1.0,
            B_rkhs=1.0,
            seed=0,
            bpe_beta_use_active_count=True,
        )
        M = dms._M_total
        n_active = 100
        sqrt_beta = dms._sqrt_beta_elimination(n_active)
        log_m = float(np.log(max(float(n_active * M / 0.1), np.e)))
        want = 1.0 + float(np.sqrt(2.0 * log_m))
        self.assertAlmostEqual(sqrt_beta, want, places=10)

    def test_dmsx0bpe_sqrt_beta_full_cover_default(self) -> None:
        """Default: ``|X°|`` in the log-term; ``noise_R=None`` ⇒ coefficient 1 in front of √(2 log …)."""
        actions = np.array([[0.0], [1.0]], dtype=float)
        dms = er.DMSX0BPE(
            actions=actions,
            T=100,
            noise_var=0.25,
            delta=0.1,
            kernel_L=1.0,
            B_rkhs=1.0,
            seed=0,
            bpe_beta_use_active_count=False,
        )
        M = dms._M_total
        n_arms = dms._n_arms
        sqrt_beta = dms._sqrt_beta_elimination(9999)
        log_m = float(np.log(max(float(n_arms * M / 0.1), np.e)))
        want = 1.0 + float(np.sqrt(2.0 * log_m))
        self.assertAlmostEqual(sqrt_beta, want, places=10)

    def test_dmsx0bpe_sqrt_beta_noise_R_factor(self) -> None:
        """Explicit ``noise_R``: ``noise_factor = R / √λ`` scales √(2 log …)."""
        actions = np.array([[0.0], [1.0]], dtype=float)
        dms = er.DMSX0BPE(
            actions=actions,
            T=100,
            noise_var=1.0,
            delta=0.1,
            kernel_L=1.0,
            B_rkhs=1.0,
            seed=0,
            noise_R=2.0,
            bpe_beta_use_active_count=False,
        )
        log_m = float(np.log(max(float(dms._n_arms * dms._M_total / 0.1), np.e)))
        want = 1.0 + 2.0 * float(np.sqrt(2.0 * log_m))
        self.assertAlmostEqual(dms._sqrt_beta_elimination(), want, places=10)

    def test_dmsx0bpe_sqrt_beta_increases_with_n_active(self) -> None:
        """Legacy: width grows with |X_active| in the log-term (same M_total)."""
        dms = er.DMSX0BPE(
            actions=np.zeros((5, 2)),
            T=50,
            noise_var=0.04,
            delta=0.1,
            B_rkhs=1.0,
            seed=0,
            bpe_beta_use_active_count=True,
        )
        b_small = dms._sqrt_beta_elimination(10)
        b_large = dms._sqrt_beta_elimination(10_000)
        self.assertGreater(b_large, b_small)

    def test_make_env_and_run_one_small(self) -> None:
        self.assertEqual(len(er.ALL_ACTIONS_NORM), int(er.N_ACTIONS) ** int(er.N_PRODUCTS))
        env, _ = er.make_default_env(cfg=er.CFG, seed=0)
        opt = env.compute_best_action_value()
        self.assertTrue(np.isfinite(opt))
        self.assertGreater(opt, 0.0)
        self.assertEqual(int(er.CFG.mc_ep) * int(er.CFG.n_baskets), 20_000)
        T = 18
        rows = er.run_one(
            T=T, env=env, seed=7, c_beta=0.5, delta=0.1, B_rkhs=1.0,
        )
        self.assertEqual(len(rows), T * len(er.ALL_ALGS))
        algs = {r["algorithm"] for r in rows}
        self.assertEqual(algs, set(er.ALL_ALGS))
        self.assertIn(er.JOINT_CONT_GP_UCB_ALG, algs)
        self.assertIn(er.ISO_MATERN_GP_UCB_ALG, algs)
        last_t = max(r["t"] for r in rows)
        self.assertEqual(last_t, T)
        for alg in er.ALL_ALGS:
            sub = [r for r in rows if r["algorithm"] == alg and r["t"] == T]
            self.assertEqual(len(sub), 1)
            self.assertGreaterEqual(sub[0]["revenue"], 0.0)
            self.assertIn("opt_joint_oracle", sub[0])
            self.assertIn("gap_joint_minus_indep_product", sub[0])
            self.assertAlmostEqual(float(sub[0]["opt_joint_oracle"]), float(opt), places=5)
            for k in er.recommendation_csv_keys():
                self.assertIn(k, sub[0])
            for k in er.played_action_csv_keys():
                self.assertIn(k, sub[0])

    def test_normalize_joint_reward_stable_scale(self) -> None:
        r = 12.3
        z = er.normalize_joint_reward(r)
        self.assertTrue(np.isfinite(z))
        self.assertAlmostEqual(er.unnormalize_joint_reward(z), r, places=10)
        self.assertLessEqual(abs(z), abs(r) / max(er.REWARD_SCALE, 1e-12) + 1e-9)

    def test_dmsx0bpe_first_batch_pulls_unique_until_active_exhausted(self) -> None:
        """Within-batch σ-exploration does not repeat an active arm until all actives were pulled."""
        grid = np.linspace(0.0, 1.0, 2)
        actions = np.array(list(itertools.product(grid, repeat=2)), dtype=float)
        dms = er.DMSX0BPE(
            actions=actions,
            T=200,
            noise_var=0.08,
            delta=0.25,
            kernel_L=1.0,
            B_rkhs=1.0,
            seed=11,
        )
        n_active = int(dms._active_mask.sum())
        L = int(dms._cur_batch_len)
        k = min(L, n_active)
        pulled: list[int] = []
        for _ in range(k):
            dms.pull()
            pulled.append(int(dms._pending_arm_idx))
            dms.update(0.25)
            self.assertTrue(dms._active_mask[int(dms._pending_arm_idx)])
        self.assertEqual(len(pulled), len(set(pulled)))

    def test_dmsx0bpe_global_history_flag_runs(self) -> None:
        """``bpe_use_global_history=True`` keeps the old full-history σ / elimination path."""
        rng = np.random.default_rng(0)
        actions = rng.uniform(0.0, 1.0, size=(25, 2))
        dms = er.DMSX0BPE(
            actions=actions,
            T=40,
            noise_var=0.05,
            delta=0.1,
            kernel_L=0.8,
            B_rkhs=1.0,
            seed=3,
            bpe_use_global_history=True,
        )
        for _ in range(18):
            dms.pull()
            dms.update(0.1)
        self.assertGreater(len(dms._all_train_idx), 0)

    def test_dmsx0bpe_recommend_idx_is_active(self) -> None:
        rng = np.random.default_rng(0)
        actions = rng.uniform(0.0, 1.0, size=(40, 2))
        dms = er.DMSX0BPE(
            actions=actions,
            T=80,
            noise_var=0.05,
            delta=0.1,
            kernel_L=0.8,
            B_rkhs=1.0,
            seed=3,
        )
        idx0 = dms.recommend_idx()
        self.assertTrue(dms._active_mask[idx0])
        dms.update(0.4)
        dms.update(0.2)
        idx1 = dms.recommend_idx()
        self.assertTrue(dms._active_mask[idx1])

    def test_dmsx0bpe_recommend_idx_preserves_rng(self) -> None:
        """Diagnostics must not advance ``_rng`` or change future ``pull()`` trajectories."""
        rng = np.random.default_rng(0)
        actions = rng.uniform(0.0, 1.0, size=(40, 2))
        dms = er.DMSX0BPE(
            actions=actions,
            T=80,
            noise_var=0.05,
            delta=0.1,
            kernel_L=0.8,
            B_rkhs=1.0,
            seed=42,
        )
        st0 = dms._rng.bit_generator.state
        dms.recommend_idx()
        self.assertEqual(dms._rng.bit_generator.state, st0)

        for _ in range(18):
            dms.pull()
            dms.update(0.35)
        st_before = dms._rng.bit_generator.state
        for _ in range(30):
            dms.recommend_idx()
        self.assertEqual(dms._rng.bit_generator.state, st_before)

        twin = er.DMSX0BPE(
            actions=actions,
            T=80,
            noise_var=0.05,
            delta=0.1,
            kernel_L=0.8,
            B_rkhs=1.0,
            seed=42,
        )
        for _ in range(18):
            twin.pull()
            twin.update(0.35)
        x_a = dms.pull()
        x_b = twin.pull()
        np.testing.assert_array_equal(x_a, x_b)

    def test_dmsx0bpe_mu_post_none_by_default_after_batches(self) -> None:
        rng = np.random.default_rng(1)
        actions = rng.uniform(0.0, 1.0, size=(28, 2))
        dms = er.DMSX0BPE(
            actions=actions,
            T=60,
            noise_var=0.05,
            delta=0.1,
            kernel_L=0.8,
            B_rkhs=1.0,
            seed=5,
            compute_full_posterior_for_debug=False,
        )
        for _ in range(24):
            dms.pull()
            dms.update(0.2)
            self.assertIsNone(dms._mu_post)

    def test_dmsx0bpe_debug_full_posterior_sets_mu_post(self) -> None:
        rng = np.random.default_rng(2)
        actions = rng.uniform(0.0, 1.0, size=(22, 2))
        dms = er.DMSX0BPE(
            actions=actions,
            T=50,
            noise_var=0.05,
            delta=0.1,
            kernel_L=0.8,
            B_rkhs=1.0,
            seed=6,
            compute_full_posterior_for_debug=True,
        )
        self.assertIsNone(dms._mu_post)
        n_first = int(dms._cur_batch_len)
        for _ in range(n_first):
            dms.pull()
            dms.update(0.15)
        self.assertIsNotNone(dms._mu_post)
        self.assertEqual(len(dms._mu_post), dms._n_arms)

    def test_joint_cont_gp_ucb_actions_in_unit_hypercube(self) -> None:
        """Joint-DMS-GP-UCB: normalised actions in [0,1]^d (catalogue snap on policy)."""
        env, _ = er.make_default_env(cfg=er.CFG, seed=0)
        rows = er.run_one(T=5, env=env, seed=0, c_beta=0.5, delta=0.1, B_rkhs=1.0)
        sub = [r for r in rows if r["algorithm"] == er.JOINT_CONT_GP_UCB_ALG]
        self.assertEqual(len(sub), 5)
        for r in sub:
            for i in range(er.N_PRODUCTS):
                a = float(r[f"a{i}"])
                self.assertGreaterEqual(a, 0.0)
                self.assertLessEqual(a, 1.0)

    def test_best_independent_product_surrogate_finite(self) -> None:
        env, _ = er.make_default_env(cfg=er.CFG, seed=0)
        v = float(er.best_independent_product_surrogate(env))
        self.assertTrue(np.isfinite(v))

    def test_dmsgpucb_li_schedule_requires_use_batches(self) -> None:
        with self.assertRaises(ValueError):
            er.DMSGPUCB(
                2,
                0.1,
                use_batches=False,
                li_scarlett_batch_sizes=np.array([2, 3], dtype=int),
            )

    def test_run_task_reproducible_with_global_rng_reset(self) -> None:
        """HGP touches ``np.random``; large-scale Pool workers need identical seeds per task."""
        arg = (12, 2, 0.5, 0.1, 1.0)
        np.random.seed(99991)
        random.seed(99991)
        a = er._run_task(arg)
        np.random.seed(99991)
        random.seed(99991)
        b = er._run_task(arg)
        self.assertEqual(a, b)

    def test_pool_two_tasks(self) -> None:
        """Multiprocessing pickle path used by full ``main()``."""
        from multiprocessing import Pool

        tasks = [(15, 0, 0.5, 0.1, 1.0), (15, 1, 0.5, 0.1, 1.0)]
        with Pool(2) as pool:
            out = pool.map(er._run_task, tasks)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 15 * len(er.ALL_ALGS))

    def test_smoke_main_subprocess(self) -> None:
        """End-to-end ``main()`` smoke (no PDFs); guards large-scale entrypoint."""
        env = os.environ.copy()
        env["EXP_SMOKE"] = "1"
        env["EXP_SMOKE_T"] = "40"
        env["EXP_SMOKE_OUT"] = str(ROOT / "results_realistic_test_run")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "experiment_realistic.py")],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}",
        )
        self.assertIn("All tasks done", proc.stdout)
        csv_path = ROOT / "results_realistic_test_run" / "results.csv"
        self.assertTrue(csv_path.is_file(), msg=f"missing {csv_path}")


if __name__ == "__main__":
    unittest.main()
