"""Exact oracle, final-commit rows, plot smoke tests (fast defaults)."""
from __future__ import annotations

import itertools
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_realistic as er
from environments.demand_generator import build_env, paired_config
from pricing_realistic.agents.dmsx0_bpe import DMSX0BPE
from pricing_realistic.config import validate_full_cartesian_x0


def _all_real_actions(cfg) -> list[np.ndarray]:
    return [np.array(p, dtype=float) for p in itertools.product(*([cfg.margin_vals] * cfg.n_products))]


class ExactOracleTests(unittest.TestCase):
    def test_given_action_exact_matches_tabulated_when_exact_oracle(self) -> None:
        cfg = paired_config(
            n_products=er.N_PRODUCTS,
            n_actions=er.N_ACTIONS,
            n_baskets=8,
            mc_ep=50,
            demand_seed=3,
        )
        env, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=cfg.mc_ep,
            demand_seed=cfg.demand_seed,
            env_seed=11,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        self.assertTrue(env.exact_oracle)
        mid = len(er.ALL_ACTIONS_REAL) // 3
        m = er.ALL_ACTIONS_REAL[mid]
        v_tab = float(env.compute_givenaction_value(m))
        v_ex = float(env.compute_givenaction_value_exact(m))
        self.assertAlmostEqual(v_tab, v_ex, places=10)

    def test_exact_independent_of_env_seed_given_same_demands(self) -> None:
        cfg = paired_config(n_products=er.N_PRODUCTS, n_actions=er.N_ACTIONS, demand_seed=42)
        env0, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=cfg.mc_ep,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        env1, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=cfg.mc_ep,
            demand_seed=cfg.demand_seed,
            env_seed=999,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        m = er.ALL_ACTIONS_REAL[len(er.ALL_ACTIONS_REAL) // 4]
        self.assertAlmostEqual(
            float(env0.compute_givenaction_value(m)),
            float(env1.compute_givenaction_value(m)),
            places=12,
        )

    def test_mc_large_matches_exact_within_tol_small_env(self) -> None:
        """Lightweight grid: 2 products × 3 margin levels (9 joint actions)."""
        cfg = paired_config(n_products=2, n_actions=3, demand_seed=7, mc_ep=4000, n_baskets=10)
        env_mc, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=cfg.mc_ep,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=False,
        )
        env_ex, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=50,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        pts = _all_real_actions(cfg)
        m = pts[len(pts) // 2]
        v_mc = float(env_mc.compute_givenaction_value(m))
        v_ex = float(env_ex.compute_givenaction_value(m))
        self.assertLess(abs(v_mc - v_ex), 0.06)

    def test_best_action_equals_bruteforce_grid_small_env(self) -> None:
        cfg = paired_config(n_products=2, n_actions=3, demand_seed=1)
        env, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=50,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        best = float(env.compute_best_action_value())
        brute = max(float(env.compute_givenaction_value(a)) for a in _all_real_actions(cfg))
        self.assertAlmostEqual(best, brute, places=10)

    def test_step_remains_stochastic(self) -> None:
        cfg = paired_config(n_products=2, n_actions=3, demand_seed=5)
        env, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=50,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        m = _all_real_actions(cfg)[3]
        env.reset(0)
        hashes = []
        for _ in range(50):
            s = env.step(m)
            hashes.append(s.sum())
        self.assertGreater(len(set(hashes)), 1)


class ValidateFullCartesianX0Tests(unittest.TestCase):
    def test_passes_on_all_actions_norm(self) -> None:
        validate_full_cartesian_x0(er.ALL_ACTIONS_NORM, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_random_subset(self) -> None:
        sub = er.ALL_ACTIONS_NORM[:: max(1, len(er.ALL_ACTIONS_NORM) // 5)]
        with self.assertRaises(ValueError):
            validate_full_cartesian_x0(sub, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_duplicated_rows(self) -> None:
        dup = np.vstack([er.ALL_ACTIONS_NORM[:10], er.ALL_ACTIONS_NORM[:10]])
        with self.assertRaises(ValueError):
            validate_full_cartesian_x0(dup, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_real_margin_values_not_normalized(self) -> None:
        with self.assertRaises(ValueError):
            validate_full_cartesian_x0(er.ALL_ACTIONS_REAL, er.N_PRODUCTS, er.N_ACTIONS)

    def test_fails_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_full_cartesian_x0(er.ALL_ACTIONS_NORM[:-1], er.N_PRODUCTS, er.N_ACTIONS)


class DMSX0DuplicatePreventionTests(unittest.TestCase):
    def test_first_pulls_in_batch_unique_until_all_actives_touched(self) -> None:
        """2×2 normalized grid → 4 arms; Li batch 1 for T=200 is 15 > 4."""
        actions = np.array(list(itertools.product(*([np.linspace(0.0, 1.0, 2)] * 2))), dtype=float)
        self.assertEqual(len(actions), 4)
        agent = DMSX0BPE(
            actions,
            T=200,
            noise_var=0.15,
            delta=0.25,
            seed=202,
            prior_mean=0.0,
            bpe_use_global_history=False,
        )
        n_arms = len(actions)
        self.assertEqual(int(agent._active_mask.sum()), n_arms)
        arms: list[int] = []
        start_bi = int(agent._batch_idx)
        while int(agent._batch_idx) == start_bi:
            agent.pull()
            arms.append(int(agent._pending_arm_idx))
            agent.update(0.42)
        bl = len(arms)
        k = min(bl, n_arms)
        self.assertEqual(len(set(arms[:k])), k)
        if bl > n_arms:
            self.assertGreaterEqual(len(set(arms)), n_arms)


class FinalCommitAndEliminationSourceTests(unittest.TestCase):
    def test_final_commit_row_and_stable_learning_pulls(self) -> None:
        cfg = paired_config(
            n_products=er.N_PRODUCTS,
            n_actions=er.N_ACTIONS,
            n_baskets=6,
            mc_ep=80,
            demand_seed=9,
        )
        env_base, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=cfg.mc_ep,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        prev = os.environ.pop("DMS_FINAL_COMMIT", None)
        try:
            rows_no = er.run_one_single_algorithm(
                er.PROPOSED_JOINT_ALG,
                T=6,
                env=build_env(
                    graph_dict=cfg.graph_dict,
                    n_products=cfg.n_products,
                    n_actions=cfg.n_actions,
                    margins_vals=cfg.margin_vals,
                    alpha=cfg.alpha,
                    n_baskets=cfg.n_baskets,
                    mc_ep=cfg.mc_ep,
                    demand_seed=cfg.demand_seed,
                    env_seed=0,
                    dependency_strength=cfg.dependency_strength,
                    exact_oracle=True,
                )[0],
                seed=3,
                c_beta=0.5,
                delta=0.1,
                B_rkhs=1.0,
            )
            os.environ["DMS_FINAL_COMMIT"] = "1"
            rows_yes = er.run_one_single_algorithm(
                er.PROPOSED_JOINT_ALG,
                T=6,
                env=env_base,
                seed=3,
                c_beta=0.5,
                delta=0.1,
                B_rkhs=1.0,
            )
        finally:
            if prev is None:
                os.environ.pop("DMS_FINAL_COMMIT", None)
            else:
                os.environ["DMS_FINAL_COMMIT"] = prev

        learn_no = [r for r in rows_no if r.get("phase", "learning") == "learning"]
        learn_yes = [r for r in rows_yes if r.get("phase", "learning") == "learning"]
        self.assertEqual(len(learn_no), len(learn_yes))
        for ra, rb in zip(learn_no, learn_yes):
            self.assertEqual(ra["regret_cumulative"], rb["regret_cumulative"])
            self.assertEqual(ra["a0"], rb["a0"])

        fc = [r for r in rows_yes if r.get("phase") == "final_commit"]
        self.assertEqual(len(fc), 1)
        self.assertEqual(fc[0]["t"], 7)
        self.assertTrue(np.isnan(fc[0]["simple_regret"]))
        idx = int(fc[0]["final_commit_idx"])
        grid = er.ALL_ACTIONS_NORM[idx]
        for i in range(er.N_PRODUCTS):
            self.assertAlmostEqual(float(fc[0][f"final_commit_a{i}"]), float(grid[i]), places=10)

    def test_eliminate_source_uses_full_active_arm_indices(self) -> None:
        src = Path(er.__file__).resolve().parent / "pricing_realistic" / "agents" / "dmsx0_bpe.py"
        text = src.read_text(encoding="utf-8")
        elim = text.split("def _eliminate")[1].split("def recommend_idx")[0]
        self.assertIn("active.astype(int)", elim)
        self.assertNotIn("query_set", elim)


class PlotHelperSmokeTests(unittest.TestCase):
    def test_bpe_recommendation_plot_uses_recommendation_column_not_played(self) -> None:
        p = Path(er.__file__).resolve().parent / "pricing_realistic" / "plots.py"
        text = p.read_text(encoding="utf-8")
        start = text.index("def _plot_bpe_recommendation_simple_regret_pdf")
        end = text.index("def _plot_bpe_final_commit_simple_regret_pdf", start)
        block = text[start:end]
        self.assertIn("recommendation_simple_regret", block)
        self.assertNotIn("_ci_band(blk, \"t\", \"simple_regret\")", block)

    def test_bpe_plot_helpers_run(self) -> None:
        import pandas as pd
        from pricing_realistic.plots import (
            _plot_bpe_final_commit_simple_regret_pdf,
            _plot_bpe_recommendation_simple_regret_pdf,
        )

        rows = []
        for t in range(1, 5):
            rows.append(
                dict(
                    seed=0,
                    t=t,
                    T=4,
                    algorithm=er.PROPOSED_JOINT_ALG,
                    phase="learning",
                    recommendation_simple_regret=0.5 / t,
                )
            )
        rows.append(
            dict(
                seed=0,
                t=5,
                T=4,
                algorithm=er.PROPOSED_JOINT_ALG,
                phase="final_commit",
                final_commit_simple_regret=0.05,
            )
        )
        df = pd.DataFrame(rows)
        base = str(ROOT / "_plot_helper_tmp")
        os.makedirs(base, exist_ok=True)
        try:
            _plot_bpe_recommendation_simple_regret_pdf(df, base, 1)
            _plot_bpe_final_commit_simple_regret_pdf(df, base, 1)
            self.assertTrue(os.path.isfile(os.path.join(base, "bpe_recommendation_simple_regret.pdf")))
            self.assertTrue(os.path.isfile(os.path.join(base, "bpe_final_commit_simple_regret.pdf")))
        finally:
            shutil.rmtree(base, ignore_errors=True)


class UnivariateDependencyStrengthTests(unittest.TestCase):
    def test_follower_mixed_uses_clipped_enhanced_demand(self) -> None:
        from environments.demand_generator import generate_demands, univariate_action

        cfg = paired_config(n_products=2, n_actions=5, demand_seed=0)
        d = generate_demands(
            cfg.graph_dict,
            cfg.n_products,
            cfg.n_actions,
            cfg.margin_vals,
            seed=cfg.demand_seed,
        )
        graph = cfg.graph_dict
        leaders = list(graph.keys())
        ldr = leaders[0]
        fol = graph[ldr][0]
        rev_l = [(0.0 + cfg.margin_vals[j]) * d[ldr, j, 0] for j in range(cfg.n_actions)]
        ml = cfg.margin_vals[int(np.argmax(rev_l))]
        ldr_idx = int(np.flatnonzero(cfg.margin_vals == ml)[0])
        D_L = float(d[ldr, ldr_idx, 0])
        base = d[fol, :, 0]
        enh = d[fol, :, 1]
        s = 3.0
        enh_eff = np.clip(base + s * (enh - base), 0.0, 1.0)
        mixed = D_L * enh_eff + (1.0 - D_L) * base
        rev_follower = [(0.0 + cfg.margin_vals[j]) * mixed[j] for j in range(cfg.n_actions)]
        expected_follower_margin = cfg.margin_vals[int(np.argmax(rev_follower))]
        u = univariate_action(
            d,
            graph,
            cfg.n_products,
            cfg.n_actions,
            cfg.margin_vals,
            alpha=0.0,
            dependency_strength=s,
        )
        self.assertAlmostEqual(float(u[fol]), float(expected_follower_margin), places=10)

    def test_strength_one_vs_three_can_differ(self) -> None:
        from environments.demand_generator import generate_demands, univariate_action

        found = False
        for ds in range(80):
            cfg = paired_config(n_products=2, n_actions=5, demand_seed=ds)
            d = generate_demands(
                cfg.graph_dict,
                cfg.n_products,
                cfg.n_actions,
                cfg.margin_vals,
                seed=cfg.demand_seed,
            )
            u1 = univariate_action(
                d,
                cfg.graph_dict,
                cfg.n_products,
                cfg.n_actions,
                cfg.margin_vals,
                0.0,
                dependency_strength=1.0,
            )
            u3 = univariate_action(
                d,
                cfg.graph_dict,
                cfg.n_products,
                cfg.n_actions,
                cfg.margin_vals,
                0.0,
                dependency_strength=3.0,
            )
            if not np.allclose(u1, u3):
                found = True
                break
        self.assertTrue(found)


class MarginLookupToleranceTests(unittest.TestCase):
    def test_compute_givenaction_value_accepts_tiny_drift(self) -> None:
        cfg = paired_config(n_products=2, n_actions=5, demand_seed=11)
        env, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=0.0,
            n_baskets=10,
            mc_ep=50,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        m0 = np.array([cfg.margin_vals[2], cfg.margin_vals[3]], dtype=float)
        v0 = float(env.compute_givenaction_value(m0))
        m_eps = m0.copy()
        m_eps[0] += 1e-11
        m_eps[1] -= 8e-12
        v1 = float(env.compute_givenaction_value(m_eps))
        self.assertAlmostEqual(v0, v1, places=9)

    def test_step_accepts_tiny_drift(self) -> None:
        cfg = paired_config(n_products=2, n_actions=5, demand_seed=11)
        env, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=0.0,
            n_baskets=20,
            mc_ep=50,
            demand_seed=cfg.demand_seed,
            env_seed=99,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        np.random.seed(10001)
        env.reset(10001)
        m = np.array([cfg.margin_vals[1], cfg.margin_vals[-2]], dtype=float)
        s0 = env.step(m.copy())
        m_eps = m.copy()
        m_eps[0] += 2e-11
        np.random.seed(10001)
        env.reset(10001)
        s1 = env.step(m_eps)
        np.testing.assert_array_equal(s0, s1)

    def test_margin_far_from_grid_raises(self) -> None:
        cfg = paired_config(n_products=2, n_actions=5, demand_seed=11)
        env, _ = build_env(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=0.0,
            n_baskets=5,
            mc_ep=50,
            demand_seed=cfg.demand_seed,
            env_seed=0,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=True,
        )
        bad = np.array([0.55555, cfg.margin_vals[0]], dtype=float)
        with self.assertRaises(ValueError):
            env.compute_givenaction_value(bad)


class MonteCarloOracleResetTests(unittest.TestCase):
    def test_mc_oracle_then_first_step_matches_duplicate_env(self) -> None:
        cfg = paired_config(
            n_products=2,
            n_actions=4,
            mc_ep=25,
            n_baskets=6,
            demand_seed=7,
        )
        kw = dict(
            graph_dict=cfg.graph_dict,
            n_products=cfg.n_products,
            n_actions=cfg.n_actions,
            margins_vals=cfg.margin_vals,
            alpha=cfg.alpha,
            n_baskets=cfg.n_baskets,
            mc_ep=cfg.mc_ep,
            demand_seed=cfg.demand_seed,
            dependency_strength=cfg.dependency_strength,
            exact_oracle=False,
        )
        env_a, _ = build_env(**kw, env_seed=123)
        env_b, _ = build_env(**kw, env_seed=123)
        m = np.array([cfg.margin_vals[1], cfg.margin_vals[2]], dtype=float)
        np.random.seed(777)
        env_a.reset(777)
        np.random.seed(777)
        env_b.reset(777)
        np.testing.assert_array_equal(env_a.step(m), env_b.step(m))


class DirectPlotSmokeTests(unittest.TestCase):
    """Fast PDF smoke tests for plot helpers (FIX 1 + FIX 6)."""

    def test_plot_simple_regret_creates_pdf(self) -> None:
        import tempfile

        import pandas as pd

        from pricing_realistic.plots import _plot_simple_regret

        rows = [
            dict(
                seed=0,
                t=t,
                T=3,
                algorithm=er.PROPOSED_JOINT_ALG,
                phase="learning",
                simple_regret=0.2 / t,
            )
            for t in range(1, 4)
        ]
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            _plot_simple_regret(df, td, 1)
            self.assertTrue(os.path.isfile(os.path.join(td, "simple_regret.pdf")))

    def test_final_commit_plot_skipped_without_rows_no_crash(self) -> None:
        import tempfile

        import pandas as pd

        from pricing_realistic.plots import _plot_bpe_final_commit_simple_regret_pdf

        df = pd.DataFrame(
            [
                dict(
                    seed=0,
                    t=1,
                    T=2,
                    algorithm=er.PROPOSED_JOINT_ALG,
                    phase="learning",
                    simple_regret=0.1,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            _plot_bpe_final_commit_simple_regret_pdf(df, td, 1)
            self.assertFalse(os.path.isfile(os.path.join(td, "bpe_final_commit_simple_regret.pdf")))


if __name__ == "__main__":
    unittest.main()
