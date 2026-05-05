"""CLI entry: full experiment run and dependency ablation."""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from environments.demand_generator import make_default_env, univariate_action

from .config import (
    ALPHA,
    CFG,
    EXP_CFG,
    ALL_ACTIONS_NORM,
    ALL_ACTIONS_REAL,
    GRAPH_DICT,
    MARGIN_VALS,
    N_PRODUCTS,
    N_ACTIONS,
    N_BASKETS,
    NOISE_VAR,
    PROPOSED_JOINT_ALG,
    JOINT_CONT_GP_UCB_ALG,
    ISO_MATERN_GP_UCB_ALG,
    COLORS,
    MARKERS,
    best_independent_product_surrogate,
    effective_gp_noise_variance,
    validate_full_cartesian_x0,
)
from .agents.dmsx0_bpe import li_22_bpe_batch_sizes
from .parallel import _run_task, _run_task_single
from .plots import (
    _add_best_simple_regret,
    _plot_best_simple_regret,
    _plot_bpe_final_commit_simple_regret_pdf,
    _plot_bpe_recommendation_simple_regret_pdf,
    _plot_heatmaps,
    _plot_regret,
    _plot_regret_per_pair,
    _plot_simple_regret,
    _print_summary,
    simple_regret_plot_source,
)
from .runs import run_one


def _print_proposed_dmsx0_single_agent_banner(T_list: list[int], n_seeds: int) -> None:
    """Startup lines for **DMS-X0-BPE** single-policy runs (full ``X°``, Li batches, batch dedupe)."""
    validate_full_cartesian_x0(ALL_ACTIONS_NORM, N_PRODUCTS, N_ACTIONS)
    nx = len(ALL_ACTIONS_NORM)
    T_max = max(T_list)
    batches = li_22_bpe_batch_sizes(T_max)
    print(
        f"DMS-X0-BPE running on full X°: |X°|={nx}, d={N_PRODUCTS}, "
        f"levels={N_ACTIONS}, full_grid=True (validated)"
    )
    print(f"  Run settings: T_list={T_list}, n_seeds={n_seeds}, T_max={T_max}")
    print(f"  Li–Scarlett BPE batch sizes (sum=T_max): {batches.tolist()}")
    print(
        "  Within-batch pulls: no duplicate active arms until each active arm has been "
        "pulled at least once in the current batch (Algorithm 1–style set accumulation)."
    )
    print(
        "  Recommendation CSV columns / recommendation plots: diagnostic & deployment only; "
        "theorem cumulative pseudo-regret uses played actions (regret_cumulative, played_*)."
    )


def _emit_simple_regret_figures(df: pd.DataFrame, base: str, n_seeds: int) -> None:
    """Instantaneous / best simple regret PDFs — source chosen by ``EXP_PLOT_SIMPLE_REGRET_SOURCE``."""
    src = simple_regret_plot_source()
    need_rec = src in ("recommendation", "both")
    if need_rec and (
        "recommendation_simple_regret" not in df.columns
        or "best_recommendation_simple_regret" not in df.columns
    ):
        raise ValueError(
            "EXP_PLOT_SIMPLE_REGRET_SOURCE=%r requires columns recommendation_simple_regret and "
            "best_recommendation_simple_regret — re-run the experiment (current runs.py CSV schema) "
            "or use EXP_PLOT_SIMPLE_REGRET_SOURCE=played."
            % (os.environ.get("EXP_PLOT_SIMPLE_REGRET_SOURCE", ""),)
        )
    if src == "played":
        _plot_simple_regret(df, base, n_seeds)
        _plot_best_simple_regret(df, base, n_seeds)
        return
    if src == "recommendation":
        _plot_simple_regret(
            df,
            base,
            n_seeds,
            y_col="recommendation_simple_regret",
            out_stem="simple_regret",
            y_label=r"BPE recommendation simple regret (diagnostic / deployment; $\hat a_t$)",
        )
        _plot_best_simple_regret(
            df,
            base,
            n_seeds,
            y_col="best_recommendation_simple_regret",
            out_stem="best_simple_regret",
            y_label="Best BPE recommendation simple regret (diagnostic / deployment)",
        )
        return
    # both
    _plot_simple_regret(
        df,
        base,
        n_seeds,
        y_col="simple_regret",
        out_stem="simple_regret_played",
        y_label="Played-action simple regret",
    )
    _plot_best_simple_regret(
        df,
        base,
        n_seeds,
        y_col="best_simple_regret",
        out_stem="best_simple_regret_played",
        y_label="Best played-action simple regret so far",
    )
    _plot_simple_regret(
        df,
        base,
        n_seeds,
        y_col="recommendation_simple_regret",
        out_stem="simple_regret_recommendation",
        y_label=r"BPE recommendation simple regret (diagnostic / deployment; $\hat a_t$)",
    )
    _plot_best_simple_regret(
        df,
        base,
        n_seeds,
        y_col="best_recommendation_simple_regret",
        out_stem="best_simple_regret_recommendation",
        y_label="Best BPE recommendation simple regret (diagnostic / deployment)",
    )


def main():
    if os.environ.get("EXP_DEP_ABLATION", "").lower() in ("1", "true", "yes"):
        run_dependency_strength_ablation()
        return

    FAST_MODE = True  # set False for full server run
    # EXP_SMOKE=1 → ~10–40 s: tiny T, 1 seed, no Pool, CSV + summary only (no PDFs).
    _smoke = os.environ.get("EXP_SMOKE", "").lower() in ("1", "true", "yes")
    # EXP_SEQ_T1000=1 → single horizon EXP_SEQ_T (default 1000); Pool across seeds (like default).
    # EXP_N_WORKERS caps parallel workers (default: min(cpu_count(), N_SEEDS)); set EXP_N_WORKERS=1 to force sequential seeds.
    _seq = os.environ.get("EXP_SEQ_T1000", "").lower() in ("1", "true", "yes")
    if _smoke:
        T_LIST     = [int(os.environ.get("EXP_SMOKE_T", "50"))]
        N_SEEDS    = 1
        N_WORKERS  = 1
        _skip_plots = True
    elif _seq:
        # EXP_SEQ_T overrides horizon (default 1000), e.g. EXP_SEQ_T=2000
        T_LIST     = [int(os.environ.get("EXP_SEQ_T", "1000"))]
        N_SEEDS    = int(os.environ.get("EXP_N_SEEDS", "3"))
        _nw_raw = os.environ.get("EXP_N_WORKERS")
        if _nw_raw is None or str(_nw_raw).strip() == "":
            N_WORKERS = min(cpu_count(), N_SEEDS)
        else:
            N_WORKERS = max(1, min(int(_nw_raw), N_SEEDS, cpu_count()))
        _skip_plots = False
    else:
        T_LIST     = [200, 1000] if FAST_MODE else [1000, 5000, 10000]
        N_SEEDS    = 3 if FAST_MODE else 10
        N_WORKERS  = min(cpu_count(), N_SEEDS)
        _skip_plots = False
    C_BETA         = 0.5   # Univariate 1-D GP-UCB exploration scale
    DELTA          = 0.1
    B_RKHS         = float(EXP_CFG.dms_rkhs_norm)
    BASE = (
        os.environ.get("EXP_SMOKE_OUT", "results_realistic_smoke")
        if _smoke
        else os.environ.get("EXP_OUT", "results_realistic")
    )
    os.makedirs(BASE, exist_ok=True)

    # ── Build environment (main process only — for diagnostics/plots) ───────
    print("Building ComplementaryPricingEnvironment …")
    env, demands = make_default_env(cfg=CFG, seed=0)

    opt = env.compute_best_action_value()
    all_vals        = np.array([env.compute_givenaction_value(a) for a in ALL_ACTIONS_REAL])
    opt_action_real = ALL_ACTIONS_REAL[int(np.argmax(all_vals))]
    print(f"  Optimal value  = {opt:.4f}")
    print(f"  Optimal action = {opt_action_real}")
    print(f"  Action space   = {len(ALL_ACTIONS_NORM):,} discrete points  "
          f"(Kleinberg needs T>>{len(ALL_ACTIONS_NORM):,} for full coverage)")
    print(f"  N_BASKETS/step = {N_BASKETS}")
    _eo_main = bool(getattr(env, "exact_oracle", True))
    if _eo_main:
        print("  Oracle/evaluation = exact closed-form expectations (tabulated action values)")
    else:
        _mc_oracle = int(CFG.mc_ep) * int(CFG.n_baskets)
        print(
            f"  Oracle MC/cell = mc_ep×n_baskets = {CFG.mc_ep}×{CFG.n_baskets} = {_mc_oracle:,}"
        )
    _fc_main = os.environ.get("DMS_FINAL_COMMIT", "").lower() in ("1", "true", "yes")
    print(f"  Evaluation oracle flag: exact_oracle={_eo_main}")
    print(f"  Final-commit CSV row (DMS_FINAL_COMMIT): {_fc_main}")
    print(
        "  Theorem cumulative regret uses played pulls only; recommendation_* / "
        "final_commit_* are diagnostic or deployment metrics (not added to cumulative regret)."
    )
    _oi = float(best_independent_product_surrogate(env))
    print(f"  Indep. product surrogate (grid) = {_oi:.4f}   joint_oracle − surrogate = {opt - _oi:.4f}")

    uni_action = univariate_action(
        demands,
        GRAPH_DICT,
        N_PRODUCTS,
        N_ACTIONS,
        MARGIN_VALS,
        ALPHA,
        dependency_strength=float(getattr(env, "dependency_strength", CFG.dependency_strength)),
    )
    val_uni = env.compute_givenaction_value(uni_action)
    print(f"  Univariate convergence = {val_uni:.4f}  at {uni_action}  "
          f"gap={100*(opt-val_uni)/opt:.1f}%")
    print(f"  BanditExperimentConfig (tune in script): {EXP_CFG!r}\n")
    if _smoke:
        print(
            f"  Mode: EXP_SMOKE — T={T_LIST[0]}, 1 seed, no Pool, plots off "
            f"(out dir {BASE})\n"
        )
    elif _seq:
        print(
            f"  Mode: EXP_SEQ_T1000 — single horizon T={T_LIST[0]}, "
            f"{N_WORKERS} parallel worker(s) for seeds (EXP_N_WORKERS; "
            f"default=min(CPU, N_SEEDS))\n"
        )

    _bpe_b = li_22_bpe_batch_sizes(max(T_LIST))
    print(
        f"  Paper method: {PROPOSED_JOINT_ALG} — full ε-cover |X°|={len(ALL_ACTIONS_NORM):,} "
        f"(not Smolyak), k_Π + Li–Scarlett BPE Alg. 1 + elimination; "
        f"L={EXP_CFG.dms_kernel_L}, λ_BPE={EXP_CFG.dms_bpe_noise_var}"
    )
    _lam_eff = float(effective_gp_noise_variance(EXP_CFG))
    print(
        f"  Baseline / ablation: {JOINT_CONT_GP_UCB_ALG} — continuous GP-UCB acquisition on [0,1]^d "
        f"(L-BFGS-B) + snap to X° for play (not finite-catalog BPE); same k_Π, Li batch lengths; "
        f"λ(eff)={_lam_eff}, δ={EXP_CFG.dms_elimination_delta}"
    )
    print(
        f"  Baseline: {ISO_MATERN_GP_UCB_ALG} — isotropic Matérn-1/2 exp(-||Δ||_2/L), "
        f"same batching as joint; L={EXP_CFG.dms_kernel_L}"
    )
    print(
        f"  BPE batch lengths N_i=⌈√(T N_{{i-1}})⌉ → {len(_bpe_b)} batches for T={max(T_LIST)} "
        f"(e.g. {_bpe_b[:min(4, len(_bpe_b))].tolist()}…)"
    )
    print(
        f"  c_beta (Univariate 1-D GP-UCB only)={C_BETA}   NOISE_VAR={NOISE_VAR:.5f}   "
        f"Ψ (RKHS norm, EXP_CFG.dms_rkhs_norm)={B_RKHS}"
    )
    print(f"  Workers: {N_WORKERS}  (total tasks: {len(T_LIST)*N_SEEDS})\n")

    # ── Build all (T, seed) tasks ────────────────────────────────────────────
    tasks = [
        (T, seed, C_BETA, DELTA, B_RKHS)
        for T in T_LIST
        for seed in range(N_SEEDS)
    ]

    t0 = time.time()
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            results = pool.map(_run_task, tasks)
    else:
        results = [_run_task(a) for a in tasks]

    all_records = [row for records in results for row in records]
    print(f"\nAll tasks done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_records)
    df.to_csv(f"{BASE}/results.csv", index=False)
    print(f"Saved {BASE}/results.csv  ({len(df):,} rows)")

    df = _add_best_simple_regret(df)
    if not _skip_plots:
        _plot_regret_per_pair(df, env, opt_action_real, BASE, N_SEEDS)
        _emit_simple_regret_figures(df, BASE, N_SEEDS)
        _plot_heatmaps(df, env, demands, opt_action_real, BASE)
    else:
        print("  (skipped regret / heatmap PDFs — EXP_SMOKE)")
    _print_summary(df)
    print(f"\nAll outputs in {BASE}/")


def main_single_agent(algorithm: str) -> None:
    """
    Like ``main()`` but each worker runs **only** ``algorithm`` via ``run_one_single_algorithm``
    (single instantiated policy — minimal CPU vs full ``run_one``).

    Environment variables (same as ``main`` where applicable):

    * ``EXP_N_WORKERS`` — cap parallel workers (default ``1`` for light CPU).
    * ``EXP_OUT`` — output directory.
    * ``EXP_SPSA_A_PAR`` — SPSA gain ``a_par`` (default ``1e-6``; smoke uses a larger value unless set).
    * ``EXP_PLOT_SIMPLE_REGRET_SOURCE`` — ``played`` (default), ``recommendation``, or ``both``;
      controls ``simple_regret*.pdf`` / ``best_simple_regret*.pdf`` (instantaneous regret on pulls
      vs on ``recommend()``). Cumulative ``regret.pdf`` always uses pulled rewards.
    """
    if os.environ.get("EXP_DEP_ABLATION", "").lower() in ("1", "true", "yes"):
        run_dependency_strength_ablation()
        return

    FAST_MODE = True
    _smoke = os.environ.get("EXP_SMOKE", "").lower() in ("1", "true", "yes")
    _seq = os.environ.get("EXP_SEQ_T1000", "").lower() in ("1", "true", "yes")
    if _smoke:
        T_LIST = [int(os.environ.get("EXP_SMOKE_T", "50"))]
        N_SEEDS = 1
        _skip_plots = True
    elif _seq:
        T_LIST = [int(os.environ.get("EXP_SEQ_T", "1000"))]
        N_SEEDS = int(os.environ.get("EXP_N_SEEDS", "3"))
        _skip_plots = False
    else:
        T_LIST = [200, 1000] if FAST_MODE else [1000, 5000, 10000]
        N_SEEDS = 3 if FAST_MODE else 10
        _skip_plots = False

    nw_default = os.environ.get("EXP_N_WORKERS", "1")
    N_WORKERS = max(1, min(int(nw_default), N_SEEDS, cpu_count()))

    C_BETA = 0.5
    DELTA = 0.1
    B_RKHS = float(EXP_CFG.dms_rkhs_norm)
    BASE = (
        os.environ.get("EXP_SMOKE_OUT", "results_realistic_smoke")
        if _smoke
        else os.environ.get("EXP_OUT", "results_realistic")
    )
    os.makedirs(BASE, exist_ok=True)

    if algorithm == PROPOSED_JOINT_ALG:
        _print_proposed_dmsx0_single_agent_banner(T_LIST, N_SEEDS)

    print("Building ComplementaryPricingEnvironment …")
    env, demands = make_default_env(cfg=CFG, seed=0)

    opt = env.compute_best_action_value()
    all_vals = np.array([env.compute_givenaction_value(a) for a in ALL_ACTIONS_REAL])
    opt_action_real = ALL_ACTIONS_REAL[int(np.argmax(all_vals))]
    print(f"  Single-agent mode: {algorithm}")
    print(f"  Optimal value  = {opt:.4f}")
    print(f"  N_BASKETS/step = {N_BASKETS}")
    _eo_sa = bool(getattr(env, "exact_oracle", True))
    if _eo_sa:
        print("  Oracle/evaluation = exact closed-form expectations (tabulated action values)")
    else:
        _mc_sa = int(CFG.mc_ep) * int(CFG.n_baskets)
        print(
            f"  Oracle MC/cell = mc_ep×n_baskets = {CFG.mc_ep}×{CFG.n_baskets} = {_mc_sa:,}"
        )
    print(f"  Evaluation oracle flag: exact_oracle={_eo_sa}")
    if algorithm == PROPOSED_JOINT_ALG:
        fc_st = os.environ.get("DMS_FINAL_COMMIT", "").lower() in ("1", "true", "yes")
        print(f"  Final-commit CSV row (DMS_FINAL_COMMIT): {fc_st}")
        print(
            "  Theorem cumulative regret uses played pulls only; recommendation_* / "
            "final_commit_* are diagnostic or deployment metrics (not added to cumulative regret)."
        )
    print(f"  Workers: {N_WORKERS}  (EXP_N_WORKERS default=1 for single-policy runs)\n")

    tasks = [
        (T, seed, C_BETA, DELTA, B_RKHS, algorithm)
        for T in T_LIST
        for seed in range(N_SEEDS)
    ]

    t0 = time.time()
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            results = pool.map(_run_task_single, tasks)
    else:
        results = [_run_task_single(a) for a in tasks]

    all_records = [row for records in results for row in records]
    print(f"\nAll tasks done in {time.time()-t0:.1f}s")

    df = pd.DataFrame(all_records)
    df.to_csv(f"{BASE}/results.csv", index=False)
    print(f"Saved {BASE}/results.csv  ({len(df):,} rows)")

    df = _add_best_simple_regret(df)
    if not _skip_plots:
        _plot_regret_per_pair(df, env, opt_action_real, BASE, N_SEEDS)
        _emit_simple_regret_figures(df, BASE, N_SEEDS)
        if PROPOSED_JOINT_ALG in set(df["algorithm"].values):
            _plot_bpe_recommendation_simple_regret_pdf(df, BASE, N_SEEDS)
            _plot_bpe_final_commit_simple_regret_pdf(df, BASE, N_SEEDS)
        _plot_heatmaps(df, env, demands, opt_action_real, BASE)
    else:
        print("  (skipped regret / heatmap PDFs — EXP_SMOKE)")
    _print_summary(df)
    print(f"\nAll outputs in {BASE}/")


def run_dependency_strength_ablation() -> None:
    """
    Sweep ``dependency_strength`` (EnvConfig → ComplementaryPricingEnvironment).

    Stronger coupling increases the value of joint pricing vs coordinate-wise play;
    we expect **DMS-X0-BPE** (paper BPE on full ``X°``) to improve **relative** to
    Univariate / Kleinberg as s → 1 — the ablation plot surfaces that trend.
    """
    raw = os.environ.get("EXP_DEP_VALUES", "0.2,0.5,0.8,1.0")
    strengths = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    T = int(os.environ.get("EXP_DEP_T", "400"))
    n_seeds = int(os.environ.get("EXP_DEP_N_SEEDS", "5"))
    base = os.environ.get("EXP_DEP_OUT", "results_ablation_dependency_strength")
    os.makedirs(base, exist_ok=True)

    print(
        f"\nDependency-strength ablation  T={T}  seeds={n_seeds}  s∈{strengths}\n"
        f"  Output directory: {base}/\n"
    )

    rows_all: list[dict] = []
    t0 = time.time()
    for s in strengths:
        cfg_s = replace(CFG, dependency_strength=float(s))
        env, _demands = make_default_env(cfg_s, seed=0)
        opt = float(env.compute_best_action_value())
        print(f"  s={s:.2f}  optimal profit = {opt:.4f}")
        for seed in range(n_seeds):
            rec = run_one(
                T=T,
                env=env,
                seed=seed,
                c_beta=0.5,
                delta=0.1,
                B_rkhs=float(EXP_CFG.dms_rkhs_norm),
            )
            for row in rec:
                r = dict(row)
                r["dependency_strength"] = float(s)
                r["opt_value"] = opt
                rows_all.append(r)

    df = pd.DataFrame(rows_all)
    df = _add_best_simple_regret(df)
    csv_path = f"{base}/results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}  ({len(df):,} rows)  wall {time.time()-t0:.1f}s")

    last = df[df["t"] == df["T"]].copy()
    summ = (
        last.groupby(["dependency_strength", "algorithm"], sort=True)["simple_regret"]
        .agg(mean_sr="mean", std_sr="std", n_seeds="count")
        .reset_index()
    )
    summ_path = f"{base}/summary_last_round.csv"
    summ.to_csv(summ_path, index=False)
    print(f"  Saved {summ_path}")
    algs_curve = [
        PROPOSED_JOINT_ALG,
        JOINT_CONT_GP_UCB_ALG,
        ISO_MATERN_GP_UCB_ALG,
        "Univariate",
        "Kleinberg",
        "HGP-UCB-CPP",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0 = axes[0]
    for alg in algs_curve:
        sub = last[last["algorithm"] == alg]
        m = sub.groupby("dependency_strength", sort=True)["simple_regret"].mean()
        se = sub.groupby("dependency_strength", sort=True)["simple_regret"].agg(
            lambda x: float(np.std(x, ddof=1) / max(np.sqrt(len(x)), 1.0))
            if len(x) > 1
            else 0.0
        )
        x = m.index.values.astype(float)
        ax0.errorbar(
            x,
            m.values,
            yerr=1.96 * se.reindex(m.index).fillna(0.0).values,
            label=alg,
            color=COLORS.get(alg, "#333"),
            marker=MARKERS.get(alg, "o"),
            capsize=3,
            lw=1.6,
            markersize=6,
        )
    ax0.set_xlabel(r"Leader–follower coupling $s$ (dependency\_strength)")
    ax0.set_ylabel(f"Mean simple regret at $T={T}$ (± 95% approx.)")
    ax0.set_title("Ablation: regret vs coupling strength")
    ax0.legend(fontsize=8, loc="best")

    ax1 = axes[1]
    adv_mean, adv_se = [], []
    for s in strengths:
        sub_s = last[last["dependency_strength"] == float(s)]
        piv = sub_s.pivot_table(
            index="seed", columns="algorithm", values="simple_regret", aggfunc="first"
        )
        if "Univariate" in piv.columns and PROPOSED_JOINT_ALG in piv.columns:
            diff = piv["Univariate"] - piv[PROPOSED_JOINT_ALG]
            adv_mean.append(float(np.mean(diff)))
            adv_se.append(
                float(np.std(diff, ddof=1) / max(np.sqrt(len(diff)), 1.0))
                if len(diff) > 1
                else 0.0
            )
        else:
            adv_mean.append(float("nan"))
            adv_se.append(0.0)
    x = np.asarray(strengths, dtype=float)
    ax1.errorbar(
        x,
        adv_mean,
        yerr=1.96 * np.asarray(adv_se),
        fmt="s-",
        color="#2ca02c",
        ecolor="#555",
        capsize=4,
        lw=2,
        markersize=7,
        label=r"$\mathbb{E}[\mathrm{SR}_{\mathrm{Uni}} - \mathrm{SR}_{\mathrm{DMS}}]$",
    )
    ax1.axhline(0.0, color="gray", ls="--", lw=1)
    ax1.set_xlabel(r"Coupling $s$ (dependency\_strength)")
    ax1.set_ylabel("Advantage of DMS over Univariate\n(positive ⇒ lower regret for DMS)")
    ax1.set_title("Paper message: BPE (full X°) gains vs Univariate grow with dependence")
    ax1.legend(loc="best")

    fig.tight_layout()
    pdf_path = f"{base}/ablation_dependency_strength.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {pdf_path}")

    print("\n=== Mean simple regret at t=T by s (DMS vs Univariate) ===")
    for s in strengths:
        sub = last[last["dependency_strength"] == float(s)]
        for alg in (PROPOSED_JOINT_ALG, "Univariate"):
            g = sub[sub["algorithm"] == alg]["simple_regret"]
            print(f"  s={s:.2f}  {alg:14s}  mean SR = {g.mean():.4f}  ± {g.std(ddof=1) if len(g) > 1 else 0.0:.4f}")

