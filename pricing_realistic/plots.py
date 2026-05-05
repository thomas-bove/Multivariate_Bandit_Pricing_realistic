"""Figures: regret curves, heatmaps, summaries."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from environments.complementary import ComplementaryPricingEnvironment
from environments.demand_generator import univariate_action

from .config import (
    ALPHA,
    COLORS,
    GRAPH_DICT,
    MARKERS,
    MARGIN_VALS,
    N_ACTIONS,
    N_PRODUCTS,
    PROPOSED_JOINT_ALG,
    JOINT_CONT_GP_UCB_ALG,
    ISO_MATERN_GP_UCB_ALG,
    ALL_ALGS,
)
from .rewards import norm_to_real


def simple_regret_plot_source() -> str:
    """
    Which instantaneous simple-regret series to plot (env ``EXP_PLOT_SIMPLE_REGRET_SOURCE``).

    * ``played`` (default) — per-round reward on the **pulled** arm (bandit behaviour).
    * ``recommendation`` — BPE **diagnostic / deployment** ``recommendation_simple_regret`` (not the
      theorem cumulative regret); baselines mirror played arms in the CSV.
    * ``both`` — emit separate PDFs with ``*_played`` and ``*_recommendation`` stems.

    Reporting recommendation regret in figures is standard when the estimator / recommendation
    is the object of interest (distinct from cumulative regret on pulls).
    """
    v = os.environ.get("EXP_PLOT_SIMPLE_REGRET_SOURCE", "played").strip().lower()
    if v in ("recommendation", "rec", "recommend", "1", "true", "yes"):
        return "recommendation"
    if v in ("both", "all"):
        return "both"
    return "played"


def _df_learning_only(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude optional ``final_commit`` evaluation rows from trajectory plots."""
    if "phase" not in df.columns:
        return df
    return df.loc[df["phase"].fillna("learning") == "learning"].copy()


def _ordered_algorithms_in_df(df: pd.DataFrame) -> list:
    """Stable legend order (same as ``ALL_ALGS``), restricted to algorithms present in ``df``."""
    present = set(df["algorithm"].unique())
    return [a for a in ALL_ALGS if a in present]


def _heatmap_columns_from_df(df: pd.DataFrame) -> list:
    """Heatmap columns: preferred ordering, plus any algorithm in ``df`` not listed (e.g. Kleinberg)."""
    preferred = [
        PROPOSED_JOINT_ALG,
        JOINT_CONT_GP_UCB_ALG,
        ISO_MATERN_GP_UCB_ALG,
        "HGP-UCB-CPP",
        "SPSA",
        "Univariate",
        "Kleinberg",
        "BZ-ETC",
    ]
    present = set(df["algorithm"].unique())
    cols = [a for a in preferred if a in present]
    rest = sorted(a for a in present if a not in preferred)
    return cols + rest


def _ci_band(grp, x_col, y_col):
    g    = grp.groupby(x_col)[y_col]
    mean = g.mean()
    std  = g.std()
    n    = g.count()
    ci95 = scipy_stats.t.ppf(0.975, df=np.maximum(n - 1, 1)) * std / np.sqrt(n)
    return mean, ci95


def _plot_regret(df, base, n_seeds):
    df = _df_learning_only(df)
    T_LIST = sorted(df["T"].unique())
    fig, axes = plt.subplots(1, len(T_LIST), figsize=(6 * len(T_LIST), 4),
                             squeeze=False)
    for ax, T in zip(axes[0], T_LIST):
        sub = df[df["T"] == T]
        for alg in _ordered_algorithms_in_df(df):
            mean, ci = _ci_band(sub[sub["algorithm"] == alg],
                                "t", "regret_cumulative")
            ax.plot(mean.index, mean.values, label=alg,
                    color=COLORS[alg], marker=MARKERS[alg],
                    markevery=max(len(mean) // 8, 1), markersize=5, lw=1.8)
            ax.fill_between(mean.index, mean.values - ci, mean.values + ci,
                            alpha=0.15, color=COLORS[alg])
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("Played-action cumulative regret")
        ax.set_title(f"Complementary pricing — $d={N_PRODUCTS}$, $T={T}$, {n_seeds} seeds")
        ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{base}/regret.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def _plot_simple_regret(
    df,
    base,
    n_seeds,
    *,
    y_col: str = "simple_regret",
    out_stem: str = "simple_regret",
    y_label: str = "Played-action simple regret",
):
    df = _df_learning_only(df)
    T_LIST = sorted(df["T"].unique())
    fig, axes = plt.subplots(1, len(T_LIST), figsize=(6 * len(T_LIST), 4),
                             squeeze=False)
    for ax, T in zip(axes[0], T_LIST):
        sub = df[df["T"] == T]
        for alg in _ordered_algorithms_in_df(df):
            mean, _ = _ci_band(sub[sub["algorithm"] == alg],
                               "t", y_col)
            ax.semilogy(mean.index, np.maximum(mean.values, 1e-6),
                        label=alg, color=COLORS[alg], lw=1.8)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel(y_label)
        ax.set_title(f"Complementary pricing — $d={N_PRODUCTS}$, $T={T}$, {n_seeds} seeds")
        ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{base}/{out_stem}.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def _add_best_simple_regret(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["algorithm", "T", "seed", "t"])
    df = df.copy()
    if "phase" in df.columns:
        learn = df["phase"].fillna("learning") == "learning"
    else:
        learn = pd.Series(True, index=df.index)

    sub = df.loc[learn].copy()
    if not sub.empty:
        df.loc[learn, "best_simple_regret"] = (
            sub.groupby(["algorithm", "T", "seed"])["simple_regret"].cummin().values
        )
    if "recommendation_simple_regret" in df.columns and not sub.empty:
        df.loc[learn, "best_recommendation_simple_regret"] = (
            sub.groupby(["algorithm", "T", "seed"])["recommendation_simple_regret"].cummin().values
        )
    return df


def _plot_best_simple_regret(
    df,
    base,
    n_seeds,
    *,
    y_col: str = "best_simple_regret",
    out_stem: str = "best_simple_regret",
    y_label: str = "Best played-action simple regret so far",
):
    df = _df_learning_only(df)
    T_max = int(df["T"].max())
    sub   = df[df["T"] == T_max]
    fig, ax = plt.subplots(figsize=(7, 4))
    for alg in _ordered_algorithms_in_df(df):
        mean, ci = _ci_band(sub[sub["algorithm"] == alg],
                            "t", y_col)
        ax.semilogy(mean.index, np.maximum(mean.values, 1e-6),
                    label=alg, color=COLORS[alg], lw=1.8)
        ax.fill_between(mean.index,
                        np.maximum(mean.values - ci, 1e-6),
                        np.maximum(mean.values + ci, 1e-6),
                        alpha=0.15, color=COLORS[alg])
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Complementary pricing — $d={N_PRODUCTS}$, $T={T_max}$, {n_seeds} seeds\n"
        f"{PROPOSED_JOINT_ALG} (paper: BPE + elimination on full X°) vs "
        f"{JOINT_CONT_GP_UCB_ALG} (continuous acquisition + snap) vs baselines"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{base}/{out_stem}.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")


def _plot_bpe_recommendation_simple_regret_pdf(df: pd.DataFrame, base: str, n_seeds: int) -> None:
    """Diagnostic only: ``recommendation_simple_regret`` vs round for **DMS-X0-BPE**."""
    if PROPOSED_JOINT_ALG not in df["algorithm"].values:
        return
    sub = _df_learning_only(df)
    sub = sub[sub["algorithm"] == PROPOSED_JOINT_ALG]
    if sub.empty or "recommendation_simple_regret" not in sub.columns:
        return
    T_LIST = sorted(sub["T"].unique())
    fig, axes = plt.subplots(1, len(T_LIST), figsize=(6 * len(T_LIST), 4),
                             squeeze=False)
    for ax, T in zip(axes[0], T_LIST):
        blk = sub[sub["T"] == T]
        mean, _ = _ci_band(blk, "t", "recommendation_simple_regret")
        ax.semilogy(mean.index, np.maximum(mean.values, 1e-6),
                    color=COLORS[PROPOSED_JOINT_ALG], lw=2.0)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel("BPE recommendation simple regret (diagnostic/deployment)")
        ax.set_title(f"{PROPOSED_JOINT_ALG} — $T={T}$, {n_seeds} seeds")
    fig.tight_layout()
    path = f"{base}/bpe_recommendation_simple_regret.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_bpe_final_commit_simple_regret_pdf(df: pd.DataFrame, base: str, n_seeds: int) -> None:
    """One bar per horizon ``T`` from optional ``final_commit`` evaluation rows."""
    if "phase" not in df.columns or "final_commit_simple_regret" not in df.columns:
        return
    fc = df[(df["algorithm"] == PROPOSED_JOINT_ALG) & (df["phase"] == "final_commit")]
    if fc.empty:
        print(f"  (skipped {base}/bpe_final_commit_simple_regret.pdf — no final-commit rows)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    T_LIST = sorted(fc["T"].unique())
    xs = np.arange(len(T_LIST))
    means: list[float] = []
    errs: list[float] = []
    for T in T_LIST:
        col = fc.loc[fc["T"] == T, "final_commit_simple_regret"].astype(float)
        means.append(float(col.mean()))
        errs.append(float(col.std(ddof=1) / np.sqrt(len(col))) if len(col) > 1 else 0.0)
    ax.bar(xs, means, yerr=errs, capsize=4,
           color=COLORS[PROPOSED_JOINT_ALG], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(t)) for t in T_LIST])
    ax.set_xlabel(r"Horizon $T$")
    ax.set_ylabel("Final commit simple regret (not included in cumulative regret)")
    ax.set_title(f"{PROPOSED_JOINT_ALG} — final deploy evaluation ({n_seeds} seeds)")
    fig.tight_layout()
    path = f"{base}/bpe_final_commit_simple_regret.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── heatmap helpers ───────────────────────────────────────────────────────────

def _plot_heatmaps(df: pd.DataFrame,
                   env: ComplementaryPricingEnvironment,
                   demands: np.ndarray,
                   opt_action_real: np.ndarray,
                   base: str):
    df = _df_learning_only(df)
    """
    5x5 revenue heatmaps (discrete margin grid) with price trajectories.

    For each leader-follower pair: contourf of E[profit] as a function of
    (margin_leader, margin_follower), **all other coordinates fixed at the global
    optimum** ``opt_action_real``.

    **Important:** the scatter path plots only the pair ``(p_L, p_F)`` taken from the
    **full** 6-D policy trajectory. The background assumes the *other* four margins are
    already at ``p^*``. If the policy is wrong on those coordinates, the point can lie
    visually near ``p^*`` in the panel while **global** ``simple_regret = f(p^*) - f(p)``
    stays large — this is not a bug in the regret series. Use ``best_simple_regret.pdf``
    to see whether the joint learner ever hits a near-optimal **full** price vector.
    Trajectory from seed=0 at T_max coloured by round.

    ★ = joint optimum   ◆ = Univariate convergence target
    """
    T_max        = int(df["T"].max())
    algs_to_plot = _heatmap_columns_from_df(df)
    pairs        = [(ldr, flw[0]) for ldr, flw in sorted(GRAPH_DICT.items())]

    # Univariate convergence target: leader at standalone opt, follower at best-given-leader
    uni_conv = univariate_action(
        demands,
        GRAPH_DICT,
        N_PRODUCTS,
        N_ACTIONS,
        MARGIN_VALS,
        ALPHA,
        dependency_strength=float(getattr(env, "dependency_strength", 1.0)),
    )

    fig, axes = plt.subplots(
        len(pairs), len(algs_to_plot),
        figsize=(5.5 * len(algs_to_plot), 4.5 * len(pairs)),
        squeeze=False,
    )

    for row, (leader, follower) in enumerate(pairs):
        # 5x5 profit surface
        hmap = np.zeros((N_ACTIONS, N_ACTIONS))
        for i, ml in enumerate(MARGIN_VALS):
            for j, mf in enumerate(MARGIN_VALS):
                a           = opt_action_real.copy()
                a[leader]   = ml
                a[follower] = mf
                hmap[i, j]  = env.compute_givenaction_value(a)

        # Joint optimum on the 5x5 grid
        oi, oj = np.unravel_index(hmap.argmax(), hmap.shape)
        opt_ml = MARGIN_VALS[oi]
        opt_mf = MARGIN_VALS[oj]

        for col, alg in enumerate(algs_to_plot):
            ax = axes[row, col]

            im = ax.contourf(MARGIN_VALS, MARGIN_VALS, hmap.T,
                             levels=20, cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=r"$\mathbb{E}[\mathrm{profit}]$")

            # Price trajectory (seed=0, T_max)
            sub = (df[(df["T"] == T_max) & (df["seed"] == 0)
                      & (df["algorithm"] == alg)]
                   .sort_values("t"))
            if not sub.empty:
                xs = norm_to_real(sub[f"a{leader}"].values)
                ys = norm_to_real(sub[f"a{follower}"].values)
                rng_j = np.random.default_rng(row * 17 + col + 42)
                jit   = 0.02
                xs_j  = xs + rng_j.uniform(-jit, jit, len(xs))
                ys_j  = ys + rng_j.uniform(-jit, jit, len(ys))
                sc = ax.scatter(xs_j, ys_j,
                                c=np.arange(len(xs)), cmap="YlOrRd",
                                s=8, alpha=0.7, linewidths=0, zorder=3)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                             label="Round $t$")

            ax.scatter([opt_ml], [opt_mf], marker="*", s=300,
                       c="lime", zorder=6, edgecolors="k", lw=0.7,
                       label=r"$p^*$ joint")
            ax.scatter([uni_conv[leader]], [uni_conv[follower]], marker="D", s=110,
                       c="red", zorder=6, edgecolors="k", lw=0.7,
                       label=r"$p_{\rm uni}$ target")
            ax.set_xlim(MARGIN_VALS[0] - 0.05, MARGIN_VALS[-1] + 0.05)
            ax.set_ylim(MARGIN_VALS[0] - 0.05, MARGIN_VALS[-1] + 0.05)
            ax.set_xlabel(rf"$p_{{{leader}}}$ (leader)", fontsize=10)
            ax.set_ylabel(rf"$p_{{{follower}}}$ (follower)", fontsize=10)
            ax.set_title(
                f"{alg} — pair ({leader},{follower}), $T={T_max}$, seed 0",
                fontsize=9,
            )
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    path = f"{base}/heatmaps.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


def _print_summary(df):
    learn_m = (
        df["phase"].fillna("learning") == "learning"
        if "phase" in df.columns
        else pd.Series(True, index=df.index)
    )
    last = df.loc[learn_m & (df["t"] == df["T"])]
    src = simple_regret_plot_source()
    has_rec = "best_recommendation_simple_regret" in last.columns
    print("\n=== Played-action cumulative pseudo-regret (theorem) — last round mean ± std ===")
    for (alg, T), grp in last.groupby(["algorithm", "T"]):
        m = grp["regret_cumulative"].mean()
        s = grp["regret_cumulative"].std()
        bp = grp["best_simple_regret"].mean() if "best_simple_regret" in grp else float("nan")

        extra = ""
        if alg == PROPOSED_JOINT_ALG and has_rec and "recommendation_simple_regret" in grp.columns:
            rs = grp["recommendation_simple_regret"].mean()
            br = grp["best_recommendation_simple_regret"].mean()
            extra = (
                f"   | rec_sr(diagnostic)={rs:.4f}  best_rec_sr(diagnostic)={br:.4f}"
            )

        if src == "both" and has_rec:
            br = grp["best_recommendation_simple_regret"].mean()
            print(
                f"  {alg:22s}  T={T:5.0f}   R_played={m:.3f} ± {s:.3f}   "
                f"best_sr_played={bp:.4f}   best_sr_rec={br:.4f}"
            )
        elif src == "recommendation" and has_rec:
            br = grp["best_recommendation_simple_regret"].mean()
            print(
                f"  {alg:22s}  T={T:5.0f}   R_played={m:.3f} ± {s:.3f}   "
                f"best_sr_rec(plot)={br:.4f}   (instantaneous plots use recommendation mode)"
            )
        else:
            print(
                f"  {alg:22s}  T={T:5.0f}   R_played={m:.3f} ± {s:.3f}   "
                f"best_sr_played={bp:.4f}{extra}"
            )

    if "phase" in df.columns:
        fc = df[(df["phase"] == "final_commit") & (df["algorithm"] == PROPOSED_JOINT_ALG)]
        if not fc.empty and "final_commit_simple_regret" in fc.columns:
            print("\n=== Final commit simple regret (deployment eval — not in theorem cumulative regret) ===")
            for T in sorted(fc["T"].unique()):
                sub = fc[fc["T"] == T]
                mm = float(sub["final_commit_simple_regret"].mean())
                ss = float(sub["final_commit_simple_regret"].std()) if len(sub) > 1 else 0.0
                print(f"  {PROPOSED_JOINT_ALG}  T={T:5.0f}   final_commit_sr={mm:.4f} ± {ss:.4f}")


def _plot_regret_per_pair(
    df: pd.DataFrame,
    env: ComplementaryPricingEnvironment,
    opt_action_real: np.ndarray,
    base: str,
    n_seeds: int,
) -> None:
    """
    Per-pair cumulative regret — two separate PDFs.

    For each leader-follower pair (L, F) and each round t:
        pair_regret_t = J(p*) - J(ã_t)
    where ã_t equals p* on all coordinates except L and F, which take the
    algorithm's played values.

    PDF 1 (regret_pair_solo.pdf): single panel for the first pair.
    PDF 2 (regret_pairs_combined.pdf): single panel whose y-axis is the
        element-wise sum of the instantaneous pair regrets for the remaining
        two pairs, then cumulated — i.e. the combined contribution of those
        pairs as if they were one.
    """
    df = _df_learning_only(df)
    T_max = int(df["T"].max())
    sub_all = df[df["T"] == T_max].copy()

    pairs = [(ldr, flw[0]) for ldr, flw in sorted(GRAPH_DICT.items())]
    algs  = _ordered_algorithms_in_df(df)
    opt   = float(env.compute_best_action_value())

    # Pre-compute 5×5 value tables for every pair
    val_tables: dict[tuple[int, int], np.ndarray] = {}
    for leader, follower in pairs:
        vt = np.zeros((N_ACTIONS, N_ACTIONS))
        for i, ml in enumerate(MARGIN_VALS):
            for j, mf in enumerate(MARGIN_VALS):
                a = opt_action_real.copy()
                a[leader]   = ml
                a[follower] = mf
                vt[i, j] = env.compute_givenaction_value(a)
        val_tables[(leader, follower)] = vt

    def _seed_curves(alg_df: pd.DataFrame, leader: int, follower: int) -> list[np.ndarray]:
        curves: list[np.ndarray] = []
        vt = val_tables[(leader, follower)]
        for _, seed_df in alg_df.groupby("seed"):
            seed_df = seed_df.sort_values("t")
            idx_L = np.round(
                seed_df[f"a{leader}"].values * (N_ACTIONS - 1)
            ).astype(int).clip(0, N_ACTIONS - 1)
            idx_F = np.round(
                seed_df[f"a{follower}"].values * (N_ACTIONS - 1)
            ).astype(int).clip(0, N_ACTIONS - 1)
            curves.append(np.cumsum(opt - vt[idx_L, idx_F]))
        return curves

    def _draw_algs(ax, alg_seed_curves: dict[str, list[np.ndarray]]) -> None:
        for alg in algs:
            sc = alg_seed_curves.get(alg)
            if not sc:
                continue
            curves = np.array(sc)
            ts   = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(axis=0)
            std  = curves.std(axis=0)
            n    = curves.shape[0]
            ci   = scipy_stats.t.ppf(0.975, df=max(n - 1, 1)) * std / np.sqrt(n)
            ax.plot(ts, mean, label=alg, color=COLORS[alg], marker=MARKERS[alg],
                    markevery=max(len(ts) // 8, 1), markersize=5, lw=1.8)
            ax.fill_between(ts, mean - ci, mean + ci, alpha=0.15, color=COLORS[alg])

    # ── PDF 1: first pair alone ───────────────────────────────────────────────
    leader0, follower0 = pairs[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _draw_algs(ax, {alg: _seed_curves(sub_all[sub_all["algorithm"] == alg], leader0, follower0)
                    for alg in algs})
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Pair cumulative regret")
    ax.set_title(
        rf"Pair ($p_{{{leader0}}}$ leader, $p_{{{follower0}}}$ follower) — "
        rf"others fixed at $p^*$,  $T={T_max}$,  {n_seeds} seeds"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path1 = f"{base}/regret_pair_solo.pdf"
    fig.savefig(path1, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path1}")

    # ── PDF 2: combined regret of remaining pairs ─────────────────────────────
    combined_pairs = pairs[1:]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    combined_curves: dict[str, list[np.ndarray]] = {}
    for alg in algs:
        alg_df = sub_all[sub_all["algorithm"] == alg]
        per_pair = [_seed_curves(alg_df, L, F) for L, F in combined_pairs]
        if not per_pair or not per_pair[0]:
            continue
        n_s = len(per_pair[0])
        combined_curves[alg] = [
            sum(per_pair[p][s] for p in range(len(combined_pairs)))
            for s in range(n_s)
        ]
    _draw_algs(ax, combined_curves)
    pair_labels = " + ".join(
        rf"($p_{{{L}}}$,$p_{{{F}}}$)" for L, F in combined_pairs
    )
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Combined pair cumulative regret")
    ax.set_title(
        rf"Combined pairs {pair_labels} — others fixed at $p^*$,  $T={T_max}$,  {n_seeds} seeds"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path2 = f"{base}/regret_pairs_combined.pdf"
    fig.savefig(path2, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path2}")
