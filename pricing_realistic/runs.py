"""Single-run loops (run_one, two-joint, BPE-only trace)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from environments.complementary import ComplementaryPricingEnvironment

from .config import (
    ALL_ACTIONS_NORM,
    ALL_ACTIONS_REAL,
    ALPHA,
    EXP_CFG,
    BanditExperimentConfig,
    ISO_MATERN_BPE_ALG,
    ISO_MATERN_GP_UCB_ALG,
    JOINT_CONT_GP_UCB_ALG,
    NOISE_VAR,
    NOISE_VAR_MARG,
    N_ACTIONS,
    N_PRODUCTS,
    PROPOSED_JOINT_ALG,
    ALL_ALGS,
    best_independent_product_surrogate,
    validate_full_cartesian_x0,
    recommendation_csv_keys,
)
from .rewards import (
    mean_joint_reward_replicated,
    norm_to_real,
    snap_to_grid,
)
from .config import effective_gp_noise_variance
from .agents.dmsx0_bpe import DMSX0BPE, li_22_bpe_batch_sizes
from .agents.dmsx0_bpe_iso import IsoMatern12BPE
from .agents.dmsgp_ucb import DMSGPUCB, IsotropicMatern12GPUCB
from .agents.kleinberg_ucb import KleinbergUCB
from .agents.univariate import UnivariateBaseline
from .agents.hgp_cpp import HGP_UCB_CPP_Wrapper
from .agents.bz_etc import BZ_ETC
from .agents.spsa import SPSAPricing


def _spsa_a_par_for_run() -> float:
    """
    ``EXP_SPSA_A_PAR`` overrides the SPSA gain scale ``a_par`` (SimpleSPSA-style).

    When unset and ``EXP_SMOKE`` is on, use a larger default than ``1e-6`` so short
    smoke runs still move ``x`` enough for ``snap_to_grid`` to hit different vertices.
    """
    raw = os.environ.get("EXP_SPSA_A_PAR", "").strip()
    if raw:
        return float(raw)
    if os.environ.get("EXP_SMOKE", "").lower() in ("1", "true", "yes"):
        return 0.08
    return 1e-6


def _with_played_metric_aliases(row: Dict[str, Any]) -> None:
    """
    Explicit aliases for **theorem / paper** cumulative pseudo-regret on **played** actions.

    ``revenue``, ``simple_regret``, ``regret_cumulative`` remain the canonical column names for
    backward compatibility; ``played_*`` duplicates document that these are pull-based, not
    recommendation-based.
    """
    row["played_revenue"] = float(row["revenue"])
    row["played_simple_regret"] = float(row["simple_regret"])
    row["played_regret_cumulative"] = float(row["regret_cumulative"])


def _final_commit_nan_payload() -> Dict[str, float]:
    """Placeholder NaNs on learning rows for final-commit CSV columns."""
    out = {
        "final_commit_revenue": float("nan"),
        "final_commit_simple_regret": float("nan"),
        "final_commit_idx": float("nan"),
    }
    out.update({f"final_commit_a{i}": float("nan") for i in range(N_PRODUCTS)})
    return out


def _recommendation_csv_row(
    env: ComplementaryPricingEnvironment,
    alg: str,
    dms_x0: Optional[DMSX0BPE],
    opt: float,
    played_norm: np.ndarray,
    revenue_played: float,
) -> Dict[str, float]:
    """
    Per-round CSV fields for **recommendation / estimator** diagnostics vs **played** pull.

    For **DMS-X0-BPE**, ``recommendation_*`` come from ``recommend_idx()`` (best current BPE
    recommendation on ``X°``). These are useful for **deployment / diagnostics** and are **not**
    the cumulative pseudo-regret sum ``sum_t J^* - J(p_t)`` from the theorem — that uses only
    played actions (see ``regret_cumulative``, ``simple_regret``, and ``played_*`` aliases).

    Non-BPE baselines: recommendation columns **mirror** the played arm (same as theorem regret).
    """
    if alg == PROPOSED_JOINT_ALG:
        if dms_x0 is None:
            raise ValueError("dms_x0 is required for BPE recommendation columns")
        ridx = int(dms_x0.recommend_idx())
        rec_norm = dms_x0.actions[ridx].copy()
        m_rec = norm_to_real(snap_to_grid(rec_norm))
        rev_r = float(env.compute_givenaction_value(m_rec))
        row: Dict[str, float] = {
            "recommendation_revenue": rev_r,
            "recommendation_simple_regret": float(opt - rev_r),
        }
        for i in range(N_PRODUCTS):
            row[f"recommendation_a{i}"] = float(rec_norm[i])
        return row
    row = {
        "recommendation_revenue": float(revenue_played),
        "recommendation_simple_regret": float(opt - revenue_played),
    }
    for i in range(N_PRODUCTS):
        row[f"recommendation_a{i}"] = float(played_norm[i])
    return row


def _bpe_round_scalar_diag(
    env: ComplementaryPricingEnvironment,
    dms_x0: DMSX0BPE,
    opt: float,
    opt_idx: int,
) -> Dict[str, float]:
    """Diagnostics after a ``DMSX0BPE`` ``update`` (grid optimum vs active set / recommend)."""
    seen = set(dms_x0._all_train_idx) | set(dms_x0._batch_intr_idx)
    ridx = int(dms_x0.recommend_idx())
    rec_n = dms_x0.actions[ridx].copy()
    mrec = norm_to_real(snap_to_grid(rec_n))
    vrec = float(env.compute_givenaction_value(mrec))
    return {
        "bpe_opt_still_active": float(bool(dms_x0._active_mask[opt_idx])),
        "bpe_opt_ever_sampled": float(opt_idx in seen),
        "bpe_n_survivors": float(int(dms_x0._active_mask.sum())),
        "bpe_recommend_idx": float(ridx),
        "bpe_recommendation_revenue": vrec,
        "bpe_recommendation_simple_regret": float(opt - vrec),
    }


def run_one(
    T      : int,
    env    : ComplementaryPricingEnvironment,
    seed   : int,
    c_beta : float,
    delta  : float,
    B_rkhs : float = 1.0,
) -> list:
    env.reset(seed)
    opt = float(env.compute_best_action_value())
    opt_indep_prod = float(best_independent_product_surrogate(env))
    gap_joint_minus_indep = float(opt - opt_indep_prod)

    R_mc = max(1, int(EXP_CFG.joint_reward_mc_replications))
    k_l = float(EXP_CFG.dms_kernel_L)
    lam_gp = float(effective_gp_noise_variance(EXP_CFG))

    validate_full_cartesian_x0(ALL_ACTIONS_NORM, N_PRODUCTS, N_ACTIONS)

    dms_x0 = DMSX0BPE(
        ALL_ACTIONS_NORM,
        T=T,
        noise_var=lam_gp,
        delta=float(EXP_CFG.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 1,
        bpe_beta_use_active_count=bool(EXP_CFG.dms_bpe_beta_use_active_count),
        noise_R=EXP_CFG.dms_bpe_noise_R,
        bpe_use_global_history=bool(EXP_CFG.dms_bpe_use_global_history),
        within_batch_ucb_c=float(EXP_CFG.dms_bpe_within_batch_ucb_c),
    )
    kl  = KleinbergUCB(ALL_ACTIONS_NORM, seed=seed * 31 + 4)
    uni = UnivariateBaseline(
        N_PRODUCTS, NOISE_VAR_MARG,
        c_beta=c_beta, delta=delta, B_rkhs=B_rkhs,
        kernel_L=k_l, seed=seed * 31 + 100,
    )

    _li_sched_joint = li_22_bpe_batch_sizes(T)
    joint_gp_ucb = DMSGPUCB(
        N_PRODUCTS,
        noise_var=lam_gp,
        c_beta=c_beta,
        delta=float(EXP_CFG.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 3,
        use_batches=True,
        batch_base=2,
        batch_ratio=2.0,
        n_restarts=max(24, 12 * N_PRODUCTS),
        li_scarlett_batch_sizes=_li_sched_joint,
    )

    iso_gp_ucb = IsotropicMatern12GPUCB(
        N_PRODUCTS,
        noise_var=lam_gp,
        c_beta=c_beta,
        delta=float(EXP_CFG.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 5,
        use_batches=True,
        batch_base=2,
        batch_ratio=2.0,
        n_restarts=max(24, 12 * N_PRODUCTS),
        li_scarlett_batch_sizes=_li_sched_joint,
    )

    hgp = HGP_UCB_CPP_Wrapper(
        T=T, alpha=ALPHA, kernel_L=float(EXP_CFG.hgp_kernel_L), known_graph=False,
    )

    # BZ-ETC sub-grid: κ ≍ T^{d/(d+3)} avoids curse-of-dimensionality on 5^d arms
    _kappa  = min(int(T ** (N_PRODUCTS / (N_PRODUCTS + 3))), len(ALL_ACTIONS_NORM))
    _bz_idx = np.random.default_rng(seed * 31 + 6).choice(
        len(ALL_ACTIONS_NORM),
        size=max(_kappa, N_PRODUCTS + 1),
        replace=False,
    )
    bz = BZ_ETC(
        ALL_ACTIONS_NORM, T=T,
        candidate_actions=ALL_ACTIONS_NORM[_bz_idx],
        tau_factor=2.0,
        seed=seed * 31 + 16,
    )

    spsa = SPSAPricing(
        d=N_PRODUCTS,
        T=T,
        a_par=_spsa_a_par_for_run(),
        noise_var=float(NOISE_VAR),
        ens_size=2,
        seed=seed * 31 + 17,
    )

    iso_bpe = IsoMatern12BPE(
        ALL_ACTIONS_NORM,
        T=T,
        noise_var=lam_gp,
        delta=float(EXP_CFG.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 7,
        bpe_beta_use_active_count=bool(EXP_CFG.dms_bpe_beta_use_active_count),
        noise_R=EXP_CFG.dms_bpe_noise_R,
        bpe_use_global_history=bool(EXP_CFG.dms_bpe_use_global_history),
        within_batch_ucb_c=float(EXP_CFG.dms_bpe_within_batch_ucb_c),
    )

    records = []
    cum = {alg: 0.0 for alg in ALL_ALGS}

    for t in range(1, T + 1):

        # Paper method: BPE Alg. 1 on full X° (ε-cover grid) with k_Π + elimination
        a_dmsx  = dms_x0.pull(snap_fn=snap_to_grid)
        m_dmsx  = norm_to_real(a_dmsx)
        if EXP_CFG.oracle_feedback_for_debug:
            r_bpe = float(env.compute_givenaction_value(m_dmsx))
        else:
            r_bpe = float(mean_joint_reward_replicated(env, m_dmsx, R_mc))
        dms_x0.update(r_bpe)

        # Competitor: Joint GP-UCB on [0,1]^d; Li batch lengths only; snap to X°
        a_jgp = joint_gp_ucb.pull(snap_fn=snap_to_grid)
        m_jgp = norm_to_real(a_jgp)
        joint_gp_ucb.update(mean_joint_reward_replicated(env, m_jgp, R_mc))

        # Baseline: isotropic Matérn GP-UCB (same Li schedule / snap as Joint-DMS competitor)
        a_iso = iso_gp_ucb.pull(snap_fn=snap_to_grid)
        m_iso = norm_to_real(a_iso)
        iso_gp_ucb.update(mean_joint_reward_replicated(env, m_iso, R_mc))

        # HGP-UCB-CPP (unknown graph; update takes raw sales matrix)
        a_hgp   = hgp.pull()
        m_hgp   = norm_to_real(a_hgp)
        s_hgp   = env.step(m_hgp)
        hgp.update(s_hgp)

        # BZ-ETC
        a_bz    = bz.pull()
        m_bz    = norm_to_real(a_bz)
        bz.update(mean_joint_reward_replicated(env, m_bz, R_mc))

        # SPSA
        a_spsa  = spsa.pull()
        m_spsa  = norm_to_real(a_spsa)
        spsa.update(mean_joint_reward_replicated(env, m_spsa, R_mc))

        # Kleinberg
        a_kl    = kl.pull()
        m_kl    = norm_to_real(a_kl)
        kl.update(mean_joint_reward_replicated(env, m_kl, R_mc))

        # Univariate: per-product marginal reward signal
        a_uni   = uni.pull()
        m_uni   = norm_to_real(a_uni)
        s_uni   = env.step(m_uni)
        uni.update(m_uni * np.mean(s_uni, axis=1))

        # BPE-Iso-Mat12: ablation — same BPE, isotropic Matérn-1/2 kernel
        a_ibpe  = iso_bpe.pull(snap_fn=snap_to_grid)
        m_ibpe  = norm_to_real(a_ibpe)
        iso_bpe.update(float(mean_joint_reward_replicated(env, m_ibpe, R_mc)))

        true_vals = {
            PROPOSED_JOINT_ALG: env.compute_givenaction_value(m_dmsx),
            JOINT_CONT_GP_UCB_ALG: env.compute_givenaction_value(m_jgp),
            ISO_MATERN_GP_UCB_ALG: env.compute_givenaction_value(m_iso),
            ISO_MATERN_BPE_ALG : env.compute_givenaction_value(m_ibpe),
            "HGP-UCB-CPP"      : env.compute_givenaction_value(m_hgp),
            "BZ-ETC"           : env.compute_givenaction_value(m_bz),
            "SPSA"             : env.compute_givenaction_value(m_spsa),
            "Kleinberg"        : env.compute_givenaction_value(m_kl),
            "Univariate"       : env.compute_givenaction_value(m_uni),
        }
        actions_norm = {
            PROPOSED_JOINT_ALG: a_dmsx,
            JOINT_CONT_GP_UCB_ALG: a_jgp,
            ISO_MATERN_GP_UCB_ALG: a_iso,
            ISO_MATERN_BPE_ALG : a_ibpe,
            "HGP-UCB-CPP"      : a_hgp,
            "BZ-ETC"           : a_bz,
            "SPSA"             : a_spsa,
            "Kleinberg"        : a_kl,
            "Univariate"       : a_uni,
        }

        for alg in ALL_ALGS:
            cum[alg] += opt - true_vals[alg]
            row = dict(
                seed=seed, t=t, T=T, algorithm=alg,
                revenue=true_vals[alg],
                regret_cumulative=cum[alg],
                simple_regret=opt - true_vals[alg],
                opt_joint_oracle=opt,
                opt_indep_product_surrogate=opt_indep_prod,
                gap_joint_minus_indep_product=gap_joint_minus_indep,
            )
            for i, v in enumerate(actions_norm[alg]):
                row[f"a{i}"] = float(v)
            row.update(
                _recommendation_csv_row(
                    env,
                    alg,
                    dms_x0,
                    opt,
                    actions_norm[alg],
                    float(true_vals[alg]),
                )
            )
            _with_played_metric_aliases(row)
            records.append(row)

    n_act = int(dms_x0._active_mask.sum())
    if seed == 0:
        print(
            f"    [{PROPOSED_JOINT_ALG}] |survivors|={n_act}  M_total={dms_x0._M_total}  "
            f"λ_BPE(eff)={lam_gp}  batches_done={dms_x0._batch_idx}  "
            f"kernel_L={EXP_CFG.dms_kernel_L}"
        )
        print(f"    [SPSA final x] {np.round(spsa.x, 3)}")
        print(f"    [BZ-ETC  tau ] {bz.tau}  committed={bz._best is not None}")
    return records


def run_one_single_algorithm(
    algorithm: str,
    T      : int,
    env    : ComplementaryPricingEnvironment,
    seed   : int,
    c_beta : float,
    delta  : float,
    B_rkhs : float = 1.0,
) -> list:
    """
    Same per-round update rules as the corresponding branch inside ``run_one``, but only one
    agent class is constructed.

    Trajectories need not match the rows of ``run_one`` for the same algorithm: the full
    experiment runs every policy each round (different number of ``env.step`` calls and possible
    global RNG side effects from other constructors). For apples-to-apples benchmarking against
    ``run_one``, use ``run_one`` and filter rows by ``algorithm``.
    """
    if algorithm not in ALL_ALGS:
        raise ValueError(f"algorithm must be one of {ALL_ALGS}, got {algorithm!r}")

    env.reset(seed)
    opt = float(env.compute_best_action_value())
    opt_indep_prod = float(best_independent_product_surrogate(env))
    gap_joint_minus_indep = float(opt - opt_indep_prod)

    R_mc = max(1, int(EXP_CFG.joint_reward_mc_replications))
    k_l = float(EXP_CFG.dms_kernel_L)
    lam_gp = float(effective_gp_noise_variance(EXP_CFG))

    dms_x0: Optional[DMSX0BPE] = None
    joint_gp_ucb: Optional[DMSGPUCB] = None
    iso_gp_ucb: Optional[IsotropicMatern12GPUCB] = None
    iso_bpe: Optional[IsoMatern12BPE] = None
    hgp: Optional[HGP_UCB_CPP_Wrapper] = None
    bz: Optional[BZ_ETC] = None
    spsa: Optional[SPSAPricing] = None
    kl: Optional[KleinbergUCB] = None
    uni: Optional[UnivariateBaseline] = None

    if algorithm == PROPOSED_JOINT_ALG:
        validate_full_cartesian_x0(ALL_ACTIONS_NORM, N_PRODUCTS, N_ACTIONS)
        dms_x0 = DMSX0BPE(
            ALL_ACTIONS_NORM,
            T=T,
            noise_var=lam_gp,
            delta=float(EXP_CFG.dms_elimination_delta),
            kernel_L=k_l,
            B_rkhs=float(B_rkhs),
            seed=seed * 31 + 1,
            bpe_beta_use_active_count=bool(EXP_CFG.dms_bpe_beta_use_active_count),
            noise_R=EXP_CFG.dms_bpe_noise_R,
            bpe_use_global_history=bool(EXP_CFG.dms_bpe_use_global_history),
            within_batch_ucb_c=float(EXP_CFG.dms_bpe_within_batch_ucb_c),
        )
    elif algorithm == JOINT_CONT_GP_UCB_ALG:
        _li_sched_joint = li_22_bpe_batch_sizes(T)
        joint_gp_ucb = DMSGPUCB(
            N_PRODUCTS,
            noise_var=lam_gp,
            c_beta=c_beta,
            delta=float(EXP_CFG.dms_elimination_delta),
            kernel_L=k_l,
            B_rkhs=float(B_rkhs),
            seed=seed * 31 + 3,
            use_batches=True,
            batch_base=2,
            batch_ratio=2.0,
            n_restarts=max(24, 12 * N_PRODUCTS),
            li_scarlett_batch_sizes=_li_sched_joint,
        )
    elif algorithm == ISO_MATERN_GP_UCB_ALG:
        _li_sched_joint = li_22_bpe_batch_sizes(T)
        iso_gp_ucb = IsotropicMatern12GPUCB(
            N_PRODUCTS,
            noise_var=lam_gp,
            c_beta=c_beta,
            delta=float(EXP_CFG.dms_elimination_delta),
            kernel_L=k_l,
            B_rkhs=float(B_rkhs),
            seed=seed * 31 + 5,
            use_batches=True,
            batch_base=2,
            batch_ratio=2.0,
            n_restarts=max(24, 12 * N_PRODUCTS),
            li_scarlett_batch_sizes=_li_sched_joint,
        )
    elif algorithm == ISO_MATERN_BPE_ALG:
        validate_full_cartesian_x0(ALL_ACTIONS_NORM, N_PRODUCTS, N_ACTIONS)
        iso_bpe = IsoMatern12BPE(
            ALL_ACTIONS_NORM,
            T=T,
            noise_var=lam_gp,
            delta=float(EXP_CFG.dms_elimination_delta),
            kernel_L=k_l,
            B_rkhs=float(B_rkhs),
            seed=seed * 31 + 7,
            bpe_beta_use_active_count=bool(EXP_CFG.dms_bpe_beta_use_active_count),
            noise_R=EXP_CFG.dms_bpe_noise_R,
            bpe_use_global_history=bool(EXP_CFG.dms_bpe_use_global_history),
            within_batch_ucb_c=float(EXP_CFG.dms_bpe_within_batch_ucb_c),
        )
    elif algorithm == "HGP-UCB-CPP":
        hgp = HGP_UCB_CPP_Wrapper(
            T=T, alpha=ALPHA, kernel_L=float(EXP_CFG.hgp_kernel_L), known_graph=False,
        )
    elif algorithm == "BZ-ETC":
        _kappa = min(int(T ** (N_PRODUCTS / (N_PRODUCTS + 3))), len(ALL_ACTIONS_NORM))
        _bz_idx = np.random.default_rng(seed * 31 + 6).choice(
            len(ALL_ACTIONS_NORM),
            size=max(_kappa, N_PRODUCTS + 1),
            replace=False,
        )
        bz = BZ_ETC(
            ALL_ACTIONS_NORM, T=T,
            candidate_actions=ALL_ACTIONS_NORM[_bz_idx],
            tau_factor=2.0,
            seed=seed * 31 + 16,
        )
    elif algorithm == "SPSA":
        spsa = SPSAPricing(
            d=N_PRODUCTS,
            T=T,
            a_par=_spsa_a_par_for_run(),
            noise_var=float(NOISE_VAR),
            ens_size=2,
            seed=seed * 31 + 17,
        )
    elif algorithm == "Kleinberg":
        kl = KleinbergUCB(ALL_ACTIONS_NORM, seed=seed * 31 + 4)
    elif algorithm == "Univariate":
        uni = UnivariateBaseline(
            N_PRODUCTS, NOISE_VAR_MARG,
            c_beta=c_beta, delta=delta, B_rkhs=B_rkhs,
            kernel_L=k_l, seed=seed * 31 + 100,
        )
    else:
        raise AssertionError(f"unhandled algorithm {algorithm}")

    records = []
    cum = 0.0

    for t in range(1, T + 1):
        if algorithm == PROPOSED_JOINT_ALG:
            assert dms_x0 is not None
            a_play = dms_x0.pull(snap_fn=snap_to_grid)
            m_play = norm_to_real(a_play)
            if EXP_CFG.oracle_feedback_for_debug:
                r_step = float(env.compute_givenaction_value(m_play))
            else:
                r_step = float(mean_joint_reward_replicated(env, m_play, R_mc))
            dms_x0.update(r_step)
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == JOINT_CONT_GP_UCB_ALG:
            assert joint_gp_ucb is not None
            a_play = joint_gp_ucb.pull(snap_fn=snap_to_grid)
            m_play = norm_to_real(a_play)
            joint_gp_ucb.update(mean_joint_reward_replicated(env, m_play, R_mc))
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == ISO_MATERN_GP_UCB_ALG:
            assert iso_gp_ucb is not None
            a_play = iso_gp_ucb.pull(snap_fn=snap_to_grid)
            m_play = norm_to_real(a_play)
            iso_gp_ucb.update(mean_joint_reward_replicated(env, m_play, R_mc))
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == ISO_MATERN_BPE_ALG:
            assert iso_bpe is not None
            a_play = iso_bpe.pull(snap_fn=snap_to_grid)
            m_play = norm_to_real(a_play)
            r_step = float(mean_joint_reward_replicated(env, m_play, R_mc))
            iso_bpe.update(r_step)
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == "HGP-UCB-CPP":
            assert hgp is not None
            a_play = hgp.pull()
            m_play = norm_to_real(a_play)
            s_hgp = env.step(m_play)
            hgp.update(s_hgp)
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == "BZ-ETC":
            assert bz is not None
            a_play = bz.pull()
            m_play = norm_to_real(a_play)
            bz.update(mean_joint_reward_replicated(env, m_play, R_mc))
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == "SPSA":
            assert spsa is not None
            a_play = spsa.pull()
            m_play = norm_to_real(a_play)
            spsa.update(mean_joint_reward_replicated(env, m_play, R_mc))
            true_val = float(env.compute_givenaction_value(m_play))
        elif algorithm == "Kleinberg":
            assert kl is not None
            a_play = kl.pull()
            m_play = norm_to_real(a_play)
            kl.update(mean_joint_reward_replicated(env, m_play, R_mc))
            true_val = float(env.compute_givenaction_value(m_play))
        else:
            assert uni is not None
            a_play = uni.pull()
            m_play = norm_to_real(a_play)
            s_uni = env.step(m_play)
            uni.update(m_play * np.mean(s_uni, axis=1))
            true_val = float(env.compute_givenaction_value(m_play))

        cum += opt - true_val
        row = dict(
            seed=seed, t=t, T=T, algorithm=algorithm,
            revenue=true_val,
            regret_cumulative=cum,
            simple_regret=opt - true_val,
            opt_joint_oracle=opt,
            opt_indep_product_surrogate=opt_indep_prod,
            gap_joint_minus_indep_product=gap_joint_minus_indep,
            phase="learning",
            is_final_commit=False,
        )
        row.update(_final_commit_nan_payload())
        for i, v in enumerate(a_play):
            row[f"a{i}"] = float(v)
        row.update(
            _recommendation_csv_row(
                env,
                algorithm,
                dms_x0,
                opt,
                a_play,
                true_val,
            )
        )
        _with_played_metric_aliases(row)
        records.append(row)

    final_commit_eval = os.environ.get("DMS_FINAL_COMMIT", "").lower() in ("1", "true", "yes")
    if algorithm == PROPOSED_JOINT_ALG and dms_x0 is not None and final_commit_eval:
        a_commit = dms_x0.recommend()
        idx_c = int(dms_x0.recommend_idx())
        m_commit = norm_to_real(snap_to_grid(a_commit))
        v_commit = float(env.compute_givenaction_value(m_commit))
        sr_commit = float(opt - v_commit)
        fc_row: Dict[str, Any] = dict(
            seed=seed,
            t=T + 1,
            T=T,
            algorithm=algorithm,
            phase="final_commit",
            is_final_commit=True,
            revenue=float("nan"),
            regret_cumulative=float(cum),
            simple_regret=float("nan"),
            opt_joint_oracle=opt,
            opt_indep_product_surrogate=opt_indep_prod,
            gap_joint_minus_indep_product=gap_joint_minus_indep,
            played_revenue=float("nan"),
            played_simple_regret=float("nan"),
            played_regret_cumulative=float(cum),
        )
        for i in range(N_PRODUCTS):
            fc_row[f"a{i}"] = float("nan")
        fc_row.update({k: float("nan") for k in recommendation_csv_keys()})
        fc_row["final_commit_revenue"] = v_commit
        fc_row["final_commit_simple_regret"] = sr_commit
        fc_row["final_commit_idx"] = float(idx_c)
        for i in range(N_PRODUCTS):
            fc_row[f"final_commit_a{i}"] = float(a_commit[i])
        records.append(fc_row)

    if seed == 0:
        if algorithm == PROPOSED_JOINT_ALG and dms_x0 is not None:
            n_act = int(dms_x0._active_mask.sum())
            print(
                f"    [{PROPOSED_JOINT_ALG}] |survivors|={n_act}  M_total={dms_x0._M_total}  "
                f"λ_BPE(eff)={lam_gp}  batches_done={dms_x0._batch_idx}  "
                f"kernel_L={EXP_CFG.dms_kernel_L}"
            )
        if algorithm == "SPSA" and spsa is not None:
            print(f"    [SPSA final x] {np.round(spsa.x, 3)}")
        if algorithm == "BZ-ETC" and bz is not None:
            print(f"    [BZ-ETC  tau ] {bz.tau}  committed={bz._best is not None}")

    return records


def run_one_two_joint(
    T      : int,
    env    : ComplementaryPricingEnvironment,
    seed   : int,
    c_beta : float = 0.5,
    delta  : float = 0.1,
    B_rkhs : float = 1.0,
    exp_cfg: Optional[BanditExperimentConfig] = None,
) -> Tuple[list, dict]:
    """
    Same environment and reward model as ``run_one``, but runs **only**:

    * ``DMS-X0-BPE`` — **paper** path: full ``X°`` cover, ``k_Π``, Li–Scarlett batches + elimination.
    * ``Joint-DMS-GP-UCB`` — **baseline / ablation**: continuous GP-UCB acquisition on ``[0,1]^d``,
      then snap to ``X°`` for play and reward (not the finite-catalog BPE algorithm).

    Returns ``(records, diag)`` where ``records`` are CSV-style rows with
    ``algorithm`` in ``{PROPOSED_JOINT_ALG, JOINT_CONT_GP_UCB_ALG}``, and
    ``diag`` contains BPE diagnostics including per-round traces for survivors, grid-optimum
    status, and recommendation simple regret.
    """
    cfg = exp_cfg if exp_cfg is not None else EXP_CFG
    env.reset(seed)
    opt = float(env.compute_best_action_value())
    opt_indep_prod = float(best_independent_product_surrogate(env))
    gap_joint_minus_indep = float(opt - opt_indep_prod)

    R_mc = max(1, int(cfg.joint_reward_mc_replications))
    k_l = float(cfg.dms_kernel_L)
    lam_gp = float(effective_gp_noise_variance(cfg))

    validate_full_cartesian_x0(ALL_ACTIONS_NORM, N_PRODUCTS, N_ACTIONS)

    dms_x0 = DMSX0BPE(
        ALL_ACTIONS_NORM,
        T=T,
        noise_var=lam_gp,
        delta=float(cfg.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 1,
        bpe_beta_use_active_count=bool(cfg.dms_bpe_beta_use_active_count),
        noise_R=cfg.dms_bpe_noise_R,
        bpe_use_global_history=bool(cfg.dms_bpe_use_global_history),
        within_batch_ucb_c=float(cfg.dms_bpe_within_batch_ucb_c),
    )
    _li_sched_joint = li_22_bpe_batch_sizes(T)
    joint_gp_ucb = DMSGPUCB(
        N_PRODUCTS,
        noise_var=lam_gp,
        c_beta=c_beta,
        delta=float(cfg.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 3,
        use_batches=True,
        batch_base=2,
        batch_ratio=2.0,
        n_restarts=max(24, 12 * N_PRODUCTS),
        li_scarlett_batch_sizes=_li_sched_joint,
    )

    all_vals = np.array(
        [float(env.compute_givenaction_value(a)) for a in ALL_ACTIONS_REAL],
        dtype=float,
    )
    opt_idx = int(np.argmax(all_vals))

    two = (PROPOSED_JOINT_ALG, JOINT_CONT_GP_UCB_ALG)
    records: list = []
    cum = {alg: 0.0 for alg in two}
    surv_trace: list[int] = []
    bpe_opt_still_active_per_round: list[float] = []
    bpe_opt_ever_sampled_per_round: list[float] = []
    bpe_n_survivors_per_round: list[float] = []
    bpe_recommend_idx_per_round: list[float] = []
    bpe_recommendation_revenue_per_round: list[float] = []
    bpe_recommendation_simple_regret_per_round: list[float] = []

    for t in range(1, T + 1):
        a_dmsx = dms_x0.pull(snap_fn=snap_to_grid)
        m_dmsx = norm_to_real(a_dmsx)
        if cfg.oracle_feedback_for_debug:
            r_bpe = float(env.compute_givenaction_value(m_dmsx))
        else:
            r_bpe = float(mean_joint_reward_replicated(env, m_dmsx, R_mc))
        dms_x0.update(r_bpe)

        a_jgp = joint_gp_ucb.pull(snap_fn=snap_to_grid)
        m_jgp = norm_to_real(a_jgp)
        joint_gp_ucb.update(mean_joint_reward_replicated(env, m_jgp, R_mc))

        surv_trace.append(int(dms_x0._active_mask.sum()))
        bpe_snap = _bpe_round_scalar_diag(env, dms_x0, opt, opt_idx)
        bpe_opt_still_active_per_round.append(bpe_snap["bpe_opt_still_active"])
        bpe_opt_ever_sampled_per_round.append(bpe_snap["bpe_opt_ever_sampled"])
        bpe_n_survivors_per_round.append(bpe_snap["bpe_n_survivors"])
        bpe_recommend_idx_per_round.append(bpe_snap["bpe_recommend_idx"])
        bpe_recommendation_revenue_per_round.append(bpe_snap["bpe_recommendation_revenue"])
        bpe_recommendation_simple_regret_per_round.append(
            bpe_snap["bpe_recommendation_simple_regret"]
        )

        true_vals = {
            PROPOSED_JOINT_ALG: env.compute_givenaction_value(m_dmsx),
            JOINT_CONT_GP_UCB_ALG: env.compute_givenaction_value(m_jgp),
        }
        actions_norm = {
            PROPOSED_JOINT_ALG: a_dmsx,
            JOINT_CONT_GP_UCB_ALG: a_jgp,
        }

        nan_bpe = {k: float("nan") for k in bpe_snap}

        for alg in two:
            cum[alg] += opt - true_vals[alg]
            row = dict(
                seed=seed, t=t, T=T, algorithm=alg,
                revenue=true_vals[alg],
                regret_cumulative=cum[alg],
                simple_regret=opt - true_vals[alg],
                opt_joint_oracle=opt,
                opt_indep_product_surrogate=opt_indep_prod,
                gap_joint_minus_indep_product=gap_joint_minus_indep,
            )
            for i, v in enumerate(actions_norm[alg]):
                row[f"a{i}"] = float(v)
            row.update(
                _recommendation_csv_row(
                    env,
                    alg,
                    dms_x0,
                    opt,
                    actions_norm[alg],
                    float(true_vals[alg]),
                )
            )
            if alg == PROPOSED_JOINT_ALG:
                row.update(bpe_snap)
            else:
                row.update(nan_bpe)
            _with_played_metric_aliases(row)
            records.append(row)

    diag = dict(
        bpe_survivors_per_round=np.asarray(surv_trace, dtype=int),
        bpe_final_survivors=int(dms_x0._active_mask.sum()),
        bpe_committed_idx=dms_x0._committed_idx,
        bpe_batches_finished=int(dms_x0._batch_idx),
        n_arms=len(ALL_ACTIONS_NORM),
        opt_grid_idx=opt_idx,
        bpe_opt_still_active_per_round=np.asarray(bpe_opt_still_active_per_round, dtype=float),
        bpe_opt_ever_sampled_per_round=np.asarray(bpe_opt_ever_sampled_per_round, dtype=float),
        bpe_n_survivors_per_round=np.asarray(bpe_n_survivors_per_round, dtype=float),
        bpe_recommend_idx_per_round=np.asarray(bpe_recommend_idx_per_round, dtype=float),
        bpe_recommendation_revenue_per_round=np.asarray(
            bpe_recommendation_revenue_per_round, dtype=float
        ),
        bpe_recommendation_simple_regret_per_round=np.asarray(
            bpe_recommendation_simple_regret_per_round, dtype=float
        ),
    )
    return records, diag


def run_bpe_x0_only_trace(
    T      : int,
    env    : ComplementaryPricingEnvironment,
    seed   : int,
    B_rkhs : float = 1.0,
    exp_cfg: Optional[BanditExperimentConfig] = None,
) -> dict:
    """
    Run **only** ``DMSX0BPE`` (paper BPE on the **full** ε-cover ``X°``, not Smolyak) for ``T``
    rounds; return the same ``diag`` keys as ``run_one_two_joint`` (no CSV rows).

    Cheaper than ``run_one_two_joint`` at the same ``T`` (no Joint GP-UCB env loop).
    """
    cfg = exp_cfg if exp_cfg is not None else EXP_CFG
    env.reset(seed)
    R_mc = max(1, int(cfg.joint_reward_mc_replications))
    k_l = float(cfg.dms_kernel_L)
    lam_gp = float(effective_gp_noise_variance(cfg))
    opt = float(env.compute_best_action_value())
    all_vals = np.array(
        [float(env.compute_givenaction_value(a)) for a in ALL_ACTIONS_REAL],
        dtype=float,
    )
    opt_idx = int(np.argmax(all_vals))

    validate_full_cartesian_x0(ALL_ACTIONS_NORM, N_PRODUCTS, N_ACTIONS)

    dms_x0 = DMSX0BPE(
        ALL_ACTIONS_NORM,
        T=T,
        noise_var=lam_gp,
        delta=float(cfg.dms_elimination_delta),
        kernel_L=k_l,
        B_rkhs=float(B_rkhs),
        seed=seed * 31 + 1,
        bpe_beta_use_active_count=bool(cfg.dms_bpe_beta_use_active_count),
        noise_R=cfg.dms_bpe_noise_R,
        bpe_use_global_history=bool(cfg.dms_bpe_use_global_history),
        within_batch_ucb_c=float(cfg.dms_bpe_within_batch_ucb_c),
    )
    surv_trace: list[int] = []
    bpe_opt_still_active_per_round: list[float] = []
    bpe_opt_ever_sampled_per_round: list[float] = []
    bpe_n_survivors_per_round: list[float] = []
    bpe_recommend_idx_per_round: list[float] = []
    bpe_recommendation_revenue_per_round: list[float] = []
    bpe_recommendation_simple_regret_per_round: list[float] = []

    for _t in range(T):
        a = dms_x0.pull(snap_fn=snap_to_grid)
        m = norm_to_real(a)
        if cfg.oracle_feedback_for_debug:
            r_bpe = float(env.compute_givenaction_value(m))
        else:
            r_bpe = float(mean_joint_reward_replicated(env, m, R_mc))
        dms_x0.update(r_bpe)
        surv_trace.append(int(dms_x0._active_mask.sum()))
        bpe_snap = _bpe_round_scalar_diag(env, dms_x0, opt, opt_idx)
        bpe_opt_still_active_per_round.append(bpe_snap["bpe_opt_still_active"])
        bpe_opt_ever_sampled_per_round.append(bpe_snap["bpe_opt_ever_sampled"])
        bpe_n_survivors_per_round.append(bpe_snap["bpe_n_survivors"])
        bpe_recommend_idx_per_round.append(bpe_snap["bpe_recommend_idx"])
        bpe_recommendation_revenue_per_round.append(bpe_snap["bpe_recommendation_revenue"])
        bpe_recommendation_simple_regret_per_round.append(
            bpe_snap["bpe_recommendation_simple_regret"]
        )

    return dict(
        bpe_survivors_per_round=np.asarray(surv_trace, dtype=int),
        bpe_final_survivors=int(dms_x0._active_mask.sum()),
        bpe_committed_idx=dms_x0._committed_idx,
        bpe_batches_finished=int(dms_x0._batch_idx),
        n_arms=len(ALL_ACTIONS_NORM),
        opt_grid_idx=opt_idx,
        bpe_opt_still_active_per_round=np.asarray(bpe_opt_still_active_per_round, dtype=float),
        bpe_opt_ever_sampled_per_round=np.asarray(bpe_opt_ever_sampled_per_round, dtype=float),
        bpe_n_survivors_per_round=np.asarray(bpe_n_survivors_per_round, dtype=float),
        bpe_recommend_idx_per_round=np.asarray(bpe_recommend_idx_per_round, dtype=float),
        bpe_recommendation_revenue_per_round=np.asarray(
            bpe_recommendation_revenue_per_round, dtype=float
        ),
        bpe_recommendation_simple_regret_per_round=np.asarray(
            bpe_recommendation_simple_regret_per_round, dtype=float
        ),
    )

