"""
Empirical calibration verifier for DMS-X0-BPE.

Hypotheses tested:
  H1 — dms_rkhs_norm (B_rkhs) < empirical RKHS norm of true reward J
  H2 — prior_mean=0.0 is well below the actual reward distribution
  H3 — kernel L=1.0 over-smooths (adjacent grid arms too similar)

Diagnosis printed at end; no fix applied here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pricing_realistic.config import (
    CFG, ALL_ACTIONS_NORM, REWARD_SCALE, EXP_CFG,
    N_PRODUCTS, N_ACTIONS, normalize_joint_reward,
    effective_gp_noise_variance,
)
from pricing_realistic.rewards import norm_to_real
from pricing_realistic.kernels import _matern12_product, KERNEL_L
from pricing_realistic.agents.dmsx0_bpe import DMSX0BPE
from environments.demand_generator import make_default_env

DIAG_T        = 1000
DIAG_SEED     = 3
RKHS_SUB      = 1000   # subsample size for RKHS norm when |X°| > 3000
SEP           = "-" * 70


# ── 1. Build environment ───────────────────────────────────────────────────────
print(SEP)
print(f"[ENV] d={N_PRODUCTS}  n_actions={N_ACTIONS}  |X°|={len(ALL_ACTIONS_NORM)}  "
      f"REWARD_SCALE={REWARD_SCALE:.3f}")
env, _ = make_default_env(cfg=CFG, seed=0)
print("[ENV] done.")


# ── 2. True objective J on full X° ────────────────────────────────────────────
print(SEP)
n_arms = len(ALL_ACTIONS_NORM)
print(f"[J_true] computing on {n_arms} arms via oracle…", flush=True)
J_true_raw  = np.array([
    env.compute_givenaction_value(norm_to_real(p)) for p in ALL_ACTIONS_NORM
], dtype=float)
J_true_norm = J_true_raw / REWARD_SCALE
opt_idx     = int(np.argmax(J_true_raw))

print(f"  J_raw : min={J_true_raw.min():.4f}  mean={J_true_raw.mean():.4f}  "
      f"median={np.median(J_true_raw):.4f}  max={J_true_raw.max():.4f}")
print(f"  J_norm: min={J_true_norm.min():.4f}  mean={J_true_norm.mean():.4f}  "
      f"median={np.median(J_true_norm):.4f}  max={J_true_norm.max():.4f}")
print(f"  Optimal arm: idx={opt_idx}  J_norm={J_true_norm[opt_idx]:.4f}  "
      f"action={ALL_ACTIONS_NORM[opt_idx]}")
top5 = np.argsort(J_true_raw)[-5:][::-1]
print("  Top-5 arms (idx | J_raw | action_norm):")
for i in top5:
    print(f"    [{i:5d}]  {J_true_raw[i]:.4f}  {ALL_ACTIONS_NORM[i]}")


# ── 3. H1 — Empirical RKHS norm ───────────────────────────────────────────────
print(SEP)
B_rkhs = float(EXP_CFG.dms_rkhs_norm)
print(f"[H1] Empirical RKHS norm of J_true_norm  (config B_rkhs={B_rkhs})")
if n_arms > RKHS_SUB:
    rng_ss = np.random.default_rng(42)
    ss_idx = rng_ss.choice(n_arms, size=RKHS_SUB, replace=False)
    X_ss, J_ss = ALL_ACTIONS_NORM[ss_idx], J_true_norm[ss_idx]
    note = f"subsample {RKHS_SUB}/{n_arms}"
else:
    X_ss, J_ss = ALL_ACTIONS_NORM, J_true_norm
    note = "full"
print(f"  Building kernel matrix ({note})…", flush=True)
K_ss  = _matern12_product(X_ss, X_ss, KERNEL_L)
K_reg = K_ss + 1e-6 * np.eye(len(X_ss))
try:
    Lch  = np.linalg.cholesky(K_reg)
    alph = np.linalg.solve(Lch.T, np.linalg.solve(Lch, J_ss))
except np.linalg.LinAlgError:
    alph = np.linalg.solve(K_reg + 1e-4 * np.eye(len(X_ss)), J_ss)
rkhs_norm   = float(np.sqrt(max(float(np.dot(J_ss, alph)), 0.0)))
h1_confirmed = rkhs_norm > B_rkhs
print(f"  ‖J‖_H ≈ {rkhs_norm:.4f}  vs  B_rkhs={B_rkhs}")
print(f"  H1: {'CONFIRMED  (rkhs_norm > B_rkhs)' if h1_confirmed else 'not confirmed  (B_rkhs is sufficient)'}")


# ── 4. H2 — Prior mean vs reward distribution ──────────────────────────────────
print(SEP)
pm    = float(EXP_CFG.dms_bpe_prior_mean)
j_mean, j_med = float(J_true_norm.mean()), float(np.median(J_true_norm))
print(f"[H2] Prior mean vs J_true_norm distribution  (prior_mean={pm})")
print(f"  mean={j_mean:.4f}  median={j_med:.4f}  prior_mean={pm}")
h2_confirmed = j_mean > 0.15
print(f"  H2: {'CONFIRMED  (mean J_norm > 0.15)' if h2_confirmed else 'not confirmed  (mean J_norm ≤ 0.15)'}")


# ── 5. H3 — Kernel resolution ─────────────────────────────────────────────────
print(SEP)
print(f"[H3] Kernel useful resolution  (L={KERNEL_L}, grid_step={1/(N_ACTIONS-1):.3f})")
step      = 1.0 / (N_ACTIONS - 1)
x0        = np.zeros((1, N_PRODUCTS))
x_adj     = x0.copy(); x_adj[0, 0]     = step
x_far1    = x0.copy(); x_far1[0, 0]    = 1.0
x_corner  = np.ones((1, N_PRODUCTS))
k_adj     = float(_matern12_product(x0, x_adj,    KERNEL_L)[0, 0])
k_far1    = float(_matern12_product(x0, x_far1,   KERNEL_L)[0, 0])
k_corner  = float(_matern12_product(x0, x_corner, KERNEL_L)[0, 0])
print(f"  k(0…0, adj-1dim) = {k_adj:.4f}   [differ by 1 step in dim-0 only]")
print(f"  k(0…0, far-1dim) = {k_far1:.4f}   [differ by full range in dim-0 only]")
print(f"  k(0…0, 1…1)      = {k_corner:.4f}   [corner-to-corner]")
h3_confirmed = k_adj > 0.7
print(f"  H3: {'CONFIRMED  (k_adj > 0.7 → over-smoothing)' if h3_confirmed else 'not confirmed  (k_adj ≤ 0.7)'}")


# ── 6. Diagnostic DMSX0BPE run ────────────────────────────────────────────────
print(SEP)
print(f"[DIAG] BPE run  T={DIAG_T}  seed={DIAG_SEED}  noiseless oracle  "
      f"global_history={EXP_CFG.dms_bpe_use_global_history}")


class _DiagBPE(DMSX0BPE):
    def __init__(self, *args, opt_arm_idx: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_arm_idx = opt_arm_idx
        self.batch_log: list = []

    def _eliminate(self):
        n_before = int(self._active_mask.sum())
        super()._eliminate()
        n_after   = int(self._active_mask.sum())
        sqrt_beta = self._sqrt_beta_elimination()
        if self._last_active_mu is not None and self._last_active_sig is not None:
            lcbs    = self._last_active_mu - sqrt_beta * self._last_active_sig
            ucbs    = self._last_active_mu + sqrt_beta * self._last_active_sig
            sig_m   = float(self._last_active_sig.mean())
            max_lcb = float(lcbs.max())
            min_ucb = float(ucbs.min())
        else:
            sig_m = max_lcb = min_ucb = float("nan")
        self.batch_log.append({
            "batch":    self._batch_idx,
            "n_before": n_before,
            "n_after":  n_after,
            "sqrt_beta": sqrt_beta,
            "sig_mean": sig_m,
            "max_lcb":  max_lcb,
            "min_ucb":  min_ucb,
            "opt_alive": bool(self._active_mask[self.opt_arm_idx]),
        })


lam_gp = effective_gp_noise_variance(EXP_CFG)
agent  = _DiagBPE(
    ALL_ACTIONS_NORM,
    T=DIAG_T,
    noise_var=lam_gp,
    delta=float(EXP_CFG.dms_elimination_delta),
    kernel_L=float(EXP_CFG.dms_kernel_L),
    B_rkhs=B_rkhs,
    seed=DIAG_SEED * 31 + 1,
    prior_mean=pm,
    bpe_beta_use_active_count=bool(EXP_CFG.dms_bpe_beta_use_active_count),
    noise_R=EXP_CFG.dms_bpe_noise_R,
    bpe_use_global_history=bool(EXP_CFG.dms_bpe_use_global_history),
    opt_arm_idx=opt_idx,
)

for _ in range(DIAG_T):
    agent.pull()
    arm = agent._pending_arm_idx
    agent.update(float(J_true_raw[arm]))   # raw scale; update() normalises internally

print(f"  Batches logged: {len(agent.batch_log)}")
hdr = f"  {'Batch':>5}  {'N_bef':>6}  {'N_aft':>6}  {'√β':>6}  {'σ_mean':>7}  {'maxLCB':>8}  {'minUCB':>8}  opt?"
print(hdr)
for r in agent.batch_log:
    print(f"  {r['batch']:>5}  {r['n_before']:>6}  {r['n_after']:>6}  "
          f"{r['sqrt_beta']:>6.3f}  {r['sig_mean']:>7.4f}  "
          f"{r['max_lcb']:>8.4f}  {r['min_ucb']:>8.4f}  "
          f"{'YES' if r['opt_alive'] else '!NO!'}")


# ── 7. Diagnosis ───────────────────────────────────────────────────────────────
print(SEP)
rec_idx       = agent.recommend_idx()
final_active  = int(agent._active_mask.sum())
opt_in_final  = bool(agent._active_mask[opt_idx])

elim_batch = next(
    (r["batch"] for r in agent.batch_log if not r["opt_alive"]), None
)
print(f"  Final |active| = {final_active} / {n_arms}")
print(f"  Recommended idx = {rec_idx}  optimal idx = {opt_idx}  "
      f"correct = {rec_idx == opt_idx}")
print(f"  Optimal in final set: {opt_in_final}")

if elim_batch is not None:
    print(f"\n  DIAGNOSIS A: Optimal arm eliminated at batch {elim_batch}.")
    print("    Cause: √β too small → LCB of opt fell below max LCB of a suboptimal arm.")
    print("    Fix: increase B_rkhs (if H1 confirmed) or widen noise_var.")
elif final_active == 1 and rec_idx == opt_idx:
    print("\n  DIAGNOSIS D: Converged correctly to optimal arm.")
    print("    Algorithm working as expected; no calibration fix needed.")
elif final_active == 1 and rec_idx != opt_idx:
    print(f"\n  DIAGNOSIS B: Converged to wrong arm {rec_idx} (optimal={opt_idx}).")
    print("    Cause: opt survived but GP ranked a suboptimal arm higher.")
    print("    Fix: check prior_mean (H2) and RKHS norm (H1).")
else:
    print(f"\n  DIAGNOSIS C: No convergence — active set never reduced (|survivors|={final_active}).")
    print("    Cause: √β too large → confidence bands prevent any elimination.")
    sqrt_beta_val = agent._sqrt_beta_elimination()
    print(f"    √β = {sqrt_beta_val:.4f}  lam_gp = {lam_gp:.6f}  B_rkhs = {B_rkhs}")
    print("    Fix: reduce B_rkhs, increase T, or reduce |X°| (fewer arms).")

print(SEP)
print("SUMMARY")
print(f"  H1 (B_rkhs too small): {'CONFIRMED' if h1_confirmed else 'not confirmed':15s}  "
      f"(‖J‖_H ≈ {rkhs_norm:.3f}  vs  B_rkhs={B_rkhs})")
print(f"  H2 (prior_mean low):   {'CONFIRMED' if h2_confirmed else 'not confirmed':15s}  "
      f"(mean J_norm={j_mean:.3f}  vs  prior_mean={pm})")
print(f"  H3 (over-smoothing):   {'CONFIRMED' if h3_confirmed else 'not confirmed':15s}  "
      f"(k_adj={k_adj:.3f}  L={KERNEL_L})")
print(SEP)
