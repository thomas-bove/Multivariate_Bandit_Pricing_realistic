"""
Estimate the RKHS norm of the (normalized) joint reward function
under the product Matérn-1/2 kernel k_Π with L=1.0.

Method: Kronecker structure of k_Π on the full Cartesian grid X° =
  MARGIN_NORM^d  (n_actions = 5, d = 6  →  5^6 = 15625 points).

  K = K₁ ⊗ K₁ ⊗ … ⊗ K₁   (d=6 times, K₁ is 5×5)
  ‖f‖²_K = f^T K⁻¹ f

Eigen-decompose K₁ = Q Λ Q^T once, then apply (Q^T)^{⊗d} to f by
successive mode-n products — O(d · n^{d+1}) instead of O(n^{3d}).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pricing_realistic.config import (
    CFG, ALL_ACTIONS_NORM, ALL_ACTIONS_REAL, MARGIN_NORM,
    N_PRODUCTS, N_ACTIONS, REWARD_SCALE,
)
from environments.demand_generator import make_default_env

# ── 1. Build environment and evaluate exact reward on full grid ────────────────
print(f"Grid: {N_ACTIONS}^{N_PRODUCTS} = {N_ACTIONS**N_PRODUCTS} points")
print(f"REWARD_SCALE = {REWARD_SCALE:.4f}  (used to normalize rewards)")

env, _ = make_default_env(CFG, seed=0)

print("Evaluating exact reward on full grid…", flush=True)
f_real = np.array(
    [float(env.compute_givenaction_value(a)) for a in ALL_ACTIONS_REAL],
    dtype=float,
)
f_norm = f_real / REWARD_SCALE   # normalized to ~O(1) scale (what the GP sees)

print(f"  f_real  min={f_real.min():.4f}  max={f_real.max():.4f}  "
      f"opt={f_real.max():.4f}")
print(f"  f_norm  min={f_norm.min():.4f}  max={f_norm.max():.4f}")

# ── 2. Build 1-D Matérn-1/2 kernel matrix K₁ (n_actions × n_actions) ─────────
L = float(CFG.n_actions)  # kernel_L = 1.0 from EXP_CFG
L = 1.0                   # always 1.0 per EXP_CFG.dms_kernel_L default

pts = MARGIN_NORM          # shape (n_actions,)  = [0, 0.25, 0.5, 0.75, 1.0]
K1 = np.exp(-np.abs(pts[:, None] - pts[None, :]) / L)   # (5, 5)
print(f"\nK₁ (1-D Matérn-1/2, L=1):\n{np.round(K1, 4)}")

# Eigen-decompose K₁ for Kronecker inversion
eigvals, Q = np.linalg.eigh(K1)   # K₁ = Q diag(λ) Q^T
print(f"\nK₁ eigenvalues: {np.round(eigvals, 6)}")
print(f"  min eigenvalue = {eigvals.min():.2e}  (conditioning)")

# ── 3. Apply (Q^T)^{⊗d} to f via successive mode-n products ──────────────────
# Reshape f to (n_actions, n_actions, …, n_actions) = 5^6 tensor
# Order matches itertools.product(*([MARGIN_NORM]*d)): last index varies fastest

d = N_PRODUCTS
n = N_ACTIONS

F = f_norm.reshape([n] * d)   # shape (5,5,5,5,5,5)

# Apply Q^T along each axis: F ← Q^T ×_j F  for j = 0 … d-1
Fc = F.copy()
for axis in range(d):
    Fc = np.tensordot(Q.T, Fc, axes=([1], [axis]))
    # tensordot contracts axis 1 of Q^T with `axis` of Fc;
    # result has the new axis first → move it back
    Fc = np.moveaxis(Fc, 0, axis)

# Now Fc holds (Q^T)^{⊗d} f.
# Eigenvalues of K^{⊗d}: outer product of the d sets of 1-D eigenvalues
lambda_grid = eigvals.copy()
for _ in range(d - 1):
    lambda_grid = np.outer(lambda_grid, eigvals).ravel()
lambda_tensor = lambda_grid.reshape([n] * d)

# ‖f‖²_K = Σ (Fc_i)² / λ_i
rkhs_norm_sq = float(np.sum(Fc**2 / lambda_tensor))
rkhs_norm    = float(np.sqrt(max(rkhs_norm_sq, 0.0)))

print(f"\n{'='*60}")
print(f"  ‖f_norm‖²_K  =  {rkhs_norm_sq:.4f}")
print(f"  ‖f_norm‖_K   =  {rkhs_norm:.4f}   ← B_rkhs should be ≥ this")
print(f"{'='*60}")
print(f"\n  Current B_rkhs in config  = 1.0")
if rkhs_norm > 1.0:
    print(f"  ⚠  True norm ({rkhs_norm:.3f}) > B_rkhs (1.0)  →  "
          f"confidence bands are too narrow; risk of premature elimination.")
else:
    print(f"  ✓  True norm ({rkhs_norm:.3f}) ≤ B_rkhs (1.0)  →  "
          f"confidence bands are correctly calibrated.")

# ── 4. Additional: norm for a few B_rkhs candidate values ─────────────────────
print("\nSuggested B_rkhs rounding (ceiling to 1 decimal):")
import math
for candidate in [1.0, round(math.ceil(rkhs_norm * 10) / 10, 1),
                  round(rkhs_norm, 1), 2.0, 3.0]:
    margin = "OK" if rkhs_norm <= candidate else "TOO SMALL"
    print(f"  B_rkhs = {candidate:.1f}   {margin}")

# ── 5. Variance cross-check ────────────────────────────────────────────────────
# Upper bound from Cauchy-Schwarz: ‖f‖² ≤ ‖f‖_K² · k(x,x) = ‖f‖_K² · 1
# k_Π(x,x) = 1 for all x, so the amplitude of f is bounded by ‖f‖_K.
print(f"\nAmplitude check: max|f_norm| = {np.abs(f_norm).max():.4f} "
      f"(should be ≤ ‖f‖_K = {rkhs_norm:.4f} by Cauchy-Schwarz)")
