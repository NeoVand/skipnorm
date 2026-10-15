"""Standardized, locked evaluation metrics for SkipNorm and Dynamical Invariant Discovery.

Implements:
1. Within-Orbit Radial Coefficient of Variation (CV_O(h)) evaluated strictly on encoded true trajectories E(x_t).
2. Orbit-Variance Ratio (OVR) with a degenerate-denominator guard based on training-set latent spread sigma_lat.
3. Monotone Isotonic Regression R^2 and Spearman Rank Correlation |rho_S| across held-out test orbits.
4. Multi-Invariant Readout R^2 (Linear Ridge and 2-Layer MLP) from single total radius vs. K-block radii.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr


def compute_orbit_radii(z_orbits: np.ndarray, num_blocks: int = 1) -> np.ndarray:
    """Compute latent radii for encoded true trajectories z_orbits of shape (M, T, d_z).

    If num_blocks == 1, returns shape (M, T) of total Euclidean norms ||z_{m,t}||_2.
    If num_blocks > 1, returns shape (M, T, num_blocks) of blockwise Euclidean norms.
    """
    z = np.asarray(z_orbits, dtype=np.float64)
    if num_blocks == 1:
        return np.linalg.norm(z, axis=-1)
    M, T, d_z = z.shape
    d_b = d_z // num_blocks
    z_blocks = z.reshape(M, T, num_blocks, d_b)
    return np.linalg.norm(z_blocks, axis=-1)


def compute_radial_cv(r_orbits: np.ndarray, eps: float = 1e-15) -> dict[str, Any]:
    """Compute within-orbit radial Coefficient of Variation CV_O(h):

        CV_m = sqrt((1/T) sum_{t=1}^T (r_{m,t} - r_bar_m)^2) / r_bar_m

    Accepts r_orbits of shape (M, T) or (M, T, K).
    Returns mean, median, std, and per-orbit CVs.
    """
    r = np.asarray(r_orbits, dtype=np.float64)
    r_bar = np.mean(r, axis=1, keepdims=True)
    r_std = np.sqrt(np.mean(np.square(r - r_bar), axis=1))
    r_bar_squeezed = np.maximum(np.squeeze(r_bar, axis=1), eps)
    cv_per_orbit = r_std / r_bar_squeezed
    return {
        "mean_cv": float(np.mean(cv_per_orbit)),
        "median_cv": float(np.median(cv_per_orbit)),
        "std_cv": float(np.std(cv_per_orbit)),
        "max_cv": float(np.max(cv_per_orbit)),
        "per_orbit_cv": cv_per_orbit.tolist(),
    }


def compute_ovr(
    r_orbits: np.ndarray,
    sigma_lat: float,
    min_spread_ratio: float = 1e-4,
) -> dict[str, Any]:
    """Compute Orbit-Variance Ratio (OVR):

        OVR = [ (1/M) sum_m (1/T) sum_t (r_{m,t} - r_bar_m)^2 ]
              / [ (1/(MT)) sum_{m,t} (r_{m,t} - r_bar_all)^2 ]

    Includes a mandatory degenerate-denominator guard comparing total variance
    against (min_spread_ratio * sigma_lat)^2 so collapsed or constant-norm
    representations are flagged as 'undefined (collapsed)' rather than producing
    spurious ratios.
    """
    r = np.asarray(r_orbits, dtype=np.float64)
    r_bar_m = np.mean(r, axis=1, keepdims=True)
    within_var = float(np.mean(np.square(r - r_bar_m)))
    total_var = float(np.var(r))
    between_var = float(np.var(np.squeeze(r_bar_m, axis=1)))

    threshold = max((min_spread_ratio * max(sigma_lat, 1e-12)) ** 2, 1e-24)
    if total_var < threshold or between_var < threshold:
        return {
            "ovr": None,
            "status": "undefined (collapsed)",
            "within_var": within_var,
            "between_var": between_var,
            "total_var": total_var,
        }

    return {
        "ovr": float(within_var / total_var),
        "status": "valid",
        "within_var": within_var,
        "between_var": between_var,
        "total_var": total_var,
    }


def compute_monotone_r2_and_spearman(
    true_c: np.ndarray,
    orbit_mean_radii: np.ndarray,
) -> dict[str, float]:
    """Compute monotone Isotonic Regression R^2 and absolute Spearman rank correlation |rho_S|
    between true orbit invariant values c_m and mean encoded radii r_bar_m.

    Checks both increasing and decreasing isotonic fits since Theorem 1 allows
    strictly increasing or strictly decreasing psi(c).
    """
    c = np.asarray(true_c, dtype=np.float64).ravel()
    r = np.asarray(orbit_mean_radii, dtype=np.float64).ravel()

    order = np.argsort(c)
    c_sorted = c[order]
    r_sorted = r[order]

    var_r = float(np.var(r_sorted))
    if var_r < 1e-20:
        return {
            "isotonic_r2": 0.0,
            "spearman_abs_rho": 0.0,
            "spearman_rho": 0.0,
            "direction": "collapsed",
        }

    iso_inc = isotonic_regression(r_sorted, increasing=True).x
    iso_dec = isotonic_regression(r_sorted, increasing=False).x

    r2_inc = 1.0 - float(np.mean(np.square(r_sorted - iso_inc))) / var_r
    r2_dec = 1.0 - float(np.mean(np.square(r_sorted - iso_dec))) / var_r

    best_r2 = max(0.0, max(r2_inc, r2_dec))
    direction = "increasing" if r2_inc >= r2_dec else "decreasing"

    rho_res = spearmanr(c_sorted, r_sorted)
    rho_val = 0.0 if np.isnan(rho_res.statistic) else float(rho_res.statistic)

    return {
        "isotonic_r2": float(best_r2),
        "spearman_abs_rho": float(abs(rho_val)),
        "spearman_rho": float(rho_val),
        "direction": direction,
    }


def compute_k_invariant_readout_r2(
    train_radii: np.ndarray,
    train_invariants: np.ndarray,
    test_radii: np.ndarray,
    test_invariants: np.ndarray,
    ridge_lambda: float = 1e-4,
) -> dict[str, Any]:
    """Fit a polynomial degree-2 Ridge readout from encoded orbit-mean radii
    (shape (M, d_r), where d_r=1 for single-sphere or d_r=K for K-block)
    to the true K invariants (shape (M, K)) on the training split and evaluate
    R^2 on the held-out test split.
    """
    X_tr = np.asarray(train_radii, dtype=np.float64)
    Y_tr = np.asarray(train_invariants, dtype=np.float64)
    X_te = np.asarray(test_radii, dtype=np.float64)
    Y_te = np.asarray(test_invariants, dtype=np.float64)

    if X_tr.ndim == 1:
        X_tr = X_tr[:, None]
    if X_te.ndim == 1:
        X_te = X_te[:, None]

    # Standardize inputs using training split
    mu_x = np.mean(X_tr, axis=0, keepdims=True)
    std_x = np.maximum(np.std(X_tr, axis=0, keepdims=True), 1e-8)
    X_tr_n = (X_tr - mu_x) / std_x
    X_te_n = (X_te - mu_x) / std_x

    def poly2_features(X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        feats = [np.ones((n, 1)), X]
        for i in range(d):
            for j in range(i, d):
                feats.append((X[:, i] * X[:, j])[:, None])
        return np.hstack(feats)

    Phi_tr = poly2_features(X_tr_n)
    Phi_te = poly2_features(X_te_n)

    # Linear ridge
    Phi_lin_tr = np.hstack([np.ones((X_tr_n.shape[0], 1)), X_tr_n])
    Phi_lin_te = np.hstack([np.ones((X_te_n.shape[0], 1)), X_te_n])
    W_lin = np.linalg.solve(
        Phi_lin_tr.T @ Phi_lin_tr + ridge_lambda * np.eye(Phi_lin_tr.shape[1]),
        Phi_lin_tr.T @ Y_tr,
    )
    Y_pred_lin = Phi_lin_te @ W_lin

    # Degree-2 ridge (smooth nonlinear readout)
    W_poly = np.linalg.solve(
        Phi_tr.T @ Phi_tr + ridge_lambda * np.eye(Phi_tr.shape[1]),
        Phi_tr.T @ Y_tr,
    )
    Y_pred_poly = Phi_te @ W_poly

    r2_lin_per_inv = []
    r2_poly_per_inv = []
    for k in range(Y_te.shape[1]):
        var_y = float(np.var(Y_te[:, k]))
        mse_lin = float(np.mean(np.square(Y_te[:, k] - Y_pred_lin[:, k])))
        mse_poly = float(np.mean(np.square(Y_te[:, k] - Y_pred_poly[:, k])))
        r2_lin_per_inv.append(max(0.0, 1.0 - mse_lin / max(var_y, 1e-16)))
        r2_poly_per_inv.append(max(0.0, 1.0 - mse_poly / max(var_y, 1e-16)))

    return {
        "linear_r2_mean": float(np.mean(r2_lin_per_inv)),
        "linear_r2_per_invariant": r2_lin_per_inv,
        "nonlinear_r2_mean": float(np.mean(r2_poly_per_inv)),
        "nonlinear_r2_per_invariant": r2_poly_per_inv,
    }
