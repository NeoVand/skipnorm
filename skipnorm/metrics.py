"""Standardized, locked evaluation metrics for SkipNorm and Dynamical Invariant Discovery.

Implements:
1. Within-Orbit Radial Coefficient of Variation (CV_O(h)) evaluated strictly on encoded true trajectories E(x_t).
2. Orbit-Variance Ratio (OVR) with a degenerate-denominator guard based on training-set latent spread sigma_lat.
3. Monotone Isotonic Regression R^2 and Spearman Rank Correlation |rho_S| across held-out test orbits.
4. Train-fitted linear and degree-two ridge readouts from total radius vs. block radii.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr


def centered_rms_spread(train_latents: np.ndarray) -> float:
    """RMS Euclidean distance from the training-set mean latent vector.

    The last axis indexes coordinates; all preceding axes index observations.
    A scalar invariant must be supplied with a singleton coordinate axis.
    This scale is invariant to translation and scales linearly under rescaling.
    """
    z = np.asarray(train_latents, dtype=np.float64)
    samples = z.reshape(-1, z.shape[-1])
    return float(np.sqrt(np.mean(np.sum((samples - samples.mean(axis=0)) ** 2, axis=-1))))


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

    if not np.isfinite(sigma_lat) or sigma_lat < 0:
        raise ValueError("sigma_lat must be a finite nonnegative training-set spread")
    threshold = (min_spread_ratio * sigma_lat) ** 2
    if total_var <= threshold or between_var <= threshold or sigma_lat == 0:
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
    train_c: np.ndarray,
    train_mean_radii: np.ndarray,
    test_c: np.ndarray,
    test_mean_radii: np.ndarray,
) -> dict[str, Any]:
    """Fit psi: invariant -> mean radius on training orbits and score test orbits.

    Increasing/decreasing direction minimizes training SSE only. Repeated training
    labels are pooled with count weights; interpolation is linear between fitted
    knots and constant beyond the training label range. Test R² is not clipped.
    Spearman is a separate descriptive correlation on the held-out orbit pairs.
    """
    c_tr = np.asarray(train_c, dtype=np.float64).ravel()
    r_tr = np.asarray(train_mean_radii, dtype=np.float64).ravel()
    c_te = np.asarray(test_c, dtype=np.float64).ravel()
    r_te = np.asarray(test_mean_radii, dtype=np.float64).ravel()
    if len(c_tr) != len(r_tr) or len(c_te) != len(r_te) or len(c_tr) < 2:
        raise ValueError("Aligned training and test orbit pairs are required")
    knots, inverse, counts = np.unique(c_tr, return_inverse=True, return_counts=True)
    pooled = np.bincount(inverse, weights=r_tr) / counts
    fitted = {}
    errors = {}
    for direction, increasing in (("increasing", True), ("decreasing", False)):
        fitted[direction] = isotonic_regression(pooled, weights=counts, increasing=increasing).x
        errors[direction] = float(np.mean((r_tr - fitted[direction][inverse]) ** 2))
    direction = min(errors, key=errors.get)
    predictions = np.interp(c_te, knots, fitted[direction])
    variance = float(np.var(r_te))
    test_r2 = None if variance == 0 else float(1.0 - np.mean((r_te - predictions) ** 2) / variance)
    rho = 0.0 if np.ptp(c_te) == 0 or np.ptp(r_te) == 0 else float(spearmanr(c_te, r_te).statistic)
    return {
        "isotonic_r2": test_r2,
        "spearman_abs_rho": abs(rho),
        "spearman_rho": rho,
        "direction": direction,
        "train_mse": errors[direction],
        "test_predictions": predictions.tolist(),
        "train_knots": knots.tolist(),
        "fitted_train_radii": fitted[direction].tolist(),
        "test_outside_training_range": int(np.sum((c_te < knots[0]) | (c_te > knots[-1]))),
    }


def compute_radius_threshold_probe(
    train_radii: np.ndarray, train_labels: np.ndarray,
    test_radii: np.ndarray, test_labels: np.ndarray,
) -> dict[str, Any]:
    """Fit a one-dimensional threshold and orientation using training labels only."""
    x = np.asarray(train_radii).ravel()
    y = np.asarray(train_labels).ravel().astype(bool)
    values = np.unique(x)
    thresholds = np.r_[np.nextafter(values[0], -np.inf),
                       (values[:-1] + values[1:]) / 2,
                       np.nextafter(values[-1], np.inf)]
    best = (-1.0, float(thresholds[0]), True)
    for threshold in thresholds:
        for high_is_one in (True, False):
            pred = (x >= threshold) if high_is_one else (x < threshold)
            score = float(np.mean(pred == y))
            if score > best[0]:
                best = (score, float(threshold), high_is_one)
    score, threshold, high_is_one = best
    x_te = np.asarray(test_radii).ravel()
    pred_te = (x_te >= threshold) if high_is_one else (x_te < threshold)
    return {"accuracy": float(np.mean(pred_te == np.asarray(test_labels).ravel())),
            "training_accuracy": score, "threshold": threshold, "high_is_one": high_is_one}


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
        if var_y == 0:
            raise ValueError("Test invariants must have nonzero variance")
        r2_lin_per_inv.append(1.0 - mse_lin / var_y)
        r2_poly_per_inv.append(1.0 - mse_poly / var_y)

    return {
        "linear_r2_mean": float(np.mean(r2_lin_per_inv)),
        "linear_r2_per_invariant": r2_lin_per_inv,
        "nonlinear_r2_mean": float(np.mean(r2_poly_per_inv)),
        "nonlinear_r2_per_invariant": r2_poly_per_inv,
        "linear_test_predictions": Y_pred_lin.tolist(),
        "nonlinear_test_predictions": Y_pred_poly.tolist(),
    }
