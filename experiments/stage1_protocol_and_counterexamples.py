"""Stage 1: Mathematical Counterexamples, Spectral Proofs, and Whole-Orbit Split Generator.

Verifies and exports JSON provenance for:
1. Proposition 1: Universal cotangent pullback J^T u_out = u_in vs. Ray-invariant forward
   singular pair J u_in = u_out (comparing 1-homogeneous LeakyReLU against smooth ELU, SiLU,
   GELU, tanh), plus the exact 2x2 twist counterexample (sigma = {phi, 1/phi}) and 1->2 spiral
   counterexample (sigma = sqrt(2)).
2. Proposition 2: Block-triangular rank identity rank(J) = 1 + rank(A|_T), the 2x2 counterexample
   where rank(A) = 1 but rank(A|_T) = 0, exact stabilized RMSNorm singular value curves
   sigma_tan and sigma_rad across epsilon, and depth-100 signal/gradient propagation.
3. Antipodal sign-flip instability in Projected Residual SkipNorm vs. strict positivity in
   Tangent-Projected Geodesic SkipNorm.
4. Theorem 1 Resonant 1:1 2D Harmonic Oscillator Control (showing why orbit density is required
   to bridge orbitwise radial constancy to level-set constancy).
5. Theorem 2 & Section 6 Translated Identity Encoder Control E_a(x) = a*e_1 + x, g(z) = z
   (showing why relative intertwining error and radial CV -> 0 as a -> infinity without
   violating the topological obstruction).
6. Whole-Orbit Disjoint Train/Val/Test Dataset Splits & Converged Reference Solvers (without
   post-step projection on DNLS), saved with SHA-256 manifests.
"""

from __future__ import annotations

import hashlib
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from skipnorm.jax_layers import (
    geodesic_skip_norm_fn,
    projected_residual_skip_norm_fn,
    skip_norm_fn,
)

RESULTS_DIR = ROOT / "results" / "stage1"
DATA_DIR = ROOT / "data" / "splits"


def verify_proposition_1() -> dict:
    """Verify Proposition 1: universal cotangent pullback, ray-invariance across activations,
    2x2 twist counterexample, and 1->2 spiral counterexample.
    """
    key = jax.random.PRNGKey(20261015)
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (6, 16), dtype=jnp.float64) / jnp.sqrt(16.0)
    x = jax.random.normal(k2, (16,), dtype=jnp.float64)
    u_in = x / jnp.linalg.norm(x)

    activations = {
        "LeakyReLU (1-homog)": lambda z: jax.nn.leaky_relu(z, negative_slope=0.2),
        "tanh (non-homog)": jnp.tanh,
        "ELU (non-homog)": jax.nn.elu,
        "SiLU (non-homog)": jax.nn.silu,
        "GELU (non-homog)": jax.nn.gelu,
    }

    act_results = {}
    for name, act in activations.items():
        def F_layer(z):
            return skip_norm_fn(z, act(W @ z), p=2.0, eps=1e-30)

        y_hat = F_layer(x)
        u_out = y_hat / jnp.linalg.norm(y_hat)
        J = jax.jacfwd(F_layer)(x)
        svs = jnp.linalg.svd(J, compute_uv=False)

        pullback_err = float(jnp.linalg.norm(J.T @ u_out - u_in))
        forward_drift = float(jnp.linalg.norm(J @ u_in - u_out))
        radial_proj = float(jnp.dot(u_out, J @ u_in))

        act_results[name] = {
            "cotangent_pullback_err": pullback_err,
            "forward_radial_drift": forward_drift,
            "radial_projection_u_out_J_u_in": radial_proj,
            "sigma_max": float(svs[0]),
            "sigma_min": float(svs[-1]),
            "singular_values": [float(s) for s in svs],
        }

    # Exact 2x2 twist counterexample: F(x) = R_{log ||x||_2} x
    def F_twist(z: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(z)
        theta = jnp.log(r)
        c, s = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[c, -s], [s, c]], dtype=jnp.float64)
        return R @ z

    x_twist = jnp.array([1.0, 0.0], dtype=jnp.float64)
    y_twist = F_twist(x_twist)
    J_twist = jax.jacfwd(F_twist)(x_twist)
    sv_twist = jnp.linalg.svd(J_twist, compute_uv=False)
    golden = (jnp.sqrt(5.0) + 1.0) / 2.0

    twist_result = {
        "J_at_e1": J_twist.tolist(),
        "singular_values": [float(s) for s in sv_twist],
        "theoretical_singular_values": [float(golden), float(1.0 / golden)],
        "cotangent_pullback_err": float(
            jnp.linalg.norm(J_twist.T @ (y_twist / jnp.linalg.norm(y_twist)) - x_twist)
        ),
        "forward_radial_drift": float(
            jnp.linalg.norm(J_twist @ x_twist - (y_twist / jnp.linalg.norm(y_twist)))
        ),
    }

    # Exact 1 -> 2 spiral counterexample: F(t) = t * (cos log t, sin log t)^T
    def F_spiral(t_vec: jnp.ndarray) -> jnp.ndarray:
        t = t_vec[0]
        return jnp.array([t * jnp.cos(jnp.log(t)), t * jnp.sin(jnp.log(t))], dtype=jnp.float64)

    t0 = jnp.array([1.0], dtype=jnp.float64)
    y_spiral = F_spiral(t0)
    J_spiral = jax.jacfwd(F_spiral)(t0)
    sv_spiral = jnp.linalg.svd(J_spiral, compute_uv=False)

    spiral_result = {
        "J_at_1": J_spiral.tolist(),
        "singular_value": float(sv_spiral[0]),
        "theoretical_singular_value": float(jnp.sqrt(2.0)),
        "cotangent_pullback_err": float(
            jnp.linalg.norm(J_spiral.T @ (y_spiral / jnp.linalg.norm(y_spiral)) - jnp.array([1.0]))
        ),
    }

    return {
        "activations": act_results,
        "twist_counterexample_2x2": twist_result,
        "spiral_counterexample_1to2": spiral_result,
    }


def verify_proposition_2() -> dict:
    """Verify Proposition 2: Block-triangular rank identity rank(J) = 1 + rank(A|_T),
    the 2x2 tangential restriction counterexample, exact stabilized RMSNorm singular values
    sigma_tan and sigma_rad, and depth-100 signal propagation.
    """
    # 1. 2x2 counterexample where rank(A) = 1 but rank(A|_T) = 0 -> rank(J) = 1 < 2
    # x = f(x) = e_1 = (1, 0)^T, J_f = [[1, 1], [1, 0]]
    u_in = np.array([[1.0], [0.0]])
    u_out = np.array([[1.0], [0.0]])
    J_f_counter = np.array([[1.0, 1.0], [1.0, 0.0]])
    P_perp = np.eye(2) - u_out @ u_out.T
    A_counter = P_perp @ J_f_counter
    J_counter = u_out @ u_in.T + A_counter
    Q_in = np.array([[0.0], [1.0]])
    A_on_T = A_counter @ Q_in

    counterexample_2x2 = {
        "J_f": J_f_counter.tolist(),
        "A": A_counter.tolist(),
        "J": J_counter.tolist(),
        "rank_A": int(np.linalg.matrix_rank(A_counter)),
        "rank_A_restricted_to_T": int(np.linalg.matrix_rank(A_on_T)),
        "rank_J": int(np.linalg.matrix_rank(J_counter)),
        "formula_1_plus_rank_A_T": int(1 + np.linalg.matrix_rank(A_on_T)),
    }

    # 2. Exact stabilized RMSNorm singular value verification across epsilon and r
    d = 16
    key = jax.random.PRNGKey(42)
    y0 = jax.random.normal(key, (d,), dtype=jnp.float64)
    y0 = y0 / jnp.linalg.norm(y0)  # unit vector, we scale by r

    eps_list = [0.0, 1e-8, 1e-5, 1e-3, 1e-2, 1e-1]
    r_grid = np.logspace(-2, 1, 40)

    rmsnorm_spectra = []
    for eps in eps_list:
        tan_theory = []
        rad_theory = []
        tan_autodiff = []
        rad_autodiff = []
        ratio_theory = []

        for r_val in r_grid:
            y = float(r_val) * y0

            def G_eps(v):
                return jnp.sqrt(d) * v / jnp.sqrt(jnp.sum(jnp.square(v)) + d * eps)

            J_rms = jax.jacfwd(G_eps)(y)
            svs = jnp.linalg.svd(J_rms, compute_uv=False)

            s_tan_th = float(np.sqrt(d) / np.sqrt(r_val**2 + d * eps))
            s_rad_th = float(np.sqrt(d) * d * eps / ((r_val**2 + d * eps) ** 1.5))

            tan_theory.append(s_tan_th)
            rad_theory.append(s_rad_th)
            tan_autodiff.append(float(svs[0]))
            rad_autodiff.append(float(svs[-1]))
            ratio_theory.append(float((d * eps) / (r_val**2 + d * eps)))

        max_tan_err = float(np.max(np.abs(np.array(tan_theory) - np.array(tan_autodiff))))
        max_rad_err = float(np.max(np.abs(np.array(rad_theory) - np.array(rad_autodiff))))

        rmsnorm_spectra.append({
            "eps": eps,
            "r_grid": r_grid.tolist(),
            "sigma_tan_theory": tan_theory,
            "sigma_rad_theory": rad_theory,
            "sigma_tan_autodiff": tan_autodiff,
            "sigma_rad_autodiff": rad_autodiff,
            "ratio_rad_to_tan": ratio_theory,
            "max_tan_err": max_tan_err,
            "max_rad_err": max_rad_err,
        })

    # 3. Jacobian Rank vs Dimension d in [4, 8, 12, 16, 24, 32, 48, 64] at eps=0 vs eps=1e-5
    dims = [4, 8, 12, 16, 24, 32, 48, 64]
    rank_comparison = []
    for dim in dims:
        k_w, k_x = jax.random.split(jax.random.PRNGKey(100 + dim))
        W = jax.random.normal(k_w, (dim, dim), dtype=jnp.float64) / jnp.sqrt(dim)
        x_vec = jax.random.normal(k_x, (dim,), dtype=jnp.float64)

        def f_skip(z):
            y = jnp.tanh(W @ z)
            return y * (jnp.linalg.norm(z) / jnp.linalg.norm(y))

        def f_rms_zero(z):
            y = jnp.tanh(W @ z)
            return jnp.sqrt(dim) * y / jnp.linalg.norm(y)

        def f_ln_zero(z):
            y = jnp.tanh(W @ z)
            yc = y - jnp.mean(y)
            return jnp.sqrt(dim) * yc / jnp.linalg.norm(yc)

        sv_skip = jnp.linalg.svd(jax.jacfwd(f_skip)(x_vec), compute_uv=False)
        sv_rms0 = jnp.linalg.svd(jax.jacfwd(f_rms_zero)(x_vec), compute_uv=False)
        sv_ln0 = jnp.linalg.svd(jax.jacfwd(f_ln_zero)(x_vec), compute_uv=False)

        tol = 1e-10
        rank_comparison.append({
            "dim": dim,
            "rank_skipnorm": int(jnp.sum(sv_skip > tol)),
            "rank_rmsnorm_eps0": int(jnp.sum(sv_rms0 > tol)),
            "rank_layernorm_eps0": int(jnp.sum(sv_ln0 > tol)),
            "min_sv_skipnorm": float(sv_skip[-1]),
            "min_sv_rmsnorm_eps0": float(sv_rms0[-1]),
            "min_sv_layernorm_eps0": float(sv_ln0[-1]),
        })

    # 4. Depth-100 signal & radial gradient norm survival
    depth = 100
    dim_d = 16
    key_d = jax.random.PRNGKey(999)
    w_keys = jax.random.split(key_d, depth + 1)
    x_init = jax.random.normal(w_keys[0], (dim_d,), dtype=jnp.float64)
    x_init = x_init / jnp.linalg.norm(x_init) * 1.5  # initial norm = 1.5
    weights = [
        jax.random.normal(w_keys[i + 1], (dim_d, dim_d), dtype=jnp.float64)
        * (0.85 / jnp.sqrt(dim_d))
        for i in range(depth)
    ]
    orth_weights = [jnp.linalg.qr(W)[0] for W in weights]

    depth_steps = list(range(0, depth + 1, 5))
    depth_traces = {
        "depths": depth_steps,
        "Geodesic SkipNorm": {"act_norm": [], "radial_grad_norm": []},
        "SkipNorm (Tanh)": {"act_norm": [], "radial_grad_norm": []},
        "RMSNorm (eps=1e-1)": {"act_norm": [], "radial_grad_norm": []},
        "RMSNorm (eps=1e-2)": {"act_norm": [], "radial_grad_norm": []},
        "Unnormalized MLP": {"act_norm": [], "radial_grad_norm": []},
    }

    for L in depth_steps:
        def run_stack(z, mode):
            h = z
            for i in range(L):
                W_i = weights[i]
                Q_i = orth_weights[i]
                if mode == "Geodesic SkipNorm":
                    h = geodesic_skip_norm_fn(h, jnp.tanh(W_i @ h), alpha=0.25, eps=1e-30)
                elif mode == "SkipNorm (Tanh)":
                    h = skip_norm_fn(
                        h, jnp.tanh(W_i @ h), p=2.0, eps=1e-30
                    )
                elif mode == "RMSNorm (eps=1e-1)":
                    y = Q_i @ h
                    h = jnp.sqrt(dim_d) * y / jnp.sqrt(jnp.sum(jnp.square(y)) + dim_d * 1e-1)
                elif mode == "RMSNorm (eps=1e-2)":
                    y = Q_i @ h
                    h = jnp.sqrt(dim_d) * y / jnp.sqrt(jnp.sum(jnp.square(y)) + dim_d * 1e-2)
                elif mode == "Unnormalized MLP":
                    h = jnp.tanh(W_i @ h)
            return h

        for mode in depth_traces:
            if mode == "depths":
                continue
            out = run_stack(x_init, mode)
            act_norm = float(jnp.linalg.norm(out))
            u_out = out / jnp.maximum(jnp.linalg.norm(out), 1e-30)
            _, vjp_fn = jax.vjp(lambda z: run_stack(z, mode), x_init)
            grad_radial = vjp_fn(u_out)[0]
            depth_traces[mode]["act_norm"].append(act_norm)
            depth_traces[mode]["radial_grad_norm"].append(float(jnp.linalg.norm(grad_radial)))

    return {
        "counterexample_2x2_tangential_rank": counterexample_2x2,
        "rmsnorm_stabilized_spectra": rmsnorm_spectra,
        "rank_vs_dimension": rank_comparison,
        "depth_100_propagation": depth_traces,
    }


def verify_antipodal_sign_flip() -> dict:
    """Compare Projected Residual SkipNorm vs. Tangent-Projected Geodesic SkipNorm
    as radial opposition alpha * <f(x), u_in> / ||x||_2 sweeps across the critical threshold 1.
    """
    x = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    # f(x) has a negative radial component -c * e_1 and a small tangential component 0.15 * e_2
    radial_oppositions = np.linspace(0.0, 2.2, 89)
    proj_cos = []
    geod_cos = []
    proj_norm_err = []
    geod_norm_err = []

    for c in radial_oppositions:
        f_x = jnp.array([-float(c), 0.15, 0.0, 0.0], dtype=jnp.float64)
        y_proj = projected_residual_skip_norm_fn(x, f_x, alpha=1.0, eps=1e-30)
        y_geod = geodesic_skip_norm_fn(x, f_x, alpha=1.0, eps=1e-30)

        proj_cos.append(float(jnp.dot(y_proj, x) / (jnp.linalg.norm(y_proj) * jnp.linalg.norm(x))))
        geod_cos.append(float(jnp.dot(y_geod, x) / (jnp.linalg.norm(y_geod) * jnp.linalg.norm(x))))
        proj_norm_err.append(float(jnp.abs(jnp.linalg.norm(y_proj) - jnp.linalg.norm(x))))
        geod_norm_err.append(float(jnp.abs(jnp.linalg.norm(y_geod) - jnp.linalg.norm(x))))

    return {
        "radial_opposition_c": radial_oppositions.tolist(),
        "projected_residual_cosine_similarity": proj_cos,
        "geodesic_skipnorm_cosine_similarity": geod_cos,
        "projected_residual_norm_err": proj_norm_err,
        "geodesic_skipnorm_norm_err": geod_norm_err,
    }


def verify_resonant_harmonic_oscillator_control() -> dict:
    """Verify the Theorem 1 Resonant 2D 1:1 Isotropic Harmonic Oscillator control in R^4.

    H(q1, p1, q2, p2) = 0.5 * (q1^2 + p1^2 + q2^2 + p2^2).
    Every orbit has period 2*pi and is a closed 1D circle inside the 3-sphere H^{-1}(c) = S^3(sqrt(2c)).
    We construct the global linear diffeomorphism used in the manuscript:
        E(q1, p1, q2, p2) = (sqrt(2)*q1, sqrt(2)*p1, q2, p2)
    which commutes EXACTLY with the 1:1 harmonic flow (so ||E(F(x)) - g(E(x))|| = 0 and
    within-orbit radial CV = 0 on every trajectory), yet ||E(x)||_2 varies between sqrt(2c)
    and 2 * sqrt(c) on the SAME energy level set H^{-1}(c).
    """
    dt = 0.1
    steps = 60
    t_grid = np.arange(steps) * dt
    c_energy = 1.0  # Fixed energy shell H = 1.0 -> ||x||_2 = sqrt(2)

    # Sample 25 orbits on the SAME energy shell H = 1.0 by varying energy partition eta in [0, 1]
    etas = np.linspace(0.0, 1.0, 25)
    orbit_radii_means = []
    orbit_cvs = []
    intertwining_errs = []

    c_dt, s_dt = np.cos(dt), np.sin(dt)
    R_step = np.array(
        [
            [c_dt, s_dt, 0.0, 0.0],
            [-s_dt, c_dt, 0.0, 0.0],
            [0.0, 0.0, c_dt, s_dt],
            [0.0, 0.0, -s_dt, c_dt],
        ],
        dtype=np.float64,
    )

    def E_map(x: np.ndarray) -> np.ndarray:
        return x * np.array([np.sqrt(2.), np.sqrt(2.), 1., 1.])

    for eta in etas:
        r1 = np.sqrt(2.0 * c_energy * eta)
        r2 = np.sqrt(2.0 * c_energy * (1.0 - eta))
        x_traj = np.stack(
            [
                r1 * np.cos(t_grid),
                -r1 * np.sin(t_grid),
                r2 * np.cos(t_grid),
                -r2 * np.sin(t_grid),
            ],
            axis=-1,
        )
        z_traj = E_map(x_traj)
        radii = np.linalg.norm(z_traj, axis=-1)
        r_mean = float(np.mean(radii))
        r_cv = float(np.std(radii) / r_mean)

        # Check exact intertwining E(F(x_t)) == R_step @ E(x_t)
        x_next = x_traj @ R_step.T
        z_next_true = E_map(x_next)
        z_next_pred = z_traj @ R_step.T
        err = float(np.max(np.linalg.norm(z_next_true - z_next_pred, axis=-1)))

        orbit_radii_means.append(r_mean)
        orbit_cvs.append(r_cv)
        intertwining_errs.append(err)

    shell_rel_spread = float(
        (np.max(orbit_radii_means) - np.min(orbit_radii_means)) / np.min(orbit_radii_means)
    )

    return {
        "energy_shell_H": c_energy,
        "partition_eta": etas.tolist(),
        "orbit_mean_radii_on_same_shell": orbit_radii_means,
        "max_within_orbit_cv": float(np.max(orbit_cvs)),
        "max_intertwining_error": float(np.max(intertwining_errs)),
        "same_shell_relative_radius_spread": shell_rel_spread,
    }


def verify_translated_encoder_control() -> dict:
    """Verify Section 6 / Appendix D Counterexample to Universal Positive Error Floors:
    On the double-well system bounded by ||x||_2 <= R, E_a(x) = a * e_1 + x, D_a(z) = z - a * e_1,
    g(z) = z has:
      - Zero reconstruction error: ||D_a(E_a(x)) - x||_2 = 0
      - Bounded absolute intertwining error: ||E_a(F(x)) - g(E_a(x))||_2 = ||F(x) - x||_2 <= D
      - Vanishing relative intertwining error: <= D / (a - R) -> 0 as a -> infinity
      - Vanishing within-orbit radial CV: <= R / (a - R) -> 0 as a -> infinity
    """
    dt = 0.05

    def double_well_rhs(state):
        q, p = state[..., 0], state[..., 1]
        dq = p
        dp = q - q**3
        return np.stack([dq, dp], axis=-1)

    def rk4_step(state, h=dt):
        k1 = double_well_rhs(state)
        k2 = double_well_rhs(state + 0.5 * h * k1)
        k3 = double_well_rhs(state + 0.5 * h * k2)
        k4 = double_well_rhs(state + h * k3)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Generate 10 trapped orbits (5 in left well q < 0, 5 in right well q > 0)
    orbits = []
    for well_sign in [-1.0, 1.0]:
        for dq0 in [0.15, 0.25, 0.35, 0.45, 0.55]:
            x = np.array([well_sign * (1.0 - dq0), 0.0], dtype=np.float64)
            traj = [x]
            for _ in range(100):
                x = rk4_step(x)
                traj.append(x)
            orbits.append(np.stack(traj, axis=0))
    orbits_arr = np.stack(orbits, axis=0)  # (10, 101, 2)
    X_curr = orbits_arr[:, :-1, :]
    X_next = orbits_arr[:, 1:, :]

    R_bound = float(np.max(np.linalg.norm(X_curr, axis=-1)))
    D_step = float(np.max(np.linalg.norm(X_next - X_curr, axis=-1)))

    a_shifts = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0]
    records = []
    for a in a_shifts:
        shift_vec = np.array([a, 0.0], dtype=np.float64)
        Z_curr = X_curr + shift_vec
        Z_next = X_next + shift_vec
        # Identity latent transition g(z) = z (exact norm-preserving map!)
        Z_pred = Z_curr
        X_recon = Z_curr - shift_vec

        recon_err = float(np.max(np.linalg.norm(X_recon - X_curr, axis=-1)))
        abs_intertwining = float(np.mean(np.linalg.norm(Z_next - Z_pred, axis=-1)))
        rel_intertwining = float(
            np.mean(
                np.linalg.norm(Z_next - Z_pred, axis=-1) / np.linalg.norm(Z_curr, axis=-1)
            )
        )
        radii = np.linalg.norm(Z_curr, axis=-1)
        r_bar = np.mean(radii, axis=-1, keepdims=True)
        cv_mean = float(np.mean(np.std(radii, axis=-1) / np.squeeze(r_bar, axis=-1)))

        records.append({
            "shift_a": a,
            "recon_err": recon_err,
            "abs_intertwining_err": abs_intertwining,
            "rel_intertwining_err": rel_intertwining,
            "theoretical_rel_err_bound": float(D_step / (a - R_bound)),
            "mean_radial_cv": cv_mean,
            "theoretical_cv_bound": float(R_bound / (a - R_bound)),
        })

    return {
        "R_state_bound": R_bound,
        "D_step_bound": D_step,
        "shift_sweep": records,
    }


def generate_whole_orbit_datasets() -> dict:
    """Generate and save deterministic whole-orbit disjoint train/val/test splits
    for all dynamical systems in Stages 2 and 3, recording SHA-256 hashes.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def save_split(name: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
        path = DATA_DIR / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[name] = {
            "file": str(path.relative_to(ROOT)),
            "sha256": sha256,
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "meta": meta,
        }

    # 1. Lotka-Volterra in log-coordinates u = log(x), v = log(y)
    # H(u, v) = exp(u) - u + exp(v) - v - 2 > 0 (shifted so minimum at (0,0) is 0)
    # Whole-orbit disjoint split: 32 train, 8 val, 16 test orbits, T=120 steps, dt=0.05
    rng_lv = np.random.default_rng(202601)
    c_all_lv = np.linspace(0.08, 1.40, 56)
    perm_lv = rng_lv.permutation(56)
    idx_tr_lv, idx_va_lv, idx_te_lv = np.sort(perm_lv[:32]), np.sort(perm_lv[32:40]), np.sort(perm_lv[40:])

    def lv_rhs(state):
        u, v = state[..., 0], state[..., 1]
        du = 1.0 - np.exp(v)
        dv = np.exp(u) - 1.0
        return np.stack([du, dv], axis=-1)

    def solve_orbit(rhs_fn, x0, dt, T_steps, substeps=10):
        h = dt / substeps
        traj = [x0.copy()]
        curr = x0.copy()
        for _ in range(T_steps - 1):
            for _ in range(substeps):
                k1 = rhs_fn(curr)
                k2 = rhs_fn(curr + 0.5 * h * k1)
                k3 = rhs_fn(curr + 0.5 * h * k2)
                k4 = rhs_fn(curr + h * k3)
                curr = curr + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(curr.copy())
        return np.stack(traj, axis=0)

    # Find initial u0 > 0 on v0 = 0 such that exp(u0) - u0 - 1 == c
    from scipy.optimize import brentq

    lv_orbits = []
    lv_invariants = []
    for c_val in c_all_lv:
        u0 = brentq(lambda u: np.exp(u) - u - 1.0 - c_val, 1e-4, 3.0)
        x0 = np.array([u0, 0.0], dtype=np.float64)
        traj = solve_orbit(lv_rhs, x0, dt=0.05, T_steps=180, substeps=10)
        lv_orbits.append(traj)
        lv_invariants.append(c_val)
    lv_orbits_arr = np.stack(lv_orbits, axis=0)
    lv_inv_arr = np.array(lv_invariants, dtype=np.float64)

    save_split(
        "lotka_volterra",
        {
            "train_orbits": lv_orbits_arr[idx_tr_lv],
            "train_invariants": lv_inv_arr[idx_tr_lv],
            "val_orbits": lv_orbits_arr[idx_va_lv],
            "val_invariants": lv_inv_arr[idx_va_lv],
            "test_orbits": lv_orbits_arr[idx_te_lv],
            "test_invariants": lv_inv_arr[idx_te_lv],
        },
        {"dt": 0.05, "T_steps": 180, "substeps": 10, "n_train": 32, "n_val": 8, "n_test": 16},
    )

    # 2. Double-Well Hamiltonian H(q, p) = 0.5*p^2 - 0.5*q^2 + 0.25*q^4 + gamma*q
    # Regime A: Balanced 50/50 sub-barrier (gamma = 0)
    # Regime B: Imbalanced 80/20 sub-barrier (gamma = 0)
    # Regime C: Tilted double-well (gamma = 0.08) breaking energy degeneracy between wells
    def make_dw_dataset(gamma: float, left_count_tr: int, right_count_tr: int, seed_offset: int):
        def dw_rhs(state):
            q, p = state[..., 0], state[..., 1]
            dq = p
            dp = q - q**3 - gamma
            return np.stack([dq, dp], axis=-1)

        # Find local minima of V(q) = -0.5*q^2 + 0.25*q^4 + gamma*q
        q_left_eq = brentq(lambda q: q**3 - q + gamma, -1.5, -0.4)
        q_right_eq = brentq(lambda q: q**3 - q + gamma, 0.4, 1.5)
        q_sad_eq = brentq(lambda q: q**3 - q + gamma, -0.35, 0.35)

        def V_pot(q):
            return -0.5 * q**2 + 0.25 * q**4 + gamma * q

        V_sad = V_pot(q_sad_eq)
        V_left_min = V_pot(q_left_eq)
        V_right_min = V_pot(q_right_eq)

        # Sample sub-barrier energies in both wells
        total_left = left_count_tr + 4 + 8
        total_right = right_count_tr + 4 + 8

        e_left = np.linspace(V_left_min + 0.02, V_sad - 0.03, total_left)
        e_right = np.linspace(V_right_min + 0.02, V_sad - 0.03, total_right)

        left_orbits, left_E = [], []
        for E_val in e_left:
            q0 = brentq(lambda q: V_pot(q) - E_val, q_left_eq, q_sad_eq - 1e-3)
            p0 = 0.0
            traj = solve_orbit(dw_rhs, np.array([q0, p0]), dt=0.05, T_steps=120, substeps=10)
            left_orbits.append(traj)
            left_E.append(E_val)

        right_orbits, right_E = [], []
        for E_val in e_right:
            q0 = brentq(lambda q: V_pot(q) - E_val, q_sad_eq + 1e-3, q_right_eq)
            p0 = 0.0
            traj = solve_orbit(dw_rhs, np.array([q0, p0]), dt=0.05, T_steps=120, substeps=10)
            right_orbits.append(traj)
            right_E.append(E_val)

        rng_dw = np.random.default_rng(202602 + seed_offset)
        p_l = rng_dw.permutation(total_left)
        p_r = rng_dw.permutation(total_right)

        l_tr, l_va, l_te = p_l[:left_count_tr], p_l[left_count_tr:left_count_tr + 4], p_l[left_count_tr + 4:]
        r_tr, r_va, r_te = p_r[:right_count_tr], p_r[right_count_tr:right_count_tr + 4], p_r[right_count_tr + 4:]

        left_orbits_arr = np.stack(left_orbits, axis=0)
        right_orbits_arr = np.stack(right_orbits, axis=0)
        left_E_arr = np.array(left_E, dtype=np.float64)
        right_E_arr = np.array(right_E, dtype=np.float64)

        return {
            "train_orbits": np.concatenate([left_orbits_arr[l_tr], right_orbits_arr[r_tr]], axis=0),
            "train_invariants": np.concatenate([left_E_arr[l_tr], right_E_arr[r_tr]], axis=0),
            "train_wells": np.concatenate([np.zeros(len(l_tr), dtype=np.int32), np.ones(len(r_tr), dtype=np.int32)]),
            "val_orbits": np.concatenate([left_orbits_arr[l_va], right_orbits_arr[r_va]], axis=0),
            "val_invariants": np.concatenate([left_E_arr[l_va], right_E_arr[r_va]], axis=0),
            "val_wells": np.concatenate([np.zeros(len(l_va), dtype=np.int32), np.ones(len(r_va), dtype=np.int32)]),
            "test_orbits": np.concatenate([left_orbits_arr[l_te], right_orbits_arr[r_te]], axis=0),
            "test_invariants": np.concatenate([left_E_arr[l_te], right_E_arr[r_te]], axis=0),
            "test_wells": np.concatenate([np.zeros(len(l_te), dtype=np.int32), np.ones(len(r_te), dtype=np.int32)]),
        }

    save_split(
        "double_well_balanced",
        make_dw_dataset(gamma=0.0, left_count_tr=16, right_count_tr=16, seed_offset=1),
        {"gamma": 0.0, "regime": "balanced_50_50", "dt": 0.05, "T_steps": 120},
    )
    save_split(
        "double_well_imbalanced",
        make_dw_dataset(gamma=0.0, left_count_tr=24, right_count_tr=6, seed_offset=2),
        {"gamma": 0.0, "regime": "imbalanced_80_20", "dt": 0.05, "T_steps": 120},
    )
    save_split(
        "double_well_tilted",
        make_dw_dataset(gamma=0.08, left_count_tr=16, right_count_tr=16, seed_offset=3),
        {"gamma": 0.08, "regime": "tilted_asymmetric", "dt": 0.05, "T_steps": 120},
    )

    # 3. 4D Integrable Coupled Nonlinear Oscillator on T^2 (K = 2 independent invariants I_1, I_2)
    # In action-angle coordinates:
    #   I_1 = 0.5 * (q_1^2 + p_1^2),  I_2 = 0.5 * (q_2^2 + p_2^2)
    #   omega_1(I) = 1.0 + 0.5 * I_1 + 0.25 * I_2
    #   omega_2(I) = sqrt(2) + 0.25 * I_1 + 0.4 * I_2
    # Embedded into observed coordinates x in R^4 via a canonical/smooth nonlinear mixing
    # diffeomorphism M_mix so that ||x||_2 and coordinate block norms in x-space are NOT constant!
    rng_tor = np.random.default_rng(202605)
    n_total_tor = 72  # 44 train, 8 val, 20 test orbits
    I1_samples = rng_tor.uniform(0.25, 1.50, size=n_total_tor)
    I2_samples = rng_tor.uniform(0.25, 1.50, size=n_total_tor)
    T_tor = 120
    dt_tor = 0.08
    t_tor = np.arange(T_tor) * dt_tor

    # Fixed orthogonal mixing + smooth cubic shear diffeomorphism in R^4 so observed coordinates
    # mix both oscillators and have non-constant Euclidean norms along every orbit.
    Q_mix = np.array(
        [
            [0.80, -0.36, 0.44, 0.18],
            [0.36, 0.80, -0.18, 0.44],
            [-0.44, 0.18, 0.80, -0.36],
            [-0.18, -0.44, 0.36, 0.80],
        ],
        dtype=np.float64,
    )

    def mix_torus_coords(u: np.ndarray) -> np.ndarray:
        v = u @ Q_mix.T
        # Smooth invertible shear:
        x0 = v[..., 0] + 0.18 * np.sin(v[..., 1] + v[..., 2])
        x1 = v[..., 1] + 0.18 * np.cos(v[..., 2] - v[..., 3])
        x2 = v[..., 2] + 0.15 * np.sin(v[..., 0] * 0.7)
        x3 = v[..., 3] + 0.15 * np.cos(v[..., 1] * 0.7)
        return np.stack([x0, x1, x2, x3], axis=-1)

    tor_orbits = []
    tor_invariants = []
    for i in range(n_total_tor):
        i1, i2 = float(I1_samples[i]), float(I2_samples[i])
        r1, r2 = np.sqrt(2.0 * i1), np.sqrt(2.0 * i2)
        om1 = 1.0 + 0.5 * i1 + 0.25 * i2
        om2 = np.sqrt(2.0) + 0.25 * i1 + 0.40 * i2
        th1_0, th2_0 = rng_tor.uniform(0, 2 * np.pi, size=2)
        th1 = th1_0 + om1 * t_tor
        th2 = th2_0 + om2 * t_tor
        u_traj = np.stack(
            [r1 * np.cos(th1), r1 * np.sin(th1), r2 * np.cos(th2), r2 * np.sin(th2)],
            axis=-1,
        )
        tor_orbits.append(mix_torus_coords(u_traj))
        tor_invariants.append([i1, i2])

    tor_orbits_arr = np.stack(tor_orbits, axis=0)
    tor_inv_arr = np.array(tor_invariants, dtype=np.float64)

    save_split(
        "coupled_torus_4d",
        {
            "train_orbits": tor_orbits_arr[:44],
            "train_invariants": tor_inv_arr[:44],
            "val_orbits": tor_orbits_arr[44:52],
            "val_invariants": tor_inv_arr[44:52],
            "test_orbits": tor_orbits_arr[52:],
            "test_invariants": tor_inv_arr[52:],
        },
        {"dt": dt_tor, "T_steps": T_tor, "K": 2, "n_train": 44, "n_val": 8, "n_test": 20},
    )

    # 4. 1D Focusing Discrete Nonlinear Schrodinger (DNLS) Equation on S=16 lattice sites (R^{32})
    # i dpsi_n/dt = -(psi_{n+1} - 2 psi_n + psi_{n-1}) - g_nl |psi_n|^2 psi_n
    # Represent state as x = [q_1..q_S, p_1..p_S] in R^{2S} where psi_n = q_n + i p_n:
    #   dq_n/dt = -(p_{n+1} - 2 p_n + p_{n-1}) - g_nl (q_n^2 + p_n^2) p_n
    #   dp_n/dt =  (q_{n+1} - 2 q_n + q_{n-1}) + g_nl (q_n^2 + p_n^2) q_n
    # Integrate with 10x sub-stepped RK4 WITHOUT ANY POST-STEP NORM PROJECTION!
    S_sites = 16
    g_nl = 1.0
    dt_dnls = 0.04
    substeps_dnls = 10

    def dnls_rhs(state):
        q = state[..., :S_sites]
        p = state[..., S_sites:]
        amp_sq = q**2 + p**2
        q_lap = np.roll(q, -1, axis=-1) - 2.0 * q + np.roll(q, 1, axis=-1)
        p_lap = np.roll(p, -1, axis=-1) - 2.0 * p + np.roll(p, 1, axis=-1)
        dq = -p_lap - g_nl * amp_sq * p
        dp = q_lap + g_nl * amp_sq * q
        return np.concatenate([dq, dp], axis=-1)

    rng_dnls = np.random.default_rng(202609)
    n_total_dnls = 48  # 28 train, 8 val, 12 test trajectories
    T_dnls_train = 60
    T_dnls_long = 250  # Long rollout horizon for test trajectories

    dnls_orbits = []
    dnls_norms = []
    max_intrinsic_rk4_drift = 0.0

    sites = np.arange(S_sites, dtype=np.float64)
    for i in range(n_total_dnls):
        if i == 36:
            # Exemplar held-out test trajectory 0: clean mobile bright sech soliton gliding from site 3 -> 11
            center = 3.0
            width = 1.9
            k_wave = 1.0 * (2.0 * np.pi / S_sites)
            target_norm = 1.38
        else:
            center = rng_dnls.uniform(1.5, S_sites - 1.5)
            width = rng_dnls.uniform(1.6, 2.4)
            k_wave = rng_dnls.choice([-1, 1]) * (2.0 * np.pi / S_sites)
            target_norm = rng_dnls.uniform(0.95, 1.85)

        dist = np.minimum(np.abs(sites - center), S_sites - np.abs(sites - center))
        env = 1.0 / np.cosh(dist / width)
        phase = k_wave * (sites - center)
        q0 = env * np.cos(phase)
        p0 = env * np.sin(phase)
        x0 = np.concatenate([q0, p0], axis=-1)
        x0 = x0 * (target_norm / np.linalg.norm(x0))

        traj = solve_orbit(dnls_rhs, x0, dt=dt_dnls, T_steps=T_dnls_long, substeps=substeps_dnls)
        norms_t = np.sum(traj**2, axis=-1)
        rel_drift = float(np.max(np.abs(norms_t - norms_t[0])) / norms_t[0])
        max_intrinsic_rk4_drift = max(max_intrinsic_rk4_drift, rel_drift)

        dnls_orbits.append(traj)
        dnls_norms.append(float(norms_t[0]))

    dnls_orbits_arr = np.stack(dnls_orbits, axis=0)
    dnls_norms_arr = np.array(dnls_norms, dtype=np.float64)

    save_split(
        "dnls_1d",
        {
            "train_orbits": dnls_orbits_arr[:28, :T_dnls_train, :],
            "train_norms": dnls_norms_arr[:28],
            "val_orbits": dnls_orbits_arr[28:36, :T_dnls_train, :],
            "val_norms": dnls_norms_arr[28:36],
            "test_orbits_short": dnls_orbits_arr[36:, :T_dnls_train, :],
            "test_orbits_long": dnls_orbits_arr[36:, :, :],
            "test_norms": dnls_norms_arr[36:],
        },
        {
            "S_sites": S_sites,
            "state_dim": 2 * S_sites,
            "dt": dt_dnls,
            "substeps": substeps_dnls,
            "post_step_projection": False,
            "max_intrinsic_rk4_relative_norm_drift": max_intrinsic_rk4_drift,
            "n_train": 28,
            "n_val": 8,
            "n_test": 12,
        },
    )

    manifest_path = DATA_DIR / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks-only", action="store_true", help="Retain the locked dataset files and rerun analytical checks only.")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stage1_results.json"
    archive_path = RESULTS_DIR / "stage1_results_pre_revision.json"
    if out_path.exists() and not archive_path.exists():
        shutil.copy2(out_path, archive_path)
    print("[Stage 1] Running Proposition 1 verifications & counterexamples...")
    prop1 = verify_proposition_1()
    print(
        f"  -> 2x2 twist singular values: {prop1['twist_counterexample_2x2']['singular_values']} "
        f"(theory: {prop1['twist_counterexample_2x2']['theoretical_singular_values']})"
    )
    print(
        f"  -> 1->2 spiral singular value: {prop1['spiral_counterexample_1to2']['singular_value']:.8f} "
        f"(theory: {prop1['spiral_counterexample_1to2']['theoretical_singular_value']:.8f})"
    )

    print("[Stage 1] Running Proposition 2 verifications, stabilized spectra & depth-100 sweep...")
    prop2 = verify_proposition_2()
    print(
        f"  -> 2x2 rank counterexample: rank(A)={prop2['counterexample_2x2_tangential_rank']['rank_A']}, "
        f"rank(A|_T)={prop2['counterexample_2x2_tangential_rank']['rank_A_restricted_to_T']}, "
        f"rank(J)={prop2['counterexample_2x2_tangential_rank']['rank_J']}"
    )
    max_spec_err = max(item["max_rad_err"] for item in prop2["rmsnorm_stabilized_spectra"])
    print(f"  -> Max stabilized RMSNorm singular value formula error: {max_spec_err:.3e}")

    print("[Stage 1] Running Antipodal Sign-Flip verification...")
    antipodal = verify_antipodal_sign_flip()

    print("[Stage 1] Running Resonant 1:1 2D Harmonic Oscillator control (Theorem 1)...")
    resonant_ho = verify_resonant_harmonic_oscillator_control()
    print(
        f"  -> Resonant 1:1 HO max within-orbit CV: {resonant_ho['max_within_orbit_cv']:.3e}, "
        f"same-shell radius spread: {resonant_ho['same_shell_relative_radius_spread']*100:.1f}%"
    )

    print("[Stage 1] Running Translated Identity Encoder control (Theorem 2 / Section 6)...")
    translated_ctrl = verify_translated_encoder_control()
    print(
        f"  -> Shift a=10000: rel_intertwining_err={translated_ctrl['shift_sweep'][-1]['rel_intertwining_err']:.3e}, "
        f"mean_radial_cv={translated_ctrl['shift_sweep'][-1]['mean_radial_cv']:.3e}"
    )

    if args.checks_only:
        splits_manifest = json.loads((DATA_DIR / "splits_manifest.json").read_text())
        for record in splits_manifest.values():
            actual = hashlib.sha256((ROOT / record["file"]).read_bytes()).hexdigest()
            if actual != record["sha256"]:
                raise RuntimeError(f"Dataset hash mismatch: {record['file']}")
    else:
        print("[Stage 1] Generating whole-orbit disjoint dataset splits with SHA-256 manifests...")
        splits_manifest = generate_whole_orbit_datasets()
    print(
        f"  -> DNLS intrinsic RK4 relative norm drift (no projection, 250 steps): "
        f"{splits_manifest['dnls_1d']['meta']['max_intrinsic_rk4_relative_norm_drift']:.3e}"
    )

    stage1_out = {
        "schema_version": 2,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "proposition_1": prop1,
        "proposition_2": prop2,
        "antipodal_sign_flip": antipodal,
        "resonant_harmonic_oscillator_control": resonant_ho,
        "translated_encoder_control": translated_ctrl,
        "splits_manifest": splits_manifest,
    }
    out_path = RESULTS_DIR / "stage1_results.json"
    out_path.write_text(json.dumps(stage1_out, indent=2))
    print(f"[Stage 1] Saved complete Stage 1 results to {out_path}")


if __name__ == "__main__":
    main()
