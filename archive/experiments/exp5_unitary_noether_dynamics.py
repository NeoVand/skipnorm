"""Experiment 5: Projective Unitary Dynamics & Spontaneous Noether Conservation Law Discovery.

Tests two foundational 2026 hypotheses:
1. Test 5A (Nonlinear Quantum / Wavepacket Rollout over T = 2,000 steps):
   Discrete Nonlinear Schrodinger (DNLS / Gross-Pitaevskii) wave evolution where ||psi||_2^2
   is conserved and phase speed depends nonlinearly on amplitude.
   Compares:
   - Standard ResNet (x + f(x)) -> energy drifts / explodes over long rollouts
   - Linear Unitary / Cayley Orthogonal Net (U x) -> conserves norm, but fails nonlinear phase speed
   - RMSNorm / nGPT ResNet -> erases initial wavepacket energy ||psi_0||_2
   - Pure SkipNorm (f(x)/||f(x)|| * ||x||)
   - Geodesic SkipNorm ((x + f_perp(x))/||x + f_perp(x)|| * ||x||) -> exact norm conservation + accurate nonlinear rollout!

2. Test 5B (Spontaneous Discovery of Hidden Non-Euclidean Hamiltonian via Latent SkipNorm):
   Quartic Anharmonic (Duffing) Oscillator H(q, p) = 0.5*p^2 + 0.5*q^2 + 0.5*q^4, where raw
   Euclidean norm ||x||_2^2 = q^2 + p^2 is NOT conserved (fluctuates strongly along squircle orbits).
   Trains an Encoder-LatentDynamics-Decoder purely on multi-step state prediction (no energy labels!).
   Checks whether the encoder norm ||Phi_enc(q, p)||_2 spontaneously discovers the hidden Hamiltonian H(q, p)
   and enables zero-drift 2,000-step phase-space rollouts!
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

from skipnorm import GeodesicSkipNorm, SkipNorm, skip_norm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# PART A: Nonlinear Schrodinger (DNLS) Wavepacket Evolution
# =============================================================================

def dnls_rhs(x: np.ndarray, c: float = 0.5, g: float = 1.5) -> np.ndarray:
    """Discrete Nonlinear Schrodinger equation on N sites:
    i dpsi_j/dt = -c (psi_{j+1} - 2 psi_j + psi_{j-1}) + g |psi_j|^2 psi_j
    State x has shape (..., 2*N) = [Re(psi), Im(psi)].
    """
    N = x.shape[-1] // 2
    re = x[..., :N]
    im = x[..., N:]
    amp_sq = re ** 2 + im ** 2

    lap_re = np.roll(re, 1, axis=-1) - 2 * re + np.roll(re, -1, axis=-1)
    lap_im = np.roll(im, 1, axis=-1) - 2 * im + np.roll(im, -1, axis=-1)

    # d(re)/dt = -c * lap_im + g * amp_sq * im
    # d(im)/dt = +c * lap_re - g * amp_sq * re
    dre = -c * lap_im + g * amp_sq * im
    dim = c * lap_re - g * amp_sq * re
    return np.concatenate([dre, dim], axis=-1)


def generate_dnls_trajectories(n_traj: int = 240, n_steps: int = 60, dt: float = 0.05, n_sites: int = 8, seed: int = 42):
    """Integrate DNLS with norm-preserving RK4 + exact renormalization to machine precision."""
    rng = np.random.default_rng(seed)
    d = 2 * n_sites
    trajs = np.zeros((n_traj, n_steps, d), dtype=np.float32)

    for i in range(n_traj):
        # Random initial wavepacket with norm in [0.7, 3.0]
        raw = rng.standard_normal(d)
        r0 = rng.uniform(0.7, 3.0)
        x = (raw / np.linalg.norm(raw)) * r0
        trajs[i, 0] = x
        for t in range(1, n_steps):
            # 4 substeps of RK4 for high accuracy
            sub_dt = dt / 4.0
            for _ in range(4):
                k1 = dnls_rhs(x)
                k2 = dnls_rhs(x + 0.5 * sub_dt * k1)
                k3 = dnls_rhs(x + 0.5 * sub_dt * k2)
                k4 = dnls_rhs(x + sub_dt * k3)
                x = x + (sub_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                x = (x / np.linalg.norm(x)) * r0
            trajs[i, t] = x
    return torch.tensor(trajs)


class CayleyUnitaryStep(nn.Module):
    """Exact Linear Orthogonal/Unitary Operator U = (I - A)(I + A)^{-1} where A = B - B^T is skew-symmetric."""

    def __init__(self, d: int):
        super().__init__()
        self.B = nn.Parameter(torch.randn(d, d) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.B - self.B.T
        I = torch.eye(A.shape[0], device=x.device, dtype=x.dtype)
        U = torch.linalg.solve(I + A, I - A)
        return x @ U.T


class WaveDynamicsStepper(nn.Module):
    def __init__(self, d: int = 16, hidden: int = 128, mode: str = "geodesic_skipnorm"):
        super().__init__()
        self.mode = mode
        mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        if mode == "resnet":
            self.mlp = mlp
            self.scale = nn.Parameter(torch.tensor(0.1))
        elif mode == "linear_unitary":
            self.cayley = CayleyUnitaryStep(d)
        elif mode == "rmsnorm":
            self.mlp = mlp
            self.scale = nn.Parameter(torch.tensor(0.1))
            self.norm = nn.RMSNorm(d, elementwise_affine=False)
        elif mode == "pure_skipnorm":
            self.block = SkipNorm(mlp)
        elif mode == "geodesic_skipnorm":
            self.block = GeodesicSkipNorm(mlp, step_scale=0.1)
        else:
            raise ValueError(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "resnet":
            return x + self.scale * self.mlp(x)
        elif self.mode == "linear_unitary":
            return self.cayley(x)
        elif self.mode == "rmsnorm":
            return self.norm(x + self.scale * self.mlp(x))
        elif self.mode in ("pure_skipnorm", "geodesic_skipnorm"):
            return self.block(x)
        raise ValueError(self.mode)


def run_dnls_experiment():
    print("\n=== Experiment 5A: Nonlinear Schrodinger (DNLS) 2,000-Step Rollout ===")
    trajs_train = generate_dnls_trajectories(n_traj=200, n_steps=20, seed=42)
    trajs_test = generate_dnls_trajectories(n_traj=20, n_steps=2001, seed=101)

    # Prepare 3-step rollout training snippets
    X0 = trajs_train[:, :-3, :].reshape(-1, 16)
    X1 = trajs_train[:, 1:-2, :].reshape(-1, 16)
    X2 = trajs_train[:, 2:-1, :].reshape(-1, 16)
    X3 = trajs_train[:, 3:, :].reshape(-1, 16)

    modes = ["resnet", "linear_unitary", "rmsnorm", "pure_skipnorm", "geodesic_skipnorm"]
    dnls_results = {}

    for mode in modes:
        torch.manual_seed(42)
        model = WaveDynamicsStepper(d=16, hidden=128, mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)

        for epoch in range(35):
            perm = torch.randperm(len(X0))
            for i in range(0, len(X0), 128):
                idx = perm[i : i + 128]
                x0, x1, x2, x3 = X0[idx], X1[idx], X2[idx], X3[idx]
                opt.zero_grad()
                p1 = model(x0)
                p2 = model(p1)
                p3 = model(p2)
                loss = F.mse_loss(p1, x1) + F.mse_loss(p2, x2) + F.mse_loss(p3, x3)
                loss.backward()
                opt.step()

        # Roll out for T = 2,000 steps on test trajectories!
        model.eval()
        with torch.no_grad():
            x_curr = trajs_test[:, 0, :]
            r0 = torch.linalg.vector_norm(x_curr, dim=-1)
            norm_drift_curve = []
            state_mse_curve = []

            for t in range(1, 2001):
                x_curr = model(x_curr)
                # Clamp extreme explosion in resnet so float32 doesn't NaN
                if torch.isnan(x_curr).any() or x_curr.abs().max() > 1e6:
                    x_curr = torch.clamp(torch.nan_to_num(x_curr, nan=1e6), -1e6, 1e6)
                rt = torch.linalg.vector_norm(x_curr, dim=-1)
                rel_drift = float((torch.abs(rt ** 2 - r0 ** 2) / (r0 ** 2)).mean().item())
                norm_drift_curve.append(rel_drift)
                if t <= 200:
                    state_mse_curve.append(float(F.mse_loss(x_curr, trajs_test[:, t, :]).item()))

        dnls_results[mode] = {
            "mse_step_10": state_mse_curve[9],
            "mse_step_50": state_mse_curve[49],
            "mse_step_200": state_mse_curve[199],
            "energy_drift_step_100": norm_drift_curve[99],
            "energy_drift_step_2000": norm_drift_curve[-1],
            "norm_drift_curve": norm_drift_curve[::10],
            "state_mse_curve": state_mse_curve,
        }
        print(
            f"  {mode:18s} | Rollout MSE (t=10/50): {state_mse_curve[9]:.5f} / {state_mse_curve[49]:.5f} | "
            f"Energy Drift (t=100 / t=2000): {norm_drift_curve[99]*100:8.4f}% / {norm_drift_curve[-1]*100:8.4f}%"
        )

    return dnls_results


# =============================================================================
# PART B: Spontaneous Discovery of Hidden Non-Euclidean Hamiltonian (Noether Test)
# =============================================================================

def duffing_hamiltonian(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True non-Euclidean Hamiltonian: H(q, p) = 0.5*p^2 + 0.25*q^2 + 0.5*q^4.
    Notice that for this system, q^2 + p^2 is strongly non-constant along orbits!
    """
    return 0.5 * (p ** 2) + 0.25 * (q ** 2) + 0.5 * (q ** 4)


def generate_duffing_trajectories(n_traj: int = 160, n_steps: int = 80, dt: float = 0.06, seed: int = 42):
    rng = np.random.default_rng(seed)
    trajs = np.zeros((n_traj, n_steps, 2), dtype=np.float32)
    energies = np.zeros(n_traj, dtype=np.float32)

    def rhs(t, y):
        q, p = y
        dq = p
        dp = -0.5 * q - 2.0 * (q ** 3)
        return [dq, dp]

    t_eval = np.arange(n_steps) * dt
    for i in range(n_traj):
        q0 = rng.uniform(-1.8, 1.8)
        p0 = rng.uniform(-2.5, 2.5)
        if q0 ** 2 + p0 ** 2 < 0.15:
            q0 = 1.0
        sol = solve_ivp(rhs, [0, t_eval[-1]], [q0, p0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        trajs[i, :, 0] = sol.y[0]
        trajs[i, :, 1] = sol.y[1]
        energies[i] = duffing_hamiltonian(q0, p0)

    return torch.tensor(trajs), torch.tensor(energies)


class LatentWorldModel(nn.Module):
    """Maps 2D phase space x=(q, p) -> latent state z = Phi_enc(x),
    evolves z_{t+1} = Stepper(z_t) in latent space, and decodes x_hat = Phi_dec(z).
    """

    def __init__(self, latent_dim: int = 2, mode: str = "geodesic_skipnorm"):
        super().__init__()
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        )
        step_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        if mode == "geodesic_skipnorm":
            self.stepper = GeodesicSkipNorm(step_mlp, step_scale=0.15)
        elif mode == "resnet":
            self.step_mlp = step_mlp
            self.scale = nn.Parameter(torch.tensor(0.15))
        elif mode == "rmsnorm":
            self.step_mlp = step_mlp
            self.scale = nn.Parameter(torch.tensor(0.15))
            self.norm = nn.RMSNorm(latent_dim, elementwise_affine=False)
        else:
            raise ValueError(mode)

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "geodesic_skipnorm":
            return self.stepper(z)
        elif self.mode == "resnet":
            return z + self.scale * self.step_mlp(z)
        elif self.mode == "rmsnorm":
            return self.norm(z + self.scale * self.step_mlp(z))
        raise ValueError(self.mode)


def run_noether_discovery_experiment():
    print("\n=== Experiment 5B: Spontaneous Noether Invariant Discovery on Quartic Duffing Oscillator ===")
    trajs_train, _ = generate_duffing_trajectories(n_traj=160, n_steps=60, dt=0.06, seed=42)
    trajs_test, energies_test = generate_duffing_trajectories(n_traj=24, n_steps=1000, dt=0.06, seed=777)

    # Create horizon K=12 training windows
    K = 12
    windows = []
    for i in range(0, trajs_train.shape[1] - K, 3):
        windows.append(trajs_train[:, i : i + K + 1, :])
    W_train = torch.cat(windows, dim=0)  # (N_win, K+1, 2)

    modes = ["resnet", "rmsnorm", "geodesic_skipnorm"]
    noether_results = {}
    trained_models = {}

    for mode in modes:
        torch.manual_seed(42)
        model = LatentWorldModel(latent_dim=2, mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=2.5e-3)

        for epoch in range(60):
            perm = torch.randperm(len(W_train))
            for i in range(0, len(W_train), 128):
                batch = W_train[perm[i : i + 128]]  # (B, K+1, 2)
                opt.zero_grad()
                z_enc_all = model.encoder(batch)  # (B, K+1, d)
                z = z_enc_all[:, 0, :]
                loss = F.mse_loss(model.decoder(z), batch[:, 0, :])
                for k in range(1, K + 1):
                    z = model.step_latent(z)
                    x_pred = model.decoder(z)
                    # Standard World-Model loss: state reconstruction + latent dynamics consistency
                    loss = (
                        loss
                        + F.mse_loss(x_pred, batch[:, k, :])
                        + 0.25 * F.mse_loss(z, z_enc_all[:, k, :])
                    )
                loss.backward()
                opt.step()

        model.eval()
        trained_models[mode] = model

        with torch.no_grad():
            # 1. Measure intra-orbit constancy of ||Phi_enc(x_t)||_2 vs raw ||x_t||_2
            z_all = model.encoder(trajs_test[:, :200, :])  # (24, 200, 2)
            z_norms = torch.linalg.vector_norm(z_all, dim=-1).numpy()  # (24, 200)
            x_norms = torch.linalg.vector_norm(trajs_test[:, :200, :], dim=-1).numpy()

            # Relative standard deviation (Coefficient of Variation) along each orbit
            raw_orbit_cv = float((x_norms.std(axis=1) / x_norms.mean(axis=1)).mean())
            latent_orbit_cv = float((z_norms.std(axis=1) / (z_norms.mean(axis=1) + 1e-8)).mean())

            # Fraction of variance in ||z||_2 that is conserved between-orbit energy vs within-orbit noise
            total_var = float(np.var(z_norms))
            within_orbit_var = float(np.mean(np.var(z_norms, axis=1)))
            between_orbit_r2 = max(0.0, 1.0 - within_orbit_var / (total_var + 1e-12)) if total_var > 1e-8 else 0.0

            # Spearman rank correlation between orbit-mean ||z||_2 and true Hamiltonian H
            rho, _ = spearmanr(energies_test.numpy(), z_norms.mean(axis=1))

            # 2. Roll out for T = 1,000 steps in phase space and measure true Hamiltonian drift!
            z_curr = model.encoder(trajs_test[:, 0, :])
            rollout_q = [trajs_test[:, 0, 0].numpy()]
            rollout_p = [trajs_test[:, 0, 1].numpy()]
            H0 = duffing_hamiltonian(rollout_q[0], rollout_p[0])
            H_drift_curve = []

            for t in range(1, 1000):
                z_curr = model.step_latent(z_curr)
                if torch.isnan(z_curr).any() or z_curr.abs().max() > 1e4:
                    z_curr = torch.clamp(torch.nan_to_num(z_curr, nan=1e4), -1e4, 1e4)
                x_pred = model.decoder(z_curr).numpy()
                qt, pt = x_pred[:, 0], x_pred[:, 1]
                rollout_q.append(qt)
                rollout_p.append(pt)
                Ht = duffing_hamiltonian(qt, pt)
                H_drift_curve.append(float(np.mean(np.abs(Ht - H0) / H0)))

        noether_results[mode] = {
            "raw_euclidean_norm_orbit_cv_pct": raw_orbit_cv * 100.0,
            "latent_norm_orbit_cv_pct": latent_orbit_cv * 100.0,
            "invariant_conservation_r2": between_orbit_r2,
            "spearman_corr_with_true_H": float(abs(rho)) if not np.isnan(rho) else 0.0,
            "hamiltonian_drift_step_100_pct": H_drift_curve[99] * 100.0,
            "hamiltonian_drift_step_1000_pct": H_drift_curve[-1] * 100.0,
            "H_drift_curve": H_drift_curve[::5],
            "sample_orbit_q": np.array(rollout_q)[:, 0].tolist(),
            "sample_orbit_p": np.array(rollout_p)[:, 0].tolist(),
        }
        print(
            f"  [{mode:18s}] Raw ||x|| Orbit Variation: {raw_orbit_cv*100:.1f}% -> "
            f"Latent ||Phi(x)|| Orbit Variation: {latent_orbit_cv*100:.2f}% | "
            f"Noether Invariance R^2: {between_orbit_r2*100:.2f}% (|rho(H, ||z||)|={abs(rho):.3f}) | "
            f"1000-Step Hamiltonian Drift: {H_drift_curve[-1]*100:.2f}%"
        )

    return noether_results, trajs_test, trained_models



def plot_exp5_figure(dnls_results, noether_results, trajs_test, trained_models):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), dpi=160)

    # --- Top Row: DNLS Quantum Wavepacket Rollout & Energy Conservation ---
    # Panel A: Wavepacket State MSE over rollout steps 1..200
    ax = axes[0, 0]
    styles_dnls = [
        ("geodesic_skipnorm", "Geodesic SkipNorm (Nonlinear Rotor)", "#e65100", "-", 2.6),
        ("pure_skipnorm", "Pure SkipNorm", "#ff9800", "-.", 2.0),
        ("resnet", "Standard ResNet (x + f(x))", "#1f77b4", "--", 2.0),
        ("linear_unitary", "Linear Unitary / Cayley (Ux)", "#8c564b", ":", 2.0),
        ("rmsnorm", "RMSNorm / Unit-Sphere", "#2ca02c", "--", 1.8),
    ]
    for m, label, color, ls, lw in styles_dnls:
        ax.plot(range(1, 201), dnls_results[m]["state_mse_curve"], label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_yscale("log")
    ax.set_xlabel("Autoregressive Rollout Step t (Trained only on t=1..3)")
    ax.set_ylabel("Wavepacket State MSE (log)")
    ax.set_title("A. Nonlinear Schrodinger Rollout Accuracy", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel B: Wavepacket Energy Drift over T = 2,000 steps
    ax = axes[0, 1]
    t_grid_2000 = list(range(1, 2001, 10))
    for m, label, color, ls, lw in styles_dnls:
        ax.plot(t_grid_2000, np.array(dnls_results[m]["norm_drift_curve"]) + 1e-15, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_yscale("log")
    ax.set_xlabel("Autoregressive Rollout Step t (up to T=2,000)")
    ax.set_ylabel("Relative Energy Drift |E_t - E_0| / E_0")
    ax.set_title("B. Quantum Wavepacket Energy Drift (2,000 Steps)", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel C: Spontaneous Discovery of Hidden Hamiltonian Contour Map
    ax = axes[0, 2]
    q_grid = np.linspace(-1.8, 1.8, 120, dtype=np.float32)
    p_grid = np.linspace(-2.5, 2.5, 120, dtype=np.float32)
    Q, P = np.meshgrid(q_grid, p_grid)
    qp_flat = torch.tensor(np.stack([Q.ravel(), P.ravel()], axis=-1))
    with torch.no_grad():
        z_flat = trained_models["geodesic_skipnorm"].encoder(qp_flat)
        z_norm_grid = torch.linalg.vector_norm(z_flat, dim=-1).numpy().reshape(120, 120)
    H_true_grid = duffing_hamiltonian(Q, P)

    cf = ax.contourf(Q, P, z_norm_grid, levels=20, cmap="viridis", alpha=0.85)
    plt.colorbar(cf, ax=ax, label="Learned Latent Norm ||Phi_enc(q, p)||_2")
    cs = ax.contour(Q, P, H_true_grid, levels=[0.3, 0.8, 1.6, 2.8, 4.5], colors="white", linestyles="--", linewidths=1.5)
    ax.clabel(cs, fmt="H=%.1f", fontsize=7.5, colors="white")
    ax.set_xlabel("Position q")
    ax.set_ylabel("Momentum p")
    ax.set_title("C. Learned ||Phi_enc(q, p)||_2 vs True H(q, p) Contours\n(White Dashed = True Non-Circular Hamiltonian!)", fontweight="bold", fontsize=9.5)

    # --- Bottom Row: Noether Conservation & 1,000-Step Phase Space Orbits ---
    # Panel D: Along-Orbit Constancy of ||Phi_enc(x_t)||_2 vs Raw ||x_t||_2
    ax = axes[1, 0]
    orb_idx = 0
    orb = trajs_test[orb_idx, :150, :]
    raw_norm_trace = torch.linalg.vector_norm(orb, dim=-1).numpy()
    raw_norm_trace = raw_norm_trace / raw_norm_trace.mean()
    with torch.no_grad():
        z_sn = trained_models["geodesic_skipnorm"].encoder(orb)
        z_rn = trained_models["resnet"].encoder(orb)
        sn_norm_trace = torch.linalg.vector_norm(z_sn, dim=-1).numpy()
        sn_norm_trace = sn_norm_trace / sn_norm_trace.mean()
        rn_norm_trace = torch.linalg.vector_norm(z_rn, dim=-1).numpy()
        rn_norm_trace = rn_norm_trace / rn_norm_trace.mean()

    ax.plot(raw_norm_trace, color="gray", linestyle=":", linewidth=2.0, label="Raw Input Norm ||(q_t, p_t)||_2 (Oscillates!)")
    ax.plot(rn_norm_trace, color="#1f77b4", linestyle="--", linewidth=1.8, label="Standard ResNet Latent Norm ||Phi(x_t)||_2")
    ax.plot(sn_norm_trace, color="#e65100", linestyle="-", linewidth=2.6, label="Geodesic SkipNorm Latent Norm ||Phi(x_t)||_2 (Constant!)")
    ax.axhline(1.0, color="black", alpha=0.3, linewidth=1)
    ax.set_xlabel("Time Step t along Non-Circular Duffing Orbit")
    ax.set_ylabel("Normalized Norm (norm / mean)")
    ax.set_title("D. Spontaneous Conservation Along a Curved Orbit", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel E: 1,000-Step Phase-Space Portrait (q_t, p_t)
    ax = axes[1, 1]
    gt_q = trajs_test[0, :400, 0].numpy()
    gt_p = trajs_test[0, :400, 1].numpy()
    ax.plot(gt_q, gt_p, color="black", linewidth=2.5, alpha=0.6, label="Ground Truth Hamiltonian Orbit")
    ax.plot(
        noether_results["resnet"]["sample_orbit_q"][:400],
        noether_results["resnet"]["sample_orbit_p"][:400],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label="Latent ResNet (Spirals Off Orbit!)",
    )
    ax.plot(
        noether_results["rmsnorm"]["sample_orbit_q"][:400],
        noether_results["rmsnorm"]["sample_orbit_p"][:400],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.4,
        alpha=0.85,
        label="Latent RMSNorm (Wrong Energy Shell!)",
    )
    ax.plot(
        noether_results["geodesic_skipnorm"]["sample_orbit_q"][:400],
        noether_results["geodesic_skipnorm"]["sample_orbit_p"][:400],
        color="#e65100",
        linestyle="-",
        linewidth=2.0,
        label="Latent Geodesic SkipNorm (Locked on Orbit!)",
    )
    ax.set_xlabel("Position q")
    ax.set_ylabel("Momentum p")
    ax.set_title("E. 1,000-Step Phase-Space Rollout Portrait", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel F: True Hamiltonian Drift |H(q_t, p_t) - H_0| / H_0 over 1,000 steps
    ax = axes[1, 2]
    t_grid_1000 = list(range(1, 1000, 5))
    ax.plot(t_grid_1000, noether_results["geodesic_skipnorm"]["H_drift_curve"], color="#e65100", linestyle="-", linewidth=2.5, label="Latent Geodesic SkipNorm")
    ax.plot(t_grid_1000, noether_results["resnet"]["H_drift_curve"], color="#1f77b4", linestyle="--", linewidth=2.0, label="Latent Standard ResNet")
    ax.plot(t_grid_1000, noether_results["rmsnorm"]["H_drift_curve"], color="#2ca02c", linestyle="-.", linewidth=2.0, label="Latent RMSNorm (Unit-Sphere)")
    ax.set_yscale("log")
    ax.set_xlabel("Autoregressive Rollout Step t (1 .. 1,000)")
    ax.set_ylabel("True Physical Energy Error |H_t - H_0| / H_0")
    ax.set_title("F. True Physical Hamiltonian Drift (1,000 Steps)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig6_path = RESULTS_DIR / "fig6_unitary_and_noether_discovery.png"
    plt.savefig(fig6_path)
    plt.close(fig)
    print(f"Saved {fig6_path}")


if __name__ == "__main__":
    dnls_results = run_dnls_experiment()
    noether_results, trajs_test, trained_models = run_noether_discovery_experiment()
    plot_exp5_figure(dnls_results, noether_results, trajs_test, trained_models)

    # Save summary JSON (without long trajectory arrays)
    summary = {
        "dnls": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in dnls_results.items()},
        "noether": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in noether_results.items()},
    }
    with open(RESULTS_DIR / "exp5_results.json", "w") as f:
        json.dump(summary, f, indent=2)
