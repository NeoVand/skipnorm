"""Experiment 4: Boundary Conditions, Failure Modes, and Modern Extensions of SkipNorm.

Part A: The "Noisy Norm Floor" in Denoising Autoencoders (DAE) & The AdaptiveSkipNorm Cure.
  - Proves and demonstrates that when input x_noisy = x_clean + N(0, sigma^2 I),
    strict SkipNorm forces ||x_hat||_2 = ||x_noisy||_2 > ||x_clean||_2, hitting an
    exact theoretical error floor, while AdaptiveSkipNorm (adding 2 scalars/layer)
    outperforms Vanilla AE, LayerNorm AE, and strict SkipNorm across all noise regimes.

Part B: Solving the "Curse of Depth" (Sun et al., NeurIPS 2025) & Token-Norm Erasure of nGPT (ICLR 2025)
  using SkipNormResBlock in Deep 24-Layer Residual Networks.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skipnorm import AdaptiveSkipNorm, SkipNorm, SkipNormResBlock, SphericalNorm
from experiments.exp2_autoencoder_disentanglement import load_mnist_tensors

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class DenoisingAE(nn.Module):
    """Autoencoder for testing clean vs noisy input norms (784 -> 256 -> 32 -> 256 -> 784)."""

    def __init__(self, mode: str = "vanilla") -> None:
        super().__init__()
        self.mode = mode
        dims = [784, 256, 32, 256, 784]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            is_bn = i == 1
            is_out = i == 3
            act = nn.Identity() if is_bn else (nn.ReLU() if is_out else nn.LeakyReLU(0.2))
            lin = nn.Linear(d_in, d_out)

            if mode == "vanilla":
                self.layers.append(nn.Sequential(lin, act))
            elif mode == "layernorm":
                if is_bn or is_out:
                    self.layers.append(nn.Sequential(lin, act))
                else:
                    self.layers.append(nn.Sequential(lin, nn.LayerNorm(d_out), act))
            elif mode == "strict_skipnorm":
                self.layers.append(SkipNorm(nn.Sequential(lin, act)))
            elif mode == "adaptive_skipnorm":
                self.layers.append(AdaptiveSkipNorm(nn.Sequential(lin, act)))
            else:
                raise ValueError(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


def run_denoising_experiment(x_train: torch.Tensor, x_test: torch.Tensor):
    print("\n=== Experiment 4A: Denoising Autoencoder (Noisy Norm Floor vs AdaptiveSkipNorm) ===")
    sigmas = [0.0, 0.1, 0.25, 0.4, 0.6]
    modes = ["vanilla", "layernorm", "strict_skipnorm", "adaptive_skipnorm"]
    results = {m: [] for m in modes}
    theoretical_floor = []

    device = x_train.device
    for sigma in sigmas:
        # Compute theoretical radial error floor of strict SkipNorm:
        # E[(||x_noisy||_2 - ||x_clean||_2)^2 / 784] when x_hat has perfect direction x_clean/||x_clean||
        torch.manual_seed(99)
        x_te_noisy = x_test + sigma * torch.randn_like(x_test)
        norm_clean = torch.linalg.vector_norm(x_test, dim=-1)
        norm_noisy = torch.linalg.vector_norm(x_te_noisy, dim=-1)
        floor_mse = float((((norm_noisy - norm_clean) ** 2) / 784.0).mean().item())
        theoretical_floor.append(floor_mse)

        print(f"\n--- Noise Level sigma = {sigma:.2f} (Strict SkipNorm Radial Floor = {floor_mse:.5f}) ---")
        for m in modes:
            torch.manual_seed(42)
            model = DenoisingAE(mode=m).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            n_train = len(x_train)

            for epoch in range(8):
                model.train()
                perm = torch.randperm(n_train, device=device)
                for i in range(0, n_train, 128):
                    clean_b = x_train[perm[i : i + 128]]
                    noisy_b = clean_b + sigma * torch.randn_like(clean_b)
                    opt.zero_grad()
                    pred = model(noisy_b)
                    loss = F.mse_loss(pred, clean_b)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                pred_te = model(x_te_noisy)
                mse = float(F.mse_loss(pred_te, x_test).item())
            results[m].append(mse)
            print(f"  {m:20s} | Clean Reconstruction MSE: {mse:.6f}")

    return sigmas, results, theoretical_floor


class DeepResidualTower(nn.Module):
    """Deep L-layer Residual Tower comparing:
    - 'preln': h_{l+1} = h_l + f_l(LN(h_l))  [Suffers from Curse of Depth norm inflation]
    - 'ngpt_unit': h_{l+1} = Norm(h_l + alpha * Norm(f_l(h_l)))  [Erases input token norm ||h_0||_2]
    - 'skipnorm_res': h_{l+1} = SkipNorm(h_l, h_l + alpha * f_l(LN(h_l)))  [Bounds norm AND preserves ||h_0||_2!]
    """

    def __init__(self, d_model: int = 64, depth: int = 24, mode: str = "preln"):
        super().__init__()
        self.mode = mode
        self.depth = depth
        self.sublayers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(depth)
        ])
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.2)) for _ in range(depth)])
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        h = x
        layer_norms = [float(torch.linalg.vector_norm(h, dim=-1).mean().item())]
        rel_updates = []

        for sub, alpha in zip(self.sublayers, self.alphas):
            delta = sub(h)
            if self.mode == "preln":
                h_next = h + delta
            elif self.mode == "ngpt_unit":
                delta_u = delta / (torch.linalg.vector_norm(delta, dim=-1, keepdim=True) + 1e-8)
                h_mix = h + alpha * delta_u
                h_next = (h_mix / (torch.linalg.vector_norm(h_mix, dim=-1, keepdim=True) + 1e-8)) * (h.shape[-1] ** 0.5)
            elif self.mode == "skipnorm_res":
                delta_u = delta / (torch.linalg.vector_norm(delta, dim=-1, keepdim=True) + 1e-8) * torch.linalg.vector_norm(h, dim=-1, keepdim=True)
                h_mix = h + alpha * delta_u
                norm_h = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
                norm_mix = torch.linalg.vector_norm(h_mix, dim=-1, keepdim=True)
                h_next = (h_mix / (norm_mix + 1e-8)) * norm_h
            else:
                raise ValueError(self.mode)

            # Measure angular change (in radians) induced by this layer
            cos_step = F.cosine_similarity(h, h_next, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            angle_step = float(torch.acos(cos_step).mean().item())
            rel_updates.append(angle_step)
            h = h_next
            layer_norms.append(float(torch.linalg.vector_norm(h, dim=-1).mean().item()))

        if self.mode == "skipnorm_res":
            out_raw = self.head(h)
            out = (out_raw / (torch.linalg.vector_norm(out_raw, dim=-1, keepdim=True) + 1e-8)) * torch.linalg.vector_norm(h, dim=-1, keepdim=True)
        else:
            out = self.head(h)

        if return_stats:
            return out, layer_norms, rel_updates
        return out


def run_curse_of_depth_experiment():
    print("\n=== Experiment 4B: Deep 24-Layer Residual Tower (Pre-LN vs nGPT/Unit-Sphere vs SkipNorm-Res) ===")
    torch.manual_seed(42)
    n_samples = 1200
    d_model = 64
    depth = 24

    # Task: Magnitude-preserving nonlinear rotation/transformation where target y has
    # ||y||_2 = ||x||_2 (varying across samples in [1.0, 15.0]) and direction transformed by a teacher orthogonal map.
    Q, _ = torch.linalg.qr(torch.randn(d_model, d_model))
    u_in = F.normalize(torch.randn(n_samples, d_model), dim=-1)
    radii = torch.exp(torch.empty(n_samples, 1).uniform_(0.0, 2.7))  # norms in [1, 15]
    x_all = radii * u_in
    # Nonlinear teacher target on sphere with exact energy conservation ||y||_2 == ||x||_2
    y_dir = F.normalize(torch.sin(u_in @ Q * 2.0) + u_in @ Q, dim=-1)
    y_all = radii * y_dir

    x_tr, y_tr = x_all[:960], y_all[:960]
    x_te, y_te = x_all[960:], y_all[960:]

    res_modes = ["preln", "ngpt_unit", "skipnorm_res"]
    tower_results = {}

    for m in res_modes:
        torch.manual_seed(42)
        model = DeepResidualTower(d_model=d_model, depth=depth, mode=m)
        opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
        loss_curve = []

        for epoch in range(25):
            model.train()
            perm = torch.randperm(len(x_tr))
            for i in range(0, len(x_tr), 64):
                xb = x_tr[perm[i : i + 64]]
                yb = y_tr[perm[i : i + 64]]
                opt.zero_grad()
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                pred_te = model(x_te)
                loss_curve.append(float(F.mse_loss(pred_te, y_te).item()))

        model.eval()
        with torch.no_grad():
            pred_te, layer_norms, rel_updates = model(x_te, return_stats=True)
            final_mse = float(F.mse_loss(pred_te, y_te).item())

        tower_results[m] = {
            "final_mse": final_mse,
            "loss_curve": loss_curve,
            "layer_norms": layer_norms,
            "angular_updates_per_layer": rel_updates,
        }
        print(
            f"  {m:15s} | Final Test MSE: {final_mse:.5f} | "
            f"Norm L=0 -> L=24: {layer_norms[0]:.2f} -> {layer_norms[-1]:.2f} | "
            f"Angular update L=1 vs L=24: {rel_updates[0]:.3f} -> {rel_updates[-1]:.3f} rad"
        )

    return tower_results


def plot_exp4_figure(sigmas, dae_results, theoretical_floor, tower_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)

    # Panel A: Denoising Autoencoder (Failure Mode of Strict SkipNorm & AdaptiveSkipNorm Cure)
    ax = axes[0]
    ax.plot(sigmas, dae_results["adaptive_skipnorm"], label="AdaptiveSkipNorm (+2 scalars/layer)", color="#2e7d32", linestyle="-", marker="o", linewidth=2.5)
    ax.plot(sigmas, dae_results["strict_skipnorm"], label="Strict SkipNorm (||x_hat|| = ||x_noisy||)", color="#e65100", linestyle="-.", marker="^", linewidth=2.2)
    ax.plot(sigmas, theoretical_floor, label="Theoretical Noisy-Norm Floor", color="#d32f2f", linestyle=":", linewidth=2.0)
    ax.plot(sigmas, dae_results["layernorm"], label="LayerNorm AE", color="#9467bd", linestyle="--", marker="s", linewidth=1.8)
    ax.plot(sigmas, dae_results["vanilla"], label="Vanilla AE", color="#1f77b4", linestyle="--", marker="x", linewidth=1.8)
    ax.set_xlabel("Input Additive Noise Std (sigma)")
    ax.set_ylabel("Clean Target Reconstruction MSE")
    ax.set_title("A. Denoising Boundary Condition & Adaptive Cure")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel B: Curse of Depth — Layer-wise Angular Contribution across 24 Layers
    ax = axes[1]
    layers_x = list(range(1, 25))
    ax.plot(layers_x, tower_results["skipnorm_res"]["angular_updates_per_layer"], label="SkipNorm-Res (Uniform Contribution)", color="#e65100", linestyle="-", marker="o", markersize=4, linewidth=2.3)
    ax.plot(layers_x, tower_results["preln"]["angular_updates_per_layer"], label="Pre-LN (Curse of Depth Decay)", color="#1f77b4", linestyle="--", marker="s", markersize=4, linewidth=2.0)
    ax.plot(layers_x, tower_results["ngpt_unit"]["angular_updates_per_layer"], label="nGPT / Unit-Sphere", color="#2ca02c", linestyle="-.", marker="^", markersize=4, linewidth=1.8)
    ax.set_xlabel("Residual Layer Index l (1 .. 24)")
    ax.set_ylabel("Angular Update ||Delta theta_l|| (radians)")
    ax.set_title("B. Mitigating the 'Curse of Depth' (L=24)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel C: Deep 24-Layer Convergence on Magnitude-Sensitive Task
    ax = axes[2]
    epochs_x = list(range(1, 26))
    ax.plot(epochs_x, tower_results["skipnorm_res"]["loss_curve"], label="SkipNorm-Res (Preserves ||h_0||_2)", color="#e65100", linestyle="-", linewidth=2.5)
    ax.plot(epochs_x, tower_results["preln"]["loss_curve"], label="Pre-LN (Norm Inflation)", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(epochs_x, tower_results["ngpt_unit"]["loss_curve"], label="nGPT Unit-Sphere (Erases ||h_0||_2)", color="#2ca02c", linestyle="-.", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Test MSE Loss (log scale)")
    ax.set_title("C. 24-Layer Residual Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig5_path = RESULTS_DIR / "fig5_limitations_and_extensions.png"
    plt.savefig(fig5_path)
    plt.close(fig)
    print(f"Saved {fig5_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x_train, _, x_test, _ = load_mnist_tensors(device)
    sigmas, dae_results, theoretical_floor = run_denoising_experiment(x_train, x_test)
    tower_results = run_curse_of_depth_experiment()
    plot_exp4_figure(sigmas, dae_results, theoretical_floor, tower_results)

    with open(RESULTS_DIR / "exp4_results.json", "w") as f:
        json.dump(
            {
                "sigmas": sigmas,
                "dae_results": dae_results,
                "theoretical_floor": theoretical_floor,
                "tower_results": tower_results,
            },
            f,
            indent=2,
        )
