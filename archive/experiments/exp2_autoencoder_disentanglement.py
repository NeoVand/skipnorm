"""Experiment 2: Autoencoder Benchmark on MNIST, Ablations, Polar Disentanglement,
and Out-of-Distribution (OOD) Scale Generalization.

Tests:
1. Latent dimension sweep d in [2, 5, 10, 15, 25, 40, 50] comparing:
   - Vanilla AE
   - BatchNorm AE
   - LayerNorm AE
   - RMSNorm AE
   - Spherical AE (unit-norm without ||x|| skip)
   - EndOnly-SkipNorm AE (ablation: SkipNorm only at bottleneck z and output x_hat)
   - SkipNorm AE (ReLU) (testing Theorem 2 orthant effect at low d)
   - SkipNorm AE (LeakyReLU / Full-Sphere NPAE)
2. 2D Latent Space Polar Disentanglement (r = ||z||_2 vs theta = atan2(z2, z1))
3. OOD Dynamic Range Generalization across scale factors c in [0.05 .. 20.0] (400x range).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from skipnorm import SkipNorm, SphericalNorm, skip_norm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist_tensors(device: torch.device):
    train_ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test_ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    x_train = (train_ds.data.float() / 255.0).view(-1, 784).to(device)
    y_train = train_ds.targets.to(device)
    x_test = (test_ds.data.float() / 255.0).view(-1, 784).to(device)
    y_test = test_ds.targets.to(device)
    return x_train, y_train, x_test, y_test


class Autoencoder(nn.Module):
    """Configurable MLP Autoencoder: 784 -> 256 -> 64 -> d -> 64 -> 256 -> 784."""

    def __init__(self, latent_dim: int, mode: str = "vanilla") -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mode = mode
        enc_dims = [784, 256, 64, latent_dim]
        dec_dims = [latent_dim, 64, 256, 784]

        self.enc_layers = nn.ModuleList()
        for i in range(len(enc_dims) - 1):
            d_in, d_out = enc_dims[i], enc_dims[i + 1]
            is_last = i == len(enc_dims) - 2
            self.enc_layers.append(self._make_layer(d_in, d_out, is_bottleneck=is_last, is_output=False))

        self.dec_layers = nn.ModuleList()
        for i in range(len(dec_dims) - 1):
            d_in, d_out = dec_dims[i], dec_dims[i + 1]
            is_last = i == len(dec_dims) - 2
            self.dec_layers.append(self._make_layer(d_in, d_out, is_bottleneck=False, is_output=is_last))

    def _make_layer(self, d_in: int, d_out: int, is_bottleneck: bool, is_output: bool) -> nn.Module:
        mode = self.mode
        use_bias = mode not in ("skipnorm_lrelu_scale_equiv", "skipnorm_relu")
        lin = nn.Linear(d_in, d_out, bias=use_bias)

        if mode == "vanilla":
            act = nn.Identity() if is_bottleneck else (nn.ReLU() if is_output else nn.LeakyReLU(0.2))
            return nn.Sequential(lin, act)

        elif mode == "layernorm":
            if is_bottleneck:
                return nn.Sequential(lin, nn.LayerNorm(d_out))
            elif is_output:
                return nn.Sequential(lin, nn.ReLU())
            else:
                return nn.Sequential(lin, nn.LayerNorm(d_out), nn.LeakyReLU(0.2))

        elif mode == "rmsnorm":
            if is_bottleneck:
                return nn.Sequential(lin, nn.RMSNorm(d_out))
            elif is_output:
                return nn.Sequential(lin, nn.ReLU())
            else:
                return nn.Sequential(lin, nn.RMSNorm(d_out), nn.LeakyReLU(0.2))

        elif mode == "batchnorm":
            if is_bottleneck:
                return nn.Sequential(lin, nn.BatchNorm1d(d_out))
            elif is_output:
                return nn.Sequential(lin, nn.ReLU())
            else:
                return nn.Sequential(lin, nn.BatchNorm1d(d_out), nn.LeakyReLU(0.2))

        elif mode == "spherical":
            act = nn.Identity() if is_bottleneck else (nn.ReLU() if is_output else nn.LeakyReLU(0.2))
            return nn.Sequential(lin, act, SphericalNorm(p=2.0, dim=-1, scale_by_dim=False))

        elif mode == "endonly_skipnorm":
            # Vanilla intermediate layers, SkipNorm only at bottleneck and final output
            act = nn.Identity() if is_bottleneck else (nn.ReLU() if is_output else nn.LeakyReLU(0.2))
            sub = nn.Sequential(lin, act)
            if is_bottleneck or is_output:
                return SkipNorm(sub, p=2.0, dim=-1)
            return sub

        elif mode == "skipnorm_relu":
            # ReLU everywhere except bottleneck if d=2, or ReLU on output (pixels >= 0)
            act = nn.Identity() if is_bottleneck else nn.ReLU()
            return SkipNorm(nn.Sequential(lin, act), p=2.0, dim=-1)

        elif mode in ("skipnorm_lrelu", "skipnorm_lrelu_scale_equiv"):
            # Full-sphere LeakyReLU in hidden layers, Identity at bottleneck (so z spans full S^{d-1}),
            # ReLU at output (since MNIST pixels x >= 0) wrapped in SkipNorm!
            if is_bottleneck:
                act = nn.Identity()
            elif is_output:
                act = nn.ReLU()
            else:
                act = nn.LeakyReLU(0.2)
            return SkipNorm(nn.Sequential(lin, act), p=2.0, dim=-1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.mode == "endonly_skipnorm":
            x0 = x
            for i, layer in enumerate(self.enc_layers):
                if i == len(self.enc_layers) - 1:
                    h = skip_norm(x0, layer.block(h))
                else:
                    h = layer(h)
            return h
        for layer in self.enc_layers:
            h = layer(h)
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        if self.mode == "endonly_skipnorm":
            z0 = z
            for i, layer in enumerate(self.dec_layers):
                if i == len(self.dec_layers) - 1:
                    h = skip_norm(z0, layer.block(h))
                else:
                    h = layer(h)
            return h
        for layer in self.dec_layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def train_and_eval_ae(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    latent_dim: int,
    mode: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[Autoencoder, dict]:
    torch.manual_seed(seed)
    device = x_train.device
    model = Autoencoder(latent_dim=latent_dim, mode=mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train = x_train.shape[0]
    epoch_test_losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            batch = x_train[perm[i : i + batch_size]]
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_hat_test, z_test = model(x_test)
            test_mse = float(F.mse_loss(x_hat_test, x_test).item())
            epoch_test_losses.append(test_mse)

    model.eval()
    with torch.no_grad():
        x_hat_test, z_test = model(x_test)
        total_mse = float(F.mse_loss(x_hat_test, x_test).item())
        norm_x = torch.linalg.vector_norm(x_test, dim=-1)
        norm_xhat = torch.linalg.vector_norm(x_hat_test, dim=-1)
        radial_mse = float((((norm_xhat - norm_x) ** 2) / 784.0).mean().item())
        cos_sim = float(F.cosine_similarity(x_hat_test, x_test, dim=-1).mean().item())

    return model, {
        "final_mse": total_mse,
        "radial_mse": radial_mse,
        "angular_mse": total_mse - radial_mse,
        "cos_sim": cos_sim,
        "epoch_test_losses": epoch_test_losses,
    }


def run_dimension_sweep(x_train, x_test):
    dims = [2, 5, 10, 15, 25, 40, 50]
    modes = [
        "vanilla",
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "endonly_skipnorm",
        "skipnorm_relu",
        "skipnorm_lrelu",
    ]
    results = {m: {} for m in modes}

    for d in dims:
        print(f"\n--- Latent Dimension d = {d} ---")
        for m in modes:
            t0 = time.time()
            _, metrics = train_and_eval_ae(x_train, x_test, latent_dim=d, mode=m, epochs=10)
            dt = time.time() - t0
            results[m][d] = metrics
            print(
                f"  {m:20s} | Test MSE: {metrics['final_mse']:.6f} "
                f"(Radial: {metrics['radial_mse']:.6f}, CosSim: {metrics['cos_sim']:.4f}) [{dt:.1f}s]"
            )

    return dims, results


def run_2d_disentanglement_and_ood(x_train, y_train, x_test, y_test):
    print("\n--- Running 2D Latent Disentanglement & OOD Scale Generalization ---")
    models_2d = {}
    for m in ["vanilla", "layernorm", "skipnorm_lrelu"]:
        model, _ = train_and_eval_ae(x_train, x_test, latent_dim=2, mode=m, epochs=15, seed=42)
        models_2d[m] = model

    disentangle_stats = {}
    for m, model in models_2d.items():
        with torch.no_grad():
            _, z_train = model(x_train)
            _, z_test = model(x_test)

        z_tr_np = z_train.cpu().numpy()
        z_te_np = z_test.cpu().numpy()
        y_tr_np = y_train.cpu().numpy()
        y_te_np = y_test.cpu().numpy()
        x_te_norm = torch.linalg.vector_norm(x_test, dim=-1).cpu().numpy()
        z_te_norm = np.linalg.norm(z_te_np, axis=-1)

        # Correlation R^2 between ||z||_2 and ||x||_2
        corr = np.corrcoef(x_te_norm, z_te_norm)[0, 1]
        r2_norm = float(corr ** 2)

        # Angular classification accuracy using ONLY unit direction z / ||z||_2
        u_tr = z_tr_np / (np.linalg.norm(z_tr_np, axis=-1, keepdims=True) + 1e-8)
        u_te = z_te_np / (np.linalg.norm(z_te_np, axis=-1, keepdims=True) + 1e-8)
        knn_angle = KNeighborsClassifier(n_neighbors=15)
        knn_angle.fit(u_tr, y_tr_np)
        acc_angle = float(knn_angle.score(u_te, y_te_np))

        # Classification using ONLY radius ||z||_2 (should be near chance/low if disentangled)
        knn_rad = KNeighborsClassifier(n_neighbors=15)
        knn_rad.fit(z_tr_np[:, :1] * 0 + np.linalg.norm(z_tr_np, axis=-1, keepdims=True), y_tr_np)
        acc_rad = float(knn_rad.score(z_te_norm[:, None], y_te_np))

        disentangle_stats[m] = {
            "r2_latent_norm_vs_input_norm": r2_norm,
            "knn_acc_pure_angle": acc_angle,
            "knn_acc_pure_radius": acc_rad,
        }
        print(
            f"  [2D {m:16s}] R^2(||z||, ||x||) = {r2_norm:.6f} | "
            f"15-NN Acc on Angle only: {acc_angle*100:.2f}% | Radius only: {acc_rad*100:.2f}%"
        )

    # OOD Scale Generalization at d=15
    print("\n--- Running OOD Dynamic Range Generalization (d=15, scale c in [0.05..20.0]) ---")
    scales = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    ood_results = {}
    for m in ["vanilla", "layernorm", "rmsnorm", "endonly_skipnorm", "skipnorm_lrelu", "skipnorm_lrelu_scale_equiv"]:
        model, _ = train_and_eval_ae(x_train, x_test, latent_dim=15, mode=m, epochs=10, seed=42)
        rel_errors = []
        with torch.no_grad():
            for c in scales:
                x_scaled = x_test * c
                x_hat, _ = model(x_scaled)
                rel_err = float(
                    (torch.linalg.vector_norm(x_hat - x_scaled, dim=-1) /
                     (torch.linalg.vector_norm(x_scaled, dim=-1) + 1e-8)).mean().item()
                )
                rel_errors.append(rel_err)
        ood_results[m] = rel_errors
        print(f"  [OOD {m:26s}] c=0.05: {rel_errors[0]:.4f} | c=1.0: {rel_errors[4]:.4f} | c=20.0: {rel_errors[-1]:.4f}")

    return models_2d, disentangle_stats, scales, ood_results


def plot_exp2_figures(dims, sweep_results, models_2d, x_test, y_test, scales, ood_results):
    # Figure 2: Dimension Sweep, Convergence, and OOD Scale Generalization
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)

    # Panel A: Test MSE vs Latent Dimension
    ax = axes[0]
    style_map = [
        ("skipnorm_lrelu", "SkipNorm AE (LeakyReLU)", "#e65100", "-", "o", 2.5),
        ("skipnorm_relu", "SkipNorm AE (ReLU)", "#ff9800", "-.", "^", 2.0),
        ("endonly_skipnorm", "EndOnly-SkipNorm (Ablation)", "#8c564b", ":", "d", 1.8),
        ("layernorm", "LayerNorm AE", "#2ca02c", "--", "s", 1.8),
        ("batchnorm", "BatchNorm AE", "#9467bd", "--", "v", 1.8),
        ("vanilla", "Vanilla AE", "#1f77b4", "--", "x", 2.0),
    ]
    for m, label, color, ls, marker, lw in style_map:
        ys = [sweep_results[m][d]["final_mse"] for d in dims]
        ax.plot(dims, ys, label=label, color=color, linestyle=ls, marker=marker, linewidth=lw, markersize=5)
    ax.set_xlabel("Latent Bottleneck Dimension (d)")
    ax.set_ylabel("Test Reconstruction MSE")
    ax.set_title("A. MNIST Reconstruction MSE vs Latent Dim")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel B: Training Convergence at d=15
    ax = axes[1]
    epochs_x = list(range(1, 11))
    for m, label, color, ls, marker, lw in style_map:
        ys = sweep_results[m][15]["epoch_test_losses"]
        ax.plot(epochs_x, ys, label=label, color=color, linestyle=ls, marker=marker, linewidth=lw, markersize=4)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Test Reconstruction MSE (d=15)")
    ax.set_title("B. Training Speed & Convergence (d=15)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # Panel C: OOD Scale Generalization across 400x Dynamic Range
    ax = axes[2]
    ood_styles = [
        ("skipnorm_lrelu_scale_equiv", "SkipNorm AE (Bias-Free, Exact Equivariant)", "#d84315", "-", "o", 2.5),
        ("skipnorm_lrelu", "SkipNorm AE (With Bias)", "#ff9800", "-.", "^", 2.0),
        ("endonly_skipnorm", "EndOnly-SkipNorm AE", "#8c564b", ":", "d", 1.8),
        ("layernorm", "LayerNorm AE", "#2ca02c", "--", "s", 1.8),
        ("vanilla", "Vanilla AE", "#1f77b4", "--", "x", 2.0),
    ]
    for m, label, color, ls, marker, lw in ood_styles:
        ax.plot(scales, ood_results[m], label=label, color=color, linestyle=ls, marker=marker, linewidth=lw, markersize=5)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7, label="Training Scale (c=1.0)")
    ax.set_xscale("log")
    ax.set_xlabel("Test Input Scale Multiplier c (log scale, 400x range)")
    ax.set_ylabel("Relative Error ||x_hat - c*x||_2 / ||c*x||_2")
    ax.set_title("C. OOD Dynamic Range Generalization (d=15)")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig2_path = RESULTS_DIR / "fig2_mnist_benchmark_and_ood.png"
    plt.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved {fig2_path}")

    # Figure 3: 2D Latent Space Comparison (Vanilla AE vs LayerNorm AE vs SkipNorm AE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    cmap = plt.get_cmap("tab10")
    titles = [
        ("vanilla", "A. Vanilla AE (2D Latent)\nNorm & Identity Entangled"),
        ("layernorm", "B. LayerNorm AE (2D Latent)\nCollapsed to 1D Circle (Norm Erased!)"),
        ("skipnorm_lrelu", "C. SkipNorm AE (2D Latent)\nRadial = Energy ||x||_2, Angular = Digit Class"),
    ]

    x_te_cpu = x_test.cpu().numpy()
    y_te_cpu = y_test.cpu().numpy()

    for ax, (m, title) in zip(axes, titles):
        model = models_2d[m]
        with torch.no_grad():
            _, z = model(x_test)
        z_np = z.cpu().numpy()

        for digit in range(10):
            mask = y_te_cpu == digit
            ax.scatter(
                z_np[mask, 0],
                z_np[mask, 1],
                s=4,
                alpha=0.5,
                color=cmap(digit),
                label=str(digit),
            )

        # Overlay sample digit thumbnails on the SkipNorm plot (just like Neo's video!)
        if m == "skipnorm_lrelu":
            np.random.seed(7)
            chosen_pts = []
            for idx in np.random.permutation(len(z_np))[:250]:
                pt = z_np[idx]
                if all(np.linalg.norm(pt - c) > 2.2 for c in chosen_pts):
                    chosen_pts.append(pt)
                    img = 1.0 - x_te_cpu[idx].reshape(28, 28)
                    # Make white background transparent
                    rgba = np.zeros((28, 28, 4))
                    rgba[..., 3] = np.clip(1.0 - img, 0, 1)
                    im = OffsetImage(rgba, zoom=0.45)
                    ab = AnnotationBbox(im, (pt[0], pt[1]), frameon=False, alpha=0.85)
                    ax.add_artist(ab)

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Latent Dim 1 (z_1)")
        ax.set_ylabel("Latent Dim 2 (z_2)")
        ax.grid(True, alpha=0.2)
        if m == "vanilla":
            ax.legend(markerscale=3, fontsize=7.5, loc="upper right", ncol=2)

    plt.tight_layout()
    fig3_path = RESULTS_DIR / "fig3_2d_latent_disentanglement.png"
    plt.savefig(fig3_path)
    plt.close(fig)
    print(f"Saved {fig3_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Running Experiment 2 on device: {device} ===")
    x_train, y_train, x_test, y_test = load_mnist_tensors(device)
    dims, sweep_results = run_dimension_sweep(x_train, x_test)
    models_2d, disentangle_stats, scales, ood_results = run_2d_disentanglement_and_ood(
        x_train, y_train, x_test, y_test
    )
    plot_exp2_figures(dims, sweep_results, models_2d, x_test, y_test, scales, ood_results)

    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(
            {
                "dims": dims,
                "sweep_results": sweep_results,
                "disentangle_stats": disentangle_stats,
                "scales": scales,
                "ood_results": ood_results,
            },
            f,
            indent=2,
        )
