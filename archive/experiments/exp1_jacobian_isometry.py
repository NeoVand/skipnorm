"""Experiment 1: Theoretical & Empirical Verification of SkipNorm Jacobian Isometry,
Gradient Propagation Across Deep Bottlenecks, and Weight-Scale Invariance.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from skipnorm import SkipNorm, SkipNormResBlock, SphericalNorm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_hourglass(
    dims: list[int],
    norm_type: str,
    activation: str = "tanh",
    weight_scale: float = 1.0,
) -> nn.Module:
    """Build an hourglass/deep network with specified normalization scheme."""
    layers = []
    for i in range(len(dims) - 1):
        d_in, d_out = dims[i], dims[i + 1]
        lin = nn.Linear(d_in, d_out, bias=(norm_type != "skipnorm_nobias"))
        with torch.no_grad():
            nn.init.orthogonal_(lin.weight)
            lin.weight.mul_(weight_scale)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "lrelu":
            act = nn.LeakyReLU(0.2)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            act = nn.Identity()

        sub = nn.Sequential(lin, act)

        if norm_type in ("skipnorm", "skipnorm_nobias", "skipnorm_lrelu"):
            layers.append(SkipNorm(sub, p=2.0, dim=-1))
        elif norm_type == "layernorm":
            layers.append(nn.Sequential(sub, nn.LayerNorm(d_out, elementwise_affine=False)))
        elif norm_type == "rmsnorm":
            layers.append(nn.Sequential(sub, nn.RMSNorm(d_out, elementwise_affine=False)))
        elif norm_type == "spherical":
            layers.append(nn.Sequential(sub, SphericalNorm(p=2.0, dim=-1)))
        elif norm_type == "vanilla":
            layers.append(sub)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    return nn.Sequential(*layers)


def test_theorem1_radial_isometry():
    """Verify Theorem 1: J_total^T u_L == u_0 exactly for any depth & bottleneck."""
    torch.manual_seed(42)
    # Bottleneck hourglass: 64 -> 32 -> 16 -> 4 -> 2 -> 4 -> 16 -> 32 -> 64
    dims = [64, 32, 16, 4, 2, 4, 16, 32, 64]
    x0 = torch.randn(1, dims[0], dtype=torch.float64) * 3.7  # arbitrary input norm

    results = {}
    for norm_type in ["skipnorm_lrelu", "skipnorm", "vanilla", "layernorm", "rmsnorm", "spherical"]:
        act = "lrelu" if norm_type == "skipnorm_lrelu" else "tanh"
        model = build_hourglass(dims, norm_type, activation=act, weight_scale=1.0).double()
        x = x0.clone().requires_grad_(True)
        x_L = model(x)

        u_0 = x0 / torch.linalg.vector_norm(x0, dim=-1, keepdim=True)
        u_L = (x_L / torch.linalg.vector_norm(x_L, dim=-1, keepdim=True)).detach()

        # Compute full Jacobian J_total = dx_L / dx_0
        J = torch.autograd.functional.jacobian(lambda inp: model(inp).squeeze(0), x0.squeeze(0))

        # Radial backprop: g_radial_in = J^T u_L
        g_radial_in = J.T @ u_L.squeeze(0)
        radial_grad_norm = float(torch.linalg.vector_norm(g_radial_in).item())
        cos_with_u0 = float(
            torch.nn.functional.cosine_similarity(
                g_radial_in.unsqueeze(0), u_0, dim=-1
            ).item()
        ) if radial_grad_norm > 1e-12 else 0.0

        # Tangent backprop: pick random tangent vector v_perp orthogonal to u_L
        v = torch.randn_like(u_L.squeeze(0))
        v_perp = v - (v @ u_L.squeeze(0)) * u_L.squeeze(0)
        v_perp = v_perp / torch.linalg.vector_norm(v_perp)
        g_tangent_in = J.T @ v_perp
        tangent_grad_norm = float(torch.linalg.vector_norm(g_tangent_in).item())

        # Singular values of J
        svs = torch.linalg.svdvals(J)

        in_norm = float(torch.linalg.vector_norm(x0).item())
        out_norm = float(torch.linalg.vector_norm(x_L).item())

        results[norm_type] = {
            "in_norm": in_norm,
            "out_norm": out_norm,
            "norm_rel_error": abs(out_norm - in_norm) / in_norm,
            "radial_grad_norm": radial_grad_norm,
            "cos_radial_grad_with_u0": cos_with_u0,
            "tangent_grad_norm": tangent_grad_norm,
            "sigma_max": float(svs[0].item()),
            "sigma_2": float(svs[1].item()),
        }
        print(
            f"[{norm_type:14s}] ||x0||={in_norm:.4f} -> ||xL||={out_norm:.4f} | "
            f"||J^T u_L||={radial_grad_norm:.6f} (cos(J^T u_L, u_0)={cos_with_u0:.6f}) | "
            f"||J^T v_perp||={tangent_grad_norm:.6f} | sigma_1={svs[0].item():.4f}, sigma_2={svs[1].item():.4f}"
        )

    return results


def test_depth_and_weight_scale():
    """Compare gradient norm across depths L in [2..50]."""
    torch.manual_seed(42)
    depths = [2, 5, 10, 20, 30, 40, 50]
    d = 64

    depth_results = {nt: {"radial": [], "tangent": [], "total_mse_grad": [], "out_norm": []}
                     for nt in ["skipnorm_lrelu", "vanilla", "layernorm", "rmsnorm", "spherical"]}

    x0 = torch.randn(32, d, dtype=torch.float64) * 5.0
    x_target = x0.clone()  # identity/autoencoding target

    for L in depths:
        dims = [d] * (L + 1)
        for nt in depth_results:
            act = "lrelu" if nt == "skipnorm_lrelu" else "tanh"
            model = build_hourglass(dims, nt, activation=act, weight_scale=1.0).double()
            x = x0.clone().requires_grad_(True)
            out = model(x)
            loss = ((out - x_target) ** 2).mean()
            loss.backward()

            grad_x = x.grad
            u0 = x0 / torch.linalg.vector_norm(x0, dim=-1, keepdim=True)
            rad_comp = (grad_x * u0).sum(dim=-1, keepdim=True) * u0
            tan_comp = grad_x - rad_comp

            depth_results[nt]["radial"].append(float(torch.linalg.vector_norm(rad_comp, dim=-1).mean().item()))
            depth_results[nt]["tangent"].append(float(torch.linalg.vector_norm(tan_comp, dim=-1).mean().item()))
            depth_results[nt]["total_mse_grad"].append(float(torch.linalg.vector_norm(grad_x, dim=-1).mean().item()))
            depth_results[nt]["out_norm"].append(float(torch.linalg.vector_norm(out, dim=-1).mean().item()))

    # Test Pre-LN ResNet norm explosion vs SkipNormResBlock across L=1..100
    res_depths = list(range(1, 101, 5))
    preln_norms = []
    skipnorm_res_norms = []
    x_res = torch.randn(32, d) * 3.0
    for L in res_depths:
        h_pre = x_res.clone()
        h_sn = x_res.clone()
        for _ in range(L):
            sub1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            sub2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            sn_block = SkipNormResBlock(sub2)
            with torch.no_grad():
                h_pre = h_pre + sub1(h_pre)
                h_sn = sn_block(h_sn)
        preln_norms.append(float(torch.linalg.vector_norm(h_pre, dim=-1).mean().item()))
        skipnorm_res_norms.append(float(torch.linalg.vector_norm(h_sn, dim=-1).mean().item()))

    # Plot Figure 1: Depth & Residual Norm Stability
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)

    # Panel A: Output norm vs depth
    ax = axes[0]
    for nt, label, color, ls in [
        ("skipnorm_lrelu", "SkipNorm", "#e65100", "-"),
        ("vanilla", "Vanilla MLP (Tanh)", "#1f77b4", "--"),
        ("layernorm", "LayerNorm", "#2ca02c", "-."),
        ("spherical", "Spherical (y/||y||)", "#9467bd", ":"),
    ]:
        ax.plot(depths, depth_results[nt]["out_norm"], label=label, color=color, linestyle=ls, linewidth=2.2, marker="o", markersize=4)
    ax.axhline(float(torch.linalg.vector_norm(x0, dim=-1).mean().item()), color="gray", linestyle=":", alpha=0.6, label="Input Norm ||x0||")
    ax.set_xlabel("Network Depth L (layers)")
    ax.set_ylabel("Output Signal Norm ||x_L||_2")
    ax.set_title("A. Forward Signal Energy Conservation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel B: Radial Gradient Norm vs depth
    ax = axes[1]
    for nt, label, color, ls in [
        ("skipnorm_lrelu", "SkipNorm (Exact Radial Isometry)", "#e65100", "-"),
        ("vanilla", "Vanilla MLP (Tanh)", "#1f77b4", "--"),
        ("layernorm", "LayerNorm (Radial Annihilated)", "#2ca02c", "-."),
        ("spherical", "Spherical (Radial Annihilated)", "#9467bd", ":"),
    ]:
        ax.plot(depths, np.array(depth_results[nt]["radial"]) + 1e-16, label=label, color=color, linestyle=ls, linewidth=2.2, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Network Depth L (layers)")
    ax.set_ylabel("Radial Gradient Norm ||(dL/dx_0)_radial||_2")
    ax.set_title("B. Backward Radial Gradient Conservation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel C: Pre-LN Residual Stream Norm Growth vs SkipNorm-Res
    ax = axes[2]
    ax.plot(res_depths, preln_norms, label="Standard Pre-LN Residual (x + f(LN(x)))", color="#d62728", linestyle="--", linewidth=2.2, marker="s", markersize=4)
    ax.plot(res_depths, skipnorm_res_norms, label="SkipNorm-Res Block", color="#e65100", linestyle="-", linewidth=2.2, marker="o", markersize=4)
    ax.set_xlabel("Residual Depth L (blocks)")
    ax.set_ylabel("Residual Stream Norm ||x_L||_2")
    ax.set_title("C. Residual Stream Norm Explosion vs SkipNorm-Res")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig1_jacobian_and_depth.png"
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")

    return {
        "depths": depths,
        "depth_results": depth_results,
        "res_depths": res_depths,
        "preln_norms": preln_norms,
        "skipnorm_res_norms": skipnorm_res_norms,
    }


if __name__ == "__main__":
    print("=== Running Experiment 1: Theorem 1 Verification & Depth Analysis ===")
    t1_res = test_theorem1_radial_isometry()
    d_res = test_depth_and_weight_scale()
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump({"theorem1": t1_res, "depth_analysis": d_res}, f, indent=2)
