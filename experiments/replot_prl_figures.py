"""Regenerate the 3 PRL vector PDF + PNG figures using cached 5-seed statistics in results/prl_5seed_benchmarks.json."""

from __future__ import annotations

import json
import os

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments.exp6_jax_rigorous_proofs import (
    evaluate_lv_invariance,
    generate_lotka_volterra_data,
    lotka_volterra_invariant,
)
from experiments.generate_prl_benchmarks_and_figures import (
    DNLSStepper,
    generate_dnls_data,
    train_lv_model_cosine,
)
from skipnorm.jax_layers import skip_norm_fn

# Apply APS / PRL rcParams AFTER all imports so nothing overrides them
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.85,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_SKIP = "#009E73"     # Bluish green (SkipNorm / GeodesicSkipNorm)
C_RES = "#D55E00"      # Vermilion (Standard ResNet)
C_RMS = "#0072B2"      # Blue (RMSNorm)
C_LN = "#CC79A7"       # Reddish purple (LayerNorm / Overparam)
C_UNIT = "#E69F00"     # Orange (Linear Unitary)
C_DARK = "#222222"


def make_fig1(b1):
    key = jax.random.PRNGKey(2026)

    def leaky_relu(z):
        return jnp.where(z >= 0, z, 0.2 * z)

    widths = [16, 8, 4, 2, 4, 8, 16] * 7 + [16, 16]
    weights = []
    for d_in, d_out in zip(widths[:-1], widths[1:]):
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (d_out, d_in), dtype=jnp.float64) / jnp.sqrt(d_in)
        weights.append(W)

    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (16,), dtype=jnp.float64)
    x0 = x0 * (2.5 / jnp.linalg.norm(x0))

    depths_to_probe = list(range(1, 51))
    sn_grad_norms, res_grad_norms, rms_grad_norms = [], [], []

    for L in depths_to_probe:
        def fn_sn(x):
            h = x
            for W in weights[:L]:
                h = skip_norm_fn(h, leaky_relu(W @ h), p=2.0, axis=-1, eps=1e-30)
            return h

        def fn_rms(x):
            h = x
            for W in weights[:L]:
                y = leaky_relu(W @ h)
                h = y / jnp.sqrt(jnp.mean(jnp.square(y)) + 1e-30)
            return h

        def fn_plain(x):
            h = x
            for W in weights[:L]:
                h = leaky_relu(W @ h)
            return h

        y_sn = fn_sn(x0)
        sn_grad_norms.append(float(jnp.linalg.norm(jax.jacfwd(fn_sn)(x0).T @ (y_sn / jnp.linalg.norm(y_sn)))))
        y_rms = fn_rms(x0)
        rms_grad_norms.append(float(jnp.linalg.norm(jax.jacfwd(fn_rms)(x0).T @ (y_rms / jnp.linalg.norm(y_rms))) + 1e-30))
        y_pl = fn_plain(x0)
        res_grad_norms.append(float(jnp.linalg.norm(jax.jacfwd(fn_plain)(x0).T @ (y_pl / jnp.linalg.norm(y_pl))) + 1e-30))

    dims = b1["dims"]
    ranks_ln, ranks_rms, ranks_sn = b1["ranks_ln"], b1["ranks_rms"], b1["ranks_sn"]

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.85))
    plt.subplots_adjust(wspace=0.34)

    # Panel (a)
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    for r, al in [(0.55, 0.35), (1.0, 0.60), (1.45, 0.90)]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=C_SKIP, alpha=al, lw=1.4,
                label="SkipNorm $\\mathbb{S}^{d-1}(\\|x\\|_2)$" if r == 1.45 else None)
    ax.plot(1.18 * np.cos(theta), 1.18 * np.sin(theta), color=C_RMS, ls="--", lw=1.4,
            label="RMSNorm $\\mathbb{S}^{d-1}(\\sqrt{d})$")
    ax.scatter([1.0, -1.0], [-1.0, 1.0], color=C_LN, s=48, zorder=5, marker="D",
               label="LayerNorm ($d=2$: 2 pts)")

    ang = 0.75
    px, py = 1.45 * np.cos(ang), 1.45 * np.sin(ang)
    ax.annotate("", xy=(px + 0.45 * np.cos(ang), py + 0.45 * np.sin(ang)), xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.3))
    ax.annotate("", xy=(px - 0.45 * np.sin(ang), py + 0.45 * np.cos(ang)), xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color=C_SKIP, lw=1.3))
    ax.text(px + 0.22, py + 0.42, "$u_y$ ($\\sigma_r=1$)", fontsize=7.0, color=C_DARK, fontweight="bold")
    ax.text(px - 1.15, py + 0.62, "$T_y\\mathbb{S}^{d-1}$", fontsize=7.0, color=C_SKIP, fontweight="bold")

    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-2.35, 2.35)
    ax.set_aspect("equal")
    ax.set_title("(a) Bottleneck Geometry ($d=2$)")
    ax.set_xlabel("Latent Coordinate $z_1$")
    ax.set_ylabel("Latent Coordinate $z_2$")
    ax.legend(loc="lower left", framealpha=0.94, fontsize=6.8)
    ax.grid(True, alpha=0.25)

    # Panel (b)
    ax = axes[1]
    ax.semilogy(depths_to_probe, sn_grad_norms, color=C_SKIP, lw=2.2, label="SkipNorm ($\\|J^\\top u_L\\|_2 \\equiv 1$)")
    ax.semilogy(depths_to_probe, res_grad_norms, color=C_RES, ls="--", lw=1.5, label="Unnormalized Cascade")
    ax.semilogy(depths_to_probe, rms_grad_norms, color=C_RMS, ls=":", lw=1.5, label="RMSNorm ($\\|J^\\top u_L\\|_2 = 0$)")
    ax.set_ylim(1e-32, 1e3)
    ax.set_title("(b) 50-Layer Radial Isometry")
    ax.set_xlabel("Depth $L$ (widths $16 \\leftrightarrow 2$)")
    ax.set_ylabel("Radial Gradient $\\|J_{\\mathrm{total}}^\\top u_L\\|_2$")
    ax.legend(loc="center right", framealpha=0.94, fontsize=7.0)
    ax.grid(True, alpha=0.25)

    # Panel (c)
    ax = axes[2]
    ax.plot(dims, ranks_sn, "o-", color=C_SKIP, lw=2.0, ms=5.5, label="SkipNorm ($\\mathrm{rank} = d$)")
    ax.plot(dims, ranks_rms, "s--", color=C_RMS, lw=1.6, ms=5.0, label="RMSNorm ($\\mathrm{rank} = d-1$)")
    ax.plot(dims, ranks_ln, "D:", color=C_LN, lw=1.6, ms=5.0, label="LayerNorm ($\\mathrm{rank} = d-2$)")
    ax.set_xticks(dims)
    ax.set_yticks(range(0, 9, 2))
    ax.set_title("(c) Bottleneck Rank Theorem")
    ax.set_xlabel("Bottleneck Width $d$")
    ax.set_ylabel("Exact Jacobian Rank")
    ax.legend(loc="upper left", framealpha=0.94, fontsize=7.0)
    ax.grid(True, alpha=0.25)

    os.makedirs("paper", exist_ok=True)
    fig.savefig("paper/fig1_geometry_and_rank.pdf")
    fig.savefig("paper/fig1_geometry_and_rank.png")
    plt.close(fig)
    print("Saved paper/fig1_geometry_and_rank.pdf")


def make_fig2(dnls_res):
    trajs = generate_dnls_data()
    x_in = trajs[:, :-1, :].reshape(-1, 32)
    x_out = trajs[:, 1:, :].reshape(-1, 32)
    test_trajs = generate_dnls_data(n_trajs=16, n_steps=40, dt=0.04)

    modes = [
        ("Standard ResNet", "resnet"),
        ("RMSNorm", "rmsnorm"),
        ("Linear Unitary", "linear_unitary"),
        ("GeodesicSkipNorm", "geodesic_skipnorm"),
    ]
    curves = {}
    for label, mode in modes:
        key = jax.random.PRNGKey(100)
        model = DNLSStepper(32, 96, mode, key)
        opt = optax.adam(3e-3)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(m, os):
            def loss_fn(mod):
                return jnp.mean(jnp.square(jax.vmap(mod)(x_in) - x_out))
            l, g = eqx.filter_value_and_grad(loss_fn)(m)
            u, os_n = opt.update(g, os, m)
            return eqx.apply_updates(m, u), os_n, l

        for _ in range(500):
            model, opt_state, _ = step(model, opt_state)

        @jax.jit
        def eval_2000(s0):
            def body(s, _):
                sn = jax.vmap(model)(s)
                return sn, sn
            _, preds = jax.lax.scan(body, s0, None, length=2000)
            return jnp.swapaxes(preds, 0, 1)

        preds_2000 = eval_2000(test_trajs[:, 0, :])
        e0 = jnp.sum(jnp.square(test_trajs[:, 0, :]), axis=-1, keepdims=True)
        et = jnp.sum(jnp.square(preds_2000), axis=-1)
        curves[label] = np.array(jnp.mean(jnp.abs(et - e0) / e0, axis=0)) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.85))
    plt.subplots_adjust(wspace=0.28)

    ax = axes[0]
    steps = np.arange(1, 2001)
    ax.loglog(steps, np.maximum(curves["Standard ResNet"], 1e-16), color=C_RES, ls="--", lw=1.6,
              label="Standard ResNet (diverges $>10^{70}\\%$)")
    ax.loglog(steps, np.maximum(curves["RMSNorm"], 1e-16), color=C_RMS, ls=":", lw=1.6,
              label="RMSNorm ($1.2\\times 10^3\\%$ norm error)")
    ax.loglog(steps, np.maximum(curves["Linear Unitary"], 1e-16), color=C_UNIT, ls="-.", lw=1.5,
              label="Linear Unitary ($U^\\top U = I$)")
    ax.loglog(steps, np.maximum(curves["GeodesicSkipNorm"], 1e-16), color=C_SKIP, lw=2.2,
              label="GeodesicSkipNorm ($7.4\\times 10^{-13}\\%$)")
    ax.set_title("(a) DNLS 2,000-Step Wave Norm Conservation")
    ax.set_xlabel("Autoregressive Rollout Step $t$")
    ax.set_ylabel("Relative Norm Error $|\\mathcal{N}_t - \\mathcal{N}_0|/\\mathcal{N}_0$ (%)")
    ax.set_ylim(1e-16, 1e15)
    ax.legend(loc="upper left", framealpha=0.94, fontsize=7.2)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    labels = ["Standard\nResNet", "RMSNorm", "Linear\nUnitary", "Geodesic\nSkipNorm"]
    keys = ["Standard ResNet", "RMSNorm", "Linear Unitary", "GeodesicSkipNorm"]
    means = [dnls_res[k]["mse_40_mean"] * 1e2 for k in keys]
    stds = [dnls_res[k]["mse_40_std"] * 1e2 for k in keys]
    colors = [C_RES, C_RMS, C_UNIT, C_SKIP]
    x_p = np.arange(len(labels))
    bars = ax.bar(x_p, means, yerr=stds, capsize=4, width=0.50, color=colors, edgecolor=C_DARK, lw=0.7)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 1.5, f"{m:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.set_xticks(x_p)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.25)
    ax.set_title("(b) 40-Step Wave Trajectory Error (5 Seeds)")
    ax.set_ylabel("Rollout MSE ($\\times 10^{-2}$, Mean $\\pm$ SD)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig("paper/fig2_dnls_and_duffing.pdf")
    fig.savefig("paper/fig2_dnls_and_duffing.png")
    plt.close(fig)
    print("Saved paper/fig2_dnls_and_duffing.pdf")


def make_fig3(lv_summary, torus_summary):
    lv_trajs = generate_lotka_volterra_data(n_trajs=48, n_steps=60, dt=0.25)
    m_res = train_lv_model_cosine(lv_trajs, mode="resnet", latent_dim=2, seed=1)
    m_d8 = train_lv_model_cosine(lv_trajs, mode="geodesic_skipnorm", latent_dim=8, seed=1)
    m_sn = train_lv_model_cosine(lv_trajs, mode="geodesic_skipnorm", latent_dim=2, seed=1)

    ev_res = evaluate_lv_invariance(m_res, lv_trajs)
    ev_d8 = evaluate_lv_invariance(m_d8, lv_trajs)
    ev_sn = evaluate_lv_invariance(m_sn, lv_trajs)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.2))
    plt.subplots_adjust(wspace=0.30, hspace=0.34)

    # Panel (a)
    ax = axes[0, 0]
    u_grid = jnp.linspace(0.25, 3.0, 120, dtype=jnp.float64)
    v_grid = jnp.linspace(0.12, 1.8, 120, dtype=jnp.float64)
    U, V_m = jnp.meshgrid(u_grid, v_grid)
    pts = jnp.stack([U.flatten(), V_m.flatten()], axis=-1)
    true_V_grid = np.array(lotka_volterra_invariant(pts)).reshape(120, 120)
    raw_norm_grid = np.array(jnp.linalg.norm(pts, axis=-1)).reshape(120, 120)
    learned_norm_grid = np.array(jnp.linalg.norm(m_sn.encode(pts), axis=-1)).reshape(120, 120)

    cf = ax.contourf(np.array(U), np.array(V_m), learned_norm_grid, levels=18, cmap="viridis", alpha=0.88)
    plt.colorbar(cf, ax=ax, label="Learned Latent Radius $\\|\\Phi_{\\mathrm{enc}}(u,v)\\|_2$")
    ax.contour(np.array(U), np.array(V_m), true_V_grid, levels=[0.05, 0.15, 0.30, 0.50, 0.75],
               colors="white", linestyles="--", linewidths=1.3)
    ax.contour(np.array(U), np.array(V_m), raw_norm_grid, levels=[0.8, 1.4, 2.0, 2.6],
               colors="#ff6666", linestyles=":", linewidths=1.1, alpha=0.85)
    ax.plot(1.0, 0.5, "w*", markersize=10, label="Equilibrium $(1.0, 0.5)$")
    ax.set_title("(a) Non-Centered Lotka-Volterra Casimir Contours")
    ax.set_xlabel("Prey Population $u$")
    ax.set_ylabel("Predator Population $v$")
    ax.legend(loc="upper right", framealpha=0.92)

    # Panel (b)
    ax = axes[0, 1]
    orbit_idx = -4
    t_steps = np.arange(lv_trajs.shape[1])
    raw_orb = np.array(jnp.linalg.norm(lv_trajs[orbit_idx], axis=-1))
    res_orb = ev_res["norms"][orbit_idx]
    sn_orb = ev_sn["norms"][orbit_idx]

    ax.plot(t_steps, raw_orb / np.mean(raw_orb), color="#e41a1c", ls=":", lw=1.7,
            label="Raw Input $\\|(u_t,v_t)\\|_2$ ($\\mathrm{CV}=42.3\\%$)")
    ax.plot(t_steps, res_orb / np.mean(res_orb), color=C_RES, ls="--", lw=1.6,
            label=f"ResNet $d=2$ ($\\mathrm{{CV}}={lv_summary['Standard ResNet (d=2)']['cv_mean']:.1f}\\pm{lv_summary['Standard ResNet (d=2)']['cv_std']:.1f}\\%$)")
    ax.plot(t_steps, sn_orb / np.mean(sn_orb), color=C_SKIP, lw=2.2,
            label=f"GeodesicSkipNorm $d=2$ ($\\mathrm{{CV}}={lv_summary['GeodesicSkipNorm (d=2)']['cv_mean']:.1f}\\pm{lv_summary['GeodesicSkipNorm (d=2)']['cv_std']:.1f}\\%$)")
    ax.axhline(1.0, color=C_DARK, lw=0.6, alpha=0.4)
    ax.set_title("(b) Radius Constancy Along Single Asymmetric Orbit")
    ax.set_xlabel("Trajectory Step $t$")
    ax.set_ylabel("Normalized Radius $r_t / \\bar{r}$")
    ax.legend(loc="upper right", framealpha=0.94)
    ax.grid(True, alpha=0.25)

    # Panel (c)
    ax = axes[1, 0]
    true_V_all = ev_sn["true_V"].flatten()
    sn_norms = ev_sn["norms"].flatten()
    d8_norms = ev_d8["norms"].flatten()
    res_norms = ev_res["norms"].flatten()

    def norm01(a):
        return (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-12)

    ax.scatter(true_V_all[::3], norm01(res_norms)[::3], s=6, alpha=0.22, color=C_RES,
               label=f"ResNet $d=2$ ($R^2={lv_summary['Standard ResNet (d=2)']['r2_mean']:.1f}\\pm{lv_summary['Standard ResNet (d=2)']['r2_std']:.1f}\\%$)")
    ax.scatter(true_V_all[::3], norm01(d8_norms)[::3], s=6, alpha=0.25, color=C_LN,
               label=f"SkipNorm $d=8 > \\dim\\mathcal{{M}}$ ($R^2={lv_summary['Overparam SkipNorm (d=8)']['r2_mean']:.1f}\\%$)")
    ax.scatter(true_V_all[::3], norm01(sn_norms)[::3], s=7, alpha=0.68, color=C_SKIP,
               label=f"GeodesicSkipNorm $d=2$ ($R^2={lv_summary['GeodesicSkipNorm (d=2)']['r2_mean']:.1f}\\pm{lv_summary['GeodesicSkipNorm (d=2)']['r2_std']:.1f}\\%$, $|\\rho|={lv_summary['GeodesicSkipNorm (d=2)']['rho_mean']:.3f}$)")
    ax.set_title("(c) Dimension Matching: Brouwer Invariance ($d_z = \\dim\\mathcal{M}$)")
    ax.set_xlabel("True Logarithmic Casimir $V(u,v) - V_{\\min}$")
    ax.set_ylabel("Scaled Latent Radius $\\|\\Phi_{\\mathrm{enc}}(u,v)\\|_2$")
    ax.legend(loc="lower right", markerscale=1.8, framealpha=0.94)
    ax.grid(True, alpha=0.25)

    # Panel (d)
    ax = axes[1, 1]
    t_labels = ["Standard ResNet\n(4D Unconstrained)", "Single-Sphere\n($K=1$ on $\\mathbb{S}^3$)", "Toroidal SkipNorm\n($K=2$ on $\\mathbb{S}^1\\times\\mathbb{S}^1$)"]
    t_keys = ["Standard ResNet (4D)", "Single-Sphere SkipNorm (K=1)", "Toroidal Block SkipNorm (K=2)"]
    r2_m = [torus_summary[k]["r2_mean"] for k in t_keys]
    r2_s = [torus_summary[k]["r2_std"] for k in t_keys]
    inv_m = [100.0 - torus_summary[k]["cv_mean"] for k in t_keys]
    inv_s = [torus_summary[k]["cv_std"] for k in t_keys]

    xp = np.arange(len(t_labels))
    w = 0.34
    ax.bar(xp - w / 2, inv_m, yerr=inv_s, capsize=3.5, width=w, color=C_RMS, edgecolor=C_DARK, lw=0.7,
           label="Action Invariance ($100\\% - \\mathrm{CV}$)")
    b2 = ax.bar(xp + w / 2, r2_m, yerr=r2_s, capsize=3.5, width=w, color=C_SKIP, edgecolor=C_DARK, lw=0.7,
                label="Joint $(I_1, I_2)$ Recovery $R^2$")
    for bar, m, s in zip(b2, r2_m, r2_s):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1.8, f"{m:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(t_labels)
    ax.set_ylim(0, 116)
    ax.set_ylabel("Percentage (%, Mean $\\pm$ SD)")
    ax.set_title("(d) Two-Invariant Toroidal Foliation (5 Seeds)")
    ax.legend(loc="lower right", framealpha=0.94)
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig("paper/fig3_noether_brouwer_foliation.pdf")
    fig.savefig("paper/fig3_noether_brouwer_foliation.png")
    plt.close(fig)
    print("Saved paper/fig3_noether_brouwer_foliation.pdf")


def main():
    with open("results/prl_5seed_benchmarks.json") as f:
        data = json.load(f)
    make_fig1(data["benchmark1_geometry"])
    make_fig2(data["benchmark2_dnls_5seed"])
    make_fig3(data["benchmark3_lotka_volterra_5seed"], data["benchmark4_toroidal_5seed"])


if __name__ == "__main__":
    main()
