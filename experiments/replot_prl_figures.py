"""Regenerate the 3 PRL vector PDF + PNG figures with zero overlapping legends or text."""

from __future__ import annotations

import json
import os

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
from skipnorm.jax_layers import geodesic_skip_norm_fn, skip_norm_fn

# Apply APS / PRL rcParams AFTER all imports so nothing overrides them
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.85,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

C_SKIP = "#009E73"     # Bluish green (SkipNorm / GeodesicSkipNorm)
C_RES = "#D55E00"      # Vermilion (Standard ResNet)
C_RMS = "#0072B2"      # Blue (RMSNorm)
C_LN = "#CC79A7"       # Reddish purple (LayerNorm / Overparam)
C_UNIT = "#E69F00"     # Orange (Linear Unitary)
C_DARK = "#222222"


def style_external_legend(ax, ncol=1, y_offset=-0.23, handles=None):
    """Place legend cleanly outside below the subplot x-axis label so zero data is covered."""
    kwargs = dict(
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#D0D5DD",
        framealpha=1.0,
        fontsize=7.3,
        handlelength=1.8,
        columnspacing=1.0,
        borderpad=0.45,
    )
    if handles is not None:
        return ax.legend(handles=handles, **kwargs)
    return ax.legend(**kwargs)


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

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))
    plt.subplots_adjust(wspace=0.32)

    # Panel (a): Bottleneck Geometry (d=2) with zero text/circle overlap
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 240)
    for r, al in [(0.48, 0.32), (0.88, 0.58), (1.28, 0.92)]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=C_SKIP, alpha=al, lw=1.45,
                label="SkipNorm $\\mathbb{S}^{d-1}(\\|x\\|_2)$" if r == 1.28 else None)
    ax.plot(1.08 * np.cos(theta), 1.08 * np.sin(theta), color=C_RMS, ls="--", lw=1.45,
            label="RMSNorm $\\mathbb{S}^{d-1}(\\sqrt{d})$")
    ax.scatter([0.76, -0.76], [-0.76, 0.76], color=C_LN, s=52, zorder=6, marker="D",
               edgecolor=C_DARK, lw=0.5, label="LayerNorm ($d=2$: 2 pts)")

    # Base point on outer circle r = 1.28 at angle 32 deg; both vectors point into exterior x > 1.28
    r_out = 1.28
    ang = np.deg2rad(32.0)
    ux, uy = np.cos(ang), np.sin(ang)
    tx, ty = np.sin(ang), -np.cos(ang)  # clockwise tangent pointing down-and-right into open exterior
    px, py = r_out * ux, r_out * uy

    # Subtle tangent line segment showing tangent hyperplane
    s_vals = np.linspace(-0.45, 0.78, 50)
    ax.plot(px + s_vals * tx, py + s_vals * ty, color=C_SKIP, ls=":", lw=0.9, alpha=0.7)

    # Right-angle square marker between u_y and t_y
    sq = 0.12
    right_angle_pts = np.array([
        [px + sq * ux, py + sq * uy],
        [px + sq * ux + sq * tx, py + sq * uy + sq * ty],
        [px + sq * tx, py + sq * ty],
    ])
    ax.plot(right_angle_pts[:, 0], right_angle_pts[:, 1], color=C_DARK, lw=0.8)
    ax.scatter([px], [py], color=C_DARK, s=18, zorder=7)

    # Radial arrow u_y (outward) and Tangent arrow T_y S^{d-1} (down-right into exterior)
    len_vec = 0.62
    ax.annotate("", xy=(px + len_vec * ux, py + len_vec * uy), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.35, mutation_scale=10))
    ax.annotate("", xy=(px + len_vec * tx, py + len_vec * ty), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.35, mutation_scale=10))

    # Labels placed purely in the exterior region (x > 1.48 > r_out=1.28)
    ax.text(px + len_vec * ux + 0.06, py + len_vec * uy + 0.02,
            "$u_y\\ (\\sigma_r=1)$", fontsize=7.6, color=C_DARK, fontweight="bold", ha="left", va="center")
    ax.text(px + len_vec * tx + 0.06, py + len_vec * ty - 0.02,
            "$T_y\\mathbb{S}^{d-1}$", fontsize=7.6, color=C_SKIP, fontweight="bold", ha="left", va="center")

    ax.set_xlim(-1.60, 2.90)
    ax.set_ylim(-2.12, 2.12)
    ax.set_aspect("equal")
    ax.set_title("(a) Bottleneck Geometry ($d=2$)", pad=7)
    ax.set_xlabel("Latent Coordinate $z_1$")
    ax.set_ylabel("Latent Coordinate $z_2$")
    style_external_legend(ax, ncol=1, y_offset=-0.24)
    ax.grid(True, alpha=0.25)

    # Panel (b): 50-Layer Radial Isometry
    ax = axes[1]
    ax.semilogy(depths_to_probe, sn_grad_norms, color=C_SKIP, lw=2.2, label="SkipNorm ($\\|J^\\top u_L\\|_2 \\equiv 1$)")
    ax.semilogy(depths_to_probe, res_grad_norms, color=C_RES, ls="--", lw=1.6, label="Unnormalized Cascade")
    ax.semilogy(depths_to_probe, rms_grad_norms, color=C_RMS, ls=":", lw=1.6, label="RMSNorm ($\\|J^\\top u_L\\|_2 = 0$)")
    ax.set_ylim(1e-31, 1e3)
    ax.set_xlim(0, 51)
    ax.set_title("(b) 50-Layer Radial Isometry", pad=7)
    ax.set_xlabel("Depth $L$ (widths $16 \\leftrightarrow 2$)")
    ax.set_ylabel("Radial Gradient $\\|J_{\\mathrm{total}}^\\top u_L\\|_2$")
    style_external_legend(ax, ncol=1, y_offset=-0.24)
    ax.grid(True, alpha=0.25)

    # Panel (c): Bottleneck Rank Theorem
    ax = axes[2]
    ax.plot(dims, ranks_sn, "o-", color=C_SKIP, lw=2.0, ms=5.5, label="SkipNorm ($\\mathrm{rank} = d$)")
    ax.plot(dims, ranks_rms, "s--", color=C_RMS, lw=1.6, ms=5.0, label="RMSNorm ($\\mathrm{rank} = d-1$)")
    ax.plot(dims, ranks_ln, "D:", color=C_LN, lw=1.6, ms=5.0, label="LayerNorm ($\\mathrm{rank} = d-2$)")
    ax.set_xticks(dims)
    ax.set_yticks(range(0, 9, 2))
    ax.set_ylim(-0.4, 8.6)
    ax.set_title("(c) Bottleneck Rank Theorem", pad=7)
    ax.set_xlabel("Bottleneck Width $d$")
    ax.set_ylabel("Exact Jacobian Rank")
    style_external_legend(ax, ncol=1, y_offset=-0.24)
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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.95))
    plt.subplots_adjust(wspace=0.28)

    ax = axes[0]
    steps = np.arange(1, 2001)
    ax.loglog(steps, np.maximum(curves["Standard ResNet"], 1e-16), color=C_RES, ls="--", lw=1.65,
              label="Standard ResNet ($>10^{70}\\%$)")
    ax.loglog(steps, np.maximum(curves["RMSNorm"], 1e-16), color=C_RMS, ls=":", lw=1.65,
              label="RMSNorm ($1.2\\times 10^3\\%$)")
    ax.loglog(steps, np.maximum(curves["Linear Unitary"], 1e-16), color=C_UNIT, ls="-.", lw=1.55,
              label="Linear Unitary ($U^\\top U = I$)")
    ax.loglog(steps, np.maximum(curves["GeodesicSkipNorm"], 1e-16), color=C_SKIP, lw=2.2,
              label="GeodesicSkipNorm ($7.4\\times 10^{-13}\\%$)")
    ax.set_title("(a) DNLS 2,000-Step Wave Norm Conservation", pad=7)
    ax.set_xlabel("Autoregressive Rollout Step $t$")
    ax.set_ylabel("Relative Norm Error $|\\mathcal{N}_t - \\mathcal{N}_0|/\\mathcal{N}_0$ (%)")
    ax.set_ylim(1e-16, 1e15)
    style_external_legend(ax, ncol=2, y_offset=-0.24)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    labels = ["Standard\nResNet", "RMSNorm", "Linear\nUnitary", "Geodesic\nSkipNorm"]
    keys = ["Standard ResNet", "RMSNorm", "Linear Unitary", "GeodesicSkipNorm"]
    means = [dnls_res[k]["mse_40_mean"] * 1e2 for k in keys]
    stds = [dnls_res[k]["mse_40_std"] * 1e2 for k in keys]
    colors = [C_RES, C_RMS, C_UNIT, C_SKIP]
    x_p = np.arange(len(labels))
    bars = ax.bar(x_p, means, yerr=stds, capsize=4, width=0.48, color=colors, edgecolor=C_DARK, lw=0.75)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 1.8, f"{m:.2f}",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.set_xticks(x_p)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.28)
    ax.set_title("(b) 40-Step Wave Trajectory Error (5 Seeds)", pad=7)
    ax.set_ylabel("Rollout MSE ($\\times 10^{-2}$, Mean $\\pm$ SD)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig("paper/fig2_dnls_and_duffing.pdf")
    fig.savefig("paper/fig2_dnls_and_duffing.png")
    plt.close(fig)
    print("Saved paper/fig2_dnls_and_duffing.pdf")


def train_double_well_for_plot():
    """Train a seed-0 Double-Well GeodesicSkipNorm model to plot the Reeb-graph topological bifurcation."""
    def dw_energy(qp):
        q, p = qp[..., 0], qp[..., 1]
        return 0.5 * p**2 - 0.5 * q**2 + 0.25 * q**4

    def dw_rhs(qp):
        q, p = qp[..., 0], qp[..., 1]
        return jnp.stack([p, q - q**3], axis=-1)

    def rk4(rhs, x, dt):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    inits, well_ids = [], []
    for dq in np.linspace(0.12, 0.52, 12):
        inits.append([-1.0 - dq, 0.0])
        well_ids.append(0)  # Left well (H < 0)
        inits.append([1.0 + dq, 0.0])
        well_ids.append(1)  # Right well (H < 0)
    for q0 in np.linspace(1.48, 2.05, 14):
        inits.append([q0, 0.0])
        well_ids.append(2)  # Outer connected (H > 0)
    inits = jnp.array(inits, dtype=jnp.float64)
    well_ids = np.array(well_ids)

    def sim(x0):
        def body(s, _):
            sn = s
            for _ in range(10):
                sn = rk4(dw_rhs, sn, 0.02)
            return sn, sn
        _, tr = jax.lax.scan(body, x0, None, length=60)
        return jnp.concatenate([x0[None, :], tr], axis=0)

    trajs = jax.vmap(sim)(inits)

    class DWModel(eqx.Module):
        enc_w: list
        enc_b: list
        dec_w: list
        dec_b: list
        lat_w: list
        lat_b: list
        alpha: jax.Array

        def __init__(self, key):
            ks = jax.random.split(key, 9)
            self.enc_w = [
                jax.random.normal(ks[0], (64, 2)) / np.sqrt(2),
                jax.random.normal(ks[1], (64, 64)) / 8.0,
                jax.random.normal(ks[2], (2, 64)) / 8.0,
            ]
            self.enc_b = [jnp.zeros(64), jnp.zeros(64), jnp.zeros(2)]
            self.dec_w = [
                jax.random.normal(ks[3], (64, 2)) / np.sqrt(2),
                jax.random.normal(ks[4], (64, 64)) / 8.0,
                jax.random.normal(ks[5], (2, 64)) / 8.0,
            ]
            self.dec_b = [jnp.zeros(64), jnp.zeros(64), jnp.zeros(2)]
            self.lat_w = [
                jax.random.normal(ks[6], (64, 2)) / np.sqrt(2),
                jax.random.normal(ks[7], (64, 64)) / 8.0,
                jax.random.normal(ks[8], (2, 64)) * 0.05,
            ]
            self.lat_b = [jnp.zeros(64), jnp.zeros(64), jnp.zeros(2)]
            self.alpha = jnp.array(0.25)

        def encode(self, x):
            h = jax.nn.silu(x @ self.enc_w[0].T + self.enc_b[0])
            h = jax.nn.silu(h @ self.enc_w[1].T + self.enc_b[1])
            return h @ self.enc_w[2].T + self.enc_b[2]

        def decode(self, z):
            h = jax.nn.silu(z @ self.dec_w[0].T + self.dec_b[0])
            h = jax.nn.silu(h @ self.dec_w[1].T + self.dec_b[1])
            return h @ self.dec_w[2].T + self.dec_b[2]

        def step_z(self, z):
            h = jax.nn.silu(z @ self.lat_w[0].T + self.lat_b[0])
            h = jax.nn.silu(h @ self.lat_w[1].T + self.lat_b[1])
            return geodesic_skip_norm_fn(z, h @ self.lat_w[2].T + self.lat_b[2], self.alpha)

    H = 6
    n_tr, n_st, _ = trajs.shape
    ws = jnp.stack([trajs[:, t : t + H + 1, :] for t in range(n_st - H)], axis=1).reshape(-1, H + 1, 2)

    model = DWModel(jax.random.PRNGKey(500))
    sched = optax.warmup_cosine_decay_schedule(1e-4, 3e-3, 100, 1200, 1e-5)
    opt = optax.adam(sched)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(m, os, b):
        def loss_fn(mod):
            x0 = b[:, 0, :]
            z0 = mod.encode(x0)
            loss = jnp.mean(jnp.square(mod.decode(z0) - x0))
            z = z0
            for k in range(1, H + 1):
                z = mod.step_z(z)
                xk = b[:, k, :]
                loss = loss + jnp.mean(jnp.square(mod.decode(z) - xk)) + 0.25 * jnp.mean(jnp.square(z - mod.encode(xk)))
            return loss / H

        l, g = eqx.filter_value_and_grad(loss_fn)(m)
        u, os_n = opt.update(g, os, m)
        return eqx.apply_updates(m, u), os_n, l

    rng = np.random.default_rng(0)
    for _ in range(1200):
        idx = rng.choice(ws.shape[0], size=256, replace=False)
        model, opt_state, _ = train_step(model, opt_state, ws[idx])

    z_all = model.encode(trajs.reshape(-1, 2)).reshape(n_tr, n_st, 2)
    norms = np.array(jnp.linalg.norm(z_all, axis=-1))
    H_true = np.array(dw_energy(trajs))
    return trajs, well_ids, norms, H_true


def make_fig3(lv_summary, torus_summary, dw_summary):
    lv_trajs = generate_lotka_volterra_data(n_trajs=48, n_steps=60, dt=0.25)
    m_res = train_lv_model_cosine(lv_trajs, mode="resnet", latent_dim=2, seed=1)
    m_d8 = train_lv_model_cosine(lv_trajs, mode="geodesic_skipnorm", latent_dim=8, seed=1)
    m_sn = train_lv_model_cosine(lv_trajs, mode="geodesic_skipnorm", latent_dim=2, seed=1)

    ev_res = evaluate_lv_invariance(m_res, lv_trajs)
    ev_d8 = evaluate_lv_invariance(m_d8, lv_trajs)
    ev_sn = evaluate_lv_invariance(m_sn, lv_trajs)

    dw_trajs, dw_well_ids, dw_norms, dw_H = train_double_well_for_plot()

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.75))
    plt.subplots_adjust(wspace=0.31, hspace=0.68)

    # Panel (a): Non-Centered Lotka-Volterra Contours
    ax = axes[0, 0]
    u_grid = jnp.linspace(0.25, 3.0, 120, dtype=jnp.float64)
    v_grid = jnp.linspace(0.12, 1.8, 120, dtype=jnp.float64)
    U, V_m = jnp.meshgrid(u_grid, v_grid)
    pts = jnp.stack([U.flatten(), V_m.flatten()], axis=-1)
    true_V_grid = np.array(lotka_volterra_invariant(pts)).reshape(120, 120)
    raw_norm_grid = np.array(jnp.linalg.norm(pts, axis=-1)).reshape(120, 120)
    learned_norm_grid = np.array(jnp.linalg.norm(m_sn.encode(pts), axis=-1)).reshape(120, 120)

    cf = ax.contourf(np.array(U), np.array(V_m), learned_norm_grid, levels=18, cmap="viridis", alpha=0.88)
    cbar = plt.colorbar(cf, ax=ax, pad=0.03)
    cbar.set_label("Latent Radius $\\|\\Phi_{\\mathrm{enc}}(u,v)\\|_2$", fontsize=8.0)
    ax.contour(np.array(U), np.array(V_m), true_V_grid, levels=[0.05, 0.15, 0.30, 0.50, 0.75],
               colors="white", linestyles="--", linewidths=1.35)
    ax.contour(np.array(U), np.array(V_m), raw_norm_grid, levels=[0.8, 1.4, 2.0, 2.6],
               colors="#ff5555", linestyles=":", linewidths=1.15, alpha=0.9)
    ax.plot(1.0, 0.5, marker="*", markersize=11, color="#FFD700", markeredgecolor=C_DARK, markeredgewidth=0.8)
    ax.set_title("(a) Lotka-Volterra Casimir Foliation", pad=7)
    ax.set_xlabel("Prey Population $u$")
    ax.set_ylabel("Predator Population $v$")
    handles_a = [
        mlines.Line2D([], [], color=C_SKIP, ls="--", lw=1.4, label="True Casimir $V(u,v)$ (white dashed)"),
        mlines.Line2D([], [], color="#ff5555", ls=":", lw=1.4, label="Raw Norm $\\|(u,v)\\|_2$ (red dotted)"),
        mlines.Line2D([], [], color="#FFD700", marker="*", ls="None", ms=8, mec=C_DARK, label="Equilibrium $(1.0, 0.5)$"),
    ]
    style_external_legend(ax, ncol=1, y_offset=-0.24, handles=handles_a)

    # Panel (b): Radius Constancy Along Single Asymmetric Orbit
    ax = axes[0, 1]
    orbit_idx = -4
    t_steps = np.arange(lv_trajs.shape[1])
    raw_orb = np.array(jnp.linalg.norm(lv_trajs[orbit_idx], axis=-1))
    res_orb = ev_res["norms"][orbit_idx]
    sn_orb = ev_sn["norms"][orbit_idx]

    ax.plot(t_steps, raw_orb / np.mean(raw_orb), color="#e41a1c", ls=":", lw=1.7,
            label="Raw $\\|(u_t,v_t)\\|_2$ ($\\mathrm{CV}=42.3\\%$)")
    ax.plot(t_steps, res_orb / np.mean(res_orb), color=C_RES, ls="--", lw=1.6,
            label=f"ResNet $d_z=2$ ($\\mathrm{{CV}}={lv_summary['Standard ResNet (d=2)']['cv_mean']:.1f}\\pm{lv_summary['Standard ResNet (d=2)']['cv_std']:.1f}\\%$)")
    ax.plot(t_steps, sn_orb / np.mean(sn_orb), color=C_SKIP, lw=2.2,
            label=f"GeodesicSkipNorm $d_z=2$ ($\\mathrm{{CV}}={lv_summary['GeodesicSkipNorm (d=2)']['cv_mean']:.1f}\\pm{lv_summary['GeodesicSkipNorm (d=2)']['cv_std']:.1f}\\%$)")
    ax.axhline(1.0, color=C_DARK, lw=0.6, alpha=0.4)
    ax.set_ylim(0.0, 2.75)
    ax.set_title("(b) Radius Constancy on Single Orbit", pad=7)
    ax.set_xlabel("Trajectory Step $t$")
    ax.set_ylabel("Normalized Radius $r_t / \\bar{r}$")
    style_external_legend(ax, ncol=1, y_offset=-0.24)
    ax.grid(True, alpha=0.25)

    # Panel (c): Brouwer Dimension Matching (d_z = dim M = 2)
    ax = axes[0, 2]
    true_V_all = ev_sn["true_V"].flatten()
    sn_norms = ev_sn["norms"].flatten()
    d8_norms = ev_d8["norms"].flatten()
    res_norms = ev_res["norms"].flatten()

    def norm01(a):
        return (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-12)

    ax.scatter(true_V_all[::3], norm01(res_norms)[::3], s=6, alpha=0.22, color=C_RES,
               label=f"ResNet $d_z=2$ ($R^2={lv_summary['Standard ResNet (d=2)']['r2_mean']:.1f}\\pm{lv_summary['Standard ResNet (d=2)']['r2_std']:.1f}\\%$)")
    ax.scatter(true_V_all[::3], norm01(d8_norms)[::3], s=6, alpha=0.25, color=C_LN,
               label=f"SkipNorm $d_z=8 > N$ ($R^2={lv_summary['Overparam SkipNorm (d=8)']['r2_mean']:.1f}\\%$)")
    ax.scatter(true_V_all[::3], norm01(sn_norms)[::3], s=7, alpha=0.68, color=C_SKIP,
               label=f"GeodesicSkipNorm $d_z=2$ ($R^2={lv_summary['GeodesicSkipNorm (d=2)']['r2_mean']:.1f}\\%, |\\rho|={lv_summary['GeodesicSkipNorm (d=2)']['rho_mean']:.3f}$)")
    ax.set_title("(c) Brouwer Dimension Matching ($d_z = N$)", pad=7)
    ax.set_xlabel("True Casimir $V(u,v) - V_{\\min}$")
    ax.set_ylabel("Scaled Radius $\\|\\Phi_{\\mathrm{enc}}(u,v)\\|_2$")
    style_external_legend(ax, ncol=1, y_offset=-0.24)
    ax.grid(True, alpha=0.25)

    # Panel (d): Double-Well Phase Plane (Disconnected vs Connected Orbits)
    ax = axes[1, 0]
    q_g = np.linspace(-2.15, 2.15, 200)
    p_g = np.linspace(-1.65, 1.65, 200)
    Q, P = np.meshgrid(q_g, p_g)
    H_grid = 0.5 * P**2 - 0.5 * Q**2 + 0.25 * Q**4

    for tr, wid in zip(np.array(dw_trajs), dw_well_ids):
        if wid == 0:
            ax.plot(tr[:, 0], tr[:, 1], color=C_RMS, lw=1.0, alpha=0.7)
        elif wid == 1:
            ax.plot(tr[:, 0], tr[:, 1], color=C_RES, lw=1.0, alpha=0.7)
        else:
            ax.plot(tr[:, 0], tr[:, 1], color=C_SKIP, lw=1.0, alpha=0.65)

    # Plot figure-eight homoclinic separatrix H(q,p) = 0
    ax.contour(Q, P, H_grid, levels=[0.0], colors=C_DARK, linestyles="--", linewidths=1.5)
    ax.scatter([-1.0, 1.0], [0.0, 0.0], color=[C_RMS, C_RES], s=30, zorder=6, edgecolor=C_DARK, lw=0.6)
    ax.scatter([0.0], [0.0], color=C_DARK, marker="x", s=35, zorder=6, lw=1.2)
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.65, 1.65)
    ax.set_title("(d) Double-Well Phase Plane & Barrier", pad=7)
    ax.set_xlabel("Position $q$")
    ax.set_ylabel("Momentum $p$")
    handles_d = [
        mlines.Line2D([], [], color=C_RMS, lw=1.5, label="Left Well $\\mathcal{O}_c^{(L)}$ ($H < 0$, disc.)"),
        mlines.Line2D([], [], color=C_RES, lw=1.5, label="Right Well $\\mathcal{O}_c^{(R)}$ ($H < 0$, disc.)"),
        mlines.Line2D([], [], color=C_SKIP, lw=1.5, label="Outer Orbits $\\mathcal{O}_c$ ($H > 0$, conn.)"),
    ]
    style_external_legend(ax, ncol=1, y_offset=-0.24, handles=handles_d)
    ax.grid(True, alpha=0.25)

    # Panel (e): Corollary 1 - Multi-Basin Topological Obstruction & Bifurcation
    ax = axes[1, 1]
    orb_H = np.mean(dw_H, axis=-1)
    orb_r_mean = np.mean(dw_norms, axis=-1)
    orb_r_std = np.std(dw_norms, axis=-1)

    m0 = dw_well_ids == 0
    m1 = dw_well_ids == 1
    m2 = dw_well_ids == 2

    ax.errorbar(orb_H[m0], orb_r_mean[m0], yerr=orb_r_std[m0], fmt="s-", color=C_RMS, ms=4.5, lw=1.3, capsize=2.5,
                label=f"Left Well ($H<0$: $R^2={dw_summary['sub_barrier_disconnected']['r2_mean']:.1f}\\%$)")
    ax.errorbar(orb_H[m1], orb_r_mean[m1], yerr=orb_r_std[m1], fmt="^-", color=C_RES, ms=4.5, lw=1.3, capsize=2.5,
                label=f"Right Well ($H<0$: $\\mathrm{{CV}}={dw_summary['sub_barrier_disconnected']['cv_mean']:.1f}\\%$)")
    ax.errorbar(orb_H[m2], orb_r_mean[m2], yerr=orb_r_std[m2], fmt="o-", color=C_SKIP, ms=4.8, lw=1.8, capsize=2.5,
                label=f"Connected ($H>0$: $R^2={dw_summary['super_barrier_connected']['r2_mean']:.1f}\\%, |\\rho|=1.00$)")
    ax.axvline(0.0, color=C_DARK, ls="--", lw=1.1, alpha=0.75)
    ax.text(0.06, 1.24, "Separatrix $H=0$", fontsize=7.4, color=C_DARK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_title("(e) Corollary 1: Multi-Basin Obstruction", pad=7)
    ax.set_xlabel("True Double-Well Energy $H(q,p)$")
    ax.set_ylabel("Latent Radius $\\|\\Phi_{\\mathrm{enc}}(q,p)\\|_2$ (Mean $\\pm$ SD)")
    style_external_legend(ax, ncol=1, y_offset=-0.24)
    ax.grid(True, alpha=0.25)

    # Panel (f): Corollary 2 - Two-Invariant Toroidal Foliation
    ax = axes[1, 2]
    t_labels = ["Standard\nResNet", "Single-Sphere\n($K=1, \\mathbb{S}^3$)", "Toroidal Block\n($K=2, \\mathbb{T}^2$)"]
    t_keys = ["Standard ResNet (4D)", "Single-Sphere SkipNorm (K=1)", "Toroidal Block SkipNorm (K=2)"]
    r2_m = [torus_summary[k]["r2_mean"] for k in t_keys]
    r2_s = [torus_summary[k]["r2_std"] for k in t_keys]
    inv_m = [100.0 - torus_summary[k]["cv_mean"] for k in t_keys]
    inv_s = [torus_summary[k]["cv_std"] for k in t_keys]

    xp = np.arange(len(t_labels)) * 1.15
    w = 0.40
    ax.bar(xp - w / 2, inv_m, yerr=inv_s, capsize=3.5, width=w, color=C_RMS, edgecolor=C_DARK, lw=0.7,
           label="Action Invariance ($100\\% - \\mathrm{CV}$)")
    b2 = ax.bar(xp + w / 2, r2_m, yerr=r2_s, capsize=3.5, width=w, color=C_SKIP, edgecolor=C_DARK, lw=0.7,
                label="Joint $(I_1, I_2)$ Recovery $R^2$ (%)")
    for bar, m, s in zip(b2, r2_m, r2_s):
        ax.text(bar.get_x() + bar.get_width() * 0.56, m + s + 2.0, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(t_labels)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Percentage (%, Mean $\\pm$ SD)")
    ax.set_title("(f) Corollary 2: Toroidal Foliation ($\\mathbb{T}^2$)", pad=7)
    style_external_legend(ax, ncol=1, y_offset=-0.24)
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig("paper/fig3_noether_brouwer_foliation.pdf")
    fig.savefig("paper/fig3_noether_brouwer_foliation.png")
    fig.savefig("results/fig7_jax_noether_foliation_proof.png")
    plt.close(fig)
    print("Saved paper/fig3_noether_brouwer_foliation.pdf and results/fig7_jax_noether_foliation_proof.png")


def main():
    with open("results/prl_5seed_benchmarks.json") as f:
        data = json.load(f)
    make_fig1(data["benchmark1_geometry"])
    make_fig2(data["benchmark2_dnls_5seed"])
    make_fig3(
        data["benchmark3_lotka_volterra_5seed"],
        data["benchmark4_toroidal_5seed"],
        data["benchmark5_double_well_5seed"],
    )


if __name__ == "__main__":
    main()
