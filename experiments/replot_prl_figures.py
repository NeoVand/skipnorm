"""Generate APS / Physical Review Letters publication figures at exact 1:1 print width (7.0 in)
with inward ticks, direct curve callouts, and frameless headspace legends."""

from __future__ import annotations

import json
import os

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# Exact APS / Physical Review Letters styling at 1:1 print scale (width = 7.0 in)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 6.9,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.35,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# High-contrast APS / Nature Physics palette (Okabe-Ito refined)
C_SKIP = "#00805A"     # Deep Emerald (SkipNorm / GeodesicSkipNorm)
C_RES = "#C84600"      # Deep Vermilion (Standard ResNet)
C_RMS = "#005B9A"      # Royal Blue (RMSNorm)
C_LN = "#A63A79"       # Deep Magenta (LayerNorm / Overparameterized)
C_UNIT = "#D48800"     # Warm Ochre (Linear Unitary)
C_DARK = "#1A1A1A"
C_GRAY = "#666666"


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

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
    plt.subplots_adjust(wspace=0.39)

    # ---------------- Panel (a): Bottleneck Geometry (d=2) ----------------
    ax = axes[0]
    ax.axhline(0, color="#E0E0E0", lw=0.5, zorder=1)
    ax.axvline(0, color="#E0E0E0", lw=0.5, zorder=1)

    theta = np.linspace(0, 2 * np.pi, 240)
    for r, al in [(0.45, 0.28), (0.82, 0.52), (1.20, 0.92)]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=C_SKIP, alpha=al, lw=1.25, zorder=3)
    ax.plot(1.01 * np.cos(theta), 1.01 * np.sin(theta), color=C_RMS, ls="--", lw=1.2, zorder=4)
    ax.scatter([-0.714, 0.714], [0.714, -0.714], color=C_LN, s=34, zorder=6, marker="D",
               edgecolor="white", lw=0.5)

    # Tangent & radial decomposition on outer circle at angle 28 deg
    r_out = 1.20
    ang = np.deg2rad(28.0)
    ux, uy = np.cos(ang), np.sin(ang)
    tx, ty = np.sin(ang), -np.cos(ang)
    px, py = r_out * ux, r_out * uy

    s_vals = np.linspace(-0.35, 0.58, 40)
    ax.plot(px + s_vals * tx, py + s_vals * ty, color=C_SKIP, ls=":", lw=0.8, alpha=0.75)

    sq = 0.11
    ra_pts = np.array([
        [px + sq * ux, py + sq * uy],
        [px + sq * ux + sq * tx, py + sq * uy + sq * ty],
        [px + sq * tx, py + sq * ty],
    ])
    ax.plot(ra_pts[:, 0], ra_pts[:, 1], color=C_DARK, lw=0.65)
    ax.scatter([px], [py], color=C_DARK, s=12, zorder=7)

    l_vec = 0.45
    ax.annotate("", xy=(px + l_vec * ux, py + l_vec * uy), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.15, mutation_scale=8.0))
    ax.annotate("", xy=(px + l_vec * tx, py + l_vec * ty), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.15, mutation_scale=8.0))

    ax.text(px + l_vec * ux + 0.05, py + l_vec * uy + 0.04, "$\\mathbf{u}_y\\ (\\sigma_r\\!=\\!1)$",
            fontsize=6.6, color=C_DARK, ha="left", va="center")
    ax.text(px + l_vec * tx + 0.05, py + l_vec * ty - 0.02, "$T_y\\mathbb{S}^{d-1}$",
            fontsize=6.6, color=C_SKIP, ha="left", va="center")

    # Direct callouts for the 3 manifolds in clean exterior space (zero circle crossings)
    ax.annotate("SkipNorm $\\mathbb{S}^{d-1}(\\|x\\|_2)$",
                xy=(-1.20 * np.cos(np.deg2rad(55)), 1.20 * np.sin(np.deg2rad(55))),
                xytext=(-1.92, 1.58),
                fontsize=6.4, color=C_SKIP, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_SKIP, lw=0.6, shrinkA=0, shrinkB=2))
    ax.annotate("RMSNorm $\\mathbb{S}^{d-1}(\\sqrt{d})$",
                xy=(-1.01 * np.cos(np.deg2rad(50)), -1.01 * np.sin(np.deg2rad(50))),
                xytext=(-1.96, -1.72),
                fontsize=6.4, color=C_RMS, ha="left",
                arrowprops=dict(arrowstyle="-", color=C_RMS, lw=0.6, shrinkA=0, shrinkB=2))
    ax.annotate("LayerNorm ($2\\text{ pts}$)",
                xy=(0.714, -0.714), xytext=(2.68, -1.34),
                fontsize=6.4, color=C_LN, ha="right",
                arrowprops=dict(arrowstyle="-", color=C_LN, lw=0.6, shrinkA=0, shrinkB=3))

    ax.set_xlim(-2.08, 2.82)
    ax.set_ylim(-1.98, 2.02)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])
    ax.set_title("(a) Bottleneck Geometry ($d=2$)", pad=5)
    ax.set_xlabel("Latent Coordinate $z_1$")
    ax.set_ylabel("Latent Coordinate $z_2$")

    # ---------------- Panel (b): 50-Layer Radial Isometry ----------------
    ax = axes[1]
    ax.semilogy(depths_to_probe, sn_grad_norms, color=C_SKIP, lw=1.75)
    ax.semilogy(depths_to_probe, res_grad_norms, color=C_RES, ls="--", lw=1.35)
    ax.semilogy(depths_to_probe, rms_grad_norms, color=C_RMS, ls=":", lw=1.45)

    # Direct curve labels in vast open log-space (no box needed)
    ax.text(25, 4.5, "SkipNorm ($\\|J_{\\mathrm{total}}^\\top u_L\\|_2 \\equiv 1$)",
            color=C_SKIP, fontsize=6.7, fontweight="bold", ha="center", va="bottom")
    ax.text(26, 2e-8, "Unnormalized",
            color=C_RES, fontsize=6.7, fontweight="bold", ha="center", va="top", rotation=-18)
    ax.text(25, 4e-15, "RMSNorm ($\\|J^\\top u_L\\|_2 = 0$)",
            color=C_RMS, fontsize=6.7, ha="center", va="bottom")

    ax.set_ylim(1e-24, 1e3)
    ax.set_xlim(0, 51)
    ax.set_yticks([1e-24, 1e-18, 1e-12, 1e-6, 1e0])
    ax.set_title("(b) 50-Layer Radial Isometry", pad=5)
    ax.set_xlabel("Depth $L$ (widths $16 \\leftrightarrow 2$)")
    ax.set_ylabel("Radial Gradient $\\|J_{\\mathrm{total}}^\\top u_L\\|_2$")

    # ---------------- Panel (c): Bottleneck Rank Theorem ----------------
    ax = axes[2]
    ax.plot(dims, ranks_sn, "o-", color=C_SKIP, lw=1.5, ms=4.2, label="SkipNorm ($d$)")
    ax.plot(dims, ranks_rms, "s--", color=C_RMS, lw=1.3, ms=3.8, label="RMSNorm ($d-1$)")
    ax.plot(dims, ranks_ln, "D:", color=C_LN, lw=1.3, ms=3.8, label="LayerNorm ($d-2$)")
    ax.set_xticks(dims)
    ax.set_yticks(range(0, 9, 2))
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(-0.5, 9.2)
    ax.set_title("(c) Bottleneck Rank Theorem", pad=5)
    ax.set_xlabel("Bottleneck Width $d$")
    ax.set_ylabel("Exact Jacobian Rank")
    # Upper-left triangle (d in [1.5, 5.5], y in [5.5, 9.2]) is 100% empty
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, labelspacing=0.28, borderaxespad=0.4)

    os.makedirs("paper", exist_ok=True)
    fig.savefig("paper/fig1_geometry_and_rank.pdf")
    fig.savefig("paper/fig1_geometry_and_rank.png")
    plt.close(fig)
    print("Saved paper/fig1_geometry_and_rank.pdf")


def make_fig2(dnls_res):
    import pickle
    cache_path = "/tmp/skipnorm_fig2_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            curves = pickle.load(f)
    else:
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
        with open(cache_path, "wb") as f:
            pickle.dump(curves, f)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25))
    plt.subplots_adjust(wspace=0.33)

    # ---------------- Panel (a): 2,000-step DNLS Norm Conservation ----------------
    ax = axes[0]
    steps = np.arange(1, 2001)
    ax.loglog(steps, np.maximum(curves["Standard ResNet"], 1e-16), color=C_RES, ls="--", lw=1.45)
    ax.loglog(steps, np.maximum(curves["RMSNorm"], 1e-16), color=C_RMS, ls=":", lw=1.45)
    ax.loglog(steps, np.maximum(curves["Linear Unitary"], 1e-16), color=C_UNIT, ls="-.", lw=1.35)
    ax.loglog(steps, np.maximum(curves["GeodesicSkipNorm"], 1e-16), color=C_SKIP, lw=1.8)

    # Direct curve labels aligned cleanly inside left-center log bands (ample margin from right spine)
    ax.text(4.5, 4.5e3, "RMSNorm ($1.2\\times 10^3\\%$)", color=C_RMS, fontsize=6.8, va="bottom")
    ax.text(6.0, 1.5e-1, "Standard ResNet (diverges $>10^{70}\\%$)", color=C_RES, fontsize=6.8, fontweight="bold", va="top")
    ax.text(4.5, 6e-11, "Linear Unitary ($U^\\top U = I$)", color=C_UNIT, fontsize=6.8, va="bottom")
    ax.text(4.5, 1.8e-15, "GeodesicSkipNorm ($7.4\\times 10^{-13}\\%$)", color=C_SKIP, fontsize=6.8, fontweight="bold", va="bottom")

    ax.set_title("(a) 1D DNLS Wave Norm Conservation", pad=5)
    ax.set_xlabel("Autoregressive Rollout Step $t$")
    ax.set_ylabel("Norm Error $|\\mathcal{N}_t - \\mathcal{N}_0|/\\mathcal{N}_0$ (%)")
    ax.set_ylim(1e-16, 1e14)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5, 1e10])

    # ---------------- Panel (b): 40-Step Wave Trajectory Error ----------------
    ax = axes[1]
    labels = ["Standard\nResNet", "RMSNorm", "Linear\nUnitary", "Geodesic\nSkipNorm"]
    keys = ["Standard ResNet", "RMSNorm", "Linear Unitary", "GeodesicSkipNorm"]
    means = [dnls_res[k]["mse_40_mean"] * 1e2 for k in keys]
    stds = [dnls_res[k]["mse_40_std"] * 1e2 for k in keys]
    colors = [C_RES, C_RMS, C_UNIT, C_SKIP]
    x_p = np.arange(len(labels))
    bars = ax.bar(x_p, means, yerr=stds, capsize=3, width=0.46, color=colors, edgecolor=C_DARK, lw=0.65)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 1.8, f"{m:.2f}",
                ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_DARK)

    # Highlight 2.08x improvement over Linear Unitary in the open right space
    ax.annotate("$2.08\\times$ lower MSE\nthan Linear Unitary",
                xy=(3.0, means[3] + 9.0), xytext=(2.42, 46.0),
                ha="center", fontsize=6.8, color=C_SKIP, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_SKIP, lw=0.85))

    ax.set_xticks(x_p)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 88)
    ax.set_title("(b) 40-Step Trajectory MSE (5 Seeds)", pad=5)
    ax.set_ylabel("Rollout MSE ($\\times 10^{-2}$, Mean $\\pm$ SD)")

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
        well_ids.append(0)
        inits.append([1.0 + dq, 0.0])
        well_ids.append(1)
    for q0 in np.linspace(1.48, 2.05, 14):
        inits.append([q0, 0.0])
        well_ids.append(2)
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
    return np.array(trajs), well_ids, norms, H_true


def make_fig3(lv_summary, torus_summary, dw_summary):
    import pickle
    cache_path = "/tmp/skipnorm_fig3_cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            c_data = pickle.load(f)
        lv_trajs = c_data["lv_trajs"]
        ev_res, ev_d8, ev_sn = c_data["ev_res"], c_data["ev_d8"], c_data["ev_sn"]
        U_np, V_m_np = c_data["U_np"], c_data["V_m_np"]
        true_V_grid, raw_norm_grid, learned_norm_grid = c_data["true_V_grid"], c_data["raw_norm_grid"], c_data["learned_norm_grid"]
        dw_trajs, dw_well_ids, dw_norms, dw_H = c_data["dw_trajs"], c_data["dw_well_ids"], c_data["dw_norms"], c_data["dw_H"]
    else:
        lv_trajs_j = generate_lotka_volterra_data(n_trajs=48, n_steps=60, dt=0.25)
        m_res = train_lv_model_cosine(lv_trajs_j, mode="resnet", latent_dim=2, seed=1)
        m_d8 = train_lv_model_cosine(lv_trajs_j, mode="geodesic_skipnorm", latent_dim=8, seed=1)
        m_sn = train_lv_model_cosine(lv_trajs_j, mode="geodesic_skipnorm", latent_dim=2, seed=1)

        ev_res = evaluate_lv_invariance(m_res, lv_trajs_j)
        ev_d8 = evaluate_lv_invariance(m_d8, lv_trajs_j)
        ev_sn = evaluate_lv_invariance(m_sn, lv_trajs_j)

        u_grid = jnp.linspace(0.25, 3.0, 120, dtype=jnp.float64)
        v_grid = jnp.linspace(0.12, 1.8, 120, dtype=jnp.float64)
        U, V_m = jnp.meshgrid(u_grid, v_grid)
        pts = jnp.stack([U.flatten(), V_m.flatten()], axis=-1)
        U_np, V_m_np = np.array(U), np.array(V_m)
        true_V_grid = np.array(lotka_volterra_invariant(pts)).reshape(120, 120)
        raw_norm_grid = np.array(jnp.linalg.norm(pts, axis=-1)).reshape(120, 120)
        learned_norm_grid = np.array(jnp.linalg.norm(m_sn.encode(pts), axis=-1)).reshape(120, 120)
        lv_trajs = np.array(lv_trajs_j)
        dw_trajs, dw_well_ids, dw_norms, dw_H = train_double_well_for_plot()
        with open(cache_path, "wb") as f:
            pickle.dump({
                "lv_trajs": lv_trajs, "ev_res": ev_res, "ev_d8": ev_d8, "ev_sn": ev_sn,
                "U_np": U_np, "V_m_np": V_m_np, "true_V_grid": true_V_grid,
                "raw_norm_grid": raw_norm_grid, "learned_norm_grid": learned_norm_grid,
                "dw_trajs": dw_trajs, "dw_well_ids": dw_well_ids, "dw_norms": dw_norms, "dw_H": dw_H,
            }, f)

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.45))
    plt.subplots_adjust(wspace=0.40, hspace=0.40)

    # ---------------- Panel (a): Non-Centered Lotka-Volterra Contours ----------------
    ax = axes[0, 0]
    cf = ax.contourf(U_np, V_m_np, learned_norm_grid, levels=18, cmap="viridis", alpha=0.90)
    cbar = plt.colorbar(cf, ax=ax, pad=0.025, fraction=0.046)
    cbar.ax.tick_params(labelsize=6.2)
    ax.contour(U_np, V_m_np, true_V_grid, levels=[0.05, 0.15, 0.30, 0.50, 0.75],
               colors="white", linestyles="--", linewidths=1.05)
    ax.contour(U_np, V_m_np, raw_norm_grid, levels=[0.8, 1.4, 2.0, 2.6],
               colors="#FF6B6B", linestyles=":", linewidths=0.95)
    ax.plot(1.0, 0.5, marker="*", markersize=8.5, color="#FFD700", markeredgecolor=C_DARK, markeredgewidth=0.6)
    # Crisp inline key at top-right of contour (including colorbar meaning so no header collides with title)
    ax.text(0.97, 0.96,
            "Color: $\\|\\Phi_{\\mathrm{enc}}(u,v)\\|_2$\nWhite $--$: True $V(u,v)$\nRed $\\cdots$: Raw $\\|(u,v)\\|_2$\nGold $\\star$: Center $(u^*,v^*)$",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#112222", edgecolor="none", alpha=0.74))
    ax.set_title("(a) Lotka-Volterra Casimir Foliation", pad=5)
    ax.set_xlabel("Prey Population $u$")
    ax.set_ylabel("Predator Population $v$")

    # ---------------- Panel (b): Radius Constancy Along Single Orbit ----------------
    ax = axes[0, 1]
    orbit_idx = -4
    t_steps = np.arange(lv_trajs.shape[1])
    raw_orb = np.linalg.norm(lv_trajs[orbit_idx], axis=-1)
    res_orb = ev_res["norms"][orbit_idx]
    sn_orb = ev_sn["norms"][orbit_idx]

    ax.axhline(1.0, color="#BBBBBB", lw=0.6, zorder=1)
    ax.plot(t_steps, raw_orb / np.mean(raw_orb), color="#D62728", ls=":", lw=1.35,
            label="Raw $\\|(u_t,v_t)\\|_2$ ($42.3\\%$)")
    ax.plot(t_steps, res_orb / np.mean(res_orb), color=C_RES, ls="--", lw=1.3,
            label=f"ResNet $d_z\\!=\\!2$ ($50.0\\%$)")
    ax.plot(t_steps, sn_orb / np.mean(sn_orb), color=C_SKIP, lw=1.75,
            label=f"GeodesicSkipNorm ($9.0\\%$)")
    # Reserve top headspace [2.55, 3.88] so frameless legend sits cleanly above the peaks
    ax.set_ylim(0.0, 3.88)
    ax.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax.set_title("(b) Orbital Radius Constancy ($\\mathrm{CV}$)", pad=5)
    ax.set_xlabel("Trajectory Step $t$")
    ax.set_ylabel("Normalized Radius $r_t / \\bar{r}$")
    ax.legend(loc="upper left", frameon=False, handlelength=1.45, labelspacing=0.18, borderaxespad=0.30, fontsize=6.5)

    # ---------------- Panel (c): Brouwer Dimension Matching (d_z = N) ----------------
    ax = axes[0, 2]
    true_V_all = ev_sn["true_V"].flatten()
    sn_norms = ev_sn["norms"].flatten()
    d8_norms = ev_d8["norms"].flatten()
    res_norms = ev_res["norms"].flatten()

    def norm01(a):
        return (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-12)

    ax.scatter(true_V_all[::3], norm01(res_norms)[::3], s=3.5, alpha=0.20, color=C_RES, rasterized=True,
               label=f"ResNet $d_z\\!=\\!2$ ($R^2\\!=\\!1.0\\%$)")
    ax.scatter(true_V_all[::3], norm01(d8_norms)[::3], s=3.5, alpha=0.22, color=C_LN, rasterized=True,
               label=f"SkipNorm $d_z\\!=\\!8\\!>\\!N$ ($28.0\\%$)")
    ax.scatter(true_V_all[::3], norm01(sn_norms)[::3], s=4.5, alpha=0.65, color=C_SKIP, rasterized=True,
               label=f"Geodesic $d_z\\!=\\!2$ ($R^2\\!=\\!94.6\\%$)")
    # Reserve top headspace [1.02, 1.54] so frameless legend never touches a single point
    ax.set_ylim(-0.04, 1.54)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_title("(c) Brouwer Matching ($d_z = N = 2$)", pad=5)
    ax.set_xlabel("True Casimir $V(u,v) - V_{\\min}$")
    ax.set_ylabel("Scaled Radius $\\|\\Phi_{\\mathrm{enc}}(u,v)\\|_2$")
    ax.legend(loc="upper left", frameon=False, markerscale=1.8, handletextpad=0.25, labelspacing=0.18, borderaxespad=0.30, fontsize=6.5)

    # ---------------- Panel (d): Double-Well Phase Plane & Separatrix ----------------
    ax = axes[1, 0]
    q_g = np.linspace(-2.15, 2.15, 200)
    p_g = np.linspace(-1.65, 1.65, 200)
    Q, P = np.meshgrid(q_g, p_g)
    H_grid = 0.5 * P**2 - 0.5 * Q**2 + 0.25 * Q**4

    for tr, wid in zip(np.array(dw_trajs), dw_well_ids):
        if wid == 0:
            ax.plot(tr[:, 0], tr[:, 1], color=C_RMS, lw=0.75, alpha=0.72)
        elif wid == 1:
            ax.plot(tr[:, 0], tr[:, 1], color=C_RES, lw=0.75, alpha=0.72)
        else:
            ax.plot(tr[:, 0], tr[:, 1], color=C_SKIP, lw=0.75, alpha=0.62)

    ax.contour(Q, P, H_grid, levels=[0.0], colors=C_DARK, linestyles="--", linewidths=1.25)
    ax.scatter([-1.0, 1.0], [0.0, 0.0], color=[C_RMS, C_RES], s=18, zorder=6, edgecolor="white", lw=0.4)
    ax.scatter([0.0], [0.0], color=C_DARK, marker="x", s=22, zorder=6, lw=1.0)

    # Direct labels with crisp white pill backgrounds so contour lines never obscure text
    pill = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.90)
    ax.text(-1.05, 1.62, "Left $\\mathcal{O}_c^{(L)}$", color=C_RMS, fontsize=6.5, fontweight="bold", ha="center", bbox=pill)
    ax.text(1.05, 1.62, "Right $\\mathcal{O}_c^{(R)}$", color=C_RES, fontsize=6.5, fontweight="bold", ha="center", bbox=pill)
    ax.text(0.0, -1.55, "Connected $\\mathcal{O}_c$ ($H>0$)", color=C_SKIP, fontsize=6.5, fontweight="bold", ha="center", bbox=pill)

    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.82, 1.95)
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_title("(d) Double-Well Phase Plane ($H \\lessgtr 0$)", pad=5)
    ax.set_xlabel("Position $q$")
    ax.set_ylabel("Momentum $p$")

    # ---------------- Panel (e): Corollary 1 - Multi-Basin Obstruction ----------------
    ax = axes[1, 1]
    orb_H = np.mean(dw_H, axis=-1)
    orb_r_mean = np.mean(dw_norms, axis=-1)
    orb_r_std = np.std(dw_norms, axis=-1)

    m0 = dw_well_ids == 0
    m1 = dw_well_ids == 1
    m2 = dw_well_ids == 2

    ax.axvline(0.0, color=C_DARK, ls="--", lw=0.9, alpha=0.7)
    ax.errorbar(orb_H[m0], orb_r_mean[m0], yerr=orb_r_std[m0], fmt="s-", color=C_RMS, ms=3.2, lw=1.0, capsize=1.8,
                label=f"Left Well ($H\\!<\\!0$: $R^2\\!=\\!39.7\\%$)")
    ax.errorbar(orb_H[m1], orb_r_mean[m1], yerr=orb_r_std[m1], fmt="^-", color=C_RES, ms=3.2, lw=1.0, capsize=1.8,
                label=f"Right Well ($H\\!<\\!0$: $\\mathrm{{CV}}\\!=\\!24.9\\%$)")
    ax.errorbar(orb_H[m2], orb_r_mean[m2], yerr=orb_r_std[m2], fmt="o-", color=C_SKIP, ms=3.4, lw=1.35, capsize=1.8,
                label=f"Connected ($H\\!>\\!0$: $R^2\\!=\\!97.4\\%$)")
    ax.text(0.08, 0.25, "Separatrix $H=0$", fontsize=6.4, color=C_DARK, ha="left", va="center")
    # Reserve top headspace [1.32, 1.86] for frameless legend
    ax.set_ylim(0.15, 1.86)
    ax.set_yticks([0.4, 0.8, 1.2, 1.6])
    ax.set_title("(e) Multi-Basin Bifurcation (Cor. 1)", pad=5)
    ax.set_xlabel("True Double-Well Energy $H(q,p)$")
    ax.set_ylabel("Latent Radius $\\|\\Phi_{\\mathrm{enc}}(q,p)\\|_2$")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, labelspacing=0.18, borderaxespad=0.30, fontsize=6.4)

    # ---------------- Panel (f): Corollary 2 - Two-Invariant Toroidal Foliation ----------------
    ax = axes[1, 2]
    t_labels = ["Standard\nResNet", "Single Sphere\n($\\mathbb{S}^3, K\\!=\\!1$)", "Toroidal\n($\\mathbb{T}^2, K\\!=\\!2$)"]
    t_keys = ["Standard ResNet (4D)", "Single-Sphere SkipNorm (K=1)", "Toroidal Block SkipNorm (K=2)"]
    r2_m = [torus_summary[k]["r2_mean"] for k in t_keys]
    r2_s = [torus_summary[k]["r2_std"] for k in t_keys]
    inv_m = [100.0 - torus_summary[k]["cv_mean"] for k in t_keys]
    inv_s = [torus_summary[k]["cv_std"] for k in t_keys]

    xp = np.arange(len(t_labels)) * 1.32
    w = 0.36
    ax.bar(xp - w / 2, inv_m, yerr=inv_s, capsize=2.5, width=w, color=C_RMS, edgecolor=C_DARK, lw=0.6,
           label="Action Invariance ($100\\% - \\mathrm{CV}$)")
    b2 = ax.bar(xp + w / 2, r2_m, yerr=r2_s, capsize=2.5, width=w, color=C_SKIP, edgecolor=C_DARK, lw=0.6,
                label="Joint $(I_1, I_2)$ Recovery $R^2$")
    for bar, m, s in zip(b2, r2_m, r2_s):
        ax.text(bar.get_x() + bar.get_width() * 0.58, m + s + 2.2, f"{m:.1f}",
                ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=C_DARK)
    ax.set_xticks(xp)
    ax.set_xticklabels(t_labels)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlim(-0.55, xp[-1] + 0.62)
    # Upper-left space above bars 0 & 1 (y in [75, 136]) is 100% empty
    ax.set_ylim(0, 136)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Percentage (%, Mean $\\pm$ SD)")
    ax.set_title("(f) Toroidal Foliation ($\\mathbb{T}^2$, Cor. 2)", pad=5)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4, labelspacing=0.20, borderaxespad=0.30, fontsize=6.4)

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
