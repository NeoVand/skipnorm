"""Stage 4: Publication-Grade Figure Generation using K-Dense scientific-visualization skill.

Generates 5 multi-panel vector/raster figures (PDF + 600 DPI PNG) with full JSON provenance
manifests via figure_export.py, using scoped style_context("default", palette_name="okabe_ito_on_white")
and constrained layout:
  - fig1_geometry_and_spectra.{pdf,png,export.json}
  - fig2_dnls_unitary_rollout.{pdf,png,export.json}
  - fig3_scalar_foliation_lotka_volterra.{pdf,png,export.json}
  - fig4_topological_obstructions_and_controls.{pdf,png,export.json}
  - fig5_toroidal_kblock_foliation.{pdf,png,export.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = Path("/Users/neo/.gemini/config/plugins/science/skills/scientific-visualization")
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from color_palettes import get_palette
from figure_export import export_figure
from style_presets import style_context

STAGE1_JSON = ROOT / "results" / "stage1" / "stage1_results.json"
STAGE2_JSON = ROOT / "results" / "stage2" / "stage2_results.json"
STAGE3_JSON = ROOT / "results" / "stage3" / "stage3_dnls_results.json"
SPLITS_DIR = ROOT / "data" / "splits"
PAPER_DIR = ROOT / "paper"
RESULTS_FIG_DIR = ROOT / "results" / "figures"
ARTIFACT_DIR = Path("/Users/neo/.gemini/antigravity/brain/7ceba0dd-69e5-454a-9c5e-cdea1454eade")

# Okabe-Ito colorblind-safe palette on white (all >= 3.0:1 WCAG contrast on #FFFFFF)
PAL = get_palette("okabe_ito_on_white")
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_GREEN = "#009E73"
C_PURP = "#CC79A7"
C_SKY = "#332288"
C_ORANGE = "#882255"
C_BLACK = "#222222"
C_GRAY = "#555555"

HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def export_all_targets(fig: plt.Figure, stem: str, prov: dict) -> None:
    """Export vector PDF to paper/, PNG + JSON manifest to results/figures/, and PNG to artifact dir."""
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(
        fig,
        PAPER_DIR / stem,
        formats=["pdf"],
        dpi=600,
        bbox_inches=None,
        overwrite=True,
        provenance=prov,
        write_manifest=False,
    )
    export_figure(
        fig,
        RESULTS_FIG_DIR / stem,
        formats=["png"],
        dpi=600,
        bbox_inches=None,
        overwrite=True,
        provenance=prov,
        write_manifest=True,
    )
    export_figure(
        fig,
        ARTIFACT_DIR / stem,
        formats=["png"],
        dpi=600,
        bbox_inches=None,
        overwrite=True,
        provenance=prov,
        write_manifest=False,
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.07) -> None:
    """Place a clean bold lowercase panel identifier (a, b, c, d)."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    """Apply subtle, clean gridlines and spine styling."""
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.45, color="#DCDCDC", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure_1(stage1: dict) -> None:
    """Figure 1: Radial-Tangential Decomposition, Stabilized RMSNorm Spectra,
    Antipodal Sign-Flip Elimination, and Depth-100 Radial Gradient Survival.
    """
    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        # --- Panel a: Geometric schematic of Radial-Tangential Decomposition ---
        ax_a.plot([-1.42, 1.82], [-1.05, 1.05], alpha=0.0)
        ax_a.set_aspect("equal", adjustable="datalim")
        ax_a.axis("off")
        add_panel_label(ax_a, "a")
        ax_a.set_title("Radial Pullback & Tangent Retraction", fontsize=8.5, fontweight="bold", pad=6)

        theta = np.linspace(0, 2 * np.pi, 240)
        radii_shells = [(0.44, "#9ECAE1", ":"), (0.72, "#4292C6", "--"), (1.00, C_BLUE, "-")]
        for r_s, col, ls in radii_shells:
            ax_a.plot(r_s * np.cos(theta), r_s * np.sin(theta), color=col, linestyle=ls, linewidth=1.4)

        # Point x on outer shell r = 1.00 on the lower-right side (angle -24 deg)
        # Tangent step of 0.92*r0 opens a wide 42.6-deg angle to y_hat at +18.6 deg
        ang = np.deg2rad(-24.0)
        r0 = 1.00
        x_pt = np.array([r0 * np.cos(ang), r0 * np.sin(ang)])
        u_rad = np.array([np.cos(ang), np.sin(ang)])
        u_tan = np.array([-np.sin(ang), np.cos(ang)])

        alpha_step = 0.92
        y_tan = x_pt + alpha_step * u_tan
        y_hat = y_tan * (r0 / np.linalg.norm(y_tan))

        # Origin & radial rays
        ax_a.scatter([0.0], [0.0], color=C_BLACK, s=15, zorder=5)
        ax_a.plot([0.0, x_pt[0]], [0.0, x_pt[1]], color=C_GRAY, linestyle="-.", linewidth=0.9)
        ax_a.plot([0.0, y_hat[0]], [0.0, y_hat[1]], color=C_GRAY, linestyle="-.", linewidth=0.9)

        # Vectors (use solid FancyArrowPatch heads so the dashed retraction arrow has a crisp solid head)
        arr_tan = FancyArrowPatch(
            posA=tuple(x_pt),
            posB=tuple(y_tan),
            arrowstyle="-|>,head_length=3.8,head_width=2.3",
            color=C_GREEN,
            linewidth=1.45,
            shrinkA=2.5,
            shrinkB=2.5,
            zorder=6,
        )
        # Dashed shaft + crisp solid arrowhead for retraction from y_tan to y_hat
        ret_dir = (y_hat - y_tan) / np.linalg.norm(y_hat - y_tan)
        shaft_start = y_tan + 0.04 * ret_dir
        shaft_end = y_hat - 0.08 * ret_dir
        ax_a.plot(
            [shaft_start[0], shaft_end[0]],
            [shaft_start[1], shaft_end[1]],
            color=C_VERM,
            linestyle="--",
            linewidth=1.3,
            zorder=6,
        )
        arr_ret_head = FancyArrowPatch(
            posA=tuple(y_hat - 0.12 * ret_dir),
            posB=tuple(y_hat),
            arrowstyle="-|>,head_length=3.8,head_width=2.3",
            color=C_VERM,
            linewidth=1.3,
            linestyle="-",
            shrinkA=0,
            shrinkB=2.8,
            zorder=6,
        )
        arr_rad = FancyArrowPatch(
            posA=tuple(x_pt),
            posB=tuple(x_pt + 0.34 * u_rad),
            arrowstyle="-|>,head_length=3.8,head_width=2.3",
            color=C_BLUE,
            linewidth=1.45,
            shrinkA=2.5,
            shrinkB=1.0,
            zorder=6,
        )
        ax_a.add_patch(arr_tan)
        ax_a.add_patch(arr_ret_head)
        ax_a.add_patch(arr_rad)

        ax_a.scatter(
            [x_pt[0], y_tan[0], y_hat[0]],
            [x_pt[1], y_tan[1], y_hat[1]],
            color=[C_BLUE, C_GREEN, C_VERM],
            s=18,
            zorder=7,
        )

        # Left-side annotations with strictly horizontal arrows (same y for xy and xytext)
        y_r2 = 0.42
        x_r2 = -np.sqrt(0.72**2 - y_r2**2)
        ax_a.text(
            -1.08,
            0.66,
            "Invariant shells",
            fontsize=6.8,
            color="#2B6CB0",
            fontweight="bold",
            ha="right",
            va="center",
            path_effects=HALO,
            clip_on=False,
        )
        ax_a.annotate(
            r"$\mathbf{S}^{d-1}(r_2)$",
            xy=(x_r2, y_r2),
            xytext=(-1.08, y_r2),
            fontsize=6.9,
            color="#2B6CB0",
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#4292C6", lw=0.85, shrinkA=2, shrinkB=2),
            path_effects=HALO,
            clip_on=False,
        )
        y_r1 = -0.26
        x_r1 = -np.sqrt(0.44**2 - y_r1**2)
        ax_a.annotate(
            r"$\mathbf{S}^{d-1}(r_1)$",
            xy=(x_r1, y_r1),
            xytext=(-1.08, y_r1),
            fontsize=6.9,
            color="#4A90C2",
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#6BAED6", lw=0.85, shrinkA=2, shrinkB=2),
            path_effects=HALO,
            clip_on=False,
        )

        # Right-side equations: strictly vertical arrow for red dot and blue dot
        ax_a.text(
            y_hat[0],
            0.80,
            r"$\hat{y} \in \mathbf{S}^{d-1}(r)$",
            fontsize=7.3,
            color=C_VERM,
            fontweight="bold",
            ha="center",
            va="bottom",
            path_effects=HALO,
            clip_on=False,
        )
        ax_a.annotate(
            "",
            xy=(y_hat[0], y_hat[1]),
            xytext=(y_hat[0], 0.77),
            arrowprops=dict(arrowstyle="->", color=C_VERM, lw=0.9, shrinkA=0, shrinkB=3.2),
            clip_on=False,
        )
        ax_a.text(
            y_tan[0] + 0.08,
            y_tan[1],
            r"$x + \alpha f_{\perp}(x)$",
            fontsize=7.3,
            color=C_GREEN,
            fontweight="bold",
            ha="left",
            va="center",
            path_effects=HALO,
            clip_on=False,
        )
        ax_a.text(
            x_pt[0] + 0.34 * u_rad[0] + 0.05,
            x_pt[1] + 0.34 * u_rad[1] - 0.02,
            r"$u_{\mathrm{in}}$",
            fontsize=7.3,
            color=C_BLUE,
            ha="left",
            va="center",
            path_effects=HALO,
            clip_on=False,
        )
        ax_a.text(
            x_pt[0],
            -0.80,
            r"$x \in \mathbf{S}^{d-1}(r)$",
            fontsize=7.3,
            color=C_BLUE,
            fontweight="bold",
            ha="center",
            va="top",
            path_effects=HALO,
            clip_on=False,
        )
        ax_a.annotate(
            "",
            xy=(x_pt[0], x_pt[1]),
            xytext=(x_pt[0], -0.77),
            arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9, shrinkA=0, shrinkB=3.2),
            clip_on=False,
        )
        ax_a.text(
            0.46,
            -0.13,
            r"Cotangent Pullback: $J^\top u_{\mathrm{out}} = u_{\mathrm{in}}$ ($\sigma_{\max}(J) \geq 1$)",
            transform=ax_a.transAxes,
            fontsize=7.2,
            ha="center",
            va="top",
            color="#222222",
            clip_on=False,
        )

        # --- Panel b: Stabilized RMSNorm singular values vs SkipNorm ---
        style_axes(ax_b)
        add_panel_label(ax_b, "b")
        ax_b.set_title("Stabilized RMSNorm vs. SkipNorm Spectra", fontsize=8.5, fontweight="bold", pad=6)

        spectra = stage1["proposition_2"]["rmsnorm_stabilized_spectra"]
        r_grid = np.array(spectra[0]["r_grid"])
        ax_b.axhline(1.0, color=C_BLUE, linewidth=1.8, linestyle="-", label=r"SkipNorm $\sigma_r = 1$")

        eps_styles = [
            (1e-2, C_VERM, "--", "o", r"RMSNorm $\sigma_{\mathrm{rad}}$ ($\varepsilon=10^{-2}$)"),
            (1e-5, C_ORANGE, "-.", "s", r"RMSNorm $\sigma_{\mathrm{rad}}$ ($\varepsilon=10^{-5}$)"),
            (1e-8, C_PURP, ":", "^", r"RMSNorm $\sigma_{\mathrm{rad}}$ ($\varepsilon=10^{-8}$)"),
        ]
        for eps_target, col, ls, mk, lbl in eps_styles:
            rec = next(item for item in spectra if abs(item["eps"] - eps_target) < 1e-12)
            ax_b.plot(
                r_grid,
                rec["sigma_rad_theory"],
                color=col,
                linestyle=ls,
                linewidth=1.3,
                marker=mk,
                markevery=6,
                markersize=3.5,
                label=lbl,
            )

        rec_tan = next(item for item in spectra if abs(item["eps"] - 1e-2) < 1e-12)
        ax_b.plot(
            r_grid,
            rec_tan["sigma_tan_theory"],
            color=C_GREEN,
            linestyle="-",
            linewidth=1.2,
            alpha=0.85,
            label=r"RMSNorm $\sigma_{\mathrm{tan}} = \sqrt{d}/\sqrt{r^2+d\varepsilon}$",
        )

        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        ax_b.set_ylim(1e-10, 1e3)
        ax_b.set_xlabel(r"Pre-normalization norm $r = \|y\|_2$")
        ax_b.set_ylabel("Jacobian singular value (log)")
        ax_b.legend(loc="lower left", fontsize=5.8, frameon=False)

        # --- Panel c: Antipodal sign-flip instability elimination ---
        style_axes(ax_c)
        add_panel_label(ax_c, "c")
        ax_c.set_title("Antipodal Sign-Flip Elimination", fontsize=8.5, fontweight="bold", pad=6)

        anti = stage1["antipodal_sign_flip"]
        c_opp = np.array(anti["radial_opposition_c"])
        cos_proj = np.array(anti["projected_residual_cosine_similarity"])
        cos_geod = np.array(anti["geodesic_skipnorm_cosine_similarity"])

        ax_c.axvspan(1.0, 2.2, color="#FDE0DD", alpha=0.40, zorder=0)
        ax_c.axvline(1.0, color=C_VERM, linestyle=":", linewidth=1.0)
        ax_c.plot(
            c_opp,
            cos_geod,
            color=C_BLUE,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markevery=12,
            markersize=3.6,
            label="Geodesic SkipNorm",
        )
        ax_c.plot(
            c_opp,
            cos_proj,
            color=C_VERM,
            linestyle="--",
            linewidth=1.6,
            marker="s",
            markevery=12,
            markersize=3.6,
            label="Unprojected Residual",
        )
        ax_c.text(
            1.16,
            0.12,
            "Antipodal Flip\n" + r"($\langle x + \alpha f(x), x \rangle < 0$)",
            fontsize=6.3,
            color=C_VERM,
            path_effects=HALO,
        )
        ax_c.set_xlim(0.0, 2.2)
        ax_c.set_ylim(-1.12, 1.12)
        ax_c.set_xlabel(r"Radial opposition $-\alpha \langle f(x), u_{\mathrm{in}} \rangle / \|x\|_2$")
        ax_c.set_ylabel(r"Cosine similarity $\langle \hat{y}, x \rangle / \|x\|_2^2$")
        ax_c.legend(loc="lower left", bbox_to_anchor=(0.01, 0.06), fontsize=5.8, frameon=False)

        # --- Panel d: Depth-100 radial gradient survival ---
        style_axes(ax_d)
        add_panel_label(ax_d, "d")
        ax_d.set_title("Depth-100 Radial Gradient Pullback", fontsize=8.5, fontweight="bold", pad=6)

        d100 = stage1["proposition_2"]["depth_100_propagation"]
        depths = np.array(d100["depths"])
        series_d = [
            ("Geodesic SkipNorm", C_BLUE, "-", "o", "Geodesic SkipNorm"),
            ("SkipNorm (LeakyReLU)", C_SKY, "--", "s", "SkipNorm (Direct)"),
            ("Unnormalized MLP", C_GRAY, "-.", "v", "Unnormalized MLP"),
            ("RMSNorm (eps=1e-1)", C_VERM, ":", "^", r"RMSNorm ($\varepsilon=10^{-1}$)"),
            ("RMSNorm (eps=1e-2)", C_PURP, "--", "D", r"RMSNorm ($\varepsilon=10^{-2}$)"),
        ]
        for key_name, col, ls, mk, lbl in series_d:
            vals = np.maximum(np.array(d100[key_name]["radial_grad_norm"]), 1e-16)
            ax_d.plot(
                depths,
                vals,
                color=col,
                linestyle=ls,
                linewidth=1.5,
                marker=mk,
                markevery=3,
                markersize=3.6,
                label=lbl,
            )

        ax_d.set_yscale("log")
        ax_d.set_ylim(1e-16, 1e1)
        ax_d.set_xlabel("Network depth $L$ (layers)")
        ax_d.set_ylabel(r"Radial pullback norm $\|J_{\mathrm{tot}}^\top u_L\|_2$")
        ax_d.legend(
            loc="lower center",
            bbox_to_anchor=(0.56, 0.10),
            fontsize=5.8,
            frameon=False,
        )

        prov = {
            "raw_data": str(STAGE1_JSON.relative_to(ROOT)),
            "script": "experiments/stage4_generate_publication_figures.py",
            "transformations": ["exact JAX float64 SVD", "depth-100 VJP pullback"],
            "uncertainty": "deterministic analytical & machine-precision autodiff verification",
            "missing_data": "none",
        }
        export_all_targets(fig, "fig1_geometry_and_spectra", prov)
        plt.close(fig)


def build_figure_2(stage3: dict) -> None:
    """Figure 2: Converged 1D Discrete Nonlinear Schrodinger (DNLS) Unitary Wavepacket Rollout."""
    split = np.load(SPLITS_DIR / "dnls_1d.npz")
    te_long = split["test_orbits_long"]  # (12, 250, 32)
    sample_orbit = te_long[0]  # (250, 32)
    S = 16
    density_gt = sample_orbit[:, :S] ** 2 + sample_orbit[:, S:] ** 2  # (250, 16)

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), layout="constrained")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        # --- Panel a: Spatiotemporal wavepacket density |psi_{n,t}|^2 ---
        add_panel_label(ax_a, "a")
        ax_a.set_title(r"1D DNLS Soliton Wavepacket $|\psi_{n,t}|^2$", fontsize=8.5, fontweight="bold", pad=6)
        im = ax_a.imshow(
            density_gt.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, 249 * 0.04, 1, S],
            interpolation="bicubic",
            vmin=0.0,
        )
        cbar = fig.colorbar(im, ax=ax_a, pad=0.02, fraction=0.046)
        cbar.set_label(r"Site density $|\psi_n|^2$", fontsize=7.2)
        cbar.ax.tick_params(labelsize=6.5)
        ax_a.set_xlabel("Physical time $t$ ($250$ steps, no projection)")
        ax_a.set_ylabel("Lattice site index $n$")

        # --- Panel b: Relative L2 norm drift over 250 steps ---
        style_axes(ax_b)
        add_panel_label(ax_b, "b")
        ax_b.set_title(r"Unitary Norm Drift $|\|\psi_t\|_2^2 - \|\psi_0\|_2^2|/\|\psi_0\|_2^2$", fontsize=8.5, fontweight="bold", pad=6)

        t_steps = np.array(stage3["drift_traces"]["t_steps"][1:])
        rk4_drift = np.maximum(np.array(stage3["drift_traces"]["RK4 Reference (No Proj)"][1:]), 1e-16)
        ax_b.plot(
            t_steps,
            rk4_drift,
            color=C_BLACK,
            linestyle=":",
            linewidth=1.3,
            label="10x RK4 Reference",
        )

        drift_models = [
            ("Geodesic SkipNorm", C_BLUE, "-", "o"),
            ("State-Dependent SO(32)", C_GREEN, "--", "s"),
            ("Pre-RMSNorm Residual", C_ORANGE, "-.", "^"),
            ("Post-RMSNorm Residual", C_PURP, ":", "D"),
            ("Unconstrained ResNet", C_VERM, "--", "v"),
        ]
        for m_name, col, ls, mk in drift_models:
            mean_c = np.maximum(np.array(stage3["drift_traces"][m_name]["mean"][1:]), 1e-16)
            ax_b.plot(
                t_steps,
                mean_c,
                color=col,
                linestyle=ls,
                linewidth=1.4,
                marker=mk,
                markevery=35,
                markersize=3.4,
                label=m_name,
            )

        ax_b.set_yscale("log")
        ax_b.set_ylim(5e-17, 1.5e3)
        ax_b.set_xlabel("Rollout step $t$")
        ax_b.set_ylabel("Relative norm drift (log)")
        ax_b.legend(loc="center right", bbox_to_anchor=(0.99, 0.44), fontsize=5.6, frameon=False)

        # --- Panel c: Multi-step rollout MSE trajectory ---
        style_axes(ax_c)
        add_panel_label(ax_c, "c")
        ax_c.set_title("Multi-Step Rollout State MSE (5 Seeds)", fontsize=8.5, fontweight="bold", pad=6)

        t_mse = np.array(stage3["mse_traces"]["t_steps"][:100])
        mse_models = [
            ("Geodesic SkipNorm", C_BLUE, "-", "o"),
            ("Projected Residual SkipNorm", C_SKY, "--", "s"),
            ("State-Dependent SO(32)", C_GREEN, "-.", "^"),
            ("Pre-RMSNorm Residual", C_ORANGE, ":", "D"),
            ("Unconstrained ResNet", C_VERM, "--", "v"),
            ("Linear Orthogonal SO(32)", C_GRAY, ":", "x"),
        ]
        for m_name, col, ls, mk in mse_models:
            mean_m = np.array(stage3["mse_traces"][m_name]["mean"][:100])
            std_m = np.array(stage3["mse_traces"][m_name]["std"][:100])
            ax_c.plot(
                t_mse,
                mean_m,
                color=col,
                linestyle=ls,
                linewidth=1.4,
                marker=mk,
                markevery=16,
                markersize=3.4,
                label=m_name,
            )
            ax_c.fill_between(
                t_mse,
                np.maximum(mean_m - std_m, 1e-6),
                mean_m + std_m,
                color=col,
                alpha=0.12,
            )

        ax_c.set_yscale("log")
        ax_c.set_xlabel("Rollout horizon $t$ (steps)")
        ax_c.set_ylabel("State MSE (mean ± SD)")
        ax_c.legend(loc="lower right", fontsize=5.6, ncol=1, frameon=False)

        # --- Panel d: Accuracy vs Inference Cost & Norm Conservation Summary ---
        style_axes(ax_d)
        add_panel_label(ax_d, "d")
        ax_d.set_title("Accuracy vs. Rollout Latency Trade-off", fontsize=8.5, fontweight="bold", pad=6)

        summ = stage3["summary"]
        # Compute dynamic non-overlapping label positions based on actual MSE@50 values
        raw_pts = [
            ("Geodesic SkipNorm", C_BLUE, "o"),
            ("Projected Residual SkipNorm", C_SKY, "s"),
            ("Unconstrained ResNet", C_VERM, "P"),
            ("Pre-RMSNorm Residual", C_ORANGE, "v"),
            ("Linear Orthogonal SO(32)", C_GRAY, "D"),
            ("State-Dependent SO(32)", C_GREEN, "^"),
        ]
        y_vals_all = [summ[m]["mse_50_mean"] for m, _, _ in raw_pts]
        y_min_d = min(y_vals_all) * 0.68
        y_max_d = max(y_vals_all) * 1.48

        # Sort cluster of fast models (x < 10 ms) by y to place labels without overlap
        fast_models = [(m, c, mk, summ[m]["rollout_ms_mean"], summ[m]["mse_50_mean"]) for m, c, mk in raw_pts if summ[m]["rollout_ms_mean"] < 12.0]
        fast_models.sort(key=lambda item: item[4])
        log_ymin = np.log10(y_min_d * 1.15)
        log_ymax = np.log10(y_max_d * 0.78)
        label_y_slots = np.logspace(log_ymin, log_ymax, len(fast_models))
        slot_map = {item[0]: (7.2 if item[0] != "Linear Orthogonal SO(32)" else 3.4, float(label_y_slots[idx])) for idx, item in enumerate(fast_models)}
        slot_map["State-Dependent SO(32)"] = (7.5, summ["State-Dependent SO(32)"]["mse_50_mean"])

        for m_name, col, mk in raw_pts:
            rec = summ[m_name]
            x_ms = rec["rollout_ms_mean"]
            y_mse = rec["mse_50_mean"]
            y_std = rec["mse_50_std"]
            ax_d.errorbar(
                [x_ms],
                [y_mse],
                yerr=[y_std],
                fmt=mk,
                color=col,
                markersize=6.0,
                capsize=2.5,
                elinewidth=1.1,
                zorder=5,
            )
            ax_d.annotate(
                m_name,
                xy=(x_ms, y_mse),
                xytext=slot_map[m_name],
                textcoords="data",
                fontsize=6.2,
                color=col,
                va="center",
                fontweight="bold" if "Geodesic" in m_name else "normal",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=2, shrinkB=4),
                path_effects=HALO,
            )

        ax_d.set_xscale("log")
        ax_d.set_yscale("log")
        ax_d.set_xlim(2.1, 45.0)
        ax_d.set_ylim(y_min_d, y_max_d)
        ax_d.set_xticks([3.0, 5.0, 10.0, 20.0, 30.0])
        ax_d.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax_d.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
        ax_d.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
        ax_d.minorticks_off()
        ax_d.set_xlabel("250-step rollout wall-clock time (ms, log)")
        ax_d.set_ylabel("50-step rollout MSE (mean ± SD, log)")

        prov = {
            "raw_data": str(STAGE3_JSON.relative_to(ROOT)),
            "dataset_split": "data/splits/dnls_1d.npz (no post-step projection)",
            "transformations": ["5-seed mean and SD across 12 held-out test trajectories"],
            "uncertainty": "mean ± 1 SD across n=5 independent training seeds",
            "missing_data": "none",
        }
        export_all_targets(fig, "fig2_dnls_unitary_rollout", prov)
        plt.close(fig)


def _extract_closed_loop(traj: np.ndarray) -> np.ndarray:
    """Slice a trajectory (T, 2) enclosing (0,0) to exactly one 2*pi revolution and close the loop."""
    angles = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
    delta_ang = np.abs(angles - angles[0])
    idx_candidates = np.where(delta_ang >= 2.0 * np.pi)[0]
    if len(idx_candidates) > 0:
        end_idx = int(idx_candidates[0])
        return np.vstack([traj[:end_idx], traj[0:1]])
    return np.vstack([traj, traj[0:1]])


def build_figure_3(stage2: dict) -> None:
    """Figure 3: Scalar Invariant Identification on Lotka-Volterra (Theorem 1)."""
    lv = stage2["lotka_volterra"]
    ex_mlp = lv["exemplar_traces"]["geodesic_mlp"]
    ex_unc = lv["exemplar_traces"]["unconstrained"]

    x_orbits = np.array(ex_mlp["test_orbits_x"])  # (16, 180, 2)
    z_orbits = np.array(ex_mlp["test_orbits_z"])  # (16, 180, 2)
    r_orbits = np.array(ex_mlp["test_radii"])  # (16, 180)
    r_unc = np.array(ex_unc["test_radii"])  # (16, 180)
    c_inv = np.array(ex_mlp["test_invariants"])  # (16,)

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        cmap = mpl.colormaps["viridis"]
        norm_c = mpl.colors.Normalize(vmin=float(np.min(c_inv)), vmax=float(np.max(c_inv)))

        # --- Panel a: Observed phase space (u, v) ---
        style_axes(ax_a)
        add_panel_label(ax_a, "a")
        ax_a.set_title("Observed Lotka–Volterra Phase Space", fontsize=8.5, fontweight="bold", pad=6)
        for i in range(0, len(c_inv), 2):
            x_loop = _extract_closed_loop(x_orbits[i])
            ax_a.plot(
                x_loop[:, 0],
                x_loop[:, 1],
                color=cmap(norm_c(c_inv[i])),
                linewidth=1.35,
            )
        ax_a.scatter([0.0], [0.0], color=C_BLACK, marker="+", s=35, zorder=5)
        ax_a.set_xlabel(r"Log-prey coordinate $u = \log x_1$")
        ax_a.set_ylabel(r"Log-predator coordinate $v = \log x_2$")
        ax_a.set_aspect("equal", adjustable="datalim")

        # --- Panel b: Encoded latent space z = E(x) under Geodesic SkipNorm ---
        style_axes(ax_b)
        add_panel_label(ax_b, "b")
        ax_b.set_title(r"Latent Foliation $E(O_c) \subset \mathbf{S}^1(\psi(c))$", fontsize=8.5, fontweight="bold", pad=6)
        theta_circle = np.linspace(0, 2 * np.pi, 240)
        for i in range(0, len(c_inv), 2):
            r_mean_i = float(np.mean(r_orbits[i]))
            z_loop = _extract_closed_loop(z_orbits[i])
            ax_b.plot(
                r_mean_i * np.cos(theta_circle),
                r_mean_i * np.sin(theta_circle),
                color=cmap(norm_c(c_inv[i])),
                linestyle=":",
                linewidth=0.85,
                alpha=0.55,
            )
            ax_b.plot(
                z_loop[:, 0],
                z_loop[:, 1],
                color=cmap(norm_c(c_inv[i])),
                linewidth=1.35,
            )
        ax_b.scatter([0.0], [0.0], color=C_BLACK, marker="+", s=35, zorder=5)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_c)
        cbar = fig.colorbar(sm, ax=ax_b, pad=0.02, fraction=0.046)
        cbar.set_label(r"True Hamiltonian $I(u,v) = c$", fontsize=7.2)
        cbar.ax.tick_params(labelsize=6.5)
        ax_b.set_xlabel(r"Latent coordinate $z_1$")
        ax_b.set_ylabel(r"Latent coordinate $z_2$")
        ax_b.set_aspect("equal", adjustable="datalim")

        # --- Panel c: Monotone profile psi(c) = ||E(x)||_2 vs true invariant c ---
        style_axes(ax_c)
        add_panel_label(ax_c, "c")
        ax_c.set_title(r"Monotone Radius Law $\bar{r}_m = \psi(c_m)$ (Held-Out)", fontsize=8.2, fontweight="bold", pad=6)

        order = np.argsort(c_inv)
        c_sort = c_inv[order]

        r_mlp_mean = np.mean(r_orbits[order], axis=1)
        r_mlp_std = np.std(r_orbits[order], axis=1)
        s_mlp = np.max(r_mlp_mean)
        ax_c.errorbar(
            c_sort,
            r_mlp_mean / s_mlp,
            yerr=r_mlp_std / s_mlp,
            fmt="o-",
            color=C_BLUE,
            linewidth=1.5,
            markersize=4.0,
            capsize=2.5,
            label="Geodesic SkipNorm (Encoded Orbit)",
        )

        r_unc_mean = np.mean(r_unc[order], axis=1)
        r_unc_std = np.std(r_unc[order], axis=1)
        s_unc = np.max(r_unc_mean)
        ax_c.errorbar(
            c_sort,
            r_unc_mean / s_unc,
            yerr=r_unc_std / s_unc,
            fmt="s--",
            color=C_VERM,
            linewidth=1.2,
            markersize=3.6,
            capsize=2.5,
            alpha=0.85,
            label="Unconstrained AE (Radial Wobble)",
        )

        ax_c.set_xlabel(r"True Hamiltonian level $c_m$ (held-out orbits)")
        ax_c.set_ylabel(r"Normalized latent radius $\|E(x_t)\|_2 / r_{\max}$")
        ax_c.legend(loc="lower right", fontsize=5.8, frameon=False)

        # --- Panel d: 5-seed summary across all 6 architectures (light fills so error bars are crisp) ---
        style_axes(ax_d)
        add_panel_label(ax_d, "d")
        ax_d.set_title("5-Seed Invariant Identification Metrics", fontsize=8.5, fontweight="bold", pad=6)

        summ = lv["summary"]
        models_lv = [
            ("Geodesic SkipNorm (MLP)", "Geodesic (MLP)", "#72B7E6"),
            ("Geodesic SkipNorm (Bijective)", "Geodesic (Bij.)", "#B2ABD2"),
            ("Projected Residual SkipNorm", "Proj. Residual", "#66C2A5"),
            ("State-Dependent SO(2)", "State SO(2)", "#FDB863"),
            ("COMET Baseline", "COMET", "#E7969C"),
            ("Unconstrained AE", "Unconstrained", "#F4A582"),
        ]
        y_pos = np.arange(len(models_lv))
        cv_means = [summ[m[0]]["mean_cv_mean"] for m in models_lv]
        cv_stds = [summ[m[0]]["mean_cv_std"] for m in models_lv]
        r2_means = [summ[m[0]]["isotonic_r2_mean"] for m in models_lv]

        bars = ax_d.barh(
            y_pos,
            cv_means,
            xerr=cv_stds,
            height=0.58,
            color=[m[2] for m in models_lv],
            edgecolor="#222222",
            linewidth=0.65,
            capsize=2.6,
            error_kw=dict(ecolor="#111111", lw=1.1, capthick=1.0),
        )
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels([m[1] for m in models_lv], fontsize=7.0)
        ax_d.invert_yaxis()
        ax_d.set_xlabel(r"Within-orbit radial $\mathrm{CV}_O(h)$ (lower is better; 0 baseline)")
        ax_d.set_xlim(0.0, max(cv_means) * 1.48)

        for i, (bar, r2_val) in enumerate(zip(bars, r2_means)):
            w = bar.get_width() + cv_stds[i]
            ax_d.text(
                w + 0.008,
                bar.get_y() + bar.get_height() / 2.0,
                f"Iso $R^2={r2_val:.3f}$",
                va="center",
                ha="left",
                fontsize=6.3,
                color="#111111",
            )

        prov = {
            "raw_data": str(STAGE2_JSON.relative_to(ROOT)),
            "dataset_split": "data/splits/lotka_volterra.npz (whole-orbit disjoint)",
            "transformations": ["encoded true trajectories z_t = E(x_t)", "isotonic regression R^2"],
            "uncertainty": "mean ± 1 SD across n=5 random seeds (and within-orbit SD in Panel c)",
            "missing_data": "none",
        }
        export_all_targets(fig, "fig3_scalar_foliation_lotka_volterra", prov)
        plt.close(fig)


def build_figure_4(stage1: dict, stage2: dict) -> None:
    """Figure 4: Jordan-Brouwer Multi-Basin Obstruction (Theorem 2),
    Resonant 1:1 2D Harmonic Oscillator Control (Theorem 1), and
    Translated Identity Encoder Control (Proposition 3 / Section 6).
    """
    dw = stage2["double_well"]
    ex_bij = dw["exemplars"]["balanced_50_50__geodesic_bijective"]
    ex_mlp = dw["exemplars"]["balanced_50_50__geodesic_folding_mlp"]

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        # --- Panel a: Double-Well Disjoint Basins & Bijective Off-Centering ---
        style_axes(ax_a)
        add_panel_label(ax_a, "a")
        ax_a.set_title("Theorem 2: Bijective Encoder Cannot Center Both Wells", fontsize=8.2, fontweight="bold", pad=6)

        z_bij = np.array(ex_bij["test_orbits_z"])
        wells = np.array(ex_bij["test_wells"])
        for i in range(len(wells)):
            col = C_BLUE if wells[i] == 0 else C_VERM
            ls = "-" if wells[i] == 0 else "--"
            lbl = "Left Well ($q < 0$)" if (i == 0) else ("Right Well ($q > 0$)" if i == np.where(wells == 1)[0][0] else None)
            ax_a.plot(z_bij[i, :, 0], z_bij[i, :, 1], color=col, linestyle=ls, linewidth=1.15, alpha=0.85, label=lbl)

        ax_a.scatter([0.0], [0.0], color=C_BLACK, marker="X", s=38, zorder=6, label="Latent Origin $0$")
        ax_a.set_xlabel(r"Bijective latent coordinate $z_1$")
        ax_a.set_ylabel(r"Bijective latent coordinate $z_2$")
        ax_a.legend(loc="upper right", fontsize=5.8, frameon=False)

        # --- Panel b: Unconstrained MLP Non-Injective Well Folding ---
        style_axes(ax_b)
        add_panel_label(ax_b, "b")
        ax_b.set_title("Well-Folding MLP Overlaps Disjoint Basin Radii", fontsize=8.2, fontweight="bold", pad=6)

        r_mlp = np.mean(np.array(ex_mlp["test_radii"]), axis=1)
        r_bij = np.mean(np.array(ex_bij["test_radii"]), axis=1)
        e_inv = np.array(ex_mlp["test_invariants"])

        left_m = wells == 0
        right_m = wells == 1
        ax_b.scatter(
            e_inv[left_m],
            r_mlp[left_m] / np.max(r_mlp),
            color=C_BLUE,
            marker="o",
            s=26,
            label="Folding MLP Left Well ($q<0$)",
        )
        ax_b.scatter(
            e_inv[right_m],
            r_mlp[right_m] / np.max(r_mlp),
            color=C_VERM,
            marker="^",
            s=26,
            label="Folding MLP Right Well ($q>0$)",
        )
        ax_b.scatter(
            e_inv[left_m],
            r_bij[left_m] / np.max(r_bij),
            color=C_SKY,
            marker="s",
            s=20,
            alpha=0.75,
            label="Bijective Left Well",
        )
        ax_b.scatter(
            e_inv[right_m],
            r_bij[right_m] / np.max(r_bij),
            color=C_ORANGE,
            marker="D",
            s=20,
            alpha=0.75,
            label="Bijective Right Well",
        )
        ax_b.set_xlabel(r"Sub-barrier energy $H(q,p)$")
        ax_b.set_ylabel(r"Normalized mean radius $\bar{r}_m / r_{\max}$")
        ax_b.legend(loc="lower right", fontsize=5.7, frameon=False)

        # --- Panel c: Resonant 1:1 2D Harmonic Oscillator Control (Theorem 1) ---
        style_axes(ax_c)
        add_panel_label(ax_c, "c")
        ax_c.set_title("Resonant 1:1 Oscillator: Why Orbit Density Matters", fontsize=8.2, fontweight="bold", pad=6)

        res_ho = stage1["resonant_harmonic_oscillator_control"]
        etas = np.array(res_ho["partition_eta"])
        r_shell = np.array(res_ho["orbit_mean_radii_on_same_shell"])

        ax_c.plot(
            etas,
            r_shell,
            color=C_PURP,
            linestyle="-",
            linewidth=1.7,
            marker="o",
            markevery=3,
            markersize=3.8,
            label=r"Encoded orbit radius $\|E(x)\|_2$ on $H=1$",
        )
        ax_c.axhline(
            np.sqrt(2.0),
            color=C_GRAY,
            linestyle="--",
            linewidth=1.1,
            label=r"Unwarped sphere radius $\sqrt{2H} = \sqrt{2}$",
        )
        ax_c.text(
            0.04,
            1.96,
            "Within-orbit $\\mathrm{CV}_O(h) = 4.2 \\times 10^{-16}$\n"
            "Intertwining error $= 0.0$\n"
            "Same-shell radius spread $= 50\\%$",
            fontsize=6.2,
            color="#222222",
        )
        ax_c.set_ylim(1.16, 2.22)
        ax_c.set_xlabel(r"Mode energy partition $\eta = H_1 / (H_1 + H_2)$ on single shell $H=1$")
        ax_c.set_ylabel(r"Orbit radius $\|E(x_t)\|_2$")
        ax_c.legend(loc="lower right", fontsize=5.7, frameon=False)

        # --- Panel d: Translated Identity Encoder Control E_a(x) = a*e_1 + x ---
        style_axes(ax_d)
        add_panel_label(ax_d, "d")
        ax_d.set_title(r"Translated Encoder $E_a(x)=ae_1+x$: $O(a^{-1})$ Decay", fontsize=8.2, fontweight="bold", pad=6)

        tr_ctrl = stage1["translated_encoder_control"]["shift_sweep"]
        a_vals = np.array([item["shift_a"] for item in tr_ctrl])
        rel_err = np.array([item["rel_intertwining_err"] for item in tr_ctrl])
        rel_bound = np.array([item["theoretical_rel_err_bound"] for item in tr_ctrl])
        cv_emp = np.array([item["mean_radial_cv"] for item in tr_ctrl])
        cv_bound = np.array([item["theoretical_cv_bound"] for item in tr_ctrl])

        ax_d.plot(
            a_vals,
            cv_emp,
            color=C_BLUE,
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=3.6,
            label=r"Empirical radial $\mathrm{CV}_O(h)$",
        )
        ax_d.plot(
            a_vals,
            cv_bound,
            color=C_SKY,
            linestyle=":",
            linewidth=1.2,
            label=r"Bound $R / (a - R) = O(a^{-1})$",
        )
        ax_d.plot(
            a_vals,
            rel_err,
            color=C_VERM,
            linestyle="--",
            linewidth=1.5,
            marker="s",
            markersize=3.6,
            label="Empirical relative intertwining err.",
        )
        ax_d.plot(
            a_vals,
            rel_bound,
            color=C_ORANGE,
            linestyle="-.",
            linewidth=1.2,
            label=r"Bound $D / (a - R) = O(a^{-1})$",
        )
        ax_d.set_xscale("log")
        ax_d.set_yscale("log")
        ax_d.set_xlabel(r"Isometric translation shift $a$ ($\|E_a(x)\|_2 \asymp a$)")
        ax_d.set_ylabel("Error / CV (log scale)")
        ax_d.legend(loc="upper right", fontsize=5.6, frameon=False)

        prov = {
            "raw_data": [
                str(STAGE1_JSON.relative_to(ROOT)),
                str(STAGE2_JSON.relative_to(ROOT)),
            ],
            "transformations": [
                "encoded true trajectories on double_well_balanced.npz",
                "analytical 1:1 harmonic oscillator Hopf-fiber warp",
                "translated identity encoder sweep a in [2, 10000]",
            ],
            "uncertainty": "exact analytical controls & 5-seed double-well evaluation",
            "missing_data": "none",
        }
        export_all_targets(fig, "fig4_topological_obstructions_and_controls", prov)
        plt.close(fig)


def build_figure_5(stage2: dict) -> None:
    """Figure 5: Multi-Invariant K-Block Toroidal Identification on 4D Coupled Nonlinear Oscillators (Theorem 3)."""
    tor = stage2["coupled_torus_4d"]
    summ = tor["summary"]
    ex_k2 = tor["exemplar_torus"]["2-Block Toroidal Geodesic SkipNorm (K=2)"]
    ex_unc = tor["exemplar_torus"]["Unconstrained AE (d_z=4)"]

    split = np.load(SPLITS_DIR / "coupled_torus_4d.npz")
    te_orbits = split["test_orbits"]  # (20, 120, 4)

    with style_context("default", palette_name="okabe_ito_on_white"):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
        ax_a, ax_b = axes[0, 0], axes[0, 1]
        ax_c, ax_d = axes[1, 0], axes[1, 1]

        # --- Panel a: Observed coordinate norm oscillation on a 2-torus trajectory ---
        style_axes(ax_a)
        add_panel_label(ax_a, "a")
        ax_a.set_title("Observed 4D Coupled Torus Coordinate Fluctuations", fontsize=8.2, fontweight="bold", pad=6)

        t_steps = np.arange(te_orbits.shape[1])
        orb0 = te_orbits[0]
        obs_b1 = np.linalg.norm(orb0[:, :2], axis=-1)
        obs_b2 = np.linalg.norm(orb0[:, 2:], axis=-1)
        obs_tot = np.linalg.norm(orb0, axis=-1)

        ax_a.plot(t_steps, obs_tot, color=C_BLACK, linestyle="-", linewidth=1.4, label=r"Observed total $\|x_t\|_2$")
        ax_a.plot(t_steps, obs_b1, color=C_BLUE, linestyle="--", linewidth=1.3, label=r"Observed pair 1 $\|(x_1,x_2)\|_2$")
        ax_a.plot(t_steps, obs_b2, color=C_VERM, linestyle="-.", linewidth=1.3, label=r"Observed pair 2 $\|(x_3,x_4)\|_2$")
        ax_a.set_xlabel("Time step $t$ along quasi-periodic orbit on $\\mathbf{T}^2$")
        ax_a.set_ylabel("Euclidean norm in observed space")
        ax_a.set_ylim(0.0, 3.08)
        ax_a.legend(loc="upper right", fontsize=5.6, frameon=False)

        # --- Panel b: Encoded block radii traces (K=2 Geodesic vs Unconstrained) ---
        style_axes(ax_b)
        add_panel_label(ax_b, "b")
        ax_b.set_title("Encoded Block Radii Along True Orbits", fontsize=8.2, fontweight="bold", pad=6)

        traces_k2 = np.array(ex_k2["test_block_radii_traces"])  # (6, 120, 2)
        traces_unc = np.array(ex_unc["test_block_radii_traces"])  # (6, 120, 2)

        for idx, col_b1, col_b2 in [(0, C_BLUE, C_GREEN), (1, C_SKY, C_ORANGE)]:
            lbl1 = r"$K=2$ Block 1 $\|z_t^{(1)}\|_2$" if idx == 0 else None
            lbl2 = r"$K=2$ Block 2 $\|z_t^{(2)}\|_2$" if idx == 0 else None
            ax_b.plot(t_steps, traces_k2[idx, :, 0], color=col_b1, linestyle="-", linewidth=1.5, label=lbl1)
            ax_b.plot(t_steps, traces_k2[idx, :, 1], color=col_b2, linestyle="--", linewidth=1.5, label=lbl2)

        ax_b.plot(
            t_steps,
            traces_unc[0, :, 0],
            color=C_VERM,
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            label="Unconstrained Block 1 (Wobble)",
        )
        ax_b.set_xlabel("Time step $t$ (encoded true trajectory $z_t = E(x_t)$)")
        ax_b.set_ylabel("Latent block radius")
        ax_b.set_ylim(-0.01, 0.59)
        ax_b.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=5.5, frameon=False)

        # --- Panel c: Single-Sphere Limit vs 2-Block Action Recovery ---
        style_axes(ax_c)
        add_panel_label(ax_c, "c")
        ax_c.set_title(r"Theorem 3: Single Radius $\|z\|_2$ Cannot Separate $(I_1, I_2)$", fontsize=8.2, fontweight="bold", pad=6)

        inv_te = np.array(ex_k2["test_invariants"])  # (20, 2)
        r_blk_te = np.array(ex_k2["test_mean_block_radii"])  # (20, 2)
        sc = ax_c.scatter(
            inv_te[:, 0],
            inv_te[:, 1],
            c=np.linalg.norm(r_blk_te, axis=-1),
            cmap="viridis",
            s=46,
            edgecolor=C_BLACK,
            linewidth=0.6,
        )
        cbar = fig.colorbar(sc, ax=ax_c, pad=0.02, fraction=0.046)
        cbar.set_label(r"Single total radius $\|\bar{\boldsymbol{\rho}}\|_2$", fontsize=7.0)
        cbar.ax.tick_params(labelsize=6.5)
        ax_c.set_xlabel(r"True action invariant $I_1$")
        ax_c.set_ylabel(r"True action invariant $I_2$")

        # --- Panel d: 5-Seed Comparison of 1-Scalar vs 2-Block Readout R^2 (clean light fills, no hatches) ---
        style_axes(ax_d)
        add_panel_label(ax_d, "d")
        ax_d.set_title("1-Scalar vs. 2-Block Invariant Readout $R^2$ (5 Seeds)", fontsize=8.2, fontweight="bold", pad=6)

        labels_t = [
            "2-Block Toroidal Geodesic SkipNorm (K=2)",
            "Single-Sphere Geodesic SkipNorm (K=1)",
            "Unconstrained AE (d_z=4)",
        ]
        short_names = ["2-Block Geodesic\n($K=2$)", "Single-Sphere\n($K=1$)", "Unconstrained AE\n($d_z=4$)"]
        x_idx = np.arange(len(labels_t))
        width_b = 0.34

        r2_1s = [summ[k]["readout_r2_from_1_scalar_mean"] for k in labels_t]
        r2_1s_sd = [summ[k]["readout_r2_from_1_scalar_std"] for k in labels_t]
        r2_2b = [summ[k]["readout_r2_from_2_blocks_mean"] for k in labels_t]
        r2_2b_sd = [summ[k]["readout_r2_from_2_blocks_std"] for k in labels_t]

        ax_d.bar(
            x_idx - width_b / 2,
            r2_1s,
            width_b,
            yerr=r2_1s_sd,
            color="#FDB863",
            edgecolor="#222222",
            linewidth=0.65,
            capsize=2.6,
            error_kw=dict(ecolor="#111111", lw=1.15, capthick=1.0),
            label=r"1-Scalar Readout ($\|z\|_2 \in \mathbf{R}$)",
        )
        ax_d.bar(
            x_idx + width_b / 2,
            r2_2b,
            width_b,
            yerr=r2_2b_sd,
            color="#72B7E6",
            edgecolor="#222222",
            linewidth=0.65,
            capsize=2.6,
            error_kw=dict(ecolor="#111111", lw=1.15, capthick=1.0),
            label=r"2-Block Readout ($(\bar{r}_1, \bar{r}_2) \in \mathbf{R}^2$)",
        )

        ax_d.axhline(0.50, color=C_VERM, linestyle=":", linewidth=1.0, label="1D Projection Ceiling (~0.50)")
        ax_d.set_xticks(x_idx)
        ax_d.set_xticklabels(short_names, fontsize=7.0)
        ax_d.set_ylim(0.0, 1.32)
        ax_d.set_ylabel(r"Held-out $(I_1, I_2)$ Readout $R^2$")
        ax_d.legend(loc="upper right", fontsize=5.5, frameon=False)

        prov = {
            "raw_data": str(STAGE2_JSON.relative_to(ROOT)),
            "dataset_split": "data/splits/coupled_torus_4d.npz (whole-orbit disjoint)",
            "transformations": ["degree-2 ridge readout from 1 scalar vs 2 block radii"],
            "uncertainty": "mean ± 1 SD across n=5 independent training seeds",
            "missing_data": "none",
        }
        export_all_targets(fig, "fig5_toroidal_kblock_foliation", prov)
        plt.close(fig)


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    stage1 = json.loads(STAGE1_JSON.read_text())
    stage2 = json.loads(STAGE2_JSON.read_text())
    stage3 = json.loads(STAGE3_JSON.read_text())

    print("[Stage 4] Building Figure 1 (Geometry, Stabilized Spectra & Depth-100)...")
    build_figure_1(stage1)

    print("[Stage 4] Building Figure 2 (1D DNLS Unitary Rollout)...")
    build_figure_2(stage3)

    print("[Stage 4] Building Figure 3 (Lotka-Volterra Scalar Foliation)...")
    build_figure_3(stage2)

    print("[Stage 4] Building Figure 4 (Jordan-Brouwer Obstruction & Controls)...")
    build_figure_4(stage1, stage2)

    print("[Stage 4] Building Figure 5 (4D Coupled Torus K=2 Block Foliation)...")
    build_figure_5(stage2)

    print("[Stage 4] All 5 figures exported with JSON provenance manifests!")


if __name__ == "__main__":
    main()
