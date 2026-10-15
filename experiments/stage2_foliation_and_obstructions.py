"""Stage 2: Core Benchmarks for Scalar Invariant Foliation (Theorem 1),
Jordan-Brouwer Multi-Basin Obstruction (Theorem 2), and K-Block Toroidal Foliation (Theorem 3).

All models are trained and evaluated on the locked whole-orbit disjoint train/val/test splits
from data/splits/ across 5 random seeds (0..4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from skipnorm.jax_layers import (
    BijectiveCouplingMap,
    block_geodesic_skip_norm_fn,
    geodesic_skip_norm_fn,
    projected_residual_skip_norm_fn,
)
from skipnorm.metrics import (
    compute_k_invariant_readout_r2,
    compute_monotone_r2_and_spearman,
    compute_orbit_radii,
    compute_ovr,
    compute_radial_cv,
)

DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "stage2"


def symplectic_tangent_blocks(z: jax.Array, num_blocks: int) -> jax.Array:
    """Given z in (R^2)^K, return the canonical orthogonal unit-norm-scaled tangent
    direction J_2 z^{(k)} = (-z_2^{(k)}, z_1^{(k)}) in each 2D block.
    """
    z_b = z.reshape(num_blocks, 2)
    j_z = jnp.stack([-z_b[:, 1], z_b[:, 0]], axis=-1)
    return j_z.reshape(-1)


class LatentDynamicsAutoencoder(eqx.Module):
    """Unified Autoencoder + Latent Dynamics operator supporting:
    - 'geodesic_bijective': Origin-pinned BijectiveCouplingMap encoder (exact inverse decoder) + Geodesic SkipNorm
    - 'geodesic_bijective_free': Unpinned BijectiveCouplingMap encoder (can translate off origin) + Geodesic SkipNorm
    - 'geodesic_mlp': MLP encoder/decoder + Geodesic SkipNorm latent operator
    - 'geodesic_folding_mlp': Even-parity / folding MLP encoder E(x) + Geodesic SkipNorm (for multi-basin folding)
    - 'block_geodesic': MLP encoder/decoder + K-Block Toroidal Geodesic SkipNorm latent operator
    - 'projected_residual': MLP encoder/decoder + Projected Residual SkipNorm (no tangent projection)
    - 'state_dep_so2': MLP encoder/decoder + State-dependent SO(d) Cayley rotation
    - 'unconstrained': MLP encoder/decoder + Unconstrained residual step z + alpha * f(z)
    """

    encoder: eqx.Module
    decoder: eqx.Module | None
    latent_net: eqx.nn.MLP
    alpha: jax.Array
    mode: str = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True, default=1)
    dim: int = eqx.field(static=True, default=2)

    def __init__(
        self,
        dim: int,
        mode: str,
        num_blocks: int = 1,
        width: int = 48,
        depth: int = 2,
        *,
        key: jax.Array,
    ):
        k_enc, k_dec, k_lat = jax.random.split(key, 3)
        self.dim = dim
        self.mode = mode
        self.num_blocks = num_blocks

        if mode in ("geodesic_bijective", "geodesic_bijective_free"):
            self.encoder = BijectiveCouplingMap(
                dim=dim, num_layers=6, width=width, depth=2, key=k_enc
            )
            self.decoder = None
        else:
            self.encoder = eqx.nn.MLP(dim, dim, width, depth, activation=jax.nn.tanh, key=k_enc)
            self.decoder = eqx.nn.MLP(dim, dim, width, depth, activation=jax.nn.tanh, key=k_dec)

        if mode == "state_dep_so2":
            self.latent_net = eqx.nn.MLP(
                dim, dim * dim, width, depth, activation=jax.nn.tanh, key=k_lat
            )
        elif mode in (
            "geodesic_bijective",
            "geodesic_bijective_free",
            "geodesic_mlp",
            "geodesic_folding_mlp",
            "block_geodesic",
        ) and (dim == 2 * num_blocks):
            # Action-angle radial frequency network: maps block radii (rho_1, ..., rho_K) -> (omega_1, ..., omega_K)
            self.latent_net = eqx.nn.MLP(
                num_blocks, num_blocks, width, depth, activation=jax.nn.tanh, key=k_lat
            )
        else:
            self.latent_net = eqx.nn.MLP(dim, dim, width, depth, activation=jax.nn.tanh, key=k_lat)

        self.alpha = jnp.asarray(0.20, dtype=jnp.float64)

    def encode(self, x: jax.Array) -> jax.Array:
        if self.mode == "geodesic_bijective":
            z0 = self.encoder(jnp.zeros_like(x))
            return self.encoder(x) - z0
        elif self.mode == "geodesic_bijective_free":
            return self.encoder(x)
        elif self.mode == "geodesic_folding_mlp":
            # Invariant under phase-space reflection (q, p) -> (-q, -p) around saddle (0, 0),
            # allowing both wells to fold onto concentric circles around 0
            z_even = 0.5 * (self.encoder(x) + self.encoder(-x))
            z0 = self.encoder(jnp.zeros_like(x))
            return z_even - z0
        else:
            z0 = self.encoder(jnp.zeros_like(x))
            return self.encoder(x) - z0

    def decode(self, z: jax.Array) -> jax.Array:
        if self.mode == "geodesic_bijective":
            z0 = self.encoder(jnp.zeros_like(z))
            return self.encoder.inverse(z + z0)
        elif self.mode == "geodesic_bijective_free":
            return self.encoder.inverse(z)
        assert self.decoder is not None
        return self.decoder(z)

    def step_latent(self, z: jax.Array) -> jax.Array:
        if self.mode in (
            "geodesic_bijective",
            "geodesic_bijective_free",
            "geodesic_mlp",
            "geodesic_folding_mlp",
        ) and (self.dim == 2 * self.num_blocks):
            z_b = z.reshape(self.num_blocks, 2)
            radii = jnp.sqrt(jnp.sum(jnp.square(z_b), axis=-1) + 1e-24)
            omegas = self.latent_net(radii)  # (K,)
            j_z = jnp.stack([-z_b[:, 1], z_b[:, 0]], axis=-1) * omegas[:, None]
            f_z = j_z.reshape(-1)
            return geodesic_skip_norm_fn(z, f_z, alpha=self.alpha, axis=-1, eps=1e-12)
        elif self.mode in ("geodesic_bijective", "geodesic_mlp"):
            f_z = self.latent_net(z)
            return geodesic_skip_norm_fn(z, f_z, alpha=self.alpha, axis=-1, eps=1e-12)
        elif self.mode == "block_geodesic":
            z_b = z.reshape(self.num_blocks, 2)
            radii = jnp.sqrt(jnp.sum(jnp.square(z_b), axis=-1) + 1e-24)
            omegas = self.latent_net(radii)  # (K,)
            j_z = jnp.stack([-z_b[:, 1], z_b[:, 0]], axis=-1) * omegas[:, None]
            f_z = j_z.reshape(-1)
            return block_geodesic_skip_norm_fn(
                z, f_z, num_blocks=self.num_blocks, alpha=self.alpha, eps=1e-12
            )
        elif self.mode == "projected_residual":
            f_z = self.latent_net(z)
            return projected_residual_skip_norm_fn(z, f_z, alpha=self.alpha, axis=-1, eps=1e-12)
        elif self.mode == "state_dep_so2":
            raw = self.latent_net(z).reshape(self.dim, self.dim)
            skew = 0.5 * (raw - raw.T)
            eye = jnp.eye(self.dim, dtype=z.dtype)
            Q = jnp.linalg.solve(eye + self.alpha * skew, eye - self.alpha * skew)
            return Q @ z
        elif self.mode == "unconstrained":
            f_z = self.latent_net(z)
            return z + self.alpha * f_z
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class COMETModel(eqx.Module):
    """COMET baseline (Otto et al., 2023): learns a scalar invariant h(x) and a dynamics
    vector field v(x) orthogonalized against grad h(x) via projection so h(x) is
    conserved along predicted trajectories.
    """

    h_net: eqx.nn.MLP
    v_net: eqx.nn.MLP
    dt: float = eqx.field(static=True, default=0.05)

    def __init__(
        self, dim: int, width: int = 48, depth: int = 2, dt: float = 0.05, *, key: jax.Array
    ):
        k1, k2 = jax.random.split(key)
        self.h_net = eqx.nn.MLP(dim, 1, width, depth, activation=jax.nn.tanh, key=k1)
        self.v_net = eqx.nn.MLP(dim, dim, width, depth, activation=jax.nn.tanh, key=k2)
        self.dt = dt

    def invariant_scalar(self, x: jax.Array) -> jax.Array:
        h0 = self.h_net(jnp.zeros_like(x))[0]
        return jnp.abs(self.h_net(x)[0] - h0) + 0.1

    def rhs_orthogonal(self, x: jax.Array) -> jax.Array:
        grad_h = jax.grad(self.invariant_scalar)(x)
        v = self.v_net(x)
        norm_sq = jnp.sum(jnp.square(grad_h)) + 1e-8
        return v - (jnp.dot(v, grad_h) / norm_sq) * grad_h

    def step(self, x: jax.Array) -> jax.Array:
        h = self.dt
        k1 = self.rhs_orthogonal(x)
        k2 = self.rhs_orthogonal(x + 0.5 * h * k1)
        return x + h * k2


def train_autoencoder_model(
    model: LatentDynamicsAutoencoder,
    train_orbits: jnp.ndarray,
    horizon: int = 12,
    stride: int = 2,
    steps: int = 1200,
    lr: float = 3e-3,
    seed: int = 0,
    fold_decode: bool = False,
) -> LatentDynamicsAutoencoder:
    """Train LatentDynamicsAutoencoder on whole-orbit trajectories using multi-step
    reconstruction + direct 1-step and multi-step latent intertwining loss.
    Using temporal stride >= 2 allows a 12-step rollout to span a large fraction of the
    periodic/quasi-periodic orbit without increasing backprop depth.
    """
    sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps, alpha=0.05)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    M, T, D = train_orbits.shape

    @eqx.filter_jit
    def loss_fn(m: LatentDynamicsAutoencoder, batch_windows: jnp.ndarray) -> jax.Array:
        # batch_windows: (B, H+1, D)
        z_true = jax.vmap(jax.vmap(m.encode))(batch_windows)
        x_recon = jax.vmap(jax.vmap(m.decode))(z_true)
        if fold_decode:
            # For symmetric well-folding MLP on double-well, reconstruct up to +/- sign parity
            recon_loss = jnp.mean(
                jnp.minimum(
                    jnp.sum(jnp.square(x_recon - batch_windows), axis=-1),
                    jnp.sum(jnp.square(x_recon + batch_windows), axis=-1),
                )
            )
        else:
            recon_loss = jnp.mean(jnp.square(x_recon - batch_windows))

        # 1-step intertwining residual eps = ||E(F(x)) - g(E(x))||_2^2
        z_step1 = jax.vmap(jax.vmap(m.step_latent))(z_true[:, :-1, :])
        step1_intertwine = jnp.mean(jnp.square(z_step1 - z_true[:, 1:, :]))

        # Multi-step rollout from t=0
        def scan_step(z_curr, _):
            z_next = jax.vmap(m.step_latent)(z_curr)
            return z_next, z_next

        _, z_roll = jax.lax.scan(scan_step, z_true[:, 0, :], jnp.arange(horizon))
        z_roll = jnp.swapaxes(z_roll, 0, 1)
        x_roll = jax.vmap(jax.vmap(m.decode))(z_roll)

        if fold_decode:
            pred_loss = jnp.mean(
                jnp.minimum(
                    jnp.sum(jnp.square(x_roll - batch_windows[:, 1:, :]), axis=-1),
                    jnp.sum(jnp.square(x_roll + batch_windows[:, 1:, :]), axis=-1),
                )
            )
        else:
            pred_loss = jnp.mean(jnp.square(x_roll - batch_windows[:, 1:, :]))

        multi_intertwine = jnp.mean(jnp.square(z_roll - z_true[:, 1:, :]))

        return recon_loss + pred_loss + 1.5 * step1_intertwine + 1.0 * multi_intertwine

    @eqx.filter_jit
    def make_step(m, opt_s, batch_w):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch_w)
        updates, opt_s = optimizer.update(grads, opt_s, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_s, loss

    rng = np.random.default_rng(10000 + seed)
    batch_size = 64
    span = horizon * stride
    max_start = T - span - 1
    tr_np = np.asarray(train_orbits)

    for _ in range(steps):
        m_idx = rng.integers(0, M, size=batch_size)
        t_idx = rng.integers(0, max_start, size=batch_size)
        windows = np.stack(
            [
                tr_np[m_idx[b], t_idx[b] : t_idx[b] + span + 1 : stride, :]
                for b in range(batch_size)
            ],
            axis=0,
        )
        model, opt_state, _ = make_step(model, opt_state, jnp.asarray(windows))

    return model


def train_comet_model(
    model: COMETModel,
    train_orbits: jnp.ndarray,
    horizon: int = 6,
    steps: int = 800,
    lr: float = 3e-3,
    seed: int = 0,
) -> COMETModel:
    """Train COMET baseline on whole-orbit trajectories."""
    sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps, alpha=0.05)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    M, T, D = train_orbits.shape

    @eqx.filter_jit
    def loss_fn(m: COMETModel, batch_w: jnp.ndarray) -> jax.Array:
        def scan_step(x_curr, _):
            x_next = jax.vmap(m.step)(x_curr)
            return x_next, x_next

        _, x_roll = jax.lax.scan(scan_step, batch_w[:, 0, :], jnp.arange(horizon))
        x_roll = jnp.swapaxes(x_roll, 0, 1)
        pred_loss = jnp.mean(jnp.square(x_roll - batch_w[:, 1:, :]))

        h_vals = jax.vmap(jax.vmap(m.invariant_scalar))(batch_w)
        h_within = jnp.mean(jnp.var(h_vals, axis=1))
        h_total_std = jnp.std(h_vals)
        reg = jax.nn.relu(0.30 - h_total_std) ** 2 + 0.5 * h_within
        return pred_loss + reg

    @eqx.filter_jit
    def make_step(m, opt_s, batch_w):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch_w)
        updates, opt_s = optimizer.update(grads, opt_s, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_s, loss

    rng = np.random.default_rng(20000 + seed)
    batch_size = 64
    max_start = T - horizon - 1
    tr_np = np.asarray(train_orbits)
    for _ in range(steps):
        m_idx = rng.integers(0, M, size=batch_size)
        t_idx = rng.integers(0, max_start, size=batch_size)
        windows = np.stack(
            [tr_np[m_idx[b], t_idx[b] : t_idx[b] + horizon + 1, :] for b in range(batch_size)],
            axis=0,
        )
        model, opt_state, _ = make_step(model, opt_state, jnp.asarray(windows))
    return model


def evaluate_lotka_volterra_suite(seeds: list[int]) -> dict:
    """Run 5-seed Lotka-Volterra benchmark across all 6 architectures and controls."""
    split = np.load(DATA_DIR / "lotka_volterra.npz")
    tr_orbits = jnp.asarray(split["train_orbits"])
    te_orbits = jnp.asarray(split["test_orbits"])
    te_inv = np.asarray(split["test_invariants"])

    modes = [
        ("Geodesic SkipNorm (Bijective)", "geodesic_bijective"),
        ("Geodesic SkipNorm (MLP)", "geodesic_mlp"),
        ("Projected Residual SkipNorm", "projected_residual"),
        ("State-Dependent SO(2)", "state_dep_so2"),
        ("COMET Baseline", "comet"),
        ("Unconstrained AE", "unconstrained"),
    ]

    summary = {}
    exemplar_traces = {}
    stride_lv = 3
    horizon_lv = 12

    for label, mode in modes:
        seed_records = []
        for s in seeds:
            key = jax.random.PRNGKey(500 + 37 * s)
            if mode == "comet":
                model = COMETModel(dim=2, width=48, depth=2, dt=0.05, key=key)
                model = train_comet_model(model, tr_orbits, horizon=6, steps=800, lr=3e-3, seed=s)

                def rollout_12(x0):
                    def step_fn(xc, _):
                        xn = model.step(xc)
                        return xn, xn
                    _, xr = jax.lax.scan(step_fn, x0, jnp.arange(12))
                    return xr

                pred_12 = jax.vmap(rollout_12)(te_orbits[:, 0, :])
                rollout_mse = float(jnp.mean(jnp.square(pred_12 - te_orbits[:, 1:13, :])))

                r_te = np.asarray(jax.vmap(jax.vmap(model.invariant_scalar))(te_orbits))
                r_tr = np.asarray(jax.vmap(jax.vmap(model.invariant_scalar))(tr_orbits))
                sigma_lat = float(np.std(r_tr))
            else:
                model = LatentDynamicsAutoencoder(dim=2, mode=mode, width=48, depth=2, key=key)
                model = train_autoencoder_model(
                    model,
                    tr_orbits,
                    horizon=horizon_lv,
                    stride=stride_lv,
                    steps=1200,
                    lr=3e-3,
                    seed=s,
                )

                def rollout_12(x0):
                    z0 = model.encode(x0)
                    def step_fn(zc, _):
                        zn = model.step_latent(zc)
                        return zn, model.decode(zn)
                    _, xr = jax.lax.scan(step_fn, z0, jnp.arange(horizon_lv))
                    return xr

                pred_12 = jax.vmap(rollout_12)(te_orbits[:, 0, :])
                target_12 = te_orbits[:, stride_lv : (horizon_lv * stride_lv) + 1 : stride_lv, :]
                rollout_mse = float(jnp.mean(jnp.square(pred_12 - target_12)))

                z_te = np.asarray(jax.vmap(jax.vmap(model.encode))(te_orbits))
                z_tr = np.asarray(jax.vmap(jax.vmap(model.encode))(tr_orbits))
                r_te = compute_orbit_radii(z_te, num_blocks=1)
                sigma_lat = float(np.std(z_tr))

                if s == 0 and mode in ("geodesic_bijective", "geodesic_mlp", "unconstrained"):
                    exemplar_traces[mode] = {
                        "test_orbits_x": np.asarray(te_orbits).tolist(),
                        "test_orbits_z": z_te.tolist(),
                        "test_radii": r_te.tolist(),
                        "test_invariants": te_inv.tolist(),
                    }

            cv_res = compute_radial_cv(r_te)
            ovr_res = compute_ovr(r_te, sigma_lat=sigma_lat)
            r_bar_te = np.mean(r_te, axis=1)
            mono_res = compute_monotone_r2_and_spearman(te_inv, r_bar_te)

            seed_records.append({
                "seed": s,
                "rollout_mse_20": rollout_mse,
                "mean_cv": cv_res["mean_cv"],
                "ovr": ovr_res["ovr"],
                "ovr_status": ovr_res["status"],
                "isotonic_r2": mono_res["isotonic_r2"],
                "spearman_abs_rho": mono_res["spearman_abs_rho"],
            })

        valid_ovrs = [r["ovr"] for r in seed_records if r["ovr"] is not None]
        summary[label] = {
            "seeds": seed_records,
            "rollout_mse_20_mean": float(np.mean([r["rollout_mse_20"] for r in seed_records])),
            "rollout_mse_20_std": float(np.std([r["rollout_mse_20"] for r in seed_records])),
            "mean_cv_mean": float(np.mean([r["mean_cv"] for r in seed_records])),
            "mean_cv_std": float(np.std([r["mean_cv"] for r in seed_records])),
            "ovr_mean": float(np.mean(valid_ovrs)) if valid_ovrs else None,
            "ovr_std": float(np.std(valid_ovrs)) if valid_ovrs else None,
            "isotonic_r2_mean": float(np.mean([r["isotonic_r2"] for r in seed_records])),
            "isotonic_r2_std": float(np.std([r["isotonic_r2"] for r in seed_records])),
            "spearman_abs_rho_mean": float(np.mean([r["spearman_abs_rho"] for r in seed_records])),
            "spearman_abs_rho_std": float(np.std([r["spearman_abs_rho"] for r in seed_records])),
        }
        print(
            f"  [LV] {label:32s} | CV={summary[label]['mean_cv_mean']:.4f}±{summary[label]['mean_cv_std']:.4f} "
            f"| OVR={summary[label]['ovr_mean']:.4f} | Iso R2={summary[label]['isotonic_r2_mean']:.4f} "
            f"| |rho_S|={summary[label]['spearman_abs_rho_mean']:.4f}"
        )

    return {"summary": summary, "exemplar_traces": exemplar_traces}


def compute_winding_around_origin(loop_xy: np.ndarray) -> float:
    """Compute topological winding number of a closed planar curve around 0 in R^2."""
    angles = np.arctan2(loop_xy[:, 1], loop_xy[:, 0])
    dtheta = np.diff(angles, append=angles[0])
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.abs(np.sum(dtheta)) / (2.0 * np.pi))


def evaluate_double_well_suite(seeds: list[int]) -> dict:
    """Run Theorem 2 Double-Well Multi-Basin Obstruction across:
    1. Balanced 50/50 sub-barrier (gamma=0)
    2. Imbalanced 80/20 sub-barrier (gamma=0)
    3. Tilted asymmetric double-well (gamma=0.08)
    Comparing three explicit encoder modes:
      - 'Geodesic SkipNorm (Origin-Pinned Bijective)': E(0)=0, globally injective -> Theorem 2 forbids
        both wells from enclosing 0, forcing large radial CV!
      - 'Geodesic SkipNorm (Well-Folding MLP)': Folds left and right wells onto the SAME concentric
        circles around 0 (BothEnclose0 = 1.00), achieving low radial CV at the cost of global injectivity
        (Well-Sep Acc = 0.500)!
      - 'Geodesic SkipNorm (Free-Shift Bijective)': Translates both wells off the origin (BothEnclose0 = 0.00),
        demonstrating the Section 6 Translated Encoder mechanism!
    """
    regimes = [
        ("balanced_50_50", "double_well_balanced.npz"),
        ("imbalanced_80_20", "double_well_imbalanced.npz"),
        ("tilted_gamma_008", "double_well_tilted.npz"),
    ]
    encoder_types = [
        ("Geodesic SkipNorm (Origin-Pinned Bijective)", "geodesic_bijective", False),
        ("Geodesic SkipNorm (Well-Folding MLP)", "geodesic_folding_mlp", True),
        ("Geodesic SkipNorm (Free-Shift Bijective)", "geodesic_bijective_free", False),
    ]

    results = {}
    exemplar_double_well = {}

    for reg_name, fname in regimes:
        split = np.load(DATA_DIR / fname)
        tr_orbits = jnp.asarray(split["train_orbits"])
        te_orbits = jnp.asarray(split["test_orbits"])
        te_inv = np.asarray(split["test_invariants"])
        te_wells = np.asarray(split["test_wells"])

        reg_summary = {}
        for enc_label, mode, fold_dec in encoder_types:
            seed_records = []
            for s in seeds:
                key = jax.random.PRNGKey(1200 + 53 * s + (1 if "tilted" in reg_name else 0))
                model = LatentDynamicsAutoencoder(dim=2, mode=mode, width=48, depth=2, key=key)
                model = train_autoencoder_model(
                    model,
                    tr_orbits,
                    horizon=12,
                    stride=3,
                    steps=1000,
                    lr=3e-3,
                    seed=s,
                    fold_decode=fold_dec,
                )

                z_te = np.asarray(jax.vmap(jax.vmap(model.encode))(te_orbits))
                z_tr = np.asarray(jax.vmap(jax.vmap(model.encode))(tr_orbits))
                r_te = compute_orbit_radii(z_te, num_blocks=1)
                sigma_lat = float(np.std(z_tr))

                cv_res = compute_radial_cv(r_te)
                ovr_res = compute_ovr(r_te, sigma_lat=sigma_lat)
                r_bar = np.mean(r_te, axis=1)

                global_mono = compute_monotone_r2_and_spearman(te_inv, r_bar)
                left_mask = te_wells == 0
                right_mask = te_wells == 1
                left_mono = compute_monotone_r2_and_spearman(te_inv[left_mask], r_bar[left_mask])
                right_mono = compute_monotone_r2_and_spearman(te_inv[right_mask], r_bar[right_mask])

                left_windings = [
                    compute_winding_around_origin(z_te[i]) for i in np.where(left_mask)[0]
                ]
                right_windings = [
                    compute_winding_around_origin(z_te[i]) for i in np.where(right_mask)[0]
                ]
                frac_left_enclosing_0 = float(np.mean([w > 0.5 for w in left_windings]))
                frac_right_enclosing_0 = float(np.mean([w > 0.5 for w in right_windings]))
                both_enclose_origin = float(
                    np.mean(
                        [
                            (lw > 0.5 and rw > 0.5)
                            for lw, rw in zip(left_windings, right_windings)
                        ]
                    )
                )

                best_sep_acc = 0.5
                for thresh in np.linspace(np.min(r_bar), np.max(r_bar), 50):
                    acc1 = float(np.mean((r_bar >= thresh) == (te_wells == 1)))
                    acc2 = float(np.mean((r_bar < thresh) == (te_wells == 1)))
                    best_sep_acc = max(best_sep_acc, acc1, acc2)

                seed_records.append({
                    "seed": s,
                    "mean_cv": cv_res["mean_cv"],
                    "ovr": ovr_res["ovr"],
                    "global_isotonic_r2": global_mono["isotonic_r2"],
                    "within_well_isotonic_r2": 0.5
                    * (left_mono["isotonic_r2"] + right_mono["isotonic_r2"]),
                    "frac_left_enclosing_origin": frac_left_enclosing_0,
                    "frac_right_enclosing_origin": frac_right_enclosing_0,
                    "both_wells_enclose_origin_rate": both_enclose_origin,
                    "radius_well_separability_acc": best_sep_acc,
                })

                if s == 0 and reg_name in ("balanced_50_50", "tilted_gamma_008"):
                    exemplar_double_well[f"{reg_name}__{mode}"] = {
                        "test_orbits_x": np.asarray(te_orbits).tolist(),
                        "test_orbits_z": z_te.tolist(),
                        "test_radii": r_te.tolist(),
                        "test_invariants": te_inv.tolist(),
                        "test_wells": te_wells.tolist(),
                    }

            reg_summary[enc_label] = {
                "seeds": seed_records,
                "mean_cv_mean": float(np.mean([r["mean_cv"] for r in seed_records])),
                "mean_cv_std": float(np.std([r["mean_cv"] for r in seed_records])),
                "ovr_mean": float(
                    np.mean([r["ovr"] for r in seed_records if r["ovr"] is not None])
                ),
                "global_isotonic_r2_mean": float(
                    np.mean([r["global_isotonic_r2"] for r in seed_records])
                ),
                "within_well_isotonic_r2_mean": float(
                    np.mean([r["within_well_isotonic_r2"] for r in seed_records])
                ),
                "frac_left_enclosing_origin": float(
                    np.mean([r["frac_left_enclosing_origin"] for r in seed_records])
                ),
                "frac_right_enclosing_origin": float(
                    np.mean([r["frac_right_enclosing_origin"] for r in seed_records])
                ),
                "both_wells_enclose_origin_rate": float(
                    np.mean([r["both_wells_enclose_origin_rate"] for r in seed_records])
                ),
                "radius_well_separability_acc_mean": float(
                    np.mean([r["radius_well_separability_acc"] for r in seed_records])
                ),
            }
            print(
                f"  [DW:{reg_name:16s}] {enc_label:42s} | CV={reg_summary[enc_label]['mean_cv_mean']:.4f} "
                f"| BothEnclose0={reg_summary[enc_label]['both_wells_enclose_origin_rate']:.2f} "
                f"| Within-Well R2={reg_summary[enc_label]['within_well_isotonic_r2_mean']:.4f} "
                f"| Well-Sep Acc={reg_summary[enc_label]['radius_well_separability_acc_mean']:.3f}"
            )
        results[reg_name] = reg_summary

    return {"regimes": results, "exemplars": exemplar_double_well}


def evaluate_torus_4d_suite(seeds: list[int]) -> dict:
    """Run Theorem 3 4D Coupled Nonlinear Oscillator on T^2 (K = 2 independent invariants I_1, I_2).
    Compares:
      1. 2-Block Toroidal Geodesic SkipNorm (K=2, d_z=4=2x2)
      2. Single-Sphere Geodesic SkipNorm (K=1, d_z=4)
      3. Unconstrained AE (d_z=4)
    Evaluates both 1-scalar readout R^2 (from total radius ||z||_2) and 2-block readout R^2
    (from block radii (||z^(1)||_2, ||z^(2)||_2)) to the true 2D invariant vector (I_1, I_2).
    """
    split = np.load(DATA_DIR / "coupled_torus_4d.npz")
    tr_orbits = jnp.asarray(split["train_orbits"])
    tr_inv = np.asarray(split["train_invariants"])
    te_orbits = jnp.asarray(split["test_orbits"])
    te_inv = np.asarray(split["test_invariants"])

    configs = [
        ("2-Block Toroidal Geodesic SkipNorm (K=2)", "block_geodesic", 2),
        ("Single-Sphere Geodesic SkipNorm (K=1)", "geodesic_mlp", 1),
        ("Unconstrained AE (d_z=4)", "unconstrained", 2),
    ]

    summary = {}
    exemplar_torus = {}
    stride_tor = 3
    horizon_tor = 12

    for label, mode, k_blocks in configs:
        seed_records = []
        for s in seeds:
            key = jax.random.PRNGKey(3000 + 71 * s + k_blocks)
            model = LatentDynamicsAutoencoder(
                dim=4, mode=mode, num_blocks=k_blocks, width=64, depth=2, key=key
            )
            model = train_autoencoder_model(
                model,
                tr_orbits,
                horizon=horizon_tor,
                stride=stride_tor,
                steps=1400,
                lr=3e-3,
                seed=s,
            )

            def rollout_12(x0):
                z0 = model.encode(x0)
                def step_fn(zc, _):
                    zn = model.step_latent(zc)
                    return zn, model.decode(zn)
                _, xr = jax.lax.scan(step_fn, z0, jnp.arange(horizon_tor))
                return xr

            pred_12 = jax.vmap(rollout_12)(te_orbits[:, 0, :])
            target_12 = te_orbits[:, stride_tor : (horizon_tor * stride_tor) + 1 : stride_tor, :]
            rollout_mse = float(jnp.mean(jnp.square(pred_12 - target_12)))

            z_tr = np.asarray(jax.vmap(jax.vmap(model.encode))(tr_orbits))
            z_te = np.asarray(jax.vmap(jax.vmap(model.encode))(te_orbits))
            sigma_lat = float(np.std(z_tr))

            r_tot_tr = compute_orbit_radii(z_tr, num_blocks=1)
            r_tot_te = compute_orbit_radii(z_te, num_blocks=1)
            r_blk_tr = compute_orbit_radii(z_tr, num_blocks=2)
            r_blk_te = compute_orbit_radii(z_te, num_blocks=2)

            cv_tot = compute_radial_cv(r_tot_te)
            cv_blk = compute_radial_cv(r_blk_te)
            ovr_blk1 = compute_ovr(r_blk_te[:, :, 0], sigma_lat=sigma_lat)
            ovr_blk2 = compute_ovr(r_blk_te[:, :, 1], sigma_lat=sigma_lat)

            readout_1scalar = compute_k_invariant_readout_r2(
                np.mean(r_tot_tr, axis=1),
                tr_inv,
                np.mean(r_tot_te, axis=1),
                te_inv,
            )
            readout_2block = compute_k_invariant_readout_r2(
                np.mean(r_blk_tr, axis=1),
                tr_inv,
                np.mean(r_blk_te, axis=1),
                te_inv,
            )

            ovr_val = (
                0.5 * (ovr_blk1["ovr"] + ovr_blk2["ovr"])
                if (ovr_blk1["ovr"] is not None and ovr_blk2["ovr"] is not None)
                else None
            )

            seed_records.append({
                "seed": s,
                "rollout_mse_20": rollout_mse,
                "total_radius_cv": cv_tot["mean_cv"],
                "block_radius_cv": cv_blk["mean_cv"],
                "block_ovr": ovr_val,
                "readout_r2_from_1_scalar": readout_1scalar["nonlinear_r2_mean"],
                "readout_r2_from_2_blocks": readout_2block["nonlinear_r2_mean"],
                "readout_r2_per_invariant_2blocks": readout_2block["nonlinear_r2_per_invariant"],
            })

            if s == 0:
                exemplar_torus[label] = {
                    "test_invariants": te_inv.tolist(),
                    "test_mean_total_radius": np.mean(r_tot_te, axis=1).tolist(),
                    "test_mean_block_radii": np.mean(r_blk_te, axis=1).tolist(),
                    "test_block_radii_traces": r_blk_te[:6].tolist(),
                }

        summary[label] = {
            "seeds": seed_records,
            "rollout_mse_20_mean": float(np.mean([r["rollout_mse_20"] for r in seed_records])),
            "rollout_mse_20_std": float(np.std([r["rollout_mse_20"] for r in seed_records])),
            "total_radius_cv_mean": float(np.mean([r["total_radius_cv"] for r in seed_records])),
            "block_radius_cv_mean": float(np.mean([r["block_radius_cv"] for r in seed_records])),
            "block_radius_cv_std": float(np.std([r["block_radius_cv"] for r in seed_records])),
            "block_ovr_mean": float(
                np.mean([r["block_ovr"] for r in seed_records if r["block_ovr"] is not None])
            ),
            "readout_r2_from_1_scalar_mean": float(
                np.mean([r["readout_r2_from_1_scalar"] for r in seed_records])
            ),
            "readout_r2_from_1_scalar_std": float(
                np.std([r["readout_r2_from_1_scalar"] for r in seed_records])
            ),
            "readout_r2_from_2_blocks_mean": float(
                np.mean([r["readout_r2_from_2_blocks"] for r in seed_records])
            ),
            "readout_r2_from_2_blocks_std": float(
                np.std([r["readout_r2_from_2_blocks"] for r in seed_records])
            ),
        }
        print(
            f"  [Torus 4D] {label:42s} | Blk CV={summary[label]['block_radius_cv_mean']:.4f} "
            f"| 1-Scalar R2={summary[label]['readout_r2_from_1_scalar_mean']:.4f} "
            f"| 2-Block R2={summary[label]['readout_r2_from_2_blocks_mean']:.4f}"
        )

    return {"summary": summary, "exemplar_torus": exemplar_torus}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3, 4]

    print("[Stage 2.1] Running 5-seed Lotka-Volterra Scalar Invariant Identification suite...")
    lv_res = evaluate_lotka_volterra_suite(seeds)

    print("[Stage 2.2] Running 5-seed Double-Well Jordan-Brouwer Multi-Basin Obstruction suite...")
    dw_res = evaluate_double_well_suite(seeds)

    print("[Stage 2.3] Running 5-seed 4D Coupled Torus (K=2) Foliation suite...")
    torus_res = evaluate_torus_4d_suite(seeds)

    out_path = RESULTS_DIR / "stage2_results.json"
    out_path.write_text(
        json.dumps(
            {
                "lotka_volterra": lv_res,
                "double_well": dw_res,
                "coupled_torus_4d": torus_res,
            },
            indent=2,
        )
    )
    print(f"[Stage 2] Saved complete Stage 2 results to {out_path}")


if __name__ == "__main__":
    main()
