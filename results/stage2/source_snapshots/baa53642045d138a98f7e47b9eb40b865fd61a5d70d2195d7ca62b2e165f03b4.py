"""Stage 2: Core Benchmarks for Scalar Invariant Foliation (Theorem 1),
Jordan-Brouwer Multi-Basin Obstruction (Theorem 2), and K-Block Toroidal Foliation (Theorem 3).

All models are trained and evaluated on the locked whole-orbit disjoint train/val/test splits
from data/splits/ across 5 random seeds (0..4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
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
    centered_rms_spread,
    compute_k_invariant_readout_r2,
    compute_monotone_r2_and_spearman,
    compute_orbit_radii,
    compute_ovr,
    compute_radial_cv,
    compute_radius_threshold_probe,
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
        elif self.mode == "block_geodesic_general":
            return block_geodesic_skip_norm_fn(
                z, self.latent_net(z), num_blocks=self.num_blocks, alpha=self.alpha, eps=1e-12
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


class ProjectedInvariantModel(eqx.Module):
    """COMET-inspired trajectory-trained projected-invariant baseline.

    This is not a reproduction of COMET (Kasim and Lim, 2022). It uses RK2,
    finite-window trajectory loss and an invariant-spread regularizer. Orthogonality
    is a continuous vector-field property; finite RK2 steps need not conserve h.
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
    model: ProjectedInvariantModel,
    train_orbits: jnp.ndarray,
    horizon: int = 6,
    steps: int = 800,
    lr: float = 3e-3,
    seed: int = 0,
    stride: int = 1,
) -> ProjectedInvariantModel:
    """Trajectory-only training; no true invariant labels enter this objective."""
    sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps, alpha=0.05)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    M, T, D = train_orbits.shape

    @eqx.filter_jit
    def loss_fn(m: ProjectedInvariantModel, batch_w: jnp.ndarray) -> jax.Array:
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
    span = horizon * stride
    max_start = T - span - 1
    tr_np = np.asarray(train_orbits)
    for _ in range(steps):
        m_idx = rng.integers(0, M, size=batch_size)
        t_idx = rng.integers(0, max_start, size=batch_size)
        windows = np.stack(
            [tr_np[m_idx[b], t_idx[b] : t_idx[b] + span + 1 : stride, :] for b in range(batch_size)],
            axis=0,
        )
        model, opt_state, _ = make_step(model, opt_state, jnp.asarray(windows))
    return model


PROTOCOL_VERSION = "stage2-v2-heldout-calibration"
RUN_SETTINGS = {"resume": True, "steps_multiplier": 1.0, "models": None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _sample_summary(records: list[dict], names: list[str]) -> dict:
    result = {"seeds": records}
    for name in names:
        values = [r[name] for r in records if r.get(name) is not None]
        result[name + "_mean"] = float(np.mean(values)) if values else None
        result[name + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result[name + "_valid_seeds"] = len(values)
    return result


def _selected(mode: str) -> bool:
    return RUN_SETTINGS["models"] is None or mode in RUN_SETTINGS["models"]


def _configuration(suite, split_file, mode, seed, *, width, dim, blocks=1, steps=1200,
                   dt=.05, stride=3, horizon=12, quotient=False, key_seed=None):
    sources = [Path(__file__), ROOT / "skipnorm/metrics.py", ROOT / "skipnorm/jax_layers.py"]
    return {
        "protocol_version": PROTOCOL_VERSION, "suite": suite, "mode": mode,
        "seed": seed, "initialization_key_seed": key_seed, "width": width, "depth": 2,
        "dim": dim, "num_blocks": blocks, "steps": max(1, round(steps * RUN_SETTINGS["steps_multiplier"])),
        "batch_size": 64, "optimizer": "Adam", "learning_rate": .003,
        "schedule": "cosine decay to 0.05 of initial rate", "dt": dt, "stride": stride,
        "rollout_horizon": horizon, "physical_eval_horizon": dt * stride * horizon,
        "reflection_quotient": quotient,
        "latent_prior": ("radius-only frequency, f_k=omega_k(radii) J z_k"
                         if mode in ("geodesic_bijective", "geodesic_bijective_free", "geodesic_mlp", "geodesic_folding_mlp", "block_geodesic") and dim == 2 * blocks
                         else "general state-dependent vector field"),
        "encoder_origin_pinned": mode not in ("geodesic_bijective_free", "projected_invariant"),
        "loss": ("trajectory MSE + relu(0.30-std(h))^2 + 0.5*within_window_variance(h)"
                 if mode == "projected_invariant" else
                 "reconstruction MSE + rollout MSE + 1.5*one-step latent MSE + multi-step latent MSE"),
        "quotient_loss_note": "min squared Euclidean error to x or -x, per sample" if quotient else None,
        "invariant_labels_used_in_training": False,
        "split_file": str((DATA_DIR / split_file).relative_to(ROOT)),
        "split_sha256": _sha256(DATA_DIR / split_file),
        "source_sha256": {str(x.relative_to(ROOT)): _sha256(x) for x in sources},
        "seed_sd_ddof": 1,
        "isotonic_protocol": "train-fitted invariant-to-mean-radius; direction by training SSE; linear interpolation; boundary clipping; untruncated test R2",
        "ovr_protocol": "within/total; total and between variances must exceed (1e-4*training centered RMS latent spread)^2",
    }


def _fit_run(model, train_orbits, config):
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    run_dir = RESULTS_DIR / "runs" / config["suite"] / f'{config["mode"]}_seed{config["seed"]}_{fingerprint}'
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "model.eqx"
    if RUN_SETTINGS["resume"] and checkpoint.exists():
        print(f"  resume {config['suite']} {config['mode']} seed={config['seed']}", flush=True)
        return eqx.tree_deserialise_leaves(checkpoint, model), run_dir
    print(f"  train {config['suite']} {config['mode']} seed={config['seed']} steps={config['steps']}", flush=True)
    start = time.perf_counter()
    kwargs = dict(horizon=config["rollout_horizon"], stride=config["stride"], steps=config["steps"],
                  lr=config["learning_rate"], seed=config["seed"])
    if config["mode"] == "projected_invariant":
        model = train_comet_model(model, train_orbits, **kwargs)
    else:
        model = train_autoencoder_model(model, train_orbits, fold_decode=config["reflection_quotient"], **kwargs)
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    metadata = {"configuration": config, "training_seconds": time.perf_counter() - start,
                "parameter_count": sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))),
                "jax_version": jax.__version__, "numpy_version": np.__version__,
                "backend": jax.default_backend(), "devices": [str(x) for x in jax.devices()],
                "platform": platform.platform(), "completed_utc": datetime.now(timezone.utc).isoformat()}
    temporary = checkpoint.with_suffix(".tmp")
    eqx.tree_serialise_leaves(temporary, model)
    temporary.replace(checkpoint)
    _write_json(run_dir / "configuration.json", metadata)
    print(f"  trained in {metadata['training_seconds']:.1f}s; checkpoint saved", flush=True)
    return model, run_dir


def _rollout(model, test_orbits, *, stride, horizon=12, direct=False):
    def one(x0):
        z0 = x0 if direct else model.encode(x0)
        def step(z, _):
            zn = model.step(z) if direct else model.step_latent(z)
            return zn, zn if direct else model.decode(zn)
        return jax.lax.scan(step, z0, jnp.arange(horizon))[1]
    predictions = np.asarray(eqx.filter_jit(lambda batch: jax.vmap(one)(batch))(test_orbits[:, 0]))
    targets = np.asarray(test_orbits[:, stride:horizon * stride + 1:stride])
    per_orbit = np.mean((predictions - targets) ** 2, axis=(1, 2))
    return float(np.mean(per_orbit)), predictions, per_orbit


def _save_run(run_dir, record, arrays, diagnostics):
    np.savez_compressed(run_dir / "encoded_trajectories.npz", **arrays)
    record["artifact_directory"] = str(run_dir.relative_to(ROOT))
    _write_json(run_dir / "evaluation.json", {"record": record, "diagnostics": diagnostics})


def _encode(model, tr, te):
    encode = eqx.filter_jit(lambda m, x: jax.vmap(jax.vmap(m.encode))(x))
    return np.asarray(encode(model, tr)), np.asarray(encode(model, te))


def evaluate_lotka_volterra_suite(seeds: list[int]) -> dict:
    split_file = "lotka_volterra.npz"
    split = np.load(DATA_DIR / split_file)
    tr, te = jnp.asarray(split["train_orbits"]), jnp.asarray(split["test_orbits"])
    tr_inv, te_inv = split["train_invariants"], split["test_invariants"]
    modes = [
        ("Geodesic SkipNorm (Bijective)", "geodesic_bijective"),
        ("Geodesic SkipNorm (MLP)", "geodesic_mlp"),
        ("Projected Residual SkipNorm", "projected_residual"),
        ("State-Dependent SO(2)", "state_dep_so2"),
        ("Projected-Invariant Baseline (COMET-inspired)", "projected_invariant"),
        ("Unconstrained AE", "unconstrained"),
    ]
    summary, exemplars = {}, {}
    for label, mode in modes:
        if not _selected(mode):
            continue
        records = []
        for seed in seeds:
            key_seed = 500 + 37 * seed
            config = _configuration("lotka_volterra", split_file, mode, seed, width=48, dim=2, key_seed=key_seed)
            key = jax.random.PRNGKey(key_seed)
            model = (ProjectedInvariantModel(dim=2, width=48, depth=2, dt=.15, key=key)
                     if mode == "projected_invariant" else
                     LatentDynamicsAutoencoder(dim=2, mode=mode, width=48, depth=2, key=key))
            model, run_dir = _fit_run(model, tr, config)
            mse, predictions, per_orbit_mse = _rollout(model, te, stride=3, direct=mode == "projected_invariant")
            if mode == "projected_invariant":
                enc = eqx.filter_jit(lambda m, x: jax.vmap(jax.vmap(m.invariant_scalar))(x))
                r_tr, r_te = np.asarray(enc(model, tr)), np.asarray(enc(model, te))
                z_tr, z_te = r_tr[..., None], r_te[..., None]
            else:
                z_tr, z_te = _encode(model, tr, te)
                r_tr, r_te = compute_orbit_radii(z_tr), compute_orbit_radii(z_te)
            sigma = centered_rms_spread(z_tr)
            cv, ovr = compute_radial_cv(r_te), compute_ovr(r_te, sigma)
            mono = compute_monotone_r2_and_spearman(tr_inv, r_tr.mean(axis=1), te_inv, r_te.mean(axis=1))
            record = {"seed": seed, "rollout_mse_20": mse, "rollout_mse": mse,
                      "physical_eval_horizon": 1.8, "mean_cv": cv["mean_cv"], "ovr": ovr["ovr"],
                      "ovr_status": ovr["status"], "isotonic_r2": mono["isotonic_r2"],
                      "spearman_abs_rho": mono["spearman_abs_rho"]}
            _save_run(run_dir, record, {
                "train_z": z_tr, "test_z": z_te, "train_radii": r_tr, "test_radii": r_te,
                "train_invariants": tr_inv, "test_invariants": te_inv,
                "test_rollout_predictions": predictions, "test_rollout_mse_per_orbit": per_orbit_mse,
            }, {"cv": cv, "ovr": ovr, "isotonic": mono, "sigma_lat_train": sigma})
            records.append(record)
            if seed == 0 and mode in ("geodesic_bijective", "geodesic_mlp", "unconstrained"):
                exemplars[mode] = {"test_orbits_x": np.asarray(te).tolist(), "test_orbits_z": z_te.tolist(),
                                   "test_radii": r_te.tolist(), "test_invariants": te_inv.tolist()}
            print(f"  LV {label} seed={seed}: CV={record['mean_cv']:.4f} R2={record['isotonic_r2']:.4f} MSE={mse:.6g}", flush=True)
        summary[label] = _sample_summary(records, ["rollout_mse_20", "mean_cv", "ovr", "isotonic_r2", "spearman_abs_rho"])
    return {"summary": summary, "exemplar_traces": exemplars,
            "evaluation": {"rollout_steps": 12, "stride": 3, "physical_horizon": 1.8,
                           "legacy_field_rollout_mse_20": "alias only; actual horizon is 12 coarse steps"}}


def compute_winding_around_origin(loop_xy: np.ndarray) -> float:
    """Absolute winding of the sampled polygon, closed by its last-to-first segment.

    This finite-trajectory diagnostic is not a certificate of a complete level set.
    """
    angles = np.arctan2(loop_xy[:, 1], loop_xy[:, 0])
    delta = (np.diff(angles, append=angles[0]) + np.pi) % (2 * np.pi) - np.pi
    return float(abs(np.sum(delta)) / (2 * np.pi))


def evaluate_double_well_suite(seeds: list[int]) -> dict:
    regimes = [("balanced_50_50", "double_well_balanced.npz"),
               ("imbalanced_80_20", "double_well_imbalanced.npz"),
               ("tilted_gamma_008", "double_well_tilted.npz")]
    types = [("Geodesic SkipNorm (Origin-Pinned Bijective)", "geodesic_bijective", False),
             ("Geodesic SkipNorm (Reflection Quotient Control)", "geodesic_folding_mlp", True),
             ("Geodesic SkipNorm (Free-Shift Bijective)", "geodesic_bijective_free", False)]
    results, exemplars, exclusions = {}, {}, []
    for regime, split_file in regimes:
        split = np.load(DATA_DIR / split_file)
        tr, te = jnp.asarray(split["train_orbits"]), jnp.asarray(split["test_orbits"])
        tr_inv, te_inv = split["train_invariants"], split["test_invariants"]
        tr_wells, te_wells = split["train_wells"], split["test_wells"]
        regime_summary = {}
        for label, mode, quotient in types:
            if not _selected(mode):
                continue
            if quotient and regime == "tilted_gamma_008":
                exclusions.append({"regime": regime, "model": label,
                                   "reason": "Reflection x -> -x is not a symmetry of the tilted Hamiltonian; quotient control is inapplicable."})
                continue
            records = []
            for seed in seeds:
                key_seed = 1200 + 53 * seed + (1 if "tilted" in regime else 0)
                config = _configuration(regime, split_file, mode, seed, width=48, dim=2, steps=1000,
                                        quotient=quotient, key_seed=key_seed)
                model = LatentDynamicsAutoencoder(dim=2, mode=mode, width=48, depth=2, key=jax.random.PRNGKey(key_seed))
                model, run_dir = _fit_run(model, tr, config)
                z_tr, z_te = _encode(model, tr, te)
                r_tr, r_te = compute_orbit_radii(z_tr), compute_orbit_radii(z_te)
                bar_tr, bar_te = r_tr.mean(axis=1), r_te.mean(axis=1)
                sigma = centered_rms_spread(z_tr)
                cv, ovr = compute_radial_cv(r_te), compute_ovr(r_te, sigma)
                mono = compute_monotone_r2_and_spearman(tr_inv, bar_tr, te_inv, bar_te)
                per_well = [compute_monotone_r2_and_spearman(tr_inv[tr_wells == w], bar_tr[tr_wells == w],
                            te_inv[te_wells == w], bar_te[te_wells == w]) for w in (0, 1)]
                windings = np.asarray([compute_winding_around_origin(x) for x in z_te])
                left, right = windings[te_wells == 0] > .5, windings[te_wells == 1] > .5
                both_any = float(np.any(left) and np.any(right))
                probe = compute_radius_threshold_probe(bar_tr, tr_wells, bar_te, te_wells)
                record = {"seed": seed, "mean_cv": cv["mean_cv"], "ovr": ovr["ovr"],
                          "ovr_status": ovr["status"], "global_isotonic_r2": mono["isotonic_r2"],
                          "within_well_isotonic_r2": float(np.mean([m["isotonic_r2"] for m in per_well])),
                          "frac_left_enclosing_origin": float(np.mean(left)),
                          "frac_right_enclosing_origin": float(np.mean(right)),
                          "both_wells_enclose_origin_rate": both_any,
                          "radius_well_separability_acc": probe["accuracy"]}
                _save_run(run_dir, record, {"train_z": z_tr, "test_z": z_te,
                    "train_radii": r_tr, "test_radii": r_te, "train_invariants": tr_inv,
                    "test_invariants": te_inv, "train_wells": tr_wells, "test_wells": te_wells},
                    {"cv": cv, "ovr": ovr, "global_isotonic": mono, "within_well_isotonic": per_well,
                     "radius_threshold_probe": probe, "test_polygon_winding": windings.tolist(), "sigma_lat_train": sigma})
                records.append(record)
                if seed == 0 and regime in ("balanced_50_50", "tilted_gamma_008"):
                    exemplars[f"{regime}__{mode}"] = {"test_orbits_x": np.asarray(te).tolist(),
                        "test_orbits_z": z_te.tolist(), "test_radii": r_te.tolist(),
                        "test_invariants": te_inv.tolist(), "test_wells": te_wells.tolist()}
                print(f"  DW {regime} {mode} seed={seed}: CV={record['mean_cv']:.4f} globalR2={record['global_isotonic_r2']:.4f} probe={probe['accuracy']:.3f}", flush=True)
            names = ["mean_cv", "ovr", "global_isotonic_r2", "within_well_isotonic_r2",
                     "frac_left_enclosing_origin", "frac_right_enclosing_origin",
                     "both_wells_enclose_origin_rate", "radius_well_separability_acc"]
            sm = _sample_summary(records, names)
            for name in ("frac_left_enclosing_origin", "frac_right_enclosing_origin", "both_wells_enclose_origin_rate"):
                sm[name] = sm[name + "_mean"]
            regime_summary[label] = sm
        results[regime] = regime_summary
    return {"regimes": results, "exemplars": exemplars, "excluded_configurations": exclusions,
            "evaluation": {"both_enclose_definition": "fraction of seeds with at least one left and one right sampled polygon enclosing zero",
                           "winding_limitation": "finite trajectories are polygon-closed; not a completeness certificate",
                           "well_separability": "training-fitted threshold/orientation; held-out accuracy",
                           "quotient_control": "explicit E(x)=E(-x), reconstruction and prediction only up to reflection; symmetric potentials only"}}


def evaluate_torus_4d_suite(seeds: list[int]) -> dict:
    split_file = "coupled_torus_4d.npz"
    split = np.load(DATA_DIR / split_file)
    tr, te = jnp.asarray(split["train_orbits"]), jnp.asarray(split["test_orbits"])
    tr_inv, te_inv = split["train_invariants"], split["test_invariants"]
    configs = [("2-Block Toroidal Geodesic SkipNorm (K=2)", "block_geodesic", 2),
               ("2-Block Geodesic SkipNorm (General Vector Field)", "block_geodesic_general", 2),
               ("Single-Sphere Geodesic SkipNorm (K=1)", "geodesic_mlp", 1),
               ("Unconstrained AE (d_z=4)", "unconstrained", 2)]
    summary, exemplars = {}, {}
    for label, mode, blocks in configs:
        if not _selected(mode):
            continue
        records = []
        for seed in seeds:
            key_seed = 3000 + 71 * seed + blocks
            config = _configuration("coupled_torus_4d", split_file, mode, seed, width=64, dim=4,
                                    blocks=blocks, steps=1400, dt=.08, key_seed=key_seed)
            model = LatentDynamicsAutoencoder(dim=4, mode=mode, num_blocks=blocks, width=64, depth=2,
                                              key=jax.random.PRNGKey(key_seed))
            model, run_dir = _fit_run(model, tr, config)
            mse, predictions, per_orbit_mse = _rollout(model, te, stride=3)
            z_tr, z_te = _encode(model, tr, te)
            sigma = centered_rms_spread(z_tr)
            total_tr, total_te = compute_orbit_radii(z_tr), compute_orbit_radii(z_te)
            block_tr, block_te = compute_orbit_radii(z_tr, 2), compute_orbit_radii(z_te, 2)
            cv_total, cv_block = compute_radial_cv(total_te), compute_radial_cv(block_te)
            ovrs = [compute_ovr(block_te[:, :, k], sigma) for k in range(2)]
            scalar = compute_k_invariant_readout_r2(total_tr.mean(axis=1), tr_inv, total_te.mean(axis=1), te_inv)
            two = compute_k_invariant_readout_r2(block_tr.mean(axis=1), tr_inv, block_te.mean(axis=1), te_inv)
            ovr = float(np.mean([v["ovr"] for v in ovrs])) if all(v["ovr"] is not None for v in ovrs) else None
            record = {"seed": seed, "rollout_mse_20": mse, "rollout_mse": mse,
                      "physical_eval_horizon": 2.88, "total_radius_cv": cv_total["mean_cv"],
                      "block_radius_cv": cv_block["mean_cv"], "block_ovr": ovr,
                      "readout_r2_from_1_scalar": scalar["nonlinear_r2_mean"],
                      "readout_r2_from_2_blocks": two["nonlinear_r2_mean"],
                      "readout_r2_per_invariant_2blocks": two["nonlinear_r2_per_invariant"]}
            _save_run(run_dir, record, {"train_z": z_tr, "test_z": z_te,
                "train_total_radii": total_tr, "test_total_radii": total_te,
                "train_block_radii": block_tr, "test_block_radii": block_te,
                "train_invariants": tr_inv, "test_invariants": te_inv,
                "test_rollout_predictions": predictions, "test_rollout_mse_per_orbit": per_orbit_mse},
                {"cv_total": cv_total, "cv_block": cv_block, "block_ovr": ovrs,
                 "scalar_readout": scalar, "two_block_readout": two, "sigma_lat_train": sigma})
            records.append(record)
            if seed == 0:
                exemplars[label] = {"test_invariants": te_inv.tolist(),
                    "test_mean_total_radius": total_te.mean(axis=1).tolist(),
                    "test_mean_block_radii": block_te.mean(axis=1).tolist(),
                    "test_block_radii_traces": block_te[:6].tolist()}
            print(f"  Torus {mode} seed={seed}: CV={record['block_radius_cv']:.4f} scalarR2={record['readout_r2_from_1_scalar']:.4f} twoR2={record['readout_r2_from_2_blocks']:.4f}", flush=True)
        summary[label] = _sample_summary(records, ["rollout_mse_20", "total_radius_cv", "block_radius_cv",
                                                  "block_ovr", "readout_r2_from_1_scalar", "readout_r2_from_2_blocks"])
    return {"summary": summary, "exemplar_torus": exemplars,
            "evaluation": {"rollout_steps": 12, "stride": 3, "physical_horizon": 2.88,
                           "scalar_readout_input": "time mean of total Euclidean radius",
                           "two_block_readout_input": "two time-mean planar block radii for every model",
                           "readout": "degree-two ridge, lambda=1e-4, training standardization and fit; untruncated test R2"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", nargs="+", choices=["all", "lv", "double_well", "torus"], default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", help="Optional mode identifiers; omitted runs all configured models")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--steps-multiplier", type=float, default=1.0, help="Smoke checks only; nonunit runs get a separate output file")
    args = parser.parse_args()
    RUN_SETTINGS.update(resume=not args.no_resume, steps_multiplier=args.steps_multiplier, models=args.models)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = args.suite == ["all"] and args.seeds == [0, 1, 2, 3, 4] and args.models is None and args.steps_multiplier == 1
    output = RESULTS_DIR / ("stage2_results.json" if full else "stage2_partial_results.json")
    if output.exists():
        archive = RESULTS_DIR / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, archive / output.name)
    result = {"metadata": {"protocol_version": PROTOCOL_VERSION, "seeds": args.seeds,
              "complete_suite": False, "seed_sd_ddof": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "steps_multiplier": args.steps_multiplier}}
    suites = [("lv", "lotka_volterra", evaluate_lotka_volterra_suite),
              ("double_well", "double_well", evaluate_double_well_suite),
              ("torus", "coupled_torus_4d", evaluate_torus_4d_suite)]
    for short, name, evaluate in suites:
        if "all" in args.suite or short in args.suite:
            print(f"[Stage 2] {name}", flush=True)
            result[name] = evaluate(args.seeds)
            _write_json(output, result)
    result["metadata"]["complete_suite"] = full
    result["metadata"]["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output, result)
    print(f"[Stage 2] Saved {output}", flush=True)


if __name__ == "__main__":
    main()
