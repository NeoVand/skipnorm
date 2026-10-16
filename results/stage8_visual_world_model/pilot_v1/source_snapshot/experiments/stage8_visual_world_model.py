"""Pixel-only pilot: persistent physical prediction from short visual context.

This is exploratory and separate from the manuscript. Physical state arrays are
used only by evaluation. The shared visual warm-start and every transition are
trained using image targets. Run artifacts retain config, code, and data hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.visual_physics_data import generate_dataset, image_positions, PIXELS_PER_POSITION
from experiments.visual_physics_models import VisualWorldModel, parameter_counts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_loss(prediction, target):
    """Equal contribution from bright mass pixels and their dark surroundings.

    Continuous foreground weights come from target pixels, never simulator state.
    Per-frame normalization prevents brightness/foreground size dominating loss.
    """
    err = (prediction - target) ** 2
    fg = jnp.clip(target, 0.0, 1.0)
    bg = 1.0 - fg
    axes = (-2, -1)
    foreground = jnp.sum(fg * err, axis=axes) / jnp.maximum(jnp.sum(fg, axis=axes), 1e-6)
    background = jnp.sum(bg * err, axis=axes) / jnp.maximum(jnp.sum(bg, axis=axes), 1e-6)
    return jnp.mean(0.5 * foreground + 0.5 * background)


def sample_windows(images, rng, batch_size, length):
    ix = rng.integers(0, len(images), size=batch_size)
    ts = rng.integers(0, images.shape[1] - length + 1, size=batch_size)
    return jnp.asarray(images[ix[:, None], ts[:, None] + np.arange(length)[None]])


def warmstart(model, images, config, seed, output):
    """Fit vision only, once per seed; all kinds receive the identical weights."""
    path = output / f"vision_seed{seed}.eqx"
    if path.exists():
        return eqx.tree_deserialise_leaves(path, model)
    optimizer = optax.adam(config["learning_rate"])
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(m, state, histories):
        def loss_fn(mm):
            pred = jax.vmap(lambda h: mm.decode(mm.encode(h)))(histories)
            return pixel_loss(pred, histories[:, -1])
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, state = optimizer.update(grads, state, m)
        return eqx.apply_updates(m, updates), state, loss

    rng = np.random.default_rng(73000 + seed)
    start = time.perf_counter()
    curve = []
    for n in range(1, config["warmup"] + 1):
        histories = sample_windows(images, rng, config["batch_size"], config["history"])
        model, state, loss = update(model, state, histories)
        if n == 1 or n % 100 == 0 or n == config["warmup"]:
            row = {"stage": "vision", "seed": seed, "update": n,
                   "loss": float(loss), "seconds": time.perf_counter() - start}
            curve.append(row)
            print(json.dumps(row), flush=True)
    eqx.tree_serialise_leaves(path, model)
    path.with_suffix(".json").write_text(json.dumps(curve, indent=2))
    return model


def train(model, images, validation, config, seed, output):
    horizon, history = config["horizon"], config["history"]
    schedule = optax.cosine_decay_schedule(config["learning_rate"], config["updates"], alpha=.1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(m, windows):
        histories = windows[:, :history]
        latent0 = jax.vmap(m.encode)(histories)
        def evolve(z, _):
            z = jax.vmap(m.step)(z)
            return z, z
        _, zs = jax.lax.scan(evolve, latent0, None, length=horizon)
        zs = jnp.swapaxes(zs, 0, 1)
        predictions = jax.vmap(jax.vmap(m.decode))(zs)
        reconstructed = jax.vmap(m.decode)(latent0)
        prediction_loss = pixel_loss(predictions, windows[:, history:])
        reconstruction_loss = pixel_loss(reconstructed, windows[:, history - 1])
        return prediction_loss + config["reconstruction_weight"] * reconstruction_loss

    @eqx.filter_jit
    def update(m, state, windows):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, windows)
        changes, state = optimizer.update(grads, state, m)
        return eqx.apply_updates(m, changes), state, loss

    rng = np.random.default_rng(81000 + seed)
    # Only short windows within the first train_steps frames are exposed here.
    val_rng = np.random.default_rng(91000)
    val_windows = sample_windows(validation[:, :config["train_steps"]], val_rng,
                                 config["batch_size"] * 2, history + horizon)
    start = time.perf_counter()
    curve = []
    best_value, best_update = float("inf"), 0
    best_model = model
    for n in range(1, config["updates"] + 1):
        windows = sample_windows(images, rng, config["batch_size"], history + horizon)
        model, state, loss = update(model, state, windows)
        if n == 1 or n % config["validate_every"] == 0 or n == config["updates"]:
            train_value = float(loss)
            val_value = float(loss_fn(model, val_windows))
            if not np.isfinite(train_value) or not np.isfinite(val_value):
                raise FloatingPointError(f"Nonfinite loss for {config['kind']} seed {seed} at {n}")
            if val_value < best_value:
                best_value, best_update, best_model = val_value, n, model
            row = {"stage": "dynamics", "kind": config["kind"], "seed": seed,
                   "update": n, "training_loss": train_value, "validation_loss": val_value,
                   "seconds": time.perf_counter() - start}
            curve.append(row)
            print(json.dumps(row), flush=True)
    stem = output / f"{config['kind']}_seed{seed}"
    eqx.tree_serialise_leaves(stem.with_suffix(".eqx"), best_model)
    return best_model, curve, {"best_validation_loss": best_value,
        "selected_update": best_update, "training_seconds": time.perf_counter() - start}


def position_metrics(predicted_q, truth_q, q_scale, dt, threshold=.3):
    """Normalize by one fixed training-derived displacement scale per mass."""
    pred = np.asarray(predicted_q, dtype=np.float64)
    truth = np.asarray(truth_q, dtype=np.float64)
    error = (pred - truth) / q_scale
    per_step = np.sqrt(np.mean(error ** 2, axis=-1))
    bad = (~np.isfinite(per_step)) | (per_step > threshold)
    # Consecutive valid forecasts before first failure; no recovery credit.
    valid = np.where(np.any(bad, axis=1), np.argmax(bad, axis=1), per_step.shape[1])
    pred_v = np.diff(pred, axis=1) / dt
    true_v = np.diff(truth, axis=1) / dt
    result = {"position_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
              "position_nrmse": float(np.sqrt(np.mean(error ** 2))),
              "velocity_fd_rmse": float(np.sqrt(np.mean((pred_v - true_v) ** 2))),
              "valid_horizon_mean": float(np.mean(valid)),
              "valid_horizon_median": float(np.median(valid)),
              "valid_horizon_threshold": threshold,
              "finite_fraction": float(np.mean(np.isfinite(pred)))}
    for length in [8, 20, 50, 100, 197]:
        if length <= truth.shape[1]:
            result[f"position_nrmse_{length}"] = float(np.sqrt(np.mean(error[:, :length] ** 2)))
    return result, {"position_nrmse_trace": np.sqrt(np.mean(error ** 2, axis=(0, 2))),
                    "per_trajectory_valid_horizon": valid}


def localization_metrics(images):
    """Detect blurred/background predictions independently of their centroid."""
    strips = np.maximum(np.asarray(images, dtype=np.float64), 0).reshape(
        images.shape[:-2] + (24, 4, 12))
    rows = strips.sum(axis=-1)
    mass = rows.sum(axis=-2)
    y = np.arange(24).reshape((1,) * (rows.ndim - 2) + (24, 1))
    center = np.sum(rows * y, axis=-2) / np.maximum(mass, 1e-12)
    variance = np.sum(rows * (y - center[..., None, :]) ** 2, axis=-2) / np.maximum(mass, 1e-12)
    spread = np.sqrt(variance)
    peak = strips.max(axis=(-3, -1))
    expected_mass = 2 * np.pi * 1.0 * .8
    localized = ((spread <= 1.8) & (mass >= .4 * expected_mass) &
                 (mass <= 2.5 * expected_mass) & (peak >= .15))
    return {"localized_object_fraction": float(np.mean(localized)),
            "mean_vertical_spread_pixels": float(np.mean(spread)),
            "mean_foreground_mass_ratio": float(np.mean(mass / expected_mass))}, localized


def evaluate(model, images, states, config, q_scale):
    history, steps = config["history"], images.shape[1] - config["history"]
    histories = jnp.asarray(images[:, :history])
    run = eqx.filter_jit(lambda m, hs: jax.vmap(lambda h: m.rollout(h, steps))(hs))
    latents, predictions = run(model, histories)
    predictions.block_until_ready()
    t0 = time.perf_counter()
    latents, predictions = run(model, histories)
    predictions.block_until_ready()
    runtime = time.perf_counter() - t0
    pred, zs = np.asarray(predictions), np.asarray(latents)
    predicted_q = image_positions(pred)
    result, traces = position_metrics(predicted_q, states[:, history:, :4], q_scale, config["dt"])
    local_metrics, localized = localization_metrics(pred)
    absolute_error = PIXELS_PER_POSITION * np.sqrt(np.mean((predicted_q - states[:, history:, :4])**2, axis=-1))
    # A complementary readable threshold; still require sharply localized spots.
    bad = (absolute_error > 1.0) | (~np.all(localized, axis=-1)) | (~np.isfinite(absolute_error))
    valid = np.where(np.any(bad, axis=1), np.argmax(bad, axis=1), steps)
    result.update(local_metrics)
    result.update({"localized_1px_valid_horizon_mean": float(np.mean(valid)),
                   "position_rmse_pixels": float(PIXELS_PER_POSITION * result["position_rmse"]),
                   "nrmse_threshold_pixels_per_mass": (q_scale * .3 * PIXELS_PER_POSITION).tolist()})
    z0 = np.asarray(jax.vmap(model.encode)(histories))
    r0 = np.linalg.norm(z0, axis=-1)
    dr = np.abs(np.linalg.norm(zs, axis=-1) - r0[:, None]) / np.maximum(r0[:, None], 1e-8)
    initial_prediction = np.asarray(jax.vmap(model.decode)(jnp.asarray(z0)))
    initial_q = image_positions(initial_prediction)
    result.update({"pixel_mse": float(np.mean((pred - images[:, history:]) ** 2)),
                   "balanced_pixel_loss": float(pixel_loss(predictions, jnp.asarray(images[:, history:]))),
                   "initial_reconstruction_position_rmse": float(np.sqrt(np.mean((initial_q - states[:, history - 1, :4]) ** 2))),
                   "mean_final_relative_latent_norm_drift": float(np.mean(dr[:, -1])),
                   "max_relative_latent_norm_drift": float(np.max(dr)),
                   "rollout_ms_per_trajectory": runtime * 1000 / len(images)})
    traces.update({"predicted_q": predicted_q, "truth_q": states[:, history:, :4],
                   "mean_relative_latent_norm_drift_trace": np.mean(dr, axis=0),
                   "example_prediction_images": pred[:3],
                   "example_truth_images": images[:3, history:]})
    return result, traces


def simple_controls(images, states, config, q_scale):
    h = config["history"]
    observed = image_positions(images[:, :h])
    steps = images.shape[1] - h
    last = np.broadcast_to(observed[:, -1:, :], (len(images), steps, 4))
    velocity = observed[:, -1:, :] - observed[:, -2:-1, :]
    extrapolated = last + velocity * np.arange(1, steps + 1)[None, :, None]
    return {kind: position_metrics(q, states[:, h:, :4], q_scale, config["dt"])[0]
            for kind, q in [("persistence", last), ("constant_velocity", extrapolated)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data/visual_physics/pilot_v1.npz")
    p.add_argument("--kinds", nargs="+", default=["residual", "prenorm", "scale", "projected", "geodesic", "rotation", "hamiltonian", "linear_orthogonal"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--updates", type=int, default=2500)
    p.add_argument("--warmup", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--eval-count", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset(args.data)
    with np.load(args.data) as d:
        arrays = {k: d[k] for k in d.files if k.endswith(("_images", "_states"))}
        data_metadata = str(d["metadata_json"])
    config = {k: v for k, v in vars(args).items() if k not in ("output", "data", "kinds", "seeds")}
    config.update({"image_shape": list(arrays["train_images"].shape[-2:]), "dt": .08,
                   "reconstruction_weight": .5, "train_steps": arrays["train_images"].shape[1],
                   "data_sha256": sha256(args.data), "selection": "minimum short-horizon ID validation pixel loss"})
    manifest = args.output / "manifest.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior["config"] != config:
            raise ValueError("Run configuration changed; use a fresh output directory.")
        for filename, value in prior["source_sha256"].items():
            if sha256(ROOT / filename) != value:
                raise ValueError(f"Source changed ({filename}); use a fresh output directory.")
    else:
        files = ["experiments/stage8_visual_world_model.py", "experiments/visual_physics_data.py",
                 "experiments/visual_physics_models.py", "skipnorm/jax_layers.py", "skipnorm/rotation_coupling.py"]
        source = args.output / "source_snapshot"
        for filename in files:
            destination = source / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / filename, destination)
        manifest.write_text(json.dumps({"config": config, "planned_kinds": args.kinds,
            "planned_seeds": args.seeds, "source_sha256": {f: sha256(ROOT / f) for f in files},
            "data_metadata": json.loads(data_metadata), "python": platform.python_version(),
            "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}, indent=2))
    q_scale = np.std(arrays["train_states"][..., :4], axis=(0, 1))
    controls = {}
    for split in ["test", "ood"]:
        n = args.eval_count
        controls[split] = simple_controls(arrays[f"{split}_images"][:n], arrays[f"{split}_states"][:n], config, q_scale)
    (args.output / "controls.json").write_text(json.dumps(controls, indent=2))
    for seed in args.seeds:
        key = jax.random.PRNGKey(3200 + seed)
        common = dict(latent_dim=args.latent_dim, width=args.width,
                      image_shape=tuple(config["image_shape"]), history=args.history, dt=.08, key=key)
        warm = warmstart(VisualWorldModel("residual", **common), arrays["train_images"], config, seed, args.output)
        for kind in args.kinds:
            stem = args.output / f"{kind}_seed{seed}"
            if stem.with_suffix(".json").exists():
                print(json.dumps({"skip_existing": str(stem)}), flush=True)
                continue
            model = VisualWorldModel(kind, **common)
            model = eqx.tree_at(lambda m: (m.encoder, m.decoder), model, (warm.encoder, warm.decoder))
            run_config = {**config, "kind": kind}
            model, curve, training = train(model, arrays["train_images"], arrays["val_images"], run_config, seed, args.output)
            result = {"kind": kind, "seed": seed, "config": run_config,
                      "parameters": parameter_counts(model), "training": training, "curve": curve}
            saved = {}
            for split in ["test", "ood"]:
                result[split], traces = evaluate(model, arrays[f"{split}_images"][:args.eval_count],
                    arrays[f"{split}_states"][:args.eval_count], run_config, q_scale)
                saved.update({f"{split}_{k}": v for k, v in traces.items()})
            np.savez_compressed(stem.with_suffix(".npz"), **saved)
            stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"completed": kind, "seed": seed, "test": result["test"], "ood": result["ood"]}), flush=True)
    summary = [json.loads(p.read_text()) for p in args.output.glob("*_seed*.json") if not p.name.startswith("vision_")]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
