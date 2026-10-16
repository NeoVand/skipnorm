"""Prospective DNLS screen of reversible rotations with controlled Jacobians.

Independent of the manuscript's Stage 1--4 outputs. Final checkpoints are used;
validation curves are recorded but do not select models or tune the protocol.
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
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

from experiments.stage3_dnls_benchmark import DNLSPropagator
from skipnorm.jax_layers import cayley_orthogonal_matrix
from skipnorm.rotation_coupling import RotationCouplingFlow


MODES = {
    "coupling_bounded": "Normalized coupling, bounded",
    "coupling_unbounded": "Normalized coupling, unconstrained",
    "coupling_raw": "Raw conditioner coupling",
    "geodesic_skipnorm": "Geodesic SkipNorm",
    "projected_residual": "Projected residual",
    "linear_orthogonal": "Linear orthogonal",
}
BASELINES = {"geodesic_skipnorm", "projected_residual", "linear_orthogonal"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CouplingPropagator(eqx.Module):
    flow: RotationCouplingFlow
    linear_raw: jax.Array
    bounded: bool = eqx.field(static=True)

    def __init__(self, mode, *, key, budget=.5):
        # Same projected parameter initialization for all three coupling variants.
        self.flow = RotationCouplingFlow(dim=32, num_stages=8, width=32, depth=2,
            budget=budget, delta=1., normalize=mode != "coupling_raw", project=True, key=key)
        self.linear_raw = jnp.zeros((32, 32), dtype=jnp.float64)
        self.bounded = mode == "coupling_bounded"

    def step(self, x):
        q = cayley_orthogonal_matrix(self.linear_raw, alpha=1.)
        return self.flow(q @ x)

    def inverse(self, y):
        q = cayley_orthogonal_matrix(self.linear_raw, alpha=1.)
        return q.T @ self.flow.inverse(y)

    def project_lipschitz(self):
        if self.bounded:
            return eqx.tree_at(lambda m: m.flow, self, self.flow.project_lipschitz())
        return self


def make_model(mode, seed, budget):
    key = jax.random.PRNGKey(7000 + 89 * seed)
    if mode in BASELINES:
        return DNLSPropagator(dim=32, mode=mode, width=96, depth=2, key=key)
    return CouplingPropagator(mode, key=key, budget=budget)


def forecast(model, initial, steps):
    def step(x, _):
        y = jax.vmap(model.step)(x)
        return y, y
    _, ys = jax.lax.scan(step, initial, None, length=steps)
    return jnp.swapaxes(ys, 0, 1)


def train(model, train_orbits, val_orbits, config, seed):
    updates, horizon = config["updates"], config["horizon"]
    schedule = optax.cosine_decay_schedule(config["learning_rate"], updates, alpha=.05)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(m, windows):
        pred = forecast(m, windows[:, 0], horizon)
        return jnp.mean(jnp.square(pred - windows[:, 1:]))

    @eqx.filter_jit
    def update(m, state, windows):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, windows)
        change, state = optimizer.update(grads, state, m)
        m = eqx.apply_updates(m, change)
        if isinstance(m, CouplingPropagator):
            m = m.project_lipschitz()
        return m, state, loss

    tr, va = np.asarray(train_orbits), np.asarray(val_orbits)
    # Follow the locked Stage 3 window sampling exactly for paired comparison.
    rng = np.random.default_rng(40000 + seed)
    max_start = tr.shape[1] - horizon - 1
    val_starts = list(range(0, va.shape[1] - horizon, horizon))
    val_windows = jnp.asarray(np.stack([orbit[t:t+horizon+1] for orbit in va for t in val_starts]))
    curve = [{"update": 0, "validation_mse": float(loss_fn(model, val_windows))}]
    started = time.perf_counter()
    for n in range(1, updates + 1):
        mi = rng.integers(0, len(tr), size=64)
        ti = rng.integers(0, max_start, size=64)
        windows = jnp.asarray(np.stack([tr[m, t:t+horizon+1] for m, t in zip(mi, ti)]))
        model, opt_state, loss = update(model, opt_state, windows)
        if n == 1 or n % 100 == 0 or n == updates:
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite training loss at update {n}")
            item = {"update": n, "training_mse": value,
                    "validation_mse": float(loss_fn(model, val_windows)),
                    "elapsed_seconds": time.perf_counter() - started}
            curve.append(item)
            print(json.dumps(item), flush=True)
    return model, curve, time.perf_counter() - started


def singular_summary(jacobians):
    js = np.asarray(jacobians)
    finite = np.all(np.isfinite(js), axis=(-2, -1))
    if not np.any(finite):
        return {"finite_fraction": 0.}
    sv = np.linalg.svd(js[finite], compute_uv=False)
    signs, logs = np.linalg.slogdet(js[finite])
    return {"finite_fraction": float(np.mean(finite)),
            "minimum_singular_value": float(sv[:, -1].min()),
            "maximum_singular_value": float(sv[:, 0].max()),
            "maximum_condition_number": float(np.max(sv[:, 0] / np.maximum(sv[:, -1], 1e-300))),
            "minimum_log_abs_det": float(logs.min()), "maximum_log_abs_det": float(logs.max()),
            "minimum_determinant_sign": float(signs.min())}


def evaluate(model, test, *, timing_repeats=10):
    test_j = jnp.asarray(test)
    run_forecast = eqx.filter_jit(lambda m, x: forecast(m, x, test.shape[1] - 1))
    prediction = run_forecast(model, test_j[:, 0]).block_until_ready()
    times = []
    for _ in range(timing_repeats):
        t0 = time.perf_counter()
        prediction = run_forecast(model, test_j[:, 0]).block_until_ready()
        times.append(1000 * (time.perf_counter() - t0))
    pred = np.asarray(prediction)
    mse = np.mean((pred - test[:, 1:])**2, axis=(0, 2))
    initial_power = np.sum(test[:, 0]**2, axis=-1)
    powers = np.sum(pred**2, axis=-1)
    drift = np.abs(powers - initial_power[:, None]) / initial_power[:, None]
    result = {f"mse_{t}": float(mse[:t].mean()) for t in [1, 20, 50, 200, 249]}
    result.update({"max_relative_power_drift": float(drift.max()),
        "mean_final_relative_power_drift": float(drift[:, -1].mean()),
        "rollout_ms": float(np.median(times)), "timing_samples_ms": times,
        "n_parameters": sum(int(a.size) for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))})

    # Uniform physical phases are not assumed; these are explicit sampled states.
    points = jnp.asarray(np.concatenate([test[::3, 0], test[::3, 50], test[::3, 150]]))
    jac = eqx.filter_jit(lambda m, xs: jax.vmap(jax.jacfwd(m.step))(xs))(model, points)
    result["one_step_jacobian"] = singular_summary(jac)
    result["origin_jacobian"] = singular_summary(
        eqx.filter_jit(lambda m: jax.jacfwd(m.step)(jnp.zeros(32)))(model)[None])
    depth = {}
    for n in [1, 10, 50, 200]:
        def terminal(m, x):
            return jax.lax.fori_loop(0, n, lambda _, z: m.step(z), x)
        js = eqx.filter_jit(lambda m, xs: jax.vmap(jax.jacfwd(lambda x: terminal(m, x)))(xs))(model, points[:2])
        depth[str(n)] = singular_summary(js)
    result["repeated_time_step_jacobian"] = depth

    if isinstance(model, CouplingPropagator):
        inv = eqx.filter_jit(lambda m, ys: jax.vmap(m.inverse)(ys))
        back = inv(model, jax.vmap(model.step)(points))
        result["one_step_inverse_relative_error"] = float(jnp.max(jnp.linalg.norm(back - points, axis=-1) / jnp.linalg.norm(points, axis=-1)))
        def inverse_rollout(m, ys):
            return jax.lax.fori_loop(0, test.shape[1]-1, lambda _, z: jax.vmap(m.inverse)(z), ys)
        recovered = eqx.filter_jit(inverse_rollout)(model, prediction[:, -1])
        result["full_rollout_inverse_relative_error"] = float(jnp.max(jnp.linalg.norm(recovered-test_j[:, 0], axis=-1)/jnp.linalg.norm(test_j[:, 0], axis=-1)))
        result["certified_log_bound_one_step"] = model.flow.certified_log_bound()
        # Stress the global bound independently of the physical test distribution.
        directions = np.random.default_rng(935).normal(size=(4, 32))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        stress = jnp.asarray(np.concatenate([directions * s for s in [1e-9, 1., 1e6]]))
        sj = eqx.filter_jit(lambda m, xs: jax.vmap(jax.jacfwd(m.step))(xs))(model, stress)
        result["scale_stress_jacobian"] = singular_summary(sj)
    return result, {"prediction": pred, "mse_trace": mse,
                    "mean_relative_power_drift_trace": drift.mean(axis=0)}


def load_baseline(mode, seed, model, config):
    stem = ROOT / "results/stage3/runs_v2" / f"{mode}_seed{seed}"
    old = json.loads(stem.with_suffix(".json").read_text())
    expected = {"updates": config["updates"], "horizon": config["horizon"],
        "learning_rate": config["learning_rate"], "batch_size": 64,
        "split_sha256": config["split_sha256"]}
    for k, value in expected.items():
        if old["config"][k] != value:
            raise ValueError(f"Existing baseline differs on {k}")
    for p, value in old["config"]["code_sha256"].items():
        if digest(ROOT / p) != value:
            raise ValueError(f"Existing baseline source differs: {p}")
    return eqx.tree_deserialise_leaves(stem.with_suffix(".eqx"), model), str(stem.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--updates", type=int, default=900)
    parser.add_argument("--budget", type=float, default=.5)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage5_rotation_coupling/pilot_v1")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_paths = ["experiments/stage5_rotation_coupling.py", "skipnorm/rotation_coupling.py",
                    "experiments/stage3_dnls_benchmark.py", "skipnorm/jax_layers.py"]
    config = {"updates": args.updates, "horizon": 8, "batch_size": 64,
        "learning_rate": .003, "optimizer": "Adam", "cosine_final_fraction": .05,
        "model_selection": "final update; validation recorded only",
        "coupling": {"stages": 8, "width": 32, "depth": 2, "budget": args.budget,
                     "delta": 1., "linear_mixing": "learned Cayley, identity initialization",
                     "initialization": "same projected weights in all coupling variants"},
        "precision": "float64", "split_sha256": digest(ROOT / "data/splits/dnls_1d.npz"),
        "source_sha256": {p: digest(ROOT / p) for p in source_paths}}
    protocol = out / "config.json"
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise RuntimeError("Configuration/source changed: preserve existing run and choose a new output directory")
    protocol.write_text(json.dumps(config, indent=2) + "\n")
    sources = out / "source_snapshot"
    for p in source_paths:
        dest = sources / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, dest)
    split = np.load(ROOT / "data/splits/dnls_1d.npz")
    for mode in args.modes:
        for seed in args.seeds:
            stem = out / f"{mode}_seed{seed}"
            record_file = stem.with_suffix(".json")
            if record_file.exists() and not args.timing_only:
                print(f"Resume: {mode} seed {seed} already complete", flush=True)
                continue
            model = make_model(mode, seed, args.budget)
            previous = json.loads(record_file.read_text()) if record_file.exists() else {}
            if stem.with_suffix(".eqx").exists():
                model = eqx.tree_deserialise_leaves(stem.with_suffix(".eqx"), model)
                curve, train_seconds = previous.get("training_curve", []), previous.get("training_seconds")
                origin = previous.get("baseline_checkpoint")
            elif mode in BASELINES and args.updates == 900:
                model, origin = load_baseline(mode, seed, model, config)
                curve, train_seconds = [], None
                eqx.tree_serialise_leaves(stem.with_suffix(".eqx"), model)
            else:
                if args.timing_only:
                    raise RuntimeError(f"Missing checkpoint: {stem}")
                print(f"Training {mode}, seed {seed}", flush=True)
                model, curve, train_seconds = train(model, split["train_orbits"], split["val_orbits"], config, seed)
                origin = None
                eqx.tree_serialise_leaves(stem.with_suffix(".eqx"), model)
            result, arrays = evaluate(model, split["test_orbits_long"])
            result.update({"mode": mode, "label": MODES[mode], "seed": seed,
                           "training_curve": curve, "training_seconds": train_seconds,
                           "baseline_checkpoint": origin, "config_sha256": digest(protocol)})
            np.savez_compressed(stem.with_suffix(".npz"), **arrays)
            record_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps({k: result[k] for k in ["mode", "seed", "mse_20", "mse_200", "rollout_ms"]}), flush=True)
    records = [json.loads(p.read_text()) for p in sorted(out.glob("*_seed*.json"))]
    summary = {}
    metrics = ["mse_1", "mse_20", "mse_50", "mse_200", "mse_249", "max_relative_power_drift", "rollout_ms"]
    for mode in MODES:
        runs = [r for r in records if r["mode"] == mode]
        if runs:
            summary[mode] = {"label": MODES[mode], "seeds": [r["seed"] for r in runs],
                "n_parameters": runs[0]["n_parameters"], **{m: {"mean": float(np.mean([r[m] for r in runs])),
                "sample_sd": float(np.std([r[m] for r in runs], ddof=1)) if len(runs)>1 else None} for m in metrics}}
    hardware = {"platform": platform.platform(), "processor": platform.processor(),
        "jax": jax.__version__, "equinox": eqx.__version__, "devices": [str(x) for x in jax.devices()]}
    (out / "summary.json").write_text(json.dumps({"config": config, "hardware": hardware,
        "status": "exploratory three-seed screen; fixed existing split; no new-dataset evidence",
        "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
