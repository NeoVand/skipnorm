"""Image-only observations of a conservative nonlinear four-mass chain.

States are ordered ``[q_0, ..., q_3, p_0, ..., p_3]``; all masses are one.
The image is an observation of positions only. Velocities and all simulator
metadata are evaluation targets and must not be supplied to visual learners.
``steps`` always means the number of saved observations, including time zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OMEGA_SQUARED = np.array([1.0, 1.15, 0.85, 1.30], dtype=np.float64)
QUARTIC = 2.0
COUPLING = 0.35
OBSERVATION_DT = 0.08
SUBSTEPS = 16
IMAGE_HEIGHT = 24
IMAGE_WIDTH = 48
COLUMN_CENTERS = np.array([5.5, 17.5, 29.5, 41.5])
PIXELS_PER_POSITION = 4.5
SPOT_SIGMA_X = 1.0
SPOT_SIGMA_Y = 0.8
TRAIN_AMPLITUDE = (0.25, 0.65)
OOD_AMPLITUDE = (0.8, 1.0)


def _validate_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.shape[-1:] != (8,):
        raise ValueError("States must have last dimension 8 (four q, four p).")
    if not np.all(np.isfinite(states)):
        raise ValueError("States must be finite.")
    return states


def energy(states: np.ndarray) -> np.ndarray:
    """Hamiltonian, with any leading batch/time dimensions preserved."""
    states = _validate_states(states)
    q, p = states[..., :4], states[..., 4:]
    return (
        0.5 * np.sum(p * p + OMEGA_SQUARED * q * q, axis=-1)
        + 0.25 * QUARTIC * np.sum(q**4, axis=-1)
        + 0.5 * COUPLING * np.sum(np.diff(q, axis=-1)**2, axis=-1)
    )


def force(q: np.ndarray) -> np.ndarray:
    """Negative position gradient of the Hamiltonian (free chain ends)."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1:] != (4,):
        raise ValueError("Positions must have last dimension 4.")
    result = -OMEGA_SQUARED * q - QUARTIC * q**3
    differences = np.diff(q, axis=-1)
    result[..., :-1] += COUPLING * differences
    result[..., 1:] -= COUPLING * differences
    return result


def generate_trajectories(
    n_trajectories: int,
    steps: int,
    seed: int | np.random.SeedSequence = 0,
    amplitude_range: tuple[float, float] = TRAIN_AMPLITUDE,
    dt: float = OBSERVATION_DT,
    substeps: int = SUBSTEPS,
    initial_states: np.ndarray | None = None,
) -> np.ndarray:
    """Return float64 trajectories ``(N, steps, 8)`` using velocity Verlet.

    Each coordinate has an independent amplitude and phase. An amplitude is
    the *initial phase-space radius* sqrt(q_i^2+p_i^2), not an assertion about
    its later maximum excursion. Its declared range defines the OOD split.
    """
    if n_trajectories < 1 or steps < 1 or substeps < 1 or dt <= 0:
        raise ValueError("n_trajectories, steps, substeps, and dt must be positive.")
    lo, hi = amplitude_range
    if not 0 < lo <= hi:
        raise ValueError("amplitude_range must satisfy 0 < low <= high.")
    if initial_states is None:
        rng = np.random.default_rng(seed)
        amplitude = rng.uniform(lo, hi, (n_trajectories, 4))
        phase = rng.uniform(-np.pi, np.pi, (n_trajectories, 4))
        initial_states = np.concatenate(
            (amplitude * np.sin(phase), amplitude * np.cos(phase)), axis=-1
        )
    initial_states = _validate_states(initial_states)
    if initial_states.shape != (n_trajectories, 8):
        raise ValueError("initial_states must have shape (n_trajectories, 8).")
    result = np.empty((n_trajectories, steps, 8), dtype=np.float64)
    q, p = initial_states[:, :4].copy(), initial_states[:, 4:].copy()
    result[:, 0] = initial_states
    h = dt / substeps
    for t in range(1, steps):
        for _ in range(substeps):
            p += 0.5 * h * force(q)
            q += h * p
            p += 0.5 * h * force(q)
        result[:, t, :4] = q
        result[:, t, 4:] = p
    return result


def rendering_diagnostics(states: np.ndarray) -> dict[str, float | int]:
    """Count off-image centers and clipping of each spot's four-sigma core."""
    states = _validate_states(states)
    y = (IMAGE_HEIGHT - 1) / 2 - PIXELS_PER_POSITION * states[..., :4]
    return {
        "position_min": float(np.min(states[..., :4])),
        "position_max": float(np.max(states[..., :4])),
        "center_row_min": float(np.min(y)),
        "center_row_max": float(np.max(y)),
        "out_of_frame_centers": int(np.count_nonzero((y < 0) | (y > IMAGE_HEIGHT - 1))),
        "clipped_four_sigma_cores": int(np.count_nonzero(
            (y - 4 * SPOT_SIGMA_Y < 0) | (y + 4 * SPOT_SIGMA_Y > IMAGE_HEIGHT - 1)
        )),
    }


def render_frames(states: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    """Render four unlabelled Gaussian masses; return ``(..., 24, 48)``.

    Rendering is deterministic with identical appearance in all splits.
    Independent appearance variation is deliberately left for a later study.
    Off-image centers are an error, rather than silently clipped examples.
    """
    states = _validate_states(states)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    diagnostics = rendering_diagnostics(states)
    if diagnostics["out_of_frame_centers"]:
        raise ValueError(f"Mass centers outside rendering field: {diagnostics}")
    flat = states.reshape(-1, 8)
    output = np.empty((len(flat), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    horizontal = np.exp(-0.5 * (
        (np.arange(IMAGE_WIDTH)[None, :] - COLUMN_CENTERS[:, None]) / SPOT_SIGMA_X
    )**2)
    rows = np.arange(IMAGE_HEIGHT)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        centers = (IMAGE_HEIGHT - 1) / 2 - PIXELS_PER_POSITION * flat[start:stop, :4]
        vertical = np.exp(-0.5 * ((rows[None, None, :] - centers[:, :, None]) / SPOT_SIGMA_Y)**2)
        output[start:stop] = np.einsum("bmh,mw->bhw", vertical, horizontal)
    np.clip(output, 0., 1., out=output)
    return output.reshape(states.shape[:-1] + (IMAGE_HEIGHT, IMAGE_WIDTH))


def image_positions(images: np.ndarray) -> np.ndarray:
    """Intensity-centroid position readout for evaluation, ``(..., 4)``.

    Four fixed, disjoint column strips are known to the evaluator. Nonpositive
    intensity is discarded. Empty or nonfinite strips yield NaN, so callers
    should explicitly count failures rather than drop them from error means.
    Predicted blur/multiple spots are not resolved: the centroid reflects that
    limitation. This helper is not an observation encoder for model training.
    """
    images = np.asarray(images)
    if images.shape[-2:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"Expected image trailing shape {(IMAGE_HEIGHT, IMAGE_WIDTH)}.")
    positive = np.maximum(images, 0).astype(np.float64)
    row_marginal = positive.reshape(images.shape[:-2] + (IMAGE_HEIGHT, 4, 12)).sum(axis=-1)
    total = row_marginal.sum(axis=-2)
    weighted = np.einsum("...hm,h->...m", row_marginal, np.arange(IMAGE_HEIGHT))
    center = np.full_like(total, np.nan)
    np.divide(weighted, total, out=center, where=np.isfinite(total) & (total > 1e-12))
    return ((IMAGE_HEIGHT - 1) / 2 - center) / PIXELS_PER_POSITION


def generate_dataset(
    path: str | Path,
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
    train_steps: int = 40,
    eval_steps: int = 200,
    seed: int = 0,
) -> dict:
    """Save independent splits and self-describing, non-pickle JSON metadata.

    The OOD split has ``n_test`` trajectories and the same duration as test.
    Returns metadata. Existing output files are not overwritten accidentally.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Dataset already exists: {path}")
    if path.suffix != ".npz":
        raise ValueError("Dataset path must end with .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    split_specs = [
        ("train", n_train, train_steps, TRAIN_AMPLITUDE),
        ("val", n_val, eval_steps, TRAIN_AMPLITUDE),
        ("test", n_test, eval_steps, TRAIN_AMPLITUDE),
        ("ood", n_test, eval_steps, OOD_AMPLITUDE),
    ]
    children = np.random.SeedSequence(seed).spawn(4)
    arrays = {}
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "state_order": ["q0", "q1", "q2", "q3", "p0", "p1", "p2", "p3"],
        "hamiltonian": "sum_i(p_i^2/2 + omega_squared_i*q_i^2/2 + quartic*q_i^4/4) + coupling/2*sum_i(q_(i+1)-q_i)^2",
        "mass": 1.0,
        "omega_squared": OMEGA_SQUARED.tolist(),
        "quartic": QUARTIC,
        "coupling": COUPLING,
        "observation_dt": OBSERVATION_DT,
        "verlet_substeps": SUBSTEPS,
        "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "column_centers": COLUMN_CENTERS.tolist(),
        "pixels_per_position": PIXELS_PER_POSITION,
        "spot_sigma_xy": [SPOT_SIGMA_X, SPOT_SIGMA_Y],
        "amplitude_definition": "Initial per-mass phase-space radius sqrt(q_i^2+p_i^2); independent uniform amplitudes and phases.",
        "training_inputs": "Images only. Physical states and metadata are evaluation-only.",
        "appearance": "Identical unlabelled Gaussian spots; no appearance OOD in this pilot.",
        "splits": {},
    }
    for (name, count, length, amplitudes), child in zip(split_specs, children):
        states = generate_trajectories(count, length, seed=child, amplitude_range=amplitudes)
        energies = energy(states)
        diagnostics = rendering_diagnostics(states)
        if diagnostics["clipped_four_sigma_cores"]:
            raise ValueError(f"Split {name} clips a four-sigma spot core: {diagnostics}")
        arrays[name + "_states"] = states.astype(np.float32)
        arrays[name + "_images"] = render_frames(states)
        metadata["splits"][name] = {
            "n_trajectories": count,
            "observations": length,
            "duration": (length - 1) * OBSERVATION_DT,
            "amplitude_range": list(amplitudes),
            "seed_spawn_key": list(child.spawn_key),
            "initial_energy_min": float(energies[:, 0].min()),
            "initial_energy_max": float(energies[:, 0].max()),
            "max_relative_energy_error": float(np.max(np.abs(energies - energies[:, :1]) / energies[:, :1])),
            **diagnostics,
        }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=40)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
