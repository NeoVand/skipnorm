"""Audit locked splits and verify DNLS reference convergence without modifying data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def power_drift(x: np.ndarray) -> float:
    powers = np.sum(x * x, axis=-1)
    return float(np.max(np.abs(powers - powers[:, :1]) / powers[:, :1]))


def dnls_reference(initial: np.ndarray, substeps: int, n_states: int = 250) -> np.ndarray:
    """Float64 RK4 with no normalization; the whole initial-condition batch is vectorized."""
    def rhs(x):
        q, p = x[..., :16], x[..., 16:]
        amp = q*q + p*p
        lap_q = np.roll(q, -1, axis=-1) - 2*q + np.roll(q, 1, axis=-1)
        lap_p = np.roll(p, -1, axis=-1) - 2*p + np.roll(p, 1, axis=-1)
        return np.concatenate((-lap_p-amp*p, lap_q+amp*q), axis=-1)
    dt = .04 / substeps
    x = initial.copy()
    states = [x.copy()]
    for _ in range(n_states-1):
        for _ in range(substeps):
            k1 = rhs(x)
            k2 = rhs(x + .5*dt*k1)
            k3 = rhs(x + .5*dt*k2)
            k4 = rhs(x + dt*k3)
            x = x + dt*(k1+2*k2+2*k3+k4)/6
        states.append(x.copy())
    return np.stack(states, axis=1)


def main():
    manifest = json.loads((ROOT / "data/splits/splits_manifest.json").read_text())
    datasets = {}
    for name, entry in manifest.items():
        path = ROOT / entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"Hash mismatch: {name}"
        with np.load(path) as data:
            tr = data["train_orbits"]
            va = data["val_orbits"]
            te = data["test_orbits_long" if name == "dnls_1d" else "test_orbits"]
            arrays = {"train": tr, "validation": va, "test": te}
            pairs = {}
            for a,b in [("train","validation"),("train","test"),("validation","test")]:
                delta = arrays[a][:,None,0,:] - arrays[b][None,:,0,:]
                ndup = int(np.sum(np.all(delta == 0, axis=-1)))
                assert ndup == 0, f"Repeated initial state across {a}/{b}: {name}"
                pairs[f"{a}_{b}"] = {"exact_initial_state_duplicates": ndup,
                                      "minimum_initial_state_distance": float(np.min(np.linalg.norm(delta, axis=-1)))}
            datasets[name] = {"sha256": digest, "shapes": {k: list(v.shape) for k,v in arrays.items()},
                              "meta": entry["meta"], "split_checks": pairs}
            label = "norms" if name == "dnls_1d" else "invariants"
            labels = {s: data[f"{s}_{label}"] for s in ["train","val","test"]}
            a = np.atleast_2d(labels["train"]).reshape(len(tr),-1)
            b = np.atleast_2d(labels["test"]).reshape(len(te),-1)
            datasets[name]["exact_train_test_invariant_duplicates"] = int(np.sum(np.all(a[:,None,:] == b[None,:,:], axis=-1)))
            if "train_wells" in data:
                same_energy = np.all(a[:, None, :] == b[None, :, :], axis=-1)
                same_well = data["train_wells"][:, None] == data["test_wells"][None, :]
                repeated_component = int(np.sum(same_energy & same_well))
                assert repeated_component == 0, f"Repeated energy/well component: {name}"
                datasets[name]["exact_train_test_energy_well_duplicates"] = repeated_component
            datasets[name]["train_invariant_range"] = [a.min(axis=0).tolist(), a.max(axis=0).tolist()]
            datasets[name]["test_invariant_range"] = [b.min(axis=0).tolist(), b.max(axis=0).tolist()]
    with np.load(ROOT / "data/splits/dnls_1d.npz") as data:
        initial = np.concatenate([data["train_orbits"][:,0], data["val_orbits"][:,0], data["test_orbits_long"][:,0]])
        coarse = dnls_reference(initial, 10)
        fine = dnls_reference(initial, 20)
        finest = dnls_reference(initial, 40)
        saved_difference = max(float(np.max(np.abs(coarse[:28,:60]-data["train_orbits"]))),
                               float(np.max(np.abs(coarse[28:36,:60]-data["val_orbits"]))),
                               float(np.max(np.abs(coarse[36:]-data["test_orbits_long"]))))
        assert saved_difference < 1e-12, "Locked reference differs from recorded generator"
        error_10_20 = float(np.max(np.abs(coarse-fine)))
        error_20_40 = float(np.max(np.abs(fine-finest)))
        reference = {"trajectories": len(initial), "states": 250, "steps": 249, "dt": .04,
                     "physical_horizon": 249*.04, "substeps": [10,20,40], "post_step_projection": False,
                     "max_absolute_state_difference_10_vs_20": error_10_20,
                     "max_absolute_state_difference_20_vs_40": error_20_40,
                     "observed_refinement_ratio": error_10_20/error_20_40,
                     "state_mse_10_vs_40": float(np.mean((coarse-finest)**2)),
                     "max_absolute_difference_from_locked_arrays": saved_difference,
                     "max_relative_power_drift_10": power_drift(coarse),
                     "max_relative_power_drift_20": power_drift(fine),
                     "max_relative_power_drift_40": power_drift(finest)}
    path = ROOT / "results/submission_validation.json"
    path.write_text(json.dumps({"datasets": datasets, "dnls_reference": reference,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")
    print(json.dumps(reference, indent=2))
    print(f"Validated {len(datasets)} locked datasets; wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
