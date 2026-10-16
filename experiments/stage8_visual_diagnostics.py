"""Exploratory, post-training motion-information checks; no model selection.

Paired scenes end at the same image, with opposite motion histories. The reverse
movie is a valid solution of this time-reversible conservative simulator. This
tests whether the model uses observed motion rather than only the last frame.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.visual_physics_data import image_positions, energy, OMEGA_SQUARED, QUARTIC, COUPLING
from experiments.visual_physics_models import VisualWorldModel


def diagnose(run_dir, data_path, output, count=16):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    c = manifest['config']
    with np.load(data_path) as data:
        images = data['test_images'][:count]
        states = data['test_states'][:count]
        q_scale = np.std(data['train_states'][..., :4], axis=(0, 1))
    history, horizon, center = c['history'], 8, 100
    forward = images[:, center-history+1:center+1]
    reverse = images[:, center:center+history][:, ::-1].copy()
    # Both histories terminate at exactly the same image.
    np.testing.assert_array_equal(forward[:, -1], reverse[:, -1])
    truth_f = states[:, center+1:center+horizon+1, :4]
    truth_r = states[:, center-horizon:center, :4][:, ::-1].copy()
    repeated = np.repeat(forward[:, -1:], history, axis=1)
    results = []
    for path in sorted(run_dir.glob('*_seed*.json')):
        if path.name.startswith('vision_'):
            continue
        r = json.loads(path.read_text())
        model = VisualWorldModel(r['kind'], latent_dim=c['latent_dim'], width=c['width'],
            image_shape=tuple(c['image_shape']), history=history, dt=c['dt'],
            key=jax.random.PRNGKey(3200+r['seed']))
        model = eqx.tree_deserialise_leaves(path.with_suffix('.eqx'), model)
        contexts = jnp.asarray(np.concatenate([forward, reverse, repeated], axis=0))
        run = eqx.filter_jit(lambda m, hs: jax.vmap(lambda h: m.rollout(h, horizon))(hs))
        _, predicted = run(model, contexts)
        qf, qr, qstatic = np.split(image_positions(np.asarray(predicted)), 3, axis=0)
        def error(a, b):
            return float(np.sqrt(np.mean(((a-b)/q_scale)**2)))
        truth_separation = truth_f - truth_r
        results.append({'kind': r['kind'], 'seed': r['seed'],
            'forward_nrmse_8': error(qf, truth_f),
            'reverse_nrmse_8': error(qr, truth_r),
            'repeated_last_frame_nrmse_8': error(qstatic, truth_f),
            'paired_future_difference_nrmse': error(qf-qr, truth_separation),
            'true_future_difference_rms': float(np.sqrt(np.mean(truth_separation**2))),
            'predicted_future_difference_rms': float(np.sqrt(np.mean((qf-qr)**2))),
            'ignoring_history_difference_nrmse': error(np.zeros_like(truth_separation), truth_separation)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'posthoc exploratory diagnostic, not model selection',
        'count': count, 'center_frame': center, 'forecast_frames': horizon,
        'description': 'Same current image, forward versus reversed valid motion histories; no retraining.',
        'results': results}, indent=2))
    for row in results:
        print(json.dumps(row))


def diagnose_energy(run_dir, data_path, output):
    """Physical energy from decoded positions and centered differences.

    This is an evaluation-only readout using known simulator parameters. It is
    not supervision and does not identify the model's latent norm with energy.
    """
    c = json.loads((run_dir/'manifest.json').read_text())['config']
    dt, h, count = c['dt'], c['history'], c['eval_count']
    with np.load(data_path) as d:
        states = {split: d[f'{split}_states'][:count] for split in ['test', 'ood']}
    def from_positions(q):
        p = (q[:, 2:]-q[:, :-2])/(2*dt)
        return np.concatenate([q[:, 1:-1], p], axis=-1)
    def components(x):
        q, p = x[..., :4], x[..., 4:]
        e = .5*p**2 + .5*OMEGA_SQUARED*q**2 + .25*QUARTIC*q**4
        shared = .25*COUPLING*np.diff(q, axis=-1)**2
        e[..., :-1] += shared
        e[..., 1:] += shared
        return e
    calibration = {}
    for split, x in states.items():
        truth = x[:, h+1:-1]
        reconstructed = from_positions(x[:, h:, :4])
        calibration[split] = float(np.mean(np.abs(energy(reconstructed)-energy(truth))/energy(truth)))
    rows = []
    for path in sorted(run_dir.glob('*_seed*.json')):
        if path.name.startswith('vision_'):
            continue
        r = json.loads(path.read_text())
        row = {'kind': r['kind'], 'seed': r['seed']}
        with np.load(path.with_suffix('.npz')) as pred:
            for split, x in states.items():
                recovered = from_positions(pred[f'{split}_predicted_q'])
                truth = x[:, h+1:-1]
                true_energy = energy(truth)
                e = energy(recovered)
                row[split] = {
                    'relative_physical_energy_mae': float(np.mean(np.abs(e-true_energy)/true_energy)),
                    'component_energy_mae_fraction_total': float(np.mean(np.sum(np.abs(
                        components(recovered)-components(truth)), axis=-1)/true_energy)),
                    'mean_physical_energy_ratio': float(np.mean(e/true_energy)),
                    'relative_latent_norm_drift': r[split]['mean_final_relative_latent_norm_drift']}
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'posthoc evaluation-only diagnostic',
        'readout': 'Image-centroid positions, centered finite-difference velocities; interior195forecast frames',
        'finite_difference_true_positions_relative_energy_mae': calibration,
        'results': rows}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT/'data/visual_physics/pilot_v1.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--energy-output', type=Path)
    a = p.parse_args()
    diagnose(a.run_dir, a.data, a.output, a.count)
    if a.energy_output:
        diagnose_energy(a.run_dir, a.data, a.energy_output)
