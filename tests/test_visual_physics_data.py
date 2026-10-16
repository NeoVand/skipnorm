"""Independent physical, rendering and split checks for the visual pilot."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.visual_physics_data import (
    OOD_AMPLITUDE, TRAIN_AMPLITUDE, energy, force, generate_dataset,
    generate_trajectories, image_positions, render_frames, rendering_diagnostics,
)


class TestVisualPhysicsData(unittest.TestCase):
    def test_force_is_negative_energy_gradient(self):
        states = np.random.default_rng(4).normal(size=(7, 8))
        step = 1e-5
        finite_difference = []
        for index in range(4):
            delta = np.zeros(8)
            delta[index] = step
            finite_difference.append(-(energy(states + delta) - energy(states - delta)) / (2 * step))
        np.testing.assert_allclose(force(states[:, :4]), np.stack(finite_difference, -1), rtol=1e-8, atol=1e-8)

    def test_verlet_energy_accuracy_and_convergence(self):
        reference = generate_trajectories(12, 300, seed=81, amplitude_range=OOD_AMPLITUDE, substeps=32)
        fine = generate_trajectories(12, 300, seed=81, amplitude_range=OOD_AMPLITUDE, substeps=16)
        coarse = generate_trajectories(12, 300, seed=81, amplitude_range=OOD_AMPLITUDE, substeps=8)
        energies = energy(fine)
        relative_drift = np.max(np.abs(energies / energies[:, :1] - 1))
        self.assertLess(relative_drift, 5e-5)
        self.assertLess(np.max(np.abs(fine - reference)), .3 * np.max(np.abs(coarse - reference)))
        self.assertEqual(rendering_diagnostics(fine)["clipped_four_sigma_cores"], 0)

    def test_rendering_position_readout(self):
        states = np.zeros((21, 8))
        states[:, :4] = np.linspace(-1.6, 1.6, 84).reshape(21, 4)
        images = render_frames(states, chunk_size=5)
        self.assertEqual(images.shape, (21, 24, 48))
        self.assertEqual(images.dtype, np.float32)
        self.assertGreaterEqual(images.min(), 0.)
        self.assertLessEqual(images.max(), 1.)
        np.testing.assert_allclose(image_positions(images), states[:, :4], atol=1e-5)
        self.assertTrue(np.isnan(image_positions(np.zeros((24, 48)))).all())
        states[:, :4] = 10.
        with self.assertRaises(ValueError):
            render_frames(states)

    def test_splits_are_independent_and_ood_is_declared(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "visual.npz"
            metadata = generate_dataset(path, n_train=5, n_val=3, n_test=4, train_steps=8, eval_steps=12, seed=9)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["train_images"].shape, (5, 8, 24, 48))
                self.assertEqual(data["test_images"].shape, (4, 12, 24, 48))
                self.assertEqual(data["ood_states"].shape, (4, 12, 8))
                initial = np.concatenate([data[name + "_states"][:, 0] for name in ("train", "val", "test", "ood")])
                self.assertEqual(len(np.unique(initial, axis=0)), len(initial))
                for name in ("train", "val", "test", "ood"):
                    states = data[name + "_states"]
                    amplitude = np.hypot(states[:, 0, :4], states[:, 0, 4:])
                    bounds = OOD_AMPLITUDE if name == "ood" else TRAIN_AMPLITUDE
                    self.assertTrue(np.all(amplitude >= bounds[0]))
                    self.assertTrue(np.all(amplitude <= bounds[1]))
                self.assertEqual(json.loads(str(data["metadata_json"])), metadata)
            self.assertEqual(json.loads(path.with_suffix(".json").read_text()), metadata)
            with self.assertRaises(FileExistsError):
                generate_dataset(path, n_train=1, n_val=1, n_test=1)


if __name__ == "__main__":
    unittest.main()
