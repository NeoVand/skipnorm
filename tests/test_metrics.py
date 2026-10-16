"""Regression tests for split isolation and scale-aware evaluation."""

import unittest

import numpy as np

from skipnorm.metrics import (
    centered_rms_spread,
    compute_k_invariant_readout_r2,
    compute_monotone_r2_and_spearman,
    compute_ovr,
    compute_radius_threshold_probe,
)


class TestMetrics(unittest.TestCase):
    def test_isotonic_uses_training_fit_and_direction(self):
        fit = compute_monotone_r2_and_spearman(
            [0, 1, 2], [0, 1, 2], [.25, 1.75], [1.75, .25]
        )
        self.assertEqual(fit["direction"], "increasing")
        np.testing.assert_allclose(fit["test_predictions"], [.25, 1.75])
        self.assertLess(fit["isotonic_r2"], 0)

    def test_isotonic_pools_duplicate_labels_and_clips_boundaries(self):
        fit = compute_monotone_r2_and_spearman(
            [0, 0, 1, 2], [1, 3, 4, 6], [-1, .5, 3], [2, 3, 6]
        )
        np.testing.assert_allclose(fit["test_predictions"], [2, 3, 6])
        self.assertEqual(fit["test_outside_training_range"], 2)
        self.assertAlmostEqual(fit["isotonic_r2"], 1)

    def test_centered_spread_and_ovr_scale_guard(self):
        z = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        sigma = centered_rms_spread(z)
        self.assertAlmostEqual(centered_rms_spread(z + [100, -20]), sigma)
        r = np.array([[1., 1.2], [2., 2.2]])
        expected = .01 / .26
        for scale in (1e-12, 1., 1e12):
            score = compute_ovr(r * scale, sigma * scale)
            self.assertEqual(score["status"], "valid")
            self.assertAlmostEqual(score["ovr"], expected)
        self.assertIsNone(compute_ovr(np.ones((3, 5)), sigma)["ovr"])
        self.assertIsNone(compute_ovr(np.array([[1., 2.], [2., 1.]]), sigma)["ovr"])

    def test_threshold_probe_does_not_optimize_test_labels(self):
        result = compute_radius_threshold_probe([0, 1, 2, 3], [0, 0, 1, 1], [.5, 2.5], [1, 0])
        self.assertEqual(result["training_accuracy"], 1)
        self.assertEqual(result["accuracy"], 0)

    def test_negative_test_r2_is_preserved(self):
        result = compute_k_invariant_readout_r2(
            np.arange(5.), np.arange(5.)[:, None], np.array([.5, 3.5]), np.array([[10.], [11.]])
        )
        self.assertLess(result["nonlinear_r2_mean"], 0)


if __name__ == "__main__":
    unittest.main()
