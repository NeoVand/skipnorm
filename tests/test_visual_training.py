"""Tests for scientific accounting in the visual pilot, not benchmark scores."""
import unittest
import numpy as np

from experiments.stage8_visual_world_model import pixel_loss, position_metrics, localization_metrics
from experiments.visual_physics_data import render_frames


class VisualTrainingTest(unittest.TestCase):
    def test_foreground_loss_penalizes_blank_prediction(self):
        target = render_frames(np.array([[.3, -.2, .1, -.4, 0, 0, 0, 0]]))
        self.assertEqual(float(pixel_loss(target, target)), 0.)
        # Background dominates ordinary MSE; the balanced loss must expose this.
        blank = np.zeros_like(target)
        self.assertGreater(float(pixel_loss(blank, target)), 10 * float(np.mean(target**2)))

    def test_valid_horizon_stops_at_first_error(self):
        truth = np.zeros((2, 10, 4))
        prediction = truth.copy()
        prediction[0, 3] = 1.
        result, traces = position_metrics(prediction, truth, np.ones(4), .08)
        np.testing.assert_array_equal(traces['per_trajectory_valid_horizon'], [3, 10])
        self.assertEqual(result['valid_horizon_mean'], 6.5)

    def test_blur_does_not_pass_localization(self):
        sharp = render_frames(np.zeros((2, 4, 8)))
        metrics, mask = localization_metrics(sharp)
        self.assertTrue(np.all(mask))
        broad = np.ones_like(sharp) * sharp.mean()
        _, broad_mask = localization_metrics(broad)
        self.assertFalse(np.any(broad_mask))


if __name__ == '__main__':
    unittest.main()
