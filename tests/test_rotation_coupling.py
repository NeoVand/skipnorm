"""Algebraic and differential checks for invertible rotation couplings."""

import math
import unittest

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from skipnorm.rotation_coupling import ConditionalRotationCoupling, RotationCouplingFlow


class TestRotationCoupling(unittest.TestCase):
    def test_scalar_shear_has_analytic_spectrum(self):
        # At x=(0,r,0), theta=alpha*a/s and shear magnitude alpha*r/s.
        alpha, radius, delta = 1.7, 2.0, 0.8
        layer = ConditionalRotationCoupling(3, depth=0, alpha=alpha, delta=delta,
                                             key=jax.random.PRNGKey(1))
        layer = eqx.tree_at(lambda m: m.conditioner.layers[0].weight, layer,
                            jnp.array([[1., 0.]]))
        layer = eqx.tree_at(lambda m: m.conditioner.layers[0].bias, layer, jnp.zeros(1))
        x = jnp.array([0., radius, 0.])
        jac = np.asarray(jax.jacfwd(layer)(x))
        beta = math.asinh(alpha * radius / math.hypot(radius, delta) / 2)
        np.testing.assert_allclose(np.linalg.svd(jac, compute_uv=False),
                                   [math.exp(beta), 1., math.exp(-beta)], rtol=1e-12, atol=1e-12)

    def test_simultaneous_pairs_inverse_volume_and_global_bound(self):
        layer = ConditionalRotationCoupling(8, width=9, depth=2, alpha=1.2,
                                             permutation=(7, 1, 5, 3, 0, 6, 4, 2),
                                             key=jax.random.PRNGKey(2))
        beta = layer.certified_log_bound()
        base = jax.random.normal(jax.random.PRNGKey(3), (8,))
        for scale in (0., 1e-120, 1., 1e120, 1e200):
            x = base * scale
            y = layer(x)
            np.testing.assert_allclose(layer.inverse(y), x, rtol=3e-13, atol=1e-130)
            divisor = max(1., scale)
            self.assertAlmostEqual(float(jnp.linalg.norm(y / divisor)),
                                   float(jnp.linalg.norm(x / divisor)), places=12)
            jac = np.asarray(jax.jacfwd(layer)(x))
            self.assertTrue(np.isfinite(jac).all())
            singular_values = np.linalg.svd(jac, compute_uv=False)
            self.assertLessEqual(singular_values[0], math.exp(beta) + 1e-12)
            self.assertGreaterEqual(singular_values[-1], math.exp(-beta) - 1e-12)
            sign, logdet = np.linalg.slogdet(jac)
            self.assertEqual(sign, 1.)
            self.assertAlmostEqual(float(logdet), 0., places=12)
            if scale == 1.:
                permutation = np.array(layer.permutation)
                angles = np.asarray(layer.alpha * layer.conditioner(layer._condition(x[permutation])))
                orthogonal = np.eye(layer.dim)
                for pair, angle in enumerate(angles):
                    start = layer.condition_dim + 2 * pair
                    orthogonal[start:start + 2, start:start + 2] = [
                        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                shear = orthogonal.T @ jac[permutation][:, permutation] - np.eye(layer.dim)
                np.testing.assert_allclose(shear @ shear, 0., atol=1e-13)
                shear_values = np.linalg.svd(shear, compute_uv=False)
                shear_values = shear_values[shear_values > 1e-12]
                pair_logs = np.arcsinh(shear_values / 2.)
                expected = np.sort(np.concatenate((np.exp(pair_logs), np.exp(-pair_logs),
                                                     np.ones(layer.dim - 2 * len(pair_logs)))))[::-1]
                np.testing.assert_allclose(singular_values, expected, rtol=1e-12, atol=1e-12)

    def test_composed_budget_and_inverse_derivative(self):
        flow = RotationCouplingFlow(dim=32, num_stages=8, width=8, depth=1,
                                     budget=.5, key=jax.random.PRNGKey(4))
        self.assertLessEqual(flow.certified_log_bound(), .5 + 1e-12)
        x = jax.random.normal(jax.random.PRNGKey(5), (32,))
        y = eqx.filter_jit(flow)(x)
        np.testing.assert_allclose(eqx.filter_jit(flow.inverse)(y), x, rtol=1e-12, atol=1e-12)
        jac = np.asarray(jax.jacfwd(flow)(x))
        inverse_jac = np.asarray(jax.jacfwd(flow.inverse)(y))
        np.testing.assert_allclose(inverse_jac @ jac, np.eye(32), rtol=1e-12, atol=1e-12)
        singular_values = np.linalg.svd(jac, compute_uv=False)
        self.assertLessEqual(singular_values[0], math.exp(.5) + 1e-12)
        self.assertGreaterEqual(singular_values[-1], math.exp(-.5) - 1e-12)
        sign, logdet = np.linalg.slogdet(jac)
        self.assertEqual(sign, 1.)
        self.assertAlmostEqual(float(logdet), 0., places=12)

    def test_finite_difference_jacobian_with_zero_pair_and_both_alpha_signs(self):
        x = jnp.array([.4, -.7, .2, .9, 0., 0., 1.2, -.3])
        for alpha in (-1.3, 1.3):
            layer = ConditionalRotationCoupling(8, width=7, depth=2, alpha=alpha,
                                                 key=jax.random.PRNGKey(10))
            columns = []
            for coordinate in range(8):
                step = 1e-5 * max(1., abs(float(x[coordinate])))
                perturbation = jnp.eye(8)[coordinate] * step
                columns.append(np.asarray(layer(x + perturbation) - layer(x - perturbation)) / (2 * step))
            finite_difference = np.column_stack(columns)
            automatic = np.asarray(jax.jacfwd(layer)(x))
            np.testing.assert_allclose(automatic, finite_difference, rtol=2e-8, atol=2e-9)
            np.testing.assert_allclose(layer.inverse(layer(x)), x, rtol=1e-12, atol=1e-12)
            singular_values = np.linalg.svd(finite_difference, compute_uv=False)
            bound = layer.certified_log_bound()
            self.assertLessEqual(singular_values[0], math.exp(bound) + 1e-8)
            self.assertGreaterEqual(singular_values[-1], math.exp(-bound) - 1e-8)

    def test_exact_svd_projection_preserves_biases_and_is_jittable(self):
        flow = RotationCouplingFlow(dim=8, num_stages=2, width=7, depth=1,
                                     budget=.5, key=jax.random.PRNGKey(6))
        weight_selector = lambda m: tuple(l.weight for s in m.layers for l in s.conditioner.layers)
        flow = eqx.tree_at(weight_selector, flow,
                           tuple(8. * w for w in weight_selector(flow)))
        biases = [np.asarray(l.bias) for s in flow.layers for l in s.conditioner.layers]
        self.assertGreater(flow.certified_log_bound(), .5)
        projected = eqx.filter_jit(lambda m: m.project_lipschitz())(flow)
        for layer, bias in zip((l for s in projected.layers for l in s.conditioner.layers), biases):
            self.assertLessEqual(np.linalg.svd(np.asarray(layer.weight), compute_uv=False)[0], 1. + 1e-12)
            np.testing.assert_array_equal(layer.bias, bias)
        self.assertLessEqual(projected.certified_log_bound(), .5 + 1e-12)

    def test_controls_share_initialization_and_inverse_without_certificate(self):
        options = dict(dim=8, num_stages=3, width=7, depth=1, key=jax.random.PRNGKey(7))
        normalized = RotationCouplingFlow(**options)
        raw = RotationCouplingFlow(**options, normalize=False)
        different_seed = RotationCouplingFlow(**dict(options, key=jax.random.PRNGKey(8)))
        for first, second, third in zip(normalized.layers, raw.layers, different_seed.layers):
            self.assertEqual(first.permutation, second.permutation)
            self.assertEqual(first.permutation, third.permutation)
            for left, right in zip(first.conditioner.layers, second.conditioner.layers):
                np.testing.assert_array_equal(left.weight, right.weight)
        self.assertIsNone(raw.certified_log_bound())
        x = jax.random.normal(jax.random.PRNGKey(9), (8,))
        np.testing.assert_allclose(raw.inverse(raw(x)), x, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(jnp.linalg.det(jax.jacfwd(raw)(x))), 1., places=12)
        for dim in (2, 3, 5, 6, 7):
            layer = ConditionalRotationCoupling(dim, key=jax.random.PRNGKey(dim))
            self.assertEqual((dim - layer.condition_dim) % 2, 0)
        with self.assertRaises(ValueError):
            ConditionalRotationCoupling(1, key=jax.random.PRNGKey(1))


if __name__ == "__main__":
    unittest.main()
