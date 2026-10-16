"""Independent derivative and structure checks for perspective phase maps."""

import math
import unittest

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from skipnorm.hamiltonian_normalization import PerspectivePhase, UnitaryMix


def omega(dim):
    zero, eye = np.zeros((dim, dim)), np.eye(dim)
    return np.block([[zero, eye], [-eye, zero]])


def phase_rotate(x, angle):
    dim = len(x)//2
    q, p = x[:dim], x[dim:]
    return jnp.concatenate((jnp.cos(angle)*q-jnp.sin(angle)*p,
                            jnp.sin(angle)*q+jnp.cos(angle)*p))


class TestHamiltonianNormalization(unittest.TestCase):
    def test_frequencies_and_perspective_hessian_match_autodiff(self):
        model = PerspectivePhase(dim=3, width=7, delta=.7, time=.4, key=jax.random.PRNGKey(1))
        actions = jnp.array([.2, .7, 1.3])
        np.testing.assert_allclose(model.phase_frequencies(actions),
                                   jax.grad(model.hamiltonian)(actions), rtol=1e-12, atol=1e-13)
        rho = model.delta + jnp.sum(actions)
        p = actions/rho
        hessian_h = jax.hessian(lambda point: model.h_and_gradient(point)[0])(p)
        A = jnp.eye(3) - p[:, None]*jnp.ones((1, 3))
        expected = A.T @ hessian_h @ A / rho
        np.testing.assert_allclose(jax.hessian(model.hamiltonian)(actions), expected, rtol=1e-12, atol=1e-13)
        self.assertLessEqual(np.linalg.norm(np.asarray(hessian_h), 2), model.hessian_bound() + 1e-12)

    def test_cartesian_properties_across_scales_and_at_origin(self):
        model = PerspectivePhase(dim=3, width=8, delta=.8, time=-.7, key=jax.random.PRNGKey(2))
        base = jax.random.normal(jax.random.PRNGKey(3), (6,))
        bound = model.certified_log_bound()
        for scale in (0., 1e-8, 1., 1e8):
            x = base*scale
            y = eqx.filter_jit(model)(x)
            np.testing.assert_allclose(model.inverse(y), x, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(model(phase_rotate(x, .731)), phase_rotate(y, .731), rtol=1e-12, atol=1e-15)
            divisor = max(1., scale)
            self.assertAlmostEqual(float(jnp.linalg.norm(y/divisor)), float(jnp.linalg.norm(x/divisor)), places=12)
            jac = np.asarray(jax.jacfwd(model)(x))
            np.testing.assert_allclose(jac.T @ omega(3) @ jac, omega(3), atol=2e-12)
            values = np.linalg.svd(jac, compute_uv=False)
            self.assertLessEqual(values[0], math.exp(bound) + 1e-12)
            self.assertGreaterEqual(values[-1], math.exp(-bound) - 1e-12)
            self.assertAlmostEqual(float(jnp.linalg.det(jnp.asarray(jac))), 1., places=12)
        np.testing.assert_allclose(jax.jacfwd(model)(jnp.zeros(6)), np.eye(6), atol=1e-14)

    def test_finite_differences_inverse_derivative_and_parameter_gradients(self):
        model = PerspectivePhase(dim=3, width=6, time=.8, key=jax.random.PRNGKey(4))
        x = jnp.array([.7, 0., -1.2, .3, 0., .6])
        step = 1e-5
        finite = np.column_stack([np.asarray(model(x+step*e)-model(x-step*e))/(2*step)
                                  for e in jnp.eye(6)])
        jac = np.asarray(jax.jacfwd(model)(x))
        np.testing.assert_allclose(jac, finite, rtol=2e-8, atol=2e-10)
        inverse_jac = np.asarray(jax.jacfwd(model.inverse)(model(x)))
        np.testing.assert_allclose(inverse_jac @ jac, np.eye(6), atol=1e-12)
        gradients = eqx.filter_grad(lambda m: jnp.sum(jnp.sin(m(x))))(model)
        self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves(gradients)))

    def test_common_phase_is_needed_for_symplecticity(self):
        corrected = PerspectivePhase(dim=2, width=2, delta=.7, time=.8, key=jax.random.PRNGKey(5))
        corrected = eqx.tree_at(lambda m: (m.W, m.a, m.b), corrected,
                                (jnp.eye(2), jnp.array([4., -3.]), jnp.array([.2, -.4])))
        naive = PerspectivePhase(dim=2, width=2, delta=.7, time=.8,
                                   omit_common_phase=True, key=jax.random.PRNGKey(5))
        naive = eqx.tree_at(lambda m: (m.W, m.a, m.b), naive, (corrected.W, corrected.a, corrected.b))
        x = jnp.array([.9, 1.4, .4, -.7])
        true_jac, naive_jac = np.asarray(jax.jacfwd(corrected)(x)), np.asarray(jax.jacfwd(naive)(x))
        np.testing.assert_allclose(true_jac.T @ omega(2) @ true_jac, omega(2), atol=1e-12)
        self.assertGreater(np.linalg.norm(naive_jac.T @ omega(2) @ naive_jac-omega(2)), 1e-3)
        self.assertIsNone(naive.certified_log_bound())
        np.testing.assert_allclose(naive.inverse(naive(x)), x, atol=1e-12)
        np.testing.assert_allclose(naive(phase_rotate(x, .4)), phase_rotate(naive(x), .4), atol=1e-12)
        self.assertAlmostEqual(float(jnp.linalg.det(jnp.asarray(naive_jac))), 1., places=12)

    def test_projection_and_sharp_hessian_certificate(self):
        model = PerspectivePhase(dim=2, width=2, time=.6, key=jax.random.PRNGKey(6))
        model = eqx.tree_at(lambda m: (m.W, m.a, m.b), model,
                            (2.*jnp.eye(2), jnp.array([8., 8.]), jnp.array([-.5, -.5])))
        model = eqx.filter_jit(lambda m: m.project_lipschitz())(model)
        np.testing.assert_allclose(model.W, np.eye(2), atol=1e-14)
        np.testing.assert_array_equal(model.a, [4., 4.])
        np.testing.assert_array_equal(model.b, [-.5, -.5])
        self.assertAlmostEqual(model.hessian_bound(), 1.)
        # Balanced large actions approach the sharp global factor, without
        # multiplying the Hessian bound by the number of hidden features.
        x = jnp.array([math.sqrt(2e6), math.sqrt(2e6), 0., 0.])
        value = np.linalg.svd(np.asarray(jax.jacfwd(model)(x)), compute_uv=False)[0]
        bound = math.exp(model.certified_log_bound())
        self.assertLessEqual(value, bound + 1e-12)
        self.assertGreater(value, bound - 1e-6)

    def test_common_phase_control_has_identical_projective_dynamics(self):
        options = dict(dim=3, width=7, time=.8, key=jax.random.PRNGKey(11))
        corrected = PerspectivePhase(**options)
        corrected = eqx.tree_at(lambda m: m.a, corrected, 2.*jnp.ones(7))
        naive = PerspectivePhase(**options, omit_common_phase=True)
        naive = eqx.tree_at(lambda m: (m.W, m.a, m.b), naive, (corrected.W, corrected.a, corrected.b))
        mix = UnitaryMix(dim=3, scale=.1, key=jax.random.PRNGKey(12))
        original = jnp.array([.8, -.5, 1.1, .4, .6, -.3])
        true_state, naive_state = original, original
        for _ in range(5):
            true_state = mix(corrected(mix(corrected(true_state))))
            naive_state = mix(naive(mix(naive(naive_state))))
        true_complex = np.asarray(true_state[:3] + 1j*true_state[3:])
        naive_complex = np.asarray(naive_state[:3] + 1j*naive_state[3:])
        np.testing.assert_allclose(abs(true_complex)**2, abs(naive_complex)**2, atol=1e-12)
        overlap = np.vdot(true_complex, naive_complex)
        aligned = naive_complex * np.conj(overlap/abs(overlap))
        np.testing.assert_allclose(aligned, true_complex, atol=1e-12)
        self.assertGreater(np.linalg.norm(naive_complex-true_complex), 1e-4)

    def test_unitary_mixing_and_composed_certificate(self):
        mix = UnitaryMix(dim=3, key=jax.random.PRNGKey(7))
        phase = PerspectivePhase(dim=3, width=7, time=.9, key=jax.random.PRNGKey(8))
        x = jax.random.normal(jax.random.PRNGKey(9), (6,))
        matrix = np.asarray(mix.matrix())
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(mix.inverse(mix(x)), x, atol=1e-12)
        np.testing.assert_allclose(mix(phase_rotate(x, .3)), phase_rotate(mix(x), .3), atol=1e-12)
        jac = np.asarray(jax.jacfwd(mix)(x))
        np.testing.assert_allclose(jac.T @ jac, np.eye(6), atol=1e-12)
        np.testing.assert_allclose(jac.T @ omega(3) @ jac, omega(3), atol=1e-12)
        flow = lambda y: mix(phase(mix(phase(y))))
        inverse = lambda y: phase.inverse(mix.inverse(phase.inverse(mix.inverse(y))))
        np.testing.assert_allclose(inverse(flow(x)), x, atol=1e-12)
        full_jac = np.asarray(jax.jacfwd(flow)(x))
        np.testing.assert_allclose(full_jac.T @ omega(3) @ full_jac, omega(3), atol=1e-12)
        values = np.linalg.svd(full_jac, compute_uv=False)
        budget = 2*phase.certified_log_bound()
        self.assertLessEqual(values[0], math.exp(budget) + 1e-12)
        self.assertGreaterEqual(values[-1], math.exp(-budget) - 1e-12)
        identity = UnitaryMix(dim=3, identity=True, key=jax.random.PRNGKey(10))
        np.testing.assert_array_equal(identity(x), x)


if __name__ == "__main__":
    unittest.main()
