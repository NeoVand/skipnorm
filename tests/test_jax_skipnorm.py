"""Unit tests for the pure JAX + Equinox SkipNorm implementation (float64) and locked metrics."""

import unittest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from skipnorm.jax_layers import (
    BijectiveCouplingMap,
    block_geodesic_skip_norm_fn,
    cayley_orthogonal_matrix,
    geodesic_skip_norm_fn,
    projected_residual_skip_norm_fn,
    skip_norm_fn,
    state_dependent_cayley_step,
    tangent_project,
)
from skipnorm.metrics import (
    compute_k_invariant_readout_r2,
    compute_monotone_r2_and_spearman,
    compute_orbit_radii,
    compute_ovr,
    compute_radial_cv,
)


class TestJaxSkipNorm(unittest.TestCase):
    def test_jax_skip_norm_exact_float64(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (32, 16), dtype=jnp.float64)
        y = jax.random.normal(k2, (32, 6), dtype=jnp.float64)

        for p in [1.0, 2.0, 3.5, jnp.inf]:
            y_hat = skip_norm_fn(x, y, p=p, axis=-1, eps=1e-30)
            if p == jnp.inf:
                nx = jnp.max(jnp.abs(x), axis=-1)
                ny = jnp.max(jnp.abs(y_hat), axis=-1)
            else:
                nx = jnp.linalg.norm(x, ord=p, axis=-1)
                ny = jnp.linalg.norm(y_hat, ord=p, axis=-1)

            rel_err = float(jnp.max(jnp.abs(nx - ny) / nx))
            self.assertLess(rel_err, 1e-14)

    def test_jax_orthogonal_jacobian_decomposition_float64(self):
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        W = jax.random.normal(k1, (5, 9), dtype=jnp.float64)
        x = jax.random.normal(k2, (9,), dtype=jnp.float64)

        def layer(z):
            y = jnp.tanh(W @ z)
            return skip_norm_fn(z, y, p=2.0, axis=-1, eps=1e-30)

        y_hat = layer(x)
        J = jax.jacfwd(layer)(x)
        u_x = x / jnp.linalg.norm(x)
        u_y = y_hat / jnp.linalg.norm(y_hat)

        self.assertLess(float(jnp.linalg.norm(J.T @ u_y - u_x)), 1e-14)
        self.assertLess(float(jnp.abs(jnp.dot(u_y, J @ u_x) - 1.0)), 1e-14)

    def test_jax_geodesic_skip_norm_no_antipodal_flip(self):
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (16, 8), dtype=jnp.float64)
        f_x = jax.random.normal(k2, (16, 8), dtype=jnp.float64) * 10.0

        f_perp = tangent_project(x, f_x, axis=-1)
        dots = jnp.sum(f_perp * x, axis=-1)
        self.assertLess(float(jnp.max(jnp.abs(dots))), 1e-13)

        y_hat = geodesic_skip_norm_fn(x, f_x, alpha=0.5, axis=-1)
        nx = jnp.linalg.norm(x, axis=-1)
        ny = jnp.linalg.norm(y_hat, axis=-1)
        self.assertLess(float(jnp.max(jnp.abs(nx - ny) / nx)), 1e-14)
        self.assertTrue(bool(jnp.all(jnp.sum(y_hat * x, axis=-1) > 0.0)))

        # Projected residual also preserves norm
        y_proj = projected_residual_skip_norm_fn(x, f_x, alpha=0.5, axis=-1)
        nproj = jnp.linalg.norm(y_proj, axis=-1)
        self.assertLess(float(jnp.max(jnp.abs(nx - nproj) / nx)), 1e-14)

    def test_jax_block_geodesic_skip_norm_multi_invariants(self):
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (10, 6), dtype=jnp.float64)
        f_x = jax.random.normal(k2, (10, 6), dtype=jnp.float64)

        y_hat = block_geodesic_skip_norm_fn(x, f_x, num_blocks=3, alpha=0.25)
        for b in range(3):
            nb_x = jnp.linalg.norm(x[:, 2 * b : 2 * (b + 1)], axis=-1)
            nb_y = jnp.linalg.norm(y_hat[:, 2 * b : 2 * (b + 1)], axis=-1)
            self.assertLess(float(jnp.max(jnp.abs(nb_x - nb_y) / nb_x)), 1e-14)

    def test_bijective_coupling_map_exact_inverse(self):
        key = jax.random.PRNGKey(4)
        k1, k2 = jax.random.split(key)
        bij = BijectiveCouplingMap(dim=4, num_layers=4, width=16, depth=2, key=k1)
        x = jax.random.normal(k2, (8, 4), dtype=jnp.float64)
        z = jax.vmap(bij)(x)
        x_rec = jax.vmap(bij.inverse)(z)
        self.assertLess(float(jnp.max(jnp.abs(x - x_rec))), 1e-13)

    def test_cayley_orthogonal_norm_conservation(self):
        key = jax.random.PRNGKey(5)
        k1, k2 = jax.random.split(key)
        raw = jax.random.normal(k1, (6, 6), dtype=jnp.float64)
        x = jax.random.normal(k2, (6,), dtype=jnp.float64)
        Q = cayley_orthogonal_matrix(raw, alpha=0.2)
        self.assertLess(float(jnp.linalg.norm(Q.T @ Q - jnp.eye(6))), 1e-14)
        y = state_dependent_cayley_step(x, raw, alpha=0.2)
        self.assertLess(float(jnp.abs(jnp.linalg.norm(y) - jnp.linalg.norm(x))), 1e-14)

    def test_metrics_and_degenerate_ovr_guard(self):
        # Constant radius across all orbits -> OVR must be flagged 'undefined (collapsed)'
        r_collapsed = np.ones((10, 50), dtype=np.float64) * 2.0
        ovr_col = compute_ovr(r_collapsed, sigma_lat=1.0)
        self.assertIsNone(ovr_col["ovr"])
        self.assertEqual(ovr_col["status"], "undefined (collapsed)")

        # Distinct concentric radii -> CV = 0, OVR = 0, Isotonic R2 = 1.0, |rho_S| = 1.0
        c_vals = np.linspace(1.0, 5.0, 12)
        r_clean = np.repeat(c_vals[:, None] ** 2, 40, axis=1)
        cv_clean = compute_radial_cv(r_clean)
        ovr_clean = compute_ovr(r_clean, sigma_lat=1.0)
        c_train = np.linspace(0.5, 5.5, 41)
        mono_clean = compute_monotone_r2_and_spearman(c_train, c_train, c_vals, c_vals)
        self.assertLess(cv_clean["mean_cv"], 1e-15)
        self.assertLess(ovr_clean["ovr"], 1e-15)
        self.assertGreater(mono_clean["isotonic_r2"], 0.999999)
        self.assertGreater(mono_clean["spearman_abs_rho"], 0.999999)

    def test_invertible_branch_does_not_imply_full_skipnorm_rank(self):
        x = jnp.array([1.0, 0.0])
        A = jnp.array([[1.0, 1.0], [1.0, 0.0]])
        def layer(z):
            y = A @ z + jnp.array([0.0, -1.0])
            return jnp.linalg.norm(z) * y / jnp.linalg.norm(y)
        J = jax.jacfwd(layer)(x)
        np.testing.assert_allclose(J, [[1., 0.], [1., 0.]], atol=1e-14)
        self.assertEqual(np.linalg.matrix_rank(np.asarray(A)), 2)
        self.assertEqual(np.linalg.matrix_rank(np.asarray(J)), 1)

    def test_radial_shear_can_coexist_with_another_unit_singular_value(self):
        def twist(x):
            a = jnp.log(jnp.linalg.norm(x))
            return jnp.array([jnp.cos(a)*x[0]-jnp.sin(a)*x[1],
                              jnp.sin(a)*x[0]+jnp.cos(a)*x[1], x[2]])
        x = jnp.array([1., 0., 0.])
        J = jax.jacfwd(twist)(x)
        self.assertGreater(float(jnp.linalg.norm(J @ x - x)), 0.5)
        self.assertLess(float(jnp.min(jnp.abs(jnp.linalg.svd(J, compute_uv=False)-1))), 1e-14)


if __name__ == "__main__":
    unittest.main()
