"""Unit tests for the pure JAX + Equinox SkipNorm implementation (float64)."""

import unittest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from skipnorm.jax_layers import (
    block_geodesic_skip_norm_fn,
    geodesic_skip_norm_fn,
    skip_norm_fn,
    tangent_project,
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


if __name__ == "__main__":
    unittest.main()
