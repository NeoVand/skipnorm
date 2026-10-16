"""Structural and differentiation checks; no training or ground-truth state input."""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.visual_physics_models import (
    MODEL_KINDS, VisualWorldModel, parameter_counts, rotation_pair_schedule,
)


class TestVisualWorldModel(unittest.TestCase):
    def make(self, kind, **kwargs):
        return VisualWorldModel(kind, latent_dim=8, width=16, image_shape=(4, 6),
                                history=3, key=jax.random.PRNGKey(7), **kwargs)

    def test_identical_visual_initialization_across_variants(self):
        reference = self.make("residual")
        expected = jax.tree_util.tree_leaves((reference.encoder, reference.decoder))
        for kind in MODEL_KINDS:
            model = self.make(kind)
            actual = jax.tree_util.tree_leaves((model.encoder, model.decoder))
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.dtype, jnp.float32)

    def test_rollout_shapes_alignment_and_batching(self):
        frames = jax.random.uniform(jax.random.PRNGKey(1), (3, 4, 6), dtype=jnp.float32)
        for kind in MODEL_KINDS:
            model = self.make(kind)
            z, images = eqx.filter_jit(lambda m, h: m.rollout(h, 3))(model, frames)
            self.assertEqual(z.shape, (3, 8))
            self.assertEqual(images.shape, (3, 4, 6))
            np.testing.assert_allclose(z[0], model.step(model.encode(frames)), atol=2e-7)
            self.assertTrue(np.isfinite(images).all())
            self.assertTrue(bool(jnp.all((images >= 0) & (images <= 1))))
            batched = jax.vmap(lambda h: model.rollout(h, 2)[1])(jnp.stack((frames, frames)))
            self.assertEqual(batched.shape, (2, 2, 4, 6))
        empty = self.make("residual").rollout(frames, 0)
        self.assertEqual(empty[0].shape, (0, 8))
        self.assertEqual(empty[1].shape, (0, 4, 6))

    def test_rotation_schedule_updates_every_coordinate_and_connects_pairs(self):
        for dim in range(2, 21):
            pair_count = (dim + 3) // 4
            adjacency = [set() for _ in range(dim)]
            schedule = rotation_pair_schedule(dim)
            self.assertEqual(len(schedule), 4)
            for permutation in schedule:
                self.assertEqual(sorted(permutation), list(range(dim)))
                rotated = permutation[-2 * pair_count:]
                for a, b in zip(rotated[::2], rotated[1::2]):
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            reached, frontier = {0}, [0]
            while frontier:
                for neighbor in adjacency[frontier.pop()]:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        frontier.append(neighbor)
            self.assertEqual(reached, set(range(dim)), dim)
        model = VisualWorldModel("rotation", key=jax.random.PRNGKey(3))
        self.assertEqual(tuple(layer.permutation for layer in model.dynamics.layers),
                         rotation_pair_schedule(16))

    def test_norm_preserving_variants_and_inverses(self):
        z = jax.random.normal(jax.random.PRNGKey(9), (8,), dtype=jnp.float32)
        for kind in ("projected", "geodesic", "rotation", "linear_orthogonal"):
            model = self.make(kind)
            final = jax.lax.fori_loop(0, 100, lambda _, x: model.step(x), z)
            np.testing.assert_allclose(jnp.linalg.norm(final), jnp.linalg.norm(z), rtol=3e-5)
        for kind in ("rotation", "linear_orthogonal", "hamiltonian"):
            model = self.make(kind)
            restored = model.dynamics.inverse(model.step(z))
            np.testing.assert_allclose(restored, z, rtol=3e-6, atol=3e-7)

    def test_end_to_end_gradients_are_finite_and_reach_dynamics(self):
        frames = jax.random.uniform(jax.random.PRNGKey(2), (3, 4, 6), dtype=jnp.float32)
        for kind in MODEL_KINDS:
            model = self.make(kind)
            def loss(m):
                return jnp.mean((m.rollout(frames, 2)[1] - frames[-1]) ** 2)
            grads = eqx.filter_jit(eqx.filter_grad(loss))(model)
            leaves = [x for x in jax.tree_util.tree_leaves(grads) if eqx.is_array(x)]
            self.assertTrue(all(np.isfinite(x).all() for x in leaves), kind)
            dynamics_grad = jax.tree_util.tree_leaves(grads.dynamics)
            self.assertGreater(sum(float(jnp.sum(x * x)) for x in dynamics_grad), 0, kind)

    def test_counts_and_guardrails(self):
        model = self.make("residual")
        counts = parameter_counts(model)
        self.assertEqual(counts["total"], sum(counts[k] for k in ("encoder", "decoder", "dynamics")))
        self.assertGreater(counts["dynamics"], 0)
        with self.assertRaises(ValueError):
            self.make("imaginary")
        with self.assertRaises(ValueError):
            model.encode(jnp.zeros((4, 4, 6)))
        with self.assertRaises(ValueError):
            model.rollout(jnp.zeros((3, 4, 6)), -1)


if __name__ == "__main__":
    unittest.main()
