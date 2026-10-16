"""Invertible norm-preserving rotation couplings with a global distortion bound.

These layers preserve Euclidean norm and volume, not generally the canonical
symplectic form.  Their Jacobians need not be orthogonal.  Certification requires
normalized conditioning and bounds the singular values of the whole Jacobian,
including derivatives of the conditioning network.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _project_weight(weight: jax.Array) -> jax.Array:
    """Euclidean projection onto the operator-norm unit ball, using exact SVD."""
    u, singular_values, vh = jnp.linalg.svd(weight, full_matrices=False)
    return (u * jnp.minimum(singular_values, 1.0)[None, :]) @ vh


def _project_mlp(network: eqx.nn.MLP) -> eqx.nn.MLP:
    weights = tuple(_project_weight(layer.weight) for layer in network.layers)
    return eqx.tree_at(lambda m: tuple(layer.weight for layer in m.layers), network, weights)


def _mlp_lipschitz_bound(network: eqx.nn.MLP) -> float:
    """Diagnostic only: exact weight norms on the host, outside traced code."""
    return math.prod(float(np.linalg.svd(np.asarray(layer.weight), compute_uv=False)[0])
                     for layer in network.layers)


class ConditionalRotationCoupling(eqx.Module):
    """Rotate disjoint pairs in ``b`` conditional on unchanged coordinates ``a``.

    In permuted coordinates x=(a,b), normalized conditioning is
    c=(a,delta)/sqrt(delta**2+||x||**2).  The MLP outputs one angle per pair.
    Both c and pair radii are unchanged by the update, giving an explicit inverse.

    If the MLP Lipschitz bound is L, normalized conditioning gives singular values
    in [exp(-beta), exp(beta)], beta=asinh(abs(alpha)*L/2).  Raw conditioning
    retains norm preservation, unit determinant, and the inverse, but has no
    corresponding global distortion certificate.

    Inputs are single vectors; use jax.vmap for batches.  ``project_lipschitz``
    returns a new module and should be called after each optimizer update for a
    fixed unit-Lipschitz conditioner budget.  Biases are intentionally unrestricted.
    """

    conditioner: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    condition_dim: int = eqx.field(static=True)
    permutation: tuple[int, ...] = eqx.field(static=True)
    inverse_permutation: tuple[int, ...] = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)

    def __init__(self, dim: int, width: int = 32, depth: int = 2, *,
                 alpha: float = 0.1, delta: float = 1.0, normalize: bool = True,
                 project: bool = True, condition_dim: int | None = None,
                 permutation: tuple[int, ...] | None = None, key: jax.Array):
        if dim < 2:
            raise ValueError("dim must be at least 2")
        if width < 1 or depth < 0:
            raise ValueError("width must be positive and depth nonnegative")
        if not math.isfinite(alpha) or not math.isfinite(delta) or delta <= 0:
            raise ValueError("alpha must be finite and delta must be finite and positive")
        if condition_dim is None:
            condition_dim = dim - 2 * max(1, dim // 4)
        if not 0 <= condition_dim <= dim - 2 or (dim - condition_dim) % 2:
            raise ValueError("rotated coordinates must consist of at least one complete pair")
        permutation = tuple(range(dim)) if permutation is None else tuple(permutation)
        if sorted(permutation) != list(range(dim)):
            raise ValueError("permutation must contain every coordinate exactly once")
        self.dim = dim
        self.condition_dim = condition_dim
        self.permutation = permutation
        self.inverse_permutation = tuple(int(v) for v in np.argsort(permutation))
        self.alpha, self.delta, self.normalize = float(alpha), float(delta), bool(normalize)
        network = eqx.nn.MLP(condition_dim + 1, (dim - condition_dim) // 2,
                             width, depth, activation=jax.nn.tanh, key=key)
        self.conditioner = _project_mlp(network) if project else network

    def _condition(self, x: jax.Array) -> jax.Array:
        a = x[:self.condition_dim]
        delta = jnp.asarray(self.delta, dtype=x.dtype)
        if not self.normalize:
            return jnp.concatenate((a, delta[None]))
        # Algebraically the stated smooth normalization. Scaling first prevents
        # squaring large inputs from overflowing, including in the conditioner.
        scale = jnp.maximum(jnp.max(jnp.abs(x)), delta)
        scaled_x, scaled_delta = x / scale, delta / scale
        denominator = jnp.sqrt(jnp.sum(scaled_x**2) + scaled_delta**2)
        return jnp.concatenate((scaled_x[:self.condition_dim], scaled_delta[None])) / denominator

    def _apply(self, x: jax.Array, sign: float) -> jax.Array:
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"expected one vector of shape ({self.dim},); use vmap for batches")
        permuted = x[jnp.asarray(self.permutation)]
        pairs = permuted[self.condition_dim:].reshape(-1, 2)
        angles = sign * self.alpha * self.conditioner(self._condition(permuted))
        cosine, sine = jnp.cos(angles), jnp.sin(angles)
        rotated = jnp.stack((cosine * pairs[:, 0] - sine * pairs[:, 1],
                             sine * pairs[:, 0] + cosine * pairs[:, 1]), axis=-1)
        output = jnp.concatenate((permuted[:self.condition_dim], rotated.reshape(-1)))
        return output[jnp.asarray(self.inverse_permutation)]

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._apply(x, 1.0)

    def step(self, x: jax.Array) -> jax.Array:
        return self(x)

    def inverse(self, y: jax.Array) -> jax.Array:
        return self._apply(y, -1.0)

    def project_lipschitz(self) -> ConditionalRotationCoupling:
        return eqx.tree_at(lambda m: m.conditioner, self, _project_mlp(self.conditioner))

    def conditioner_lipschitz_bound(self) -> float:
        return _mlp_lipschitz_bound(self.conditioner)

    def certified_log_bound(self) -> float | None:
        """Global log singular-value bound using actual weight norms; not JIT-safe."""
        if not self.normalize:
            return None
        return math.asinh(abs(self.alpha) * self.conditioner_lipschitz_bound() / 2.0)


class RotationCouplingFlow(eqx.Module):
    """Composition with a fixed log-distortion budget after weight projection.

    For ``num_stages=n`` set alpha=2*sinh(budget/n).  Unit-Lipschitz conditioners
    then give every singular value in [exp(-budget), exp(budget)] globally.
    ``certified_log_bound`` uses actual weights and also works for unprojected
    normalized controls; such controls need not satisfy the requested budget.

    Permutations use a separate fixed seed, so data/model initialization seeds
    and normalized/raw controls share the same coordinate mixing pattern.
    """

    layers: tuple[ConditionalRotationCoupling, ...]
    dim: int = eqx.field(static=True)
    budget: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, dim: int = 32, num_stages: int = 8, width: int = 32,
                 depth: int = 2, *, budget: float = 0.5, delta: float = 1.0,
                 normalize: bool = True, project: bool = True,
                 condition_dim: int | None = None, permutation_seed: int = 8675309,
                 key: jax.Array):
        if num_stages < 1 or not math.isfinite(budget) or budget < 0:
            raise ValueError("num_stages must be positive and budget finite and nonnegative")
        self.dim, self.budget = dim, float(budget)
        self.alpha = 2.0 * math.sinh(budget / num_stages)
        keys = jax.random.split(key, num_stages)
        rng = np.random.default_rng(permutation_seed)
        permutations = [tuple(int(v) for v in rng.permutation(dim)) for _ in range(num_stages)]
        self.layers = tuple(ConditionalRotationCoupling(
            dim, width, depth, alpha=self.alpha, delta=delta, normalize=normalize,
            project=project, condition_dim=condition_dim, permutation=permutation,
            key=layer_key) for permutation, layer_key in zip(permutations, keys))

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x: jax.Array) -> jax.Array:
        return self(x)

    def inverse(self, y: jax.Array) -> jax.Array:
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def project_lipschitz(self) -> RotationCouplingFlow:
        return eqx.tree_at(lambda m: m.layers, self,
                           tuple(layer.project_lipschitz() for layer in self.layers))

    def certified_log_bound(self) -> float | None:
        """Sum the layer certificates (host-side diagnostic, outside JIT)."""
        bounds = [layer.certified_log_bound() for layer in self.layers]
        return None if any(bound is None for bound in bounds) else sum(bounds)
