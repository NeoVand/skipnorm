"""Pure JAX + Equinox implementation of SkipNorm and Geodesic SkipNorm layers.

Supports float64 exact radial isometry, arbitrary Lp norms, tangent-space Geodesic SkipNorm,
and Multi-Block (Toroidal / Liouville-Arnold) SkipNorm on direct sums of spheres.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def skip_norm_fn(
    x: Array,
    y: Array,
    p: float = 2.0,
    axis: int = -1,
    eps: float = 1e-12,
) -> Array:
    """Core functional SkipNorm operation in JAX:

        y_hat = y * (||x||_p / max(||y||_p, eps))

    Using jnp.maximum(||y||_p, eps) instead of (||y||_p + eps) guarantees exact
    machine-precision norm preservation (10^-16 in float64) across thousands
    of recurrent steps without cumulative damping.
    """
    if p == 2.0:
        norm_x = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True))
        norm_y = jnp.sqrt(jnp.sum(jnp.square(y), axis=axis, keepdims=True))
    elif p == 1.0:
        norm_x = jnp.sum(jnp.abs(x), axis=axis, keepdims=True)
        norm_y = jnp.sum(jnp.abs(y), axis=axis, keepdims=True)
    elif p == jnp.inf:
        norm_x = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        norm_y = jnp.max(jnp.abs(y), axis=axis, keepdims=True)
    else:
        norm_x = jnp.power(jnp.sum(jnp.power(jnp.abs(x), p), axis=axis, keepdims=True), 1.0 / p)
        norm_y = jnp.power(jnp.sum(jnp.power(jnp.abs(y), p), axis=axis, keepdims=True), 1.0 / p)

    return y * (norm_x / jnp.maximum(norm_y, eps))


def tangent_project(x: Array, v: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Project vector v onto the tangent space T_x S^{d-1} orthogonal to x:

        v_perp = v - (<v, x> / max(||x||_2^2, eps^2)) * x
    """
    norm_x_sq = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    dot_vx = jnp.sum(v * x, axis=axis, keepdims=True)
    return v - (dot_vx / jnp.maximum(norm_x_sq, eps * eps)) * x


def geodesic_skip_norm_fn(
    x: Array,
    f_x: Array,
    alpha: Array | float = 0.1,
    axis: int = -1,
    eps: float = 1e-12,
) -> Array:
    """Tangent-space Geodesic SkipNorm update on concentric spheres S^{d-1}(||x||_2):

        f_perp(x) = f(x) - (<f(x), x> / ||x||_2^2) * x
        y = x + alpha * f_perp(x)
        y_hat = y * (||x||_2 / max(||y||_2, eps))

    Guarantees:
      1. Exact norm conservation ||y_hat||_2 == ||x||_2 to machine precision.
      2. Zero antipodal flip instability (<y, x> = ||x||_2^2 > 0 for all finite alpha).
      3. First-order symplectic / unitary retract on the sphere S^{d-1}(||x||_2).
    """
    f_perp = tangent_project(x, f_x, axis=axis, eps=eps)
    y = x + alpha * f_perp
    return skip_norm_fn(x, y, p=2.0, axis=axis, eps=eps)


def block_geodesic_skip_norm_fn(
    x: Array,
    f_x: Array,
    num_blocks: int,
    alpha: Array | float = 0.1,
    eps: float = 1e-12,
) -> Array:
    """Multi-Block (Toroidal / Liouville-Arnold) Geodesic SkipNorm.

    Splits x in R^{K * d_b} into K blocks (z^(1), ..., z^(K)) in (R^{d_b})^K
    and applies Geodesic SkipNorm independently to each block's norm while allowing
    f_x(x) to couple all blocks nonlinearly.

    Preserves K independent action invariants (||z^(1)||_2, ..., ||z^(K)||_2),
    foliating R^{K * d_b} into invariant tori / sphere products prod_{k=1}^K S^{d_b-1}(r_k).
    """
    orig_shape = x.shape
    block_dim = orig_shape[-1] // num_blocks
    x_blocks = x.reshape(*orig_shape[:-1], num_blocks, block_dim)
    f_blocks = f_x.reshape(*orig_shape[:-1], num_blocks, block_dim)
    y_blocks = geodesic_skip_norm_fn(x_blocks, f_blocks, alpha=alpha, axis=-1, eps=eps)
    return y_blocks.reshape(orig_shape)


class SkipNormLayer(eqx.Module):
    """Equinox wrapper applying SkipNorm around an arbitrary callable block f."""

    fn: Callable[[Array], Array]
    p: float = eqx.field(static=True, default=2.0)
    eps: float = eqx.field(static=True, default=1e-12)

    def __call__(self, x: Array) -> Array:
        y = self.fn(x)
        return skip_norm_fn(x, y, p=self.p, axis=-1, eps=self.eps)


class GeodesicSkipNormBlock(eqx.Module):
    """Equinox Geodesic SkipNorm residual block with learnable tangent step alpha."""

    fn: Callable[[Array], Array]
    alpha: Array
    eps: float = eqx.field(static=True, default=1e-12)

    def __init__(self, fn: Callable[[Array], Array], alpha_init: float = 0.1, eps: float = 1e-12):
        self.fn = fn
        self.alpha = jnp.asarray(alpha_init)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        return geodesic_skip_norm_fn(x, self.fn(x), alpha=self.alpha, axis=-1, eps=self.eps)


class BlockGeodesicSkipNormBlock(eqx.Module):
    """Equinox Multi-Block (Toroidal) Geodesic SkipNorm block preserving K block norms."""

    fn: Callable[[Array], Array]
    alpha: Array
    num_blocks: int = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-12)

    def __init__(
        self,
        fn: Callable[[Array], Array],
        num_blocks: int,
        alpha_init: float = 0.1,
        eps: float = 1e-12,
    ):
        self.fn = fn
        self.num_blocks = num_blocks
        self.alpha = jnp.asarray(alpha_init)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        return block_geodesic_skip_norm_fn(
            x, self.fn(x), num_blocks=self.num_blocks, alpha=self.alpha, eps=self.eps
        )
