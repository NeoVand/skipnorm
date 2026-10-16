"""Pure JAX + Equinox implementation of SkipNorm, Projected Residual SkipNorm,
Tangent-Projected Geodesic SkipNorm, Multi-Block Toroidal SkipNorm,
State-Dependent Orthogonal Rotations, and Bijective Coupling Encoders.
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

    Using jnp.maximum(||y||_p, eps) guarantees exact machine-precision norm
    preservation (10^-16 in float64) across thousands of recurrent steps.
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


def projected_residual_skip_norm_fn(
    x: Array,
    f_x: Array,
    alpha: Array | float = 0.1,
    axis: int = -1,
    eps: float = 1e-12,
) -> Array:
    """Projected Residual SkipNorm (without tangent projection):

        y = x + alpha * f(x)
        y_hat = y * (||x||_2 / max(||y||_2, eps))

    Can experience antipodal sign-flip if f(x) has a strong negative radial
    component opposing x (e.g., alpha * <f(x), x> < -||x||_2^2).
    """
    y = x + alpha * f_x
    return skip_norm_fn(x, y, p=2.0, axis=axis, eps=eps)


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
    """Tangent-Projected Geodesic SkipNorm update on concentric spheres S^{d-1}(||x||_2):

        f_perp(x) = f(x) - (<f(x), x> / ||x||_2^2) * x
        y = x + alpha * f_perp(x)
        y_hat = y * (||x||_2 / max(||y||_2, eps))

    Guarantees:
      1. Exact norm conservation ||y_hat||_2 == ||x||_2 to machine precision.
      2. Zero antipodal flip instability (<y, x> = ||x||_2^2 > 0 for all finite alpha).
      3. Retraction onto S^{d-1}(||x||_2) along tangent vector alpha * f_perp(x).
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

    Preserves K independent block radii (||z^(1)||_2, ..., ||z^(K)||_2),
    foliating R^{K * d_b} into sphere products prod_{k=1}^K S^{d_b-1}(r_k).
    """
    orig_shape = x.shape
    block_dim = orig_shape[-1] // num_blocks
    x_blocks = x.reshape(*orig_shape[:-1], num_blocks, block_dim)
    f_blocks = f_x.reshape(*orig_shape[:-1], num_blocks, block_dim)
    y_blocks = geodesic_skip_norm_fn(x_blocks, f_blocks, alpha=alpha, axis=-1, eps=eps)
    return y_blocks.reshape(orig_shape)


def cayley_orthogonal_matrix(raw_matrix: Array, alpha: Array | float = 0.1) -> Array:
    """Construct an exact orthogonal matrix Q in SO(d) via the Cayley transform:

        S = (M - M^T) / 2
        Q = (I - alpha * S) @ (I + alpha * S)^{-1}
    """
    d = raw_matrix.shape[-1]
    skew = 0.5 * (raw_matrix - jnp.swapaxes(raw_matrix, -1, -2))
    eye = jnp.eye(d, dtype=raw_matrix.dtype)
    return jnp.linalg.solve(eye + alpha * skew, eye - alpha * skew)


def state_dependent_cayley_step(x: Array, raw_matrix: Array, alpha: Array | float = 0.1) -> Array:
    """Apply a state-dependent orthogonal rotation O_phi(x) x in SO(d):

        S(x) = (M(x) - M(x)^T) / 2
        y_hat = (I + alpha * S(x))^{-1} (I - alpha * S(x)) x

    Preserves ||y_hat||_2 == ||x||_2 to machine precision while allowing
    state-dependent angular velocities.
    """
    Q = cayley_orthogonal_matrix(raw_matrix, alpha=alpha)
    return Q @ x


class SkipNormLayer(eqx.Module):
    """Equinox wrapper applying SkipNorm around an arbitrary callable block f."""

    fn: Callable[[Array], Array]
    p: float = eqx.field(static=True, default=2.0)
    eps: float = eqx.field(static=True, default=1e-12)

    def __call__(self, x: Array) -> Array:
        y = self.fn(x)
        return skip_norm_fn(x, y, p=self.p, axis=-1, eps=self.eps)


class ProjectedResidualSkipNormBlock(eqx.Module):
    """Equinox Projected Residual SkipNorm block without tangent projection."""

    fn: Callable[[Array], Array]
    alpha: Array
    eps: float = eqx.field(static=True, default=1e-12)

    def __init__(self, fn: Callable[[Array], Array], alpha_init: float = 0.1, eps: float = 1e-12):
        self.fn = fn
        self.alpha = jnp.asarray(alpha_init)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        return projected_residual_skip_norm_fn(x, self.fn(x), alpha=self.alpha, axis=-1, eps=self.eps)


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


class AffineCouplingLayer(eqx.Module):
    """Single bijective affine coupling layer on R^d (RealNVP style).

    Guaranteed globally homeomorphic (continuous and globally injective with
    exact analytical inverse) for any neural conditioner network.
    """

    scale_net: eqx.nn.MLP
    shift_net: eqx.nn.MLP
    d_split: int = eqx.field(static=True)
    flip: bool = eqx.field(static=True)
    scale_bound: float = eqx.field(static=True, default=1.5)

    def __init__(
        self,
        dim: int,
        width: int,
        depth: int,
        flip: bool,
        key: PRNGKeyArray,
        scale_bound: float = 1.5,
    ):
        k1, k2 = jax.random.split(key)
        self.d_split = dim // 2
        d_cond = self.d_split if not flip else (dim - self.d_split)
        d_trans = (dim - self.d_split) if not flip else self.d_split
        self.scale_net = eqx.nn.MLP(d_cond, d_trans, width, depth, activation=jax.nn.tanh, key=k1)
        self.shift_net = eqx.nn.MLP(d_cond, d_trans, width, depth, activation=jax.nn.tanh, key=k2)
        self.flip = flip
        self.scale_bound = scale_bound

    def forward(self, x: Array) -> Array:
        if not self.flip:
            x_a, x_b = x[: self.d_split], x[self.d_split :]
            s = self.scale_bound * jnp.tanh(self.scale_net(x_a))
            t = self.shift_net(x_a)
            y_b = x_b * jnp.exp(s) + t
            return jnp.concatenate([x_a, y_b], axis=-1)
        else:
            x_a, x_b = x[: self.d_split], x[self.d_split :]
            s = self.scale_bound * jnp.tanh(self.scale_net(x_b))
            t = self.shift_net(x_b)
            y_a = x_a * jnp.exp(s) + t
            return jnp.concatenate([y_a, x_b], axis=-1)

    def inverse(self, y: Array) -> Array:
        if not self.flip:
            y_a, y_b = y[: self.d_split], y[self.d_split :]
            s = self.scale_bound * jnp.tanh(self.scale_net(y_a))
            t = self.shift_net(y_a)
            x_b = (y_b - t) * jnp.exp(-s)
            return jnp.concatenate([y_a, x_b], axis=-1)
        else:
            y_a, y_b = y[: self.d_split], y[self.d_split :]
            s = self.scale_bound * jnp.tanh(self.scale_net(y_b))
            t = self.shift_net(y_b)
            x_a = (y_a - t) * jnp.exp(-s)
            return jnp.concatenate([x_a, y_b], axis=-1)


class BijectiveCouplingMap(eqx.Module):
    """Stack of alternating AffineCouplingLayers forming a guaranteed global
    diffeomorphism on R^d (satisfying the global injectivity hypothesis of Theorem 1).
    """

    layers: tuple[AffineCouplingLayer, ...]

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        width: int = 32,
        depth: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_layers)
        self.layers = tuple(
            AffineCouplingLayer(dim, width, depth, flip=(i % 2 == 1), key=keys[i])
            for i in range(num_layers)
        )

    def __call__(self, x: Array) -> Array:
        z = x
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def inverse(self, z: Array) -> Array:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
