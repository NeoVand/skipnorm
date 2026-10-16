"""Perspective Hamiltonian phase maps and complex-unitary mixing.

Real inputs are ordered as all real parts followed by all imaginary parts.
The corrected phase map is smooth, canonical symplectic, norm-preserving and
U(1)-equivariant, with an explicit inverse and a global singular-value bound.
It does not, by itself, conserve a separately specified physical Hamiltonian.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _spectral_projection(weight: jax.Array) -> jax.Array:
    u, s, vh = jnp.linalg.svd(weight, full_matrices=False)
    return (u * jnp.minimum(s, 1.)[None, :]) @ vh


def _check_vector(x: jax.Array, dim: int) -> None:
    if x.ndim != 1 or x.shape[0] != 2 * dim:
        raise ValueError(f"expected a real vector of shape ({2 * dim},); use vmap for batches")


class PerspectivePhase(eqx.Module):
    """Exact phase flow of K(I)=rho*h(I/rho), rho=delta+sum(I).

    ``dim`` counts complex coordinates.  I_j=(q_j**2+p_j**2)/2.  With
    h(p)=sum_j a_j[softplus(w_j.p+b_j)-softplus(b_j)-sigmoid(b_j)*w_j.p],
    the Hessian of h has operator norm at most
    L=(1/4)||W||_2**2*max_j|a_j|.  The corrected frequencies are
    grad(h)+(h-p.grad(h))*1 = grad_I K.  The flow is z*exp(-i*time*grad_I K).

    Its full Cartesian singular values lie in [exp(-beta), exp(beta)] for
    beta=asinh(abs(time)*L/2).  Centering makes the phase map's derivative at
    the origin the identity, so a separate unitary map is needed for general
    linear dynamics.  ``omit_common_phase`` is a deliberate nonsymplectic
    control; it retains norm, volume, equivariance and its inverse.  This API
    does not expose the corrected map's certificate for that control.

    Projection returns a new module: clip singular values of W to one and a
    to [-4,4], leaving b unchanged.  Call it after optimizer updates to keep
    L<=1.  No SVD or Hessian evaluation occurs in the forward computation.
    """

    W: jax.Array
    b: jax.Array
    a: jax.Array
    dim: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    delta: float = eqx.field(static=True)
    time: float = eqx.field(static=True)
    omit_common_phase: bool = eqx.field(static=True)

    def __init__(self, dim: int = 16, width: int = 32, *, delta: float = 1.,
                 time: float = .1, omit_common_phase: bool = False,
                 project: bool = True, key: jax.Array):
        if dim < 1 or width < 1:
            raise ValueError("dim and width must be positive")
        if not math.isfinite(delta) or delta <= 0 or not math.isfinite(time):
            raise ValueError("delta must be finite and positive, and time finite")
        self.dim, self.width = dim, width
        self.delta, self.time = float(delta), float(time)
        self.omit_common_phase = bool(omit_common_phase)
        w_key, b_key, a_key = jax.random.split(key, 3)
        weight = jax.random.normal(w_key, (width, dim)) / math.sqrt(dim)
        self.W = _spectral_projection(weight) if project else weight
        self.b = .05 * jax.random.normal(b_key, (width,))
        self.a = .1 * jax.random.normal(a_key, (width,))
        if project:
            self.a = jnp.clip(self.a, -4., 4.)

    def h_and_gradient(self, p: jax.Array) -> tuple[jax.Array, jax.Array]:
        wp = self.W @ p
        base_sigmoid = jax.nn.sigmoid(self.b)
        centered = jax.nn.softplus(wp + self.b) - jax.nn.softplus(self.b) - base_sigmoid * wp
        h = jnp.dot(self.a, centered)
        gradient = self.W.T @ (self.a * (jax.nn.sigmoid(wp + self.b) - base_sigmoid))
        return h, gradient

    def hamiltonian(self, actions: jax.Array) -> jax.Array:
        """Scalar perspective Hamiltonian, defined on nonnegative actions."""
        rho = self.delta + jnp.sum(actions)
        h, _ = self.h_and_gradient(actions / rho)
        return rho * h

    def phase_frequencies(self, actions: jax.Array) -> jax.Array:
        rho = self.delta + jnp.sum(actions)
        p = actions / rho
        h, gradient = self.h_and_gradient(p)
        if self.omit_common_phase:
            return gradient
        return gradient + h - jnp.dot(p, gradient)

    def _apply(self, x: jax.Array, sign: float) -> jax.Array:
        _check_vector(x, self.dim)
        q, p = x[:self.dim], x[self.dim:]
        actions = .5 * (q*q + p*p)
        angle = -sign * self.time * self.phase_frequencies(actions)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        return jnp.concatenate((cosine*q - sine*p, sine*q + cosine*p))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._apply(x, 1.)

    def step(self, x: jax.Array) -> jax.Array:
        return self(x)

    def inverse(self, y: jax.Array) -> jax.Array:
        return self._apply(y, -1.)

    def project_lipschitz(self) -> PerspectivePhase:
        return eqx.tree_at(lambda m: (m.W, m.a), self,
                           (_spectral_projection(self.W), jnp.clip(self.a, -4., 4.)))

    def hessian_bound(self) -> float:
        """Host-side actual-parameter upper bound, outside JIT/gradient traces."""
        norm = float(np.linalg.svd(np.asarray(self.W), compute_uv=False)[0])
        return .25 * norm**2 * float(np.max(np.abs(np.asarray(self.a))))

    def certified_log_bound(self) -> float | None:
        if self.omit_common_phase:
            return None
        return math.asinh(abs(self.time) * self.hessian_bound() / 2.)


class UnitaryMix(eqx.Module):
    """State-independent complex-unitary Cayley mixing on real (q,p) vectors.

    The realification is orthogonal and canonical symplectic, and commutes with
    every global complex phase.  The raw matrix has redundant trainable entries;
    parameter counts should include them.  ``identity=True`` initializes zeros.
    """

    raw_real: jax.Array
    raw_imag: jax.Array
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int = 16, *, scale: float = .02, identity: bool = False,
                 key: jax.Array):
        if dim < 1 or not math.isfinite(scale) or scale < 0:
            raise ValueError("dim must be positive and scale finite and nonnegative")
        self.dim = dim
        real_key, imag_key = jax.random.split(key)
        self.raw_real = jnp.zeros((dim, dim)) if identity else scale * jax.random.normal(real_key, (dim, dim))
        self.raw_imag = jnp.zeros((dim, dim)) if identity else scale * jax.random.normal(imag_key, (dim, dim))

    def matrix(self) -> jax.Array:
        raw = self.raw_real + 1j*self.raw_imag
        skew = .5*(raw - raw.conj().T)
        eye = jnp.eye(self.dim, dtype=raw.dtype)
        return jnp.linalg.solve(eye + skew, eye - skew)

    def _apply(self, x: jax.Array, inverse: bool) -> jax.Array:
        _check_vector(x, self.dim)
        matrix = self.matrix()
        if inverse:
            matrix = matrix.conj().T
        z = matrix @ (x[:self.dim] + 1j*x[self.dim:])
        return jnp.concatenate((z.real, z.imag))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self._apply(x, False)

    def step(self, x: jax.Array) -> jax.Array:
        return self(x)

    def inverse(self, y: jax.Array) -> jax.Array:
        return self._apply(y, True)

    def project_lipschitz(self) -> UnitaryMix:
        return self

    def certified_log_bound(self) -> float:
        return 0.
