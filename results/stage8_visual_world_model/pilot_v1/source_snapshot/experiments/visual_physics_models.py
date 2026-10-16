"""Small image-history world models for the visual physics pilot.

Only the latent transition differs between variants. The encoder and decoder
have identical initialization for a given key, irrespective of transition kind.
All methods operate on one example; use ``jax.vmap`` to batch them.

``rotation`` is invertible in latent space. This does *not* make the visual
encoder, decoder, or complete world model information preserving. No transition
is claimed to conserve physical energy merely because it preserves latent norm.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from skipnorm.jax_layers import (
    cayley_orthogonal_matrix,
    geodesic_skip_norm_fn,
    projected_residual_skip_norm_fn,
)
from skipnorm.rotation_coupling import ConditionalRotationCoupling, RotationCouplingFlow


MODEL_KINDS = (
    "residual", "prenorm", "scale", "projected", "geodesic", "rotation",
    "linear_orthogonal", "hamiltonian",
)


def rotation_pair_schedule(dim: int) -> tuple[tuple[int, ...], ...]:
    """Four deterministic coupling permutations with a connected pair graph.

    Split the two alternating matchings of a coordinate path (cycle for even
    dimension) over four stages. Pad short matchings with disjoint pairs. The
    union includes every path edge, so no latent coordinate is permanently
    frozen. This schedule is independent of training data and random seed.
    """
    if dim < 2:
        raise ValueError("rotation dimension must be at least 2")
    pair_count = math.ceil(dim / 4)
    matchings = [list(zip(range(0, dim - 1, 2), range(1, dim, 2))),
                 list(zip(range(1, dim - 1, 2), range(2, dim, 2)))]
    if dim % 2 == 0:
        matchings[1].append((dim - 1, 0))
    permutations = []
    for matching in matchings:
        for start in (0, pair_count):
            pairs = list(matching[start:start + pair_count])
            used = {coordinate for pair in pairs for coordinate in pair}
            free = [coordinate for coordinate in range(dim) if coordinate not in used]
            missing = pair_count - len(pairs)
            pairs.extend(zip(free[:2 * missing:2], free[1:2 * missing:2]))
            rotated = tuple(coordinate for pair in pairs for coordinate in pair)
            conditioned = tuple(coordinate for coordinate in range(dim) if coordinate not in rotated)
            permutations.append(conditioned + rotated)
    return tuple(permutations)


def _float32(tree):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, tree)


def _small_final(network: eqx.nn.MLP, scale: float = 0.1) -> eqx.nn.MLP:
    """Initialize updates near identity without freezing earlier layers."""
    return eqx.tree_at(
        lambda m: (m.layers[-1].weight, m.layers[-1].bias), network,
        (scale * network.layers[-1].weight,
         scale * network.layers[-1].bias))


class DenseStack(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]

    def __init__(self, sizes: tuple[int, ...], *, key: jax.Array,
                 final_bias: float | None = None):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(_float32(eqx.nn.Linear(a, b, key=k))
                            for a, b, k in zip(sizes[:-1], sizes[1:], keys))
        if final_bias is not None:
            self.layers = (*self.layers[:-1], eqx.tree_at(
                lambda m: m.bias, self.layers[-1],
                jnp.full((sizes[-1],), final_bias, dtype=jnp.float32)))

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)


class ResidualTransition(eqx.Module):
    body: eqx.nn.MLP
    kind: str = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, kind: str, dim: int, width: int, dt: float, *, key):
        self.kind, self.dt = kind, float(dt)
        self.body = _float32(_small_final(eqx.nn.MLP(
            dim + (kind == "scale"), dim, width, depth=2,
            activation=jax.nn.tanh, key=key)))

    def __call__(self, z: jax.Array) -> jax.Array:
        inputs = z
        if self.kind in ("prenorm", "scale"):
            rms = jnp.sqrt(jnp.mean(z * z) + jnp.asarray(1e-6, z.dtype))
            inputs = z / rms
            if self.kind == "scale":
                inputs = jnp.concatenate((inputs, jnp.log(rms)[None]))
        update = self.body(inputs)
        if self.kind == "projected":
            return projected_residual_skip_norm_fn(z, update, alpha=self.dt)
        if self.kind == "geodesic":
            return geodesic_skip_norm_fn(z, update, alpha=self.dt)
        return z + self.dt * update


class OrthogonalTransition(eqx.Module):
    raw_matrix: jax.Array
    dt: float = eqx.field(static=True)

    def __init__(self, dim: int, dt: float, *, key):
        self.raw_matrix = 0.1 * jax.random.normal(key, (dim, dim), dtype=jnp.float32)
        self.dt = float(dt)

    def __call__(self, z: jax.Array) -> jax.Array:
        return cayley_orthogonal_matrix(self.raw_matrix, alpha=self.dt) @ z

    def inverse(self, z: jax.Array) -> jax.Array:
        return cayley_orthogonal_matrix(self.raw_matrix, alpha=-self.dt) @ z


class SeparableHamiltonianTransition(eqx.Module):
    """Learn T(p)+V(q); one explicit symmetric kick-drift-kick step.

    Leapfrog is symplectic for these smooth separable functions, but does not
    exactly conserve the learned energy at finite step size, nor latent norm.
    """

    kinetic: eqx.nn.MLP
    potential: eqx.nn.MLP
    dt: float = eqx.field(static=True)
    half_dim: int = eqx.field(static=True)

    def __init__(self, dim: int, width: int, dt: float, *, key):
        if dim % 2:
            raise ValueError("Hamiltonian latent dimension must be even")
        self.dt, self.half_dim = float(dt), dim // 2
        k1, k2 = jax.random.split(key)
        # Approximately match the two-hidden-layer residual transition count.
        target = dim * width + width + width * width + width + width * dim + dim
        candidates = range(1, max(2, 2 * width))
        hwidth = min(candidates, key=lambda w: abs(
            2 * (self.half_dim * w + w + w * w + w + w + 1) - target))
        self.kinetic, self.potential = tuple(
            _float32(_small_final(eqx.nn.MLP(
                self.half_dim, "scalar", hwidth, depth=2,
                activation=jax.nn.tanh, key=k))) for k in (k1, k2))

    def energy(self, z: jax.Array) -> jax.Array:
        q, p = z[:self.half_dim], z[self.half_dim:]
        return self.kinetic(p) + self.potential(q)

    def _step(self, z: jax.Array, dt: float) -> jax.Array:
        q, p = z[:self.half_dim], z[self.half_dim:]
        p_half = p - 0.5 * dt * jax.grad(self.potential)(q)
        q_next = q + dt * jax.grad(self.kinetic)(p_half)
        p_next = p_half - 0.5 * dt * jax.grad(self.potential)(q_next)
        return jnp.concatenate((q_next, p_next))

    def __call__(self, z: jax.Array) -> jax.Array:
        return self._step(z, self.dt)

    def inverse(self, z: jax.Array) -> jax.Array:
        return self._step(z, -self.dt)


class VisualWorldModel(eqx.Module):
    encoder: DenseStack
    decoder: DenseStack
    dynamics: ResidualTransition | RotationCouplingFlow | OrthogonalTransition | SeparableHamiltonianTransition
    kind: str = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    image_shape: tuple[int, int] = eqx.field(static=True)
    history: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, kind: str, latent_dim: int = 16, width: int = 64,
                 image_shape: tuple[int, int] = (24, 48), history: int = 3,
                 dt: float = 0.08, *, key: jax.Array):
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
        if latent_dim < 2 or width < 1 or history < 1:
            raise ValueError("latent_dim >= 2, width >= 1, and history >= 1 required")
        if len(image_shape) != 2 or min(image_shape) < 1:
            raise ValueError("image_shape must contain two positive sizes")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.kind, self.latent_dim = kind, int(latent_dim)
        self.image_shape, self.history, self.dt = tuple(image_shape), int(history), float(dt)
        enc_key, dec_key, dyn_key = jax.random.split(key, 3)
        pixels = math.prod(image_shape)
        self.encoder = DenseStack((history * pixels, 128, 64, latent_dim), key=enc_key)
        self.decoder = DenseStack((latent_dim, 64, 128, pixels), key=dec_key, final_bias=-2.0)
        if kind == "rotation":
            stages = 4
            condition_dim = latent_dim - 2 * math.ceil(latent_dim / 4)
            # Each of four substeps has alpha=dt/4. No SVD projection/certificate
            # is imposed during training. Norm and inverse hold independently.
            flow = RotationCouplingFlow(
                dim=latent_dim, num_stages=stages, width=max(1, width // 2), depth=2,
                budget=stages * math.asinh(dt / (2 * stages)), delta=1.0,
                normalize=True, project=False, condition_dim=condition_dim, key=dyn_key)
            # A small random stack can leave coordinates permanently untouched.
            # Use a fixed connected schedule instead of relying on seed luck.
            layers = tuple(ConditionalRotationCoupling(
                latent_dim, width=max(1, width // 2), depth=2, alpha=flow.alpha,
                delta=1.0, normalize=True, project=False, condition_dim=condition_dim,
                permutation=permutation, key=layer_key)
                for permutation, layer_key in zip(rotation_pair_schedule(latent_dim),
                                                  jax.random.split(dyn_key, stages)))
            self.dynamics = _float32(eqx.tree_at(
                lambda m: m.layers, flow, tuple(eqx.tree_at(
                    lambda layer: layer.conditioner, layer, _small_final(layer.conditioner))
                    for layer in layers)))
        elif kind == "linear_orthogonal":
            self.dynamics = OrthogonalTransition(latent_dim, dt, key=dyn_key)
        elif kind == "hamiltonian":
            self.dynamics = SeparableHamiltonianTransition(latent_dim, width, dt, key=dyn_key)
        else:
            self.dynamics = ResidualTransition(kind, latent_dim, width, dt, key=dyn_key)

    def encode(self, history: jax.Array) -> jax.Array:
        expected = (self.history, *self.image_shape)
        if history.shape != expected:
            raise ValueError(f"expected frame history {expected}, got {history.shape}")
        return self.encoder(jnp.asarray(history, jnp.float32).reshape(-1))

    def decode(self, z: jax.Array) -> jax.Array:
        return jax.nn.sigmoid(self.decoder(z)).reshape(self.image_shape)

    def step(self, z: jax.Array) -> jax.Array:
        return self.dynamics(z)

    def rollout(self, history: jax.Array, steps: int) -> tuple[jax.Array, jax.Array]:
        """Return future states and frames, excluding the observed initial state.

        ``steps`` is a nonnegative Python integer/static argument under JIT.
        No future image enters the encoder or the open-loop transition.
        """
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        initial = self.encode(history)

        def advance(z, _):
            next_z = self.step(z)
            return next_z, next_z

        _, latents = jax.lax.scan(advance, initial, xs=None, length=steps)
        return latents, jax.vmap(self.decode)(latents)


def parameter_counts(model: VisualWorldModel) -> dict[str, int]:
    def count(tree):
        return sum(int(x.size) for x in jax.tree_util.tree_leaves(tree)
                   if eqx.is_inexact_array(x))

    counts = {name: count(getattr(model, name)) for name in ("encoder", "decoder", "dynamics")}
    return {**counts, "total": sum(counts.values())}


def normalization_details(model: VisualWorldModel) -> dict[str, object]:
    """Host-side report metadata; these are architectural, not empirical claims."""
    return {
        "kind": model.kind,
        "dt": model.dt,
        "latent_dim": model.latent_dim,
        "preserves_latent_norm": model.kind in ("projected", "geodesic", "rotation", "linear_orthogonal"),
        "explicit_latent_inverse": model.kind in ("rotation", "linear_orthogonal", "hamiltonian"),
        "branch_rms_normalized": model.kind in ("prenorm", "scale"),
        "log_rms_side_channel": model.kind == "scale",
        "normalization_epsilon": 1e-6 if model.kind in ("prenorm", "scale") else None,
        "rotation_weight_projection": False if model.kind == "rotation" else None,
        "rotation_pair_schedule": "connected_path_four_stages" if model.kind == "rotation" else None,
        "physical_energy_conservation_claimed": False,
        "invertible_visual_encoder_claimed": False,
        "parameters": parameter_counts(model),
    }
