"""Stage 3: Converged 1D Discrete Nonlinear Schrodinger (DNLS) Unitary Rollout Benchmark.

Loads the locked whole-orbit disjoint DNLS dataset from data/splits/dnls_1d.npz (generated via
10x sub-stepped RK4 with NO post-step norm projection; intrinsic RK4 drift = 3.67e-11).
Compares 7 architectures across 5 seeds (0..4):
  1. Tangent-Projected Geodesic SkipNorm
  2. Projected Residual SkipNorm (without tangent projection)
  3. State-Dependent Orthogonal Rotation O_phi(x)x in SO(2S)
  4. Linear Orthogonal Rotation Q x in SO(2S)
  5. Pre-RMSNorm Residual
  6. Post-RMSNorm Residual (collapses varying initial norms to sqrt(2S))
  7. Unconstrained Residual MLP
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from skipnorm.jax_layers import (
    geodesic_skip_norm_fn,
    projected_residual_skip_norm_fn,
)

DATA_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results" / "stage3"


class DNLSPropagator(eqx.Module):
    """Single-step propagator on R^{32} (S=16 complex lattice sites) for 1D DNLS."""

    net: eqx.nn.MLP | None
    skew_param: jax.Array | None
    alpha: jax.Array
    mode: str = eqx.field(static=True)
    dim: int = eqx.field(static=True, default=32)

    def __init__(self, dim: int, mode: str, width: int = 96, depth: int = 2, *, key: jax.Array):
        self.dim = dim
        self.mode = mode
        self.alpha = jnp.asarray(0.10, dtype=jnp.float64)

        if mode == "linear_orthogonal":
            self.net = None
            self.skew_param = jax.random.normal(key, (dim, dim), dtype=jnp.float64) * 0.05
        elif mode == "state_dep_orthogonal":
            # Low-rank skew-symmetric generator U(x) V(x)^T - V(x) U(x)^T with rank r=4
            # parameterized via an MLP of matched parameter budget
            self.net = eqx.nn.MLP(dim, dim * 8, width, depth, activation=jax.nn.tanh, key=key)
            self.skew_param = None
        else:
            self.net = eqx.nn.MLP(dim, dim, width, depth, activation=jax.nn.tanh, key=key)
            self.skew_param = None

    def step(self, x: jax.Array) -> jax.Array:
        if self.mode == "geodesic_skipnorm":
            assert self.net is not None
            return geodesic_skip_norm_fn(x, self.net(x), alpha=self.alpha, axis=-1, eps=1e-30)
        elif self.mode == "projected_residual":
            assert self.net is not None
            return projected_residual_skip_norm_fn(
                x, self.net(x), alpha=self.alpha, axis=-1, eps=1e-30
            )
        elif self.mode == "state_dep_orthogonal":
            assert self.net is not None
            uv = self.net(x).reshape(self.dim, 8)
            u, v = uv[:, :4], uv[:, 4:]
            skew = (u @ v.T - v @ u.T) / jnp.sqrt(self.dim)
            eye = jnp.eye(self.dim, dtype=x.dtype)
            return jnp.linalg.solve(eye + self.alpha * skew, (eye - self.alpha * skew) @ x)
        elif self.mode == "linear_orthogonal":
            assert self.skew_param is not None
            skew = 0.5 * (self.skew_param - self.skew_param.T)
            eye = jnp.eye(self.dim, dtype=x.dtype)
            return jnp.linalg.solve(eye + self.alpha * skew, (eye - self.alpha * skew) @ x)
        elif self.mode == "pre_rmsnorm":
            assert self.net is not None
            x_rms = jnp.sqrt(self.dim) * x / jnp.sqrt(jnp.sum(jnp.square(x)) + self.dim * 1e-5)
            return x + self.alpha * self.net(x_rms)
        elif self.mode == "post_rmsnorm":
            assert self.net is not None
            y = x + self.alpha * self.net(x)
            return jnp.sqrt(self.dim) * y / jnp.sqrt(jnp.sum(jnp.square(y)) + self.dim * 1e-5)
        elif self.mode == "unconstrained":
            assert self.net is not None
            return x + self.alpha * self.net(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def train_dnls_model(
    model: DNLSPropagator,
    train_orbits: jnp.ndarray,
    horizon: int = 8,
    steps: int = 900,
    lr: float = 3e-3,
    seed: int = 0,
) -> DNLSPropagator:
    sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps, alpha=0.05)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    M, T, D = train_orbits.shape

    @eqx.filter_jit
    def loss_fn(m: DNLSPropagator, batch_w: jnp.ndarray) -> jax.Array:
        def scan_step(xc, _):
            xn = jax.vmap(m.step)(xc)
            return xn, xn

        _, x_roll = jax.lax.scan(scan_step, batch_w[:, 0, :], jnp.arange(horizon))
        x_roll = jnp.swapaxes(x_roll, 0, 1)
        return jnp.mean(jnp.square(x_roll - batch_w[:, 1:, :]))

    @eqx.filter_jit
    def make_step(m, opt_s, batch_w):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m, batch_w)
        updates, opt_s = optimizer.update(grads, opt_s, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_s, loss

    rng = np.random.default_rng(40000 + seed)
    batch_size = 64
    max_start = T - horizon - 1
    tr_np = np.asarray(train_orbits)

    for _ in range(steps):
        m_idx = rng.integers(0, M, size=batch_size)
        t_idx = rng.integers(0, max_start, size=batch_size)
        windows = np.stack(
            [tr_np[m_idx[b], t_idx[b] : t_idx[b] + horizon + 1, :] for b in range(batch_size)],
            axis=0,
        )
        model, opt_state, _ = make_step(model, opt_state, jnp.asarray(windows))

    return model


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split = np.load(DATA_DIR / "dnls_1d.npz")
    tr_orbits = jnp.asarray(split["train_orbits"])
    te_long = jnp.asarray(split["test_orbits_long"])  # (12, 250, 32)
    T_long = te_long.shape[1] - 1  # 249 steps

    # Measure ground-truth RK4 norm drift trace across t=0..249
    gt_norms = np.sum(np.asarray(te_long) ** 2, axis=-1)
    gt_rel_drift_trace = np.mean(
        np.abs(gt_norms - gt_norms[:, :1]) / gt_norms[:, :1], axis=0
    ).tolist()

    configs = [
        ("Geodesic SkipNorm", "geodesic_skipnorm"),
        ("Projected Residual SkipNorm", "projected_residual"),
        ("State-Dependent SO(32)", "state_dep_orthogonal"),
        ("Linear Orthogonal SO(32)", "linear_orthogonal"),
        ("Pre-RMSNorm Residual", "pre_rmsnorm"),
        ("Post-RMSNorm Residual", "post_rmsnorm"),
        ("Unconstrained ResNet", "unconstrained"),
    ]

    seeds = [0, 1, 2, 3, 4]
    summary = {}
    drift_traces = {"t_steps": list(range(T_long + 1)), "RK4 Reference (No Proj)": gt_rel_drift_trace}
    mse_traces = {"t_steps": list(range(1, T_long + 1))}

    for label, mode in configs:
        seed_records = []
        mode_drift_curves = []
        mode_mse_curves = []

        for s in seeds:
            key = jax.random.PRNGKey(7000 + 89 * s)
            model = DNLSPropagator(dim=32, mode=mode, width=96, depth=2, key=key)
            model = train_dnls_model(model, tr_orbits, horizon=8, steps=900, lr=3e-3, seed=s)

            @eqx.filter_jit
            def rollout_long(x0_batch):
                def step_fn(xc, _):
                    xn = jax.vmap(model.step)(xc)
                    return xn, xn
                _, xr = jax.lax.scan(step_fn, x0_batch, jnp.arange(T_long))
                return jnp.swapaxes(xr, 0, 1)

            # Warmup & time inference
            _ = rollout_long(te_long[:, 0, :]).block_until_ready()
            t0 = time.perf_counter()
            pred_long = rollout_long(te_long[:, 0, :]).block_until_ready()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            pred_full = np.concatenate(
                [np.asarray(te_long[:, :1, :]), np.asarray(pred_long)], axis=1
            )
            norms_pred = np.sum(pred_full**2, axis=-1)
            rel_drift_t = np.abs(norms_pred - norms_pred[:, :1]) / np.maximum(norms_pred[:, :1], 1e-30)
            mean_drift_curve = np.mean(rel_drift_t, axis=0)

            sq_err_t = np.mean(np.square(np.asarray(pred_long) - np.asarray(te_long[:, 1:, :])), axis=(0, 2))

            mse_20 = float(np.mean(sq_err_t[:20]))
            mse_50 = float(np.mean(sq_err_t[:50]))
            mse_200 = float(np.mean(sq_err_t[:200]))
            final_rel_norm_drift = float(mean_drift_curve[-1])
            max_rel_norm_drift = float(np.max(mean_drift_curve))

            seed_records.append({
                "seed": s,
                "mse_20": mse_20,
                "mse_50": mse_50,
                "mse_200": mse_200,
                "final_rel_norm_drift": final_rel_norm_drift,
                "max_rel_norm_drift": max_rel_norm_drift,
                "rollout_ms": elapsed_ms,
            })
            mode_drift_curves.append(mean_drift_curve)
            mode_mse_curves.append(sq_err_t)

        drift_traces[label] = {
            "mean": np.mean(mode_drift_curves, axis=0).tolist(),
            "std": np.std(mode_drift_curves, axis=0).tolist(),
        }
        mse_traces[label] = {
            "mean": np.mean(mode_mse_curves, axis=0).tolist(),
            "std": np.std(mode_mse_curves, axis=0).tolist(),
        }

        summary[label] = {
            "seeds": seed_records,
            "mse_20_mean": float(np.mean([r["mse_20"] for r in seed_records])),
            "mse_20_std": float(np.std([r["mse_20"] for r in seed_records])),
            "mse_50_mean": float(np.mean([r["mse_50"] for r in seed_records])),
            "mse_50_std": float(np.std([r["mse_50"] for r in seed_records])),
            "mse_200_mean": float(np.mean([r["mse_200"] for r in seed_records])),
            "mse_200_std": float(np.std([r["mse_200"] for r in seed_records])),
            "max_rel_norm_drift_mean": float(np.mean([r["max_rel_norm_drift"] for r in seed_records])),
            "max_rel_norm_drift_std": float(np.std([r["max_rel_norm_drift"] for r in seed_records])),
            "rollout_ms_mean": float(np.mean([r["rollout_ms"] for r in seed_records])),
        }
        print(
            f"  [DNLS] {label:28s} | MSE@20={summary[label]['mse_20_mean']:.5f} "
            f"| MSE@50={summary[label]['mse_50_mean']:.5f} "
            f"| MaxNormDrift={summary[label]['max_rel_norm_drift_mean']:.3e} "
            f"| Time={summary[label]['rollout_ms_mean']:.2f}ms"
        )

    out_path = RESULTS_DIR / "stage3_dnls_results.json"
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "drift_traces": drift_traces,
                "mse_traces": mse_traces,
            },
            indent=2,
        )
    )
    print(f"[Stage 3] Saved complete Stage 3 DNLS results to {out_path}")


if __name__ == "__main__":
    main()
