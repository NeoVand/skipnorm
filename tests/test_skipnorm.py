"""Unit and property tests for SkipNorm layers and mathematical invariants."""

import torch
import torch.nn as nn
from skipnorm import AdaptiveSkipNorm, GeodesicSkipNorm, SkipNorm, SkipNormResBlock, SkipNormRNN, skip_norm


def test_exact_norm_conservation_across_dimensions():
    torch.manual_seed(0)
    x = torch.randn(16, 128) * 7.3
    block = nn.Sequential(nn.Linear(128, 5), nn.LeakyReLU(0.2))
    sn = SkipNorm(block, p=2.0, dim=-1)
    y_hat = sn(x)
    assert y_hat.shape == (16, 5)
    norm_x = torch.linalg.vector_norm(x, ord=2, dim=-1)
    norm_yhat = torch.linalg.vector_norm(y_hat, ord=2, dim=-1)
    assert torch.allclose(norm_x, norm_yhat, rtol=1e-5, atol=1e-6)


def test_radial_jacobian_isometry_theorem1():
    torch.manual_seed(1)
    dims = [32, 16, 4, 16, 32]
    layers = [
        SkipNorm(nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU(0.2)))
        for i in range(len(dims) - 1)
    ]
    net = nn.Sequential(*layers).double()
    x0 = torch.randn(1, 32, dtype=torch.float64) * 4.2
    xL = net(x0)
    u0 = (x0 / torch.linalg.vector_norm(x0)).squeeze(0)
    uL = (xL / torch.linalg.vector_norm(xL)).squeeze(0)
    J = torch.autograd.functional.jacobian(lambda inp: net(inp).squeeze(0), x0.squeeze(0))
    grad_radial = J.T @ uL
    assert torch.allclose(grad_radial, u0, rtol=1e-6, atol=1e-6)


def test_scale_equivariance_bias_free():
    torch.manual_seed(2)
    block = nn.Sequential(nn.Linear(64, 16, bias=False), nn.LeakyReLU(0.2))
    sn = SkipNorm(block, p=2.0, dim=-1).double()
    x = torch.randn(8, 64, dtype=torch.float64)
    for c in [0.01, 0.5, 13.7, 100.0]:
        out_cx = sn(c * x)
        c_out_x = c * sn(x)
        assert torch.allclose(out_cx, c_out_x, rtol=1e-6, atol=1e-6)


def test_geodesic_skipnorm_long_rollout():
    torch.manual_seed(3)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
    geo = GeodesicSkipNorm(mlp, step_scale=0.2).double()
    x0 = torch.randn(4, 16, dtype=torch.float64) * 5.5
    norm0 = torch.linalg.vector_norm(x0, dim=-1)
    x = x0.clone()
    with torch.no_grad():
        for _ in range(500):
            x = geo(x)
    norm_final = torch.linalg.vector_norm(x, dim=-1)
    assert torch.allclose(norm0, norm_final, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_exact_norm_conservation_across_dimensions()
    test_radial_jacobian_isometry_theorem1()
    test_scale_equivariance_bias_free()
    test_geodesic_skipnorm_long_rollout()
    print("All SkipNorm mathematical property tests passed!")

