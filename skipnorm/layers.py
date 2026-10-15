"""Core SkipNorm layers and blocks for energy-conserving neural networks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def skip_norm(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 2.0,
    dim: int | tuple[int, ...] = -1,
    eps: float = 1e-8,
    rms: bool = False,
) -> torch.Tensor:
    """Apply the SkipNorm operation: y_hat = (y / ||y||_p) * ||x||_p.

    Conserves the L_p norm (or root-mean-p norm if rms=True) of the input `x`
    at the output `y_hat`, even when `x` and `y` have different dimensions.

    Args:
        x: Input tensor to the block (provides target norm ||x||_p).
        y: Output tensor of the block f(x) (provides direction y / ||y||_p).
        p: Order of the norm (default: 2.0 for Euclidean energy).
        dim: Dimension(s) along which to compute the norm (default: -1).
        eps: Small constant for numerical stability when ||y||_p -> 0.
        rms: If True, conserves per-dimension RMS_p norm instead of total L_p norm
            when dim sizes of x and y differ. (Identical when d_in == d_out).

    Returns:
        Normalized and rescaled tensor y_hat with ||y_hat||_p == ||x||_p.
    """
    if p == 2.0:
        norm_x = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
        norm_y = torch.linalg.vector_norm(y, ord=2, dim=dim, keepdim=True)
    else:
        norm_x = torch.linalg.vector_norm(x, ord=p, dim=dim, keepdim=True)
        norm_y = torch.linalg.vector_norm(y, ord=p, dim=dim, keepdim=True)

    if rms:
        # Adjust by (d_out / d_in)^(1/p) so per-element energy is conserved
        if isinstance(dim, int):
            d_in = x.shape[dim]
            d_out = y.shape[dim]
        else:
            d_in = 1
            d_out = 1
            for d in dim:
                d_in *= x.shape[d]
                d_out *= y.shape[d]
        if d_in != d_out:
            norm_x = norm_x * ((d_out / d_in) ** (1.0 / p))

    return (y / (norm_y + eps)) * norm_x


class SkipNorm(nn.Module):
    """Wraps any neural network module `block` with a SkipNorm connection:

        y_hat = (block(x) / ||block(x)||_p) * ||x||_p
    """

    def __init__(
        self,
        block: nn.Module,
        p: float = 2.0,
        dim: int | tuple[int, ...] = -1,
        eps: float = 1e-8,
        rms: bool = False,
    ) -> None:
        super().__init__()
        self.block = block
        self.p = p
        self.dim = dim
        self.eps = eps
        self.rms = rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return skip_norm(x, y, p=self.p, dim=self.dim, eps=self.eps, rms=self.rms)


class AdaptiveSkipNorm(nn.Module):
    """Generalized Gated/Affine SkipNorm for tasks where target norm != input norm
    (e.g., denoising, forecasting, or cross-modality translation):

        r_out = gate * ||x||_p + (1 - gate) * softplus(scale) * sqrt(d_out)
        y_hat = (block(x) / ||block(x)||_p) * r_out

    When gate -> 1 (default init: gate_logit = 2.0), behaves like pure SkipNorm.
    When gate -> 0, behaves like learned-scale SphericalNorm / RMSNorm.
    Adds only 2 scalar parameters per layer.
    """

    def __init__(
        self,
        block: nn.Module,
        p: float = 2.0,
        dim: int = -1,
        eps: float = 1e-8,
        init_gate_logit: float = 2.0,
    ) -> None:
        super().__init__()
        self.block = block
        self.p = p
        self.dim = dim
        self.eps = eps
        self.gate_logit = nn.Parameter(torch.tensor(float(init_gate_logit)))
        self.learned_scale = nn.Parameter(torch.tensor(1.0))
        self.norm_power = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        norm_x = torch.linalg.vector_norm(x, ord=self.p, dim=self.dim, keepdim=True)
        norm_y = torch.linalg.vector_norm(y, ord=self.p, dim=self.dim, keepdim=True)
        u_y = y / (norm_y + self.eps)

        gate = torch.sigmoid(self.gate_logit)
        # Power law + affine fallback allows denoising noisy norms while preserving scale cues
        r_skip = torch.pow(norm_x + self.eps, self.norm_power)
        r_const = F.softplus(self.learned_scale) * (y.shape[self.dim] ** (1.0 / self.p))
        r_out = gate * r_skip + (1.0 - gate) * r_const
        return u_y * r_out


class SkipNormResBlock(nn.Module):
    """Spherical Residual Block combining residual addition with SkipNorm:

        y_hat = ((x + sublayer(x)) / ||x + sublayer(x)||_2) * ||x||_2

    Prevents the O(sqrt(L)) activation norm explosion of Pre-LN Transformers/ResNets
    while preserving per-token / per-sample input embedding energy ||x_0||_2.
    """

    def __init__(
        self,
        sublayer: nn.Module,
        p: float = 2.0,
        dim: int = -1,
        eps: float = 1e-8,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.p = p
        self.dim = dim
        self.eps = eps
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.res_scale * self.sublayer(x)
        return skip_norm(x, y, p=self.p, dim=self.dim, eps=self.eps)


class SphericalNorm(nn.Module):
    """Projects activations onto the unit (or sqrt(d)-scaled) hypersphere,
    discarding the input norm ||x||_p. Used as an ablation baseline.
    """

    def __init__(self, p: float = 2.0, dim: int = -1, eps: float = 1e-8, scale_by_dim: bool = False) -> None:
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
        self.scale_by_dim = scale_by_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        norm_y = torch.linalg.vector_norm(y, ord=self.p, dim=self.dim, keepdim=True)
        out = y / (norm_y + self.eps)
        if self.scale_by_dim:
            out = out * (y.shape[self.dim] ** (1.0 / self.p))
        return out


class SkipNormRNN(nn.Module):
    """Unidirectional Vanilla RNN with Frame-wise SkipNorm:
    Applies a vanilla Elman RNN to sequence x (B, T, D_in) -> y (B, T, D_out),
    then applies SkipNorm at each time step t so ||y_t||_p = ||x_t||_p.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        p: float = 2.0,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.p = p

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.rnn(x, h0)
        return skip_norm(x, y, p=self.p, dim=-1)
