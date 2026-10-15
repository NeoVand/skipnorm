"""SkipNorm package (PyTorch and JAX + Equinox backends)."""

from .layers import (
    AdaptiveSkipNorm,
    GeodesicSkipNorm,
    SkipNorm,
    SkipNormResBlock,
    SkipNormRNN,
    SphericalNorm,
    skip_norm,
)

__all__ = [
    "skip_norm",
    "SkipNorm",
    "AdaptiveSkipNorm",
    "GeodesicSkipNorm",
    "SkipNormResBlock",
    "SphericalNorm",
    "SkipNormRNN",
]

try:
    from .jax_layers import (
        BlockGeodesicSkipNormBlock,
        GeodesicSkipNormBlock,
        SkipNormLayer,
        block_geodesic_skip_norm_fn,
        geodesic_skip_norm_fn,
        skip_norm_fn,
        tangent_project,
    )

    __all__.extend([
        "skip_norm_fn",
        "tangent_project",
        "geodesic_skip_norm_fn",
        "block_geodesic_skip_norm_fn",
        "SkipNormLayer",
        "GeodesicSkipNormBlock",
        "BlockGeodesicSkipNormBlock",
    ])
except ImportError:
    pass
