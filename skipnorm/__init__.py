"""SkipNorm package (JAX + Equinox layers, differential geometry, and foliation metrics)."""

from .jax_layers import (
    AffineCouplingLayer,
    BijectiveCouplingMap,
    BlockGeodesicSkipNormBlock,
    GeodesicSkipNormBlock,
    ProjectedResidualSkipNormBlock,
    SkipNormLayer,
    block_geodesic_skip_norm_fn,
    cayley_orthogonal_matrix,
    geodesic_skip_norm_fn,
    projected_residual_skip_norm_fn,
    skip_norm_fn,
    state_dependent_cayley_step,
    tangent_project,
)
from .metrics import (
    compute_k_invariant_readout_r2,
    compute_monotone_r2_and_spearman,
    compute_orbit_radii,
    compute_ovr,
    compute_radial_cv,
)

__all__ = [
    "skip_norm_fn",
    "tangent_project",
    "geodesic_skip_norm_fn",
    "projected_residual_skip_norm_fn",
    "block_geodesic_skip_norm_fn",
    "cayley_orthogonal_matrix",
    "state_dependent_cayley_step",
    "SkipNormLayer",
    "GeodesicSkipNormBlock",
    "ProjectedResidualSkipNormBlock",
    "BlockGeodesicSkipNormBlock",
    "AffineCouplingLayer",
    "BijectiveCouplingMap",
    "compute_orbit_radii",
    "compute_radial_cv",
    "compute_ovr",
    "compute_monotone_r2_and_spearman",
    "compute_k_invariant_readout_r2",
]
