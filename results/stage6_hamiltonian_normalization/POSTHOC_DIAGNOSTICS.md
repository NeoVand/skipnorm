# Additional diagnostics, declared during the pilot

After training began, the mathematical audit established that the corrected
Hamiltonian phase and the omitted-common-phase control, at identical parameters,
differ only by an overall complex phase. Unitary mixing and action-only updates
preserve this relation under composition and repeated evolution. Consequently
all intensities and U(1)-invariant observables are identical at matched weights.
The difference in full canonical symplecticity is real, but cannot be described
as changing the reduced/projective dynamics at matched parameters.

Therefore add the following diagnostics without retraining or selecting models:

* Cumulative phase-aligned state MSE, with one optimal common phase per trajectory
  and time, alongside the original unaligned MSE. This is a different target
  metric and does not replace the prespecified one.
* Cumulative intensity MSE, averaged over sites, trajectories and predicted times.
* For each trained corrected bounded network, create an otherwise identical
  omitted-common-phase network and check intensity equality, phase-aligned state
  equality, and physical energy equality through 249 steps.
* Compute the ratio |||psi|² psi||/||Delta psi|| on the existing train/test states
  to qualify the earlier inference from the strong fitted linear baseline.

The post-hoc script records its own source hash and the immutable pilot config
hash. All original metrics remain unchanged. These checks are not fresh-data
evidence and will not be used to select a candidate or tune hyperparameters.
