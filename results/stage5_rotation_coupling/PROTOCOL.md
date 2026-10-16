# Prospective rotation-coupling screen

This protocol was written before the 900-update pilot. A two-update smoke run
checked execution only; it is retained separately and is not experimental evidence.

## Questions

1. Do the implemented layers satisfy their norm, inverse, volume, and global
   state-Jacobian guarantees at arbitrary and trained parameters?
2. Does enforcing a finite distortion budget retain useful predictive accuracy
   on a real norm- and volume-preserving dynamical system?
3. What changes when spectral constraints or invariant-normalized conditioning
   are removed?

## Fixed design

Use the existing locked DNLS split: 28 training, 8 validation, 12 test trajectories,
32 real coordinates, dt=0.04. Use float64, 900 Adam updates, batch64, horizon8,
learning rate0.003 with cosine decay to5%, seeds0,1,2. The batch sampling matches
Stage3. Evaluate the final model; do not select checkpoints on validation or test.

Each new propagator has8 rotation-coupling stages, width32, two hidden layers,
delta1, and a learned linear Cayley rotation initialized to identity. Each stage
rotates8 disjoint pairs, with the other16 coordinates and augmented constant as
conditioner inputs. Fixed permutations are identical across seeds. All three
variants start from the same spectrally projected parameters. They are:

- normalized conditioner + exact SVD spectral projection after every update;
- normalized conditioner, no subsequent spectral projection;
- raw conditioner, no subsequent spectral projection.

The prescribed whole-step log-distortion budget is B=0.5 for the first variant.
The others retain the same angle multiplier, but do not retain this budget.
The vector-output conditioner uses a joint Euclidean Lipschitz bound; each
linear weight, including the output weight, is constrained in operator norm.

Compare against Geodesic SkipNorm, projected residual, and linear orthogonal
models. Reuse existing checkpoints only after checking dataset, protocol and
source hashes. They used the identical900-update training budget and seeds;
reevaluate all checkpoints using the same new diagnostics and timing procedure.
The new model has a comparable but not identical parameter count; report counts.

## Measurements

Cumulative rollout MSE over1,20,50,200,249 transitions; maximum relative power
error; energy drift; warmed rollout time. For the new family, measure forward-
inverse recovery over one and249 steps, actual spectral-weight certificates,
and Jacobians on states spanning radii1e-9 to1e6. For all models, measure sampled
one-step spectra, repeated-step spectra, origin regularity, global-phase
commutation error and canonical symplectic defect. The last two are diagnostics,
not claimed guarantees. Report sample SD across optimization seeds.

## Interpretation and stopping rule

This is an exploratory screen on one existing dataset split, not independent
new-dataset evidence or a final publication result. No method is declared better
from conservation alone. A successful follow-up needs the promised algebraic
properties, competitive forecasting and an identifiable benefit of the explicit
conditioning constraint. Weak prediction or runtime regressions are retained.
Do not tune architecture or budget against test results; record any subsequent
experiment as a separately motivated protocol.

The bound applies to one learned time step. An n-step rollout has worst-case
budget nB. Norm and volume preservation do not guarantee physical energy,
canonical symplecticity, phase equivariance, arbitrary expressivity, or useful
invariant discovery. The layers preserve every radial probability density.
