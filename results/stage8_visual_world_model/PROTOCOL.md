# Visual physics pilot — prospective protocol

This exploratory experiment is separate from the manuscript and does not establish
a publication claim. Its question is whether existing SkipNorm transitions help a
small model learn sustained physical motion from images. No new theorem is needed.

## System and observations

Four unit masses move vertically in fixed image columns. They have unequal onsite
quadratic potentials, a common quartic potential, and nearest-neighbor quadratic
couplings. Velocity Verlet uses 16 internal steps per 0.08-second observation.
The renderer draws four identical Gaussian spots at 24 × 48 resolution. There are
no velocity labels, physical state inputs, object discovery, appearance variation,
actions, damping, or literal occluding objects in this first pilot. The observation
gap is implemented by withholding all frames after the initial three.

Whole trajectories are independently sampled: 256 training, 64 validation, 64 ID
test, 64 higher-amplitude test. Training trajectories contain 40 observations.
Evaluation trajectories contain 200. Initial per-mass phase-space radii are drawn
from [0.25, 0.65] for ID and [0.8, 1.0] for OOD, with independent random phases.
The dataset records simulation convergence and clipping diagnostics and a hash.

## Training

All models use the same learned three-frame image encoder (latent dimension 16)
and image decoder. For each initialization seed a shared visual warm-start learns
to reconstruct the last observed frame. Those visual weights initialize every
model. Subsequent training updates the full model using image targets only.

The locked pilot uses 4,000 visual warm-start updates followed by 8,000 updates per
transition, batch size 32, Adam initial learning rate 0.001 with cosine decay during
joint training, gradient clipping at norm 1, an eight-step prediction loss plus
0.5 times current-image reconstruction loss. Pixel loss balances bright foreground
and dark background continuously using target image intensity. There is no state,
energy, velocity, or latent consistency supervision.

Training windows and initialization seeds are paired. Seeds are 0, 1, 2. Models:
raw residual, pre-normalized residual, pre-normalized residual with explicit
log-scale input, projected residual SkipNorm, tangent-retracted geodesic SkipNorm,
invertible rotation coupling, learned linear orthogonal map, and a separable
learned Hamiltonian with leapfrog integration. Transition parameter counts are
reported (linear orthogonal is deliberately much smaller).

Checkpoints are selected by the minimum eight-step ID validation image loss,
evaluated every 200 updates on fixed windows from the first 40 validation frames.
Neither long-horizon nor amplitude-OOD results select checkpoints. Equal optimizer
settings define a controlled pilot; they do not establish optimal tuning for every
architecture. The 200-update smoke run only checks execution and is not included
in scientific comparisons.

## Evaluation and interpretation

The model receives the first three frames, then forecasts 197 frames (15.76s)
without access to future images. The trained forecast horizon is eight frames
(0.64s), not the full 40-frame training clip. Primary pilot metrics are decoded
position error at 8/20/50/100/197 frames and full-forecast position RMSE. Physical
positions are extracted from generated images in the four known columns; true
states enter only evaluation and the fixed training-derived normalization scale.

Complementary measurements: finite-difference velocity error, pixel error,
foreground mass, vertical spread, localized-spot fraction, initial reconstruction
floor, runtime, parameter count, and latent norm drift. A stable latent norm alone
is never counted as physical prediction success.

Two validity horizons are recorded: first normalized position RMSE above 0.3
(training-standard-deviation units), and first absolute RMS position error above
one pixel or failed spot localization. The latter requires every spot to have
vertical spread ≤1.8 pixels, brightness mass 0.4–2.5 times a true spot, and peak
intensity ≥0.15. No recovery after the first failure is credited. Report zero
horizons honestly when perception itself fails. Persistence and constant-velocity
controls use only observed images.

For the initial pilot, evaluate the first 32 preselected trajectories in each test
split. Report each seed and descriptive mean/SD, not significance claims. All
test inspection is exploratory; promising findings require a fresh, larger
confirmation with stronger visual architectures and individually tuned baselines.

Norm preservation, information preservation of a latent transition, physical
energy conservation, and correct physical prediction are distinct properties.
Fixed latent radius is a candidate bias for this conservative setup and is not a
claim about general forced, dissipative, or partially observed physical systems.
