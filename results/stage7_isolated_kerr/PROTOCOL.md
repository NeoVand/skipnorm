# Isolated Kerr phase identification: fixed fresh-data experiment

Declared on 15 October 2026 before fitting any Stage 7 model. This is a mechanism
experiment motivated by earlier results, using independently generated data and
an exact target. No prior dataset is loaded. No choices will be retuned on these
test states, and all final models will be reported.

## Target and states

Use 16 complex coordinates, represented as all real followed by all imaginary
parts. The exact step is z_j -> exp(+i*0.04*|z_j|^2)*z_j. Its powers are evaluated
analytically, not with a numerical integrator. All predictions here are pure
phase dynamics, not full DNLS propagation.

Generate 1,024 training, 256 validation, 512 in-distribution test and 512
out-of-distribution test states. Separate NumPy generator seeds are 817001,
817002, 817003 and 817004, respectively. Total power is uniform on [0.5,3.4]
for the first three splits and [3.4,4.0] for the fourth. Exactly half of each
split consists of independent complex Gaussian vectors normalized to that
power. The other half uses Dirichlet(0.2) intensities scaled to total power and
independent uniform phases. Diffuse states occupy the first half, concentrated
states the second half. Jacobians use ID test indices 0..15 and 256..271,
fixed before fitting.

## Models and training

Compare three initialization seeds, 0,1,2, for each of four models (12 runs):

1. Four centered PerspectivePhase layers, complex dimension 16, width 64,
   delta=1, phase time 2*sinh(0.5/4), and no coordinate mixers. Project W to
   spectral norm <=1 and a to [-4,4] initially and after every update.
2. The same model and identical projected initialization, without subsequent
   projection. It has an actual-weight certificate but no enforced B=0.5 cap.
3. One learned scalar gamma=2*tanh(raw), initialized at raw=0, with exact
   Kerr phase +0.04*gamma*|z_j|^2. This control knows the nonlinear law's form.
4. A shared scalar phase theta(s)=sinh(0.5)*g(log(1+s)), s=|z_j|^2, using a
   1->32->32->1 MLP with tanh hidden activations and a linear output. Exact SVD
   projection caps every matrix's spectral norm at 1 initially and after each
   update; biases are unrestricted. Its global log-distortion certificate is
   asinh(sinh(0.5)*product_of_weight_norms) <= 0.5. It is norm preserving,
   symplectic, U(1) equivariant and explicitly invertible, but uses a different
   coordinate restriction and parameter count from the perspective models.

The fourth model was added before any fitting or evaluation on generated data,
after a theoretical restriction of the perspective-coordinate class was
identified. Only synthetic finite-difference/inverse checks had been run.
No model is being selected on earlier Stage 7 test results; none existed.

Use final parameters after 900 Adam updates, batch size 64, learning rate 0.003
with cosine decay to 5%. Neural initialization uses JAX seed 12000+89*seed;
paired minibatch streams use NumPy seed 52000+seed. Validation is recorded at
initialization, update 1, and every 100 updates; it neither selects checkpoints
nor changes training. Sample training states uniformly with replacement.

The loss is raw state MSE averaged over the exact first eight powers of the
learned phase map. Since actions remain constant, angle multiplication is
exact. The equivalent stable expression is mean[2*|z_j|^2*sin^2(k*(theta_model-
theta_target)/2)], k=1..8, including averaging over complex coordinates. It
equals MSE over the 32 real coordinates.

## Evaluation and interpretation

For both test splits, report one-step MSE, cumulative MSE through 200 steps,
and endpoint MSE at step 200, with raw and independently phase-aligned versions.
Report phase-increment errors, including amplitude-weighted and wrapped errors.
For the fixed 32 ID states, compare the full Cartesian Jacobian with the exact
target at steps 1 and 200 using relative Frobenius error. Also retain singular
values and actual-weight certificates. A smaller condition number is not an
accuracy measure. The unbounded cubic scalar control has no global certificate;
its norm-ball certificate uses the predeclared test power cap 4.

Save the dataset and its hash, source snapshots/configuration hashes, final
checkpoints, training curves, learned angles, endpoint predictions, per-state
errors and Jacobians. Summaries use all three seeds and sample SD (ddof=1).
An initial numerical check compares the exact target Jacobian with finite
differences and verifies the analytical repeated-phase implementation against
repeated application; it performs no fitting.

This isolates approximation and optimization of the potential in a fixed basis.
It cannot establish impossibility, generic forecasting superiority, or novelty
of Kerr splitting. A successful scalar control is expected because its model
contains the exact target. OOD results assess the specified modest power shift,
not unrestricted extrapolation.
