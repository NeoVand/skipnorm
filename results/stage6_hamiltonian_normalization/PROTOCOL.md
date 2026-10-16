# Hamiltonian normalization: second exploratory screen

Written before training the Stage 6 candidates. This screen follows the Stage 5
results on an already inspected DNLS split. It is explicitly adaptive exploratory
research, not a fresh confirmatory benchmark or a publication result.

## Question

Does deriving invariant-normalized phase updates from one scalar Hamiltonian
recover canonical symplecticity and phase equivariance while retaining the
explicit inverse, exact norm and volume preservation, and a global two-sided
state-Jacobian bound? Is its prediction useful compared with much simpler learned
structure-preserving models?

## Fixed candidates

Use the locked DNLS split, seeds 0, 1, 2, 900 Adam updates, batch 64, horizon 8,
learning rate .003 with cosine decay to 5%, float64, final checkpoints. Use the
same trajectory-window sampling as Stages 3 and 5. Record validation without
using it to select checkpoints. Do not tune the following choices on this test.

1. Four Hamiltonian phase stages, each conjugated by a learned complex unitary
   map, and one global learned complex unitary mixer. Each phase uses
   I_j=|z_j|²/2, rho=1+sum(I), p=I/rho, K=rho*h(p). Its exact phase frequency is
   grad(h)+(h-p.dot(grad(h)))*ones. The potential has 64 signed softplus features,
   with its affine Taylor part at zero subtracted. Project its feature matrix
   to spectral norm <=1 and each output coefficient to [-4,4]. This gives
   ||Hessian(h)||<=1. Set the phase time multiplier to 2*sinh(.5/4), giving
   a composition log-distortion budget B=.5. Initialize all variants identically.
2. Same corrected Hamiltonian architecture without projection after initialization.
   It retains a finite actual-weight certificate, not the prescribed B=.5.
3. Same projected architecture omitting the common phase h-p.dot(grad(h)). This
   is a mechanistic control: it retains norm, invertibility and U(1) symmetry,
   but generally loses symplecticity. Do not give it the Hamiltonian certificate.
4. Learned symmetric Kerr splitting: one learned complex unitary half-step,
   a shared learned scalar Kerr coefficient gamma=2*tanh(raw), and the same
   unitary half-step. The nonlinear phase is +.04*gamma*f_4(|z_j|²), where f_S(s)=s
   for s<=S, and S*(1+log(s/S)) otherwise. This is an established architecture
   class, with a C1 exact-on-domain globally bounded continuation. S=4 exceeds
   the original generator's maximum specified total power 1.85². Initial gamma=0.
5. Learned complex linear unitary control, identical half-step parameterization
   and initialization, with gamma identically zero.

Re-use the previously evaluated Stage 5 baselines only as contextual comparisons.
The new architectures do not have matched parameter counts; report them. All
learned models above share the update budget, not an equal optimization difficulty.

## Controls and metrics

Include the known-equation Strang split-step DNLS solver with one and two
substeps as zero-training oracle controls. These know the equation and coefficients;
their accuracy is a diagnostic, not a learned-method result. Keep the distinction
explicit. The reference arrays use finer RK4 integration.

Evaluate cumulative MSE through 20 and 200 steps, norm drift, explicit inverse
recovery, symplectic defect, phase equivariance, actual-weight certificates,
200-step Jacobian condition numbers on the same two test initial states, and
noncancelling-scale physical-energy errors. Recheck inference timing after all
training is complete. Record runtime/source/dataset hashes and per-seed artifacts.

## Interpretation and stopping

Algebraic identities are not evidence of predictive superiority or architectural
novelty. In particular, nonlinear phase/unitary splitting and normalized-action
Hamiltonians have close prior art. The tested contribution is the certificate and
its practical tradeoff. Report all failures and simpler-model wins. No candidate
will be retuned against this test split. A useful outcome must survive comparison
with the learned Kerr baseline before motivating a more expensive fresh-data study.

Composition of distinct exact Hamiltonian subflows does not exactly conserve
the physical DNLS energy. The per-step budget grows under recurrent composition;
it does not promise an indefinitely bounded long-time gradient.
