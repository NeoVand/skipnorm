# SkipNorm Revision & Execution Plan for TMLR

**Prepared for:** Mostafa Mohsenvand  
**Date:** October 15, 2026  
**Status:** Reconciled Pre-Execution Blueprint (Incorporating `skipnorm_author_revision_guide.md`)  
**Target Venue:** *Transactions on Machine Learning Research (TMLR)* (using the official anonymized TMLR LaTeX template, with companion arXiv release following TMLR preprint policy)

---

## 1. Core Positioning & Resolved Decisions

### Working Title & Central Question
- **Working Title:** *Identifying Physical Invariants through Conserved Latent Coordinates*
- **Central Question:** *When do conserved latent coordinates in a neural network identify physical first integrals, and which geometric, topological, or dynamical assumptions make identification possible?*

We treat the normalization and retraction operators as attributed geometric constructions and establish our contribution through:
1. **Precise conditional identification theorems and topological obstructions** (Scalar Identification, Jordan–Brouwer Multi-Basin Obstruction, and $K$-Block Toroidal Identification), paired with explicit analytic counterexamples showing that each hypothesis (tangent-rank, ray-invariance, global injectivity, non-resonant sampled density, and enclosed-basin connectedness) is necessary.
2. **Rigorous negative results on error floors** (showing via both scalar rescaling $E_\lambda = \lambda E$ and isometric translation $E_a(x) = a e_1 + x$ why neither absolute commutation error, relative commutation error, nor orbital $\operatorname{CV}$ admits a universal positive lower bound from topological impossibility alone).
3. **Reproducible held-out empirical diagnostics** that separate physical trajectory prediction, reconstruction, true-orbit radial variation, out-of-sample invariant recovery, and assumption diagnostics.

### Summary of Reconciled Decisions (From Section 1 of the Author Guide)

| Issue / Recommendation | Consolidated Decision in This Plan |
| :--- | :--- |
| **Scalar & $K$-block identification theorems** | **Adopt** with full domain, regularity ($I_k \ge I_{\min} > 0$), global injectivity, exact intertwining, and sampled-density assumptions. |
| **Jordan–Brouwer multi-basin obstruction** | **Adopt.** Two disjoint enclosed basins ($\overline{D_1}, \overline{D_2} \subset \mathcal{M}$, $D_1 \cap D_2 = \emptyset$) cannot both have even one constant-radius enclosing leaf ($r_1, r_2 > 0$). |
| **Unavoidable error / CV floor from obstruction** | **Reject.** Include both the scalar rescaling counterexample ($E_\lambda = \lambda E$) and the translated isometric encoder counterexample ($E_a(x) = a e_1 + x, g(z)=z$) proving neither absolute nor relative error nor $\operatorname{CV}$ has a universal positive floor. |
| **Fixed $10^{-10}$ OVR cutoff** | **Reject.** A fixed cutoff is scale-dependent. Report within-orbit variance and total variance separately under a training-set scale convention, and flag degenerate denominators as `undefined (collapsed)`. |
| **6D two-invariant torus sweep** | **Remove from core plan.** In 6D with $K=2$ invariants, regular leaves are 4-manifolds (not 2-tori), and $\sum_k(d_k-1) = N-K$ does not imply equal topology ($\mathbb{S}^2\times\mathbb{S}^2 \not\cong \mathbb{S}^1\times\mathbb{S}^3 \not\cong \mathbb{T}^3$). |
| **Automatic invariant-count or chaos estimators** | **Do not claim as theorems.** Keep dimension/partition sweeps or nonintegrable controls strictly as exploratory sensitivity analyses with preregistered measurements. |
| **Resonant sampling & distorted action grids** | **Adopt** as controlled analytic/empirical diagnostics (including the exact half-period harmonic oscillator counterexample $F(x)=-x, E(q,p)=(2q,p)$). |
| **Promised empirical outcomes in abstract/captions** | **Remove.** State only what the conditional theorems prove; write empirical claims and figure captions strictly from measured held-out data after execution. |

---

## 2. Attribution, Terminology, and Baseline Specifications

### 2.1 Result-Specific Attribution
1. **Sphere Retraction & Cotangent Pullback:** Credit **Zhang et al. (Aug 2026, *Sphere Retraction Normalizations*, arXiv:2608.02668)** for the projection-retraction update matching Geodesic SkipNorm away from its numerical cutoff and for the cotangent pullback identity $J^\top x_{\text{next}} = x$.
2. **Exponential-Map & Hyperspherical Updates:** Credit **Zheng et al. (Jan 2026, *GeoNorm*, arXiv:2601.22095)** for sphere exponential-map normalization, **Loshchilov et al. (ICLR 2025, *nGPT*, arXiv:2410.01131)** for normalized interpolation updates and its Appendix A.3 tangent-projected variant (noting the main interpolation update is not generally identical to GSN), and **Absil, Mahony, & Sepulchre (2008)** for classical projection retractions on manifolds.
3. **Scale-Invariant Weight Gradients:** Credit **Salimans & Kingma (NeurIPS 2016, *Weight Normalization*)** and **Arora, Li, & Lyu (ICLR 2019, Lemma 2.4)** for orthogonal weight gradients under scale invariance, keeping activation normalization distinct from weight normalization.
4. **2020 Public SkipNorm Video:** Cite **Mohsenvand (22 Oct 2020, `https://www.youtube.com/watch?v=hiZTIHOSmFQ`)** with verified timestamps strictly for what it discloses: the projective skip-normalization block $\hat{y} = (y/\|y\|)\|x\|$ where $y = f(x)$ (at `0:15`) and its $p=2$ norm-preserving autoencoder (`NPAE`) radial disentanglement on 2D MNIST (at `1:05`). Do not attribute the tangent-retraction GSN formula or topological identification theorems to the 2020 video.
5. **Physics-Informed & Invariant-Discovery Literature:**
   - Credit **COMET (Kasim & Lim, NeurIPS 2022)** for jointly learning constants of motion and dynamics via orthogonalization against invariant gradients, and **ConCerNet (Zhang et al., ICML 2023)** as a distinct conservation-discovery method.
   - Credit **Lusch, Kutz, & Brunton (*Nat. Commun.* 2018)** for state-dependent latent rotation frequencies $\omega(\rho)$.
   - Credit **Noether Networks (Alet et al., NeurIPS 2021)**, **Neural Conservation Laws (Richter-Powell et al., NeurIPS 2022)**, and **Lu, Dangovski, & Soljačić (*Nat. Commun.* 2023)** accurately according to their mechanisms.
   - Note accurately that **Lagrangian Neural Networks (Cranmer et al., 2020)** do *not* require canonical coordinates, **Poisson Neural Networks (Jin et al., 2020)** learn noncanonical skew-symmetric structure, and **Discrete Energy-Based Models (Matsubara et al., NeurIPS 2020)** enforce exact discrete conservation.

### 2.2 Accurate Physical Terminology
- **Lotka–Volterra $V(u,v)$:** On $\mathbb{R}_{>0}^2$, the natural Poisson tensor $P(u,v) = uv\begin{psmallmatrix}0 & -1 \\ 1 & 0\end{psmallmatrix}$ has $\det P = u^2v^2 > 0$ (invertible). Thus $V(u,v) = u - \ln u + \frac{4}{3}v - \frac{2}{3}\ln v$ is a **first integral and non-canonical Hamiltonian**, *not* a Casimir. We replace "Casimir" and "Noether–Brouwer" throughout.
- **Conserved Block Radii:** We refer to learned block radii $\|E^{(k)}(x)\|_2$ (and $\|E^{(k)}(x)\|_2^2/2$) as **invariant coordinates** (which identify physical invariants up to a continuous invertible reparameterization $\boldsymbol{\rho}: A \to (0,\infty)^K$ that may mix coordinates), *not* canonical symplectic actions, and we call $z^{(k)}/\|z^{(k)}\|_2$ **latent phase coordinates** rather than canonical angle variables.
- **Canonical 4D Torus Generator:** We use the consistent canonical Hamiltonian
  $$H(I_1, I_2) = I_1 + \sqrt{2}\,I_2 + 0.30\,I_1^2 + 0.25\,I_2^2 + 0.20\,I_1 I_2, \qquad I_k = \frac{1}{2}(q_k^2 + p_k^2) \ge I_{\min} > 0,$$
  whose frequencies $\omega_1 = 1 + 0.60\,I_1 + 0.20\,I_2$ and $\omega_2 = \sqrt{2} + 0.50\,I_2 + 0.20\,I_1$ satisfy the exact symmetry $\frac{\partial \omega_1}{\partial I_2} = \frac{\partial \omega_2}{\partial I_1} = 0.20$.

---

## 3. Scoped Mathematical Statements, Counterexamples, and Proofs

All exact statements assume nonzero inputs/outputs on open domains where the unregularized maps are $C^1$, with numerical stabilizers ($\varepsilon > 0$) and boundary/origin extensions analyzed explicitly.

### 3.1 Differential Properties & Stabilized Normalization Spectra (Propositions 1 & 2)

#### Proposition 1 (Universal Radial Pullback vs. Ray-Invariant Singular Pair)
1. **Universal Cotangent Pullback:** For any $C^1$ map $F: \mathbb{R}^{d_{\text{in}}}\setminus\{0\} \to \mathbb{R}^{d_{\text{out}}}\setminus\{0\}$ satisfying $\|F(x)\|_2 = \|x\|_2$, differentiating $\|F(x)\|_2^2 = \|x\|_2^2$ gives
   $$J(x)^\top u_{\text{out}} = u_{\text{in}}, \qquad J_{\text{total}}^\top u_L = u_0, \qquad \sigma_{\max}(J(x)) \ge 1,$$
   across compositions of arbitrary widths $d_0, \dots, d_L$.
2. **Condition for Forward Radial Preservation ($J u_{\text{in}} = u_{\text{out}}$):** For $\operatorname{SN}_2(x; f) = \|x\|_2 f(x)/\|f(x)\|_2$ ($f(x)\neq 0$), writing $J = A + u_{\text{out}} u_{\text{in}}^\top$ with $A = \frac{\|x\|_2}{\|f(x)\|_2} P_{\perp u_{\text{out}}} J_f(x)$, we have $J u_{\text{in}} = u_{\text{out}}$ (making $(u_{\text{out}}, u_{\text{in}})$ a singular vector pair with singular value $1$) **if and only if** $J_f(x) u_{\text{in}} \in \operatorname{span}(f(x))$, i.e., the unit output direction $f(x)/\|f(x)\|_2$ has zero radial derivative along $u_{\text{in}}$. Positive $1$-homogeneity ($f(cx) = cf(x)$ for $c>0$, e.g., bias-free LeakyReLU or bias-free maxout at differentiability points; *excluding* ELU, SiLU, and GELU) is sufficient, though not necessary (e.g., $f(x) = \|x\|_2 x$ is degree $2$ yet gives $\operatorname{SN}_2(x; f) = x$).
3. **Analytic Counterexamples:**
   - *Planar shear ($d_{\text{in}}=d_{\text{out}}=2$):* $F(x) = R_{\log\|x\|_2}x$ obeys $\|F(x)\|_2 = \|x\|_2$ and $J^\top u_{\text{out}} = u_{\text{in}}$, but at $x=(1,0)^\top$, $J = \begin{psmallmatrix}1 & 0 \\ 1 & 1\end{psmallmatrix}$ has singular values $\frac{\sqrt{5}\pm 1}{2} \approx \{1.618, 0.618\}$, neither equal to $1$.
   - *Rectangular expansion ($d_{\text{in}}=1, d_{\text{out}}=2$):* $F(t) = t(\cos\log t, \sin\log t)^\top$ ($t>0$) obeys $\|F(t)\|_2 = t$ and $J^\top u_{\text{out}} = 1$, yet at $t=1$ its sole singular value is $\sqrt{2} > 1$.

#### Proposition 2 (Block-Triangular Rank Formula & Stabilized Normalization Spectra)
1. **Exact Tangential Rank Formula:** In orthonormal bases $(u_{\text{in}}, Q_{\text{in}})$ of $\mathbb{R}^{d_{\text{in}}}$ and $(u_{\text{out}}, Q_{\text{out}})$ of $\mathbb{R}^{d_{\text{out}}}$ with $T = u_{\text{in}}^\perp$:
   $$\widetilde{J} = \begin{pmatrix} 1 & 0 \\ Q_{\text{out}}^\top A u_{\text{in}} & Q_{\text{out}}^\top A Q_{\text{in}} \end{pmatrix} \implies \operatorname{rank}(J) = 1 + \operatorname{rank}\!\big(A|_T\big).$$
   For square width $d$, $\operatorname{rank}(J) = d \iff \operatorname{rank}(A|_T) = d - 1$. *(Counterexample showing $\operatorname{rank}(A)=d-1$ is insufficient: at $x = f(x) = e_1 \in \mathbb{R}^2$, $J_f = \begin{psmallmatrix}1 & 1 \\ 1 & 0\end{psmallmatrix}$ gives $A = \begin{psmallmatrix}0 & 0 \\ 1 & 0\end{psmallmatrix}$ with $\operatorname{rank}(A) = 1 = d-1$, but $A|_T = 0$ and $J = \begin{psmallmatrix}1 & 0 \\ 1 & 0\end{psmallmatrix}$ has $\operatorname{rank}(J) = 1 < 2$.)*
2. **Idealized ($\varepsilon = 0$) vs. Stabilized ($\varepsilon > 0$) Normalizations:**
   - For the unscaled stabilized RMS normalization map $G_\varepsilon(y) = \frac{\sqrt{d}\,y}{\sqrt{r^2 + d\varepsilon}}$ ($r = \|y\|_2$, no affine gain), the Jacobian $\partial G_\varepsilon / \partial y$ has exact tangential and radial singular values:
     $$\sigma_{\text{tan}}\!\left(\frac{\partial G_\varepsilon}{\partial y}\right) = \frac{\sqrt{d}}{\sqrt{r^2 + d\varepsilon}}, \qquad \sigma_{\text{rad}}\!\left(\frac{\partial G_\varepsilon}{\partial y}\right) = \frac{\sqrt{d}\,d\varepsilon}{(r^2 + d\varepsilon)^{3/2}}, \qquad \frac{\sigma_{\text{rad}}}{\sigma_{\text{tan}}} = \frac{d\varepsilon}{r^2 + d\varepsilon}.$$
     When $\varepsilon > 0$, $\partial G_\varepsilon/\partial y$ has full algebraic rank $d$ (though ill-conditioned radially by $d\varepsilon/(r^2+d\varepsilon)$); when $\varepsilon = 0$, $\sigma_{\text{rad}} = 0$ and $\operatorname{rank}(\partial G_0/\partial y) = d - 1$.
   - For stabilized LayerNorm on $\mathbb{R}^d \setminus \operatorname{span}(\mathbf{1})$, the mean direction $\mathbf{1}$ remains an exact null vector ($\operatorname{rank} = d-1$ for $\varepsilon > 0$ and invertible $J_f$), while its centered-radial singular value scales analogously and vanishes at $\varepsilon = 0$ ($\operatorname{rank} = d-2$, collapsing $d=2$ to $\{\pm(1,-1)^\top\}$ on the circle of radius $\sqrt{2}$).

---

### 3.2 Topological Identification & Obstructions (Theorems 1–3 & Proposition 3)

1. **Theorem 1 (Scalar Invariant Identification — Appendix A):**
   - *Hypotheses:* $\mathcal{M} \subset \mathbb{R}^N$ ($N \ge 2$) open; $I \in C^1(\mathcal{M}, \mathbb{R})$; $C = (c_{\min}, c_{\max})$ an open interval of regular values whose fibers $\mathcal{O}_c = I^{-1}(c)$ are nonempty, compact, connected $(N-1)$-manifolds without boundary; $E: \mathcal{M} \to \mathbb{R}^N$ continuous and globally injective.
   - *Static Conclusion:* If $h = \|E\|_2$ is constant on each $\mathcal{O}_c$ ($c \in C$), there exists a continuous, strictly monotonic $\psi: C \to (0, \infty)$ such that $E(\mathcal{O}_c) = \mathbb{S}^{N-1}(\psi(c))$, $h = \psi \circ I$ on $I^{-1}(C)$, and each restriction $E|_{\mathcal{O}_c}$ is a homeomorphism onto its sphere.
   - *Sufficient Dynamical Condition:* Exact intertwining $E \circ F = g \circ E$ where $F$ preserves $I$, $g$ preserves Euclidean norm, and $F$ has a dense sampled orbit on $\mathcal{O}_c$ for a dense subset $C' \subset C$ (extended to all $c \in C$ via local submersion sections).
   - *Separate Endpoint Alignment:* If $\sup_{x \in \mathcal{O}_c} \|x - x_{\text{eq}}\|_2 \to 0$ as $c \downarrow c_{\min}$ for an equilibrium $x_{\text{eq}} \in \mathcal{M}$, then $\psi(c) \to 0$, $E(x_{\text{eq}}) = 0$, and $\psi$ is strictly increasing.

2. **Theorem 2 (Jordan–Brouwer Multi-Basin Obstruction — Appendix B):**
   - *Ball Mapping of Enclosed Domains:* If $O \subset \mathcal{M} \subset \mathbb{R}^N$ ($N \ge 2$) is a compact, connected, embedded $(N-1)$-manifold without boundary on which $\|E\|_2 \equiv r > 0$ is constant, and its bounded Jordan–Brouwer complementary region $D$ satisfies $\overline{D} \subset \mathcal{M}$, then $E(D) = B_N(0, r)$ and $E(\overline{D}) = \overline{B_N(0, r)}$, with a unique point in $D$ mapping to $0$.
   - *At Most One Foliated Basin:* Two such hypersurfaces $O_1, O_2 \subset \mathcal{M}$ with disjoint enclosed interiors $D_1 \cap D_2 = \emptyset$ ($\overline{D_1}, \overline{D_2} \subset \mathcal{M}$) cannot both have constant encoded norm, even at different energies or radii.
   - *No Exact Planar Two-Well Conjugacy:* For $N = 2$, if two periodic orbit loops in $\mathcal{M} \subset \mathbb{R}^2$ with disjoint enclosed interiors ($\overline{D_1}, \overline{D_2} \subset \mathcal{M}$) each carry a dense sampled orbit of $F$, no continuous injective $E: \mathcal{M} \to \mathbb{R}^2$ can satisfy $E \circ F = g \circ E$ on both loops for a norm-preserving $g$. *(For $N > 2$, this requires $(N-1)$-dimensional invariant hypersurfaces rather than 1D periodic loops.)*
   - *Super-Barrier Orientation Remark:* On a super-barrier annulus ($H > 0$), Theorem 1 gives strict monotonicity of $\psi(c)$; if $E$ is globally injective on the full enclosed interiors (containing both wells and the saddle), nested interiors map to nested centered balls $B_N(0, \psi(c))$, forcing $\psi(c)$ to increase with $c$.

3. **Theorem 3 ($K$-Block Toroidal Identification & Single-Radius Limit — Appendix C):**
   - *Positive $K$-Block Identification:* For $\mathcal{M} \subset \mathbb{R}^{2K}$ ($K \ge 1$) open and a $C^1$ submersion $I: \mathcal{M} \to \mathbb{R}^K$ with open image $A = I(\mathcal{M})$ and nonempty compact connected fibers $T_a = I^{-1}(a)$, if $F$ preserves $I$ and has a dense sampled orbit on $T_a$ for a dense subset $A' \subset A$, and a continuous injective $E: \mathcal{M} \to \bigoplus_{k=1}^K \mathbb{R}^2$ satisfies $E \circ F = g \circ E$ with block-norm-preserving $g$, then there exists a continuous map $\boldsymbol{\rho}: A \to (0, \infty)^K$ such that $E(T_a) = \prod_{k=1}^K \mathbb{S}^1(\rho_k(a))$ and $\|E^{(k)}(x)\|_2 = \rho_k(I(x))$. Each $E|_{T_a}$ is a homeomorphism onto its product torus, and $\boldsymbol{\rho}$ is a homeomorphism of $A$ onto an open subset of $\mathbb{R}^K$.
   - *Single-Radius Information Limit vs. Embeddability ($K \ge 2$):* A $K$-torus embeds directly into $\mathbb{S}^{2K-1}(\sqrt{\sum_k r_k^2})$ via $\theta \mapsto (r_1\cos\theta_1, r_1\sin\theta_1, \dots, r_K\cos\theta_K, r_K\sin\theta_K)$ ($r_k > 0$), so single-sphere preservation does not prohibit representing toroidal dynamics. However, under single-sphere preservation $\|E(x)\|_2 = \psi(I(x))$, the continuous scalar map $\psi: A \subset \mathbb{R}^K \to \mathbb{R}$ cannot be injective for $K \ge 2$ by Invariance of Domain, prohibiting unique recovery of $I(x)$ from the single scalar radius $\|E(x)\|_2$ alone.

4. **Proposition 3 (Finite-Error Coverage Bound & Counterexamples to Universal Error Floors — Appendix D & Section 6):**
   - *Finite-$M$ & Periodic Bounds:* On a compact invariant leaf $O$ where $h = \|E\|_2$ is $L$-Lipschitz and $\{F^j(x_0)\}_{j=0}^M$ is $\delta_M$-dense with one-step residual $\le \varepsilon$, $\operatorname{osc}_O(h) \le M\varepsilon + 2L\delta_M$. For a planar periodic orbit of period $T$ and speed $\le v_{\max}$ with $M = \lceil T/\Delta t \rceil$, $\operatorname{osc}_O(h) \le (T/\Delta t + 1)\varepsilon + L v_{\max}\Delta t =: B$ and $\operatorname{CV}_O(h) \le B/(2\mu)$.
   - *Why Exact Impossibility Gives Neither an Absolute Nor a Relative Error Floor:*
     1. *Scalar rescaling:* $E_\lambda = \lambda E, g_\lambda(z) = \lambda g(z/\lambda), D_\lambda(z) = D(z/\lambda)$ multiplies the absolute residual $\|E_\lambda(F(x)) - g_\lambda(E_\lambda(x))\|_2$ by $\lambda \to 0$ while leaving decoded predictions unchanged.
     2. *Isometric translation:* On $\|x\|_2 \le R$ with $\|F(x) - x\|_2 \le D$, the globally injective isometric encoder $E_a(x) = a e_1 + x$ ($a > R$) with $g(z) = z$ has $\partial E_a/\partial x = I$, yet
        $$\frac{\|E_a(F(x)) - g(E_a(x))\|_2}{\|E_a(x)\|_2} \le \frac{D}{a - R} \to 0, \qquad \operatorname{CV}(\|E_a\|_2) \le \frac{R}{a - R} \to 0 \quad (a \to \infty),$$
        even though the latent dynamics are stationary ($g = \text{id}$) and physical predictions fail. Thus we always report physical rollout MSE, reconstruction MSE, between-orbit spread, and latent mean offset alongside centered-spread-normalized intertwining error.

---

## 4. Data, Metrics, and Reproducibility Protocol

### 4.1 Whole-Orbit Disjoint Splits (Train / Validation / Test)
- **Disjoint Invariant Levels & Phases:** Trajectory generation uses explicit, separate `PRNGKey` seeds for `train`, `val`, and `test`. Crucially, for all invariant-identification benchmarks (Lotka–Volterra, Double-Well, 4D Torus), we partition the **invariant values / action vectors** $I$ into disjoint sets ($I_{\text{train}} \cap I_{\text{val}} \cap I_{\text{test}} = \emptyset$) before sampling initial phases, and we evaluate both:
  1. **Held-out Interpolation** (test invariant levels interleaved strictly between training levels, with independent random initial phases), and
  2. **Held-out Extrapolation** (test invariant levels in an outer ring beyond the training range).
- **Automated Split Assertions:** A pre-training verification script asserts zero shared invariant levels, zero shared initial conditions, and exact energy/basin membership (e.g., verifying $H(q_0, p_0) < 0$ for every sub-barrier Double-Well orbit), writing a signed split manifest to `results/split_manifests.json`.

### 4.2 Metrics & Assumption Diagnostics (Evaluated on Encoded True Trajectories)
1. **Physical Prediction & Reconstruction:** Held-out physical rollout $\operatorname{MSE}(x_{t+1:t+H}, \hat{x}_{t+1:t+H})$, reconstruction $\operatorname{MSE}(x_t, D(E(x_t)))$, and physical invariant drift on generated rollouts.
2. **True-Orbit Radial Statistics:** Evaluated strictly on **encoded true test trajectories $r_t = \|E(x_t)\|_2$** (never on generated latent rollouts, whose radii are constant by construction):
   - Per-orbit mean radius $\mu_i$, temporal standard deviation $\operatorname{SD}_i$, and $\operatorname{CV}_i = \operatorname{SD}_i / \mu_i$ (with explicit near-zero guard if $\mu_i < 10^{-8} \sigma_{\text{lat}}$).
   - Between-orbit radius spread $\operatorname{SD}_{\text{between}}(\mu_i)$ and relative offset $\|\bar{E}_{\text{train}}\|_2 / \sigma_{\text{lat}}$, where $\sigma_{\text{lat}} = \sqrt{\mathbb{E}_{\text{train}}[\|E(x) - \bar{E}_{\text{train}}\|_2^2]}$.
   - **Orbit Variance Ratio (OVR):** $1 - \frac{\mathbb{E}_i[\operatorname{Var}_t(r_{i,t})]}{\operatorname{Var}_{\text{all}}(r_{i,t})}$, reporting numerator and denominator explicitly under training-normalized scale and flagging $\operatorname{Var}_{\text{all}}(r)/\sigma_{\text{lat}}^2 < 10^{-6}$ as `undefined (collapsed)`.
3. **Out-of-Sample Invariant Recovery ($R_{\text{test}}^2$):** Readout models (ridge-regularized polynomial or monotone spline) are fitted strictly on `train` trajectories, hyperparameter-selected on `val` trajectories, and evaluated on `test` trajectories (reporting both per-invariant $R_{\text{test}}^2$ and joint multivariate $R_{\text{test}}^2$, plus Spearman $|\rho_{\text{test}}|$ for scalar invariants).
4. **Intertwining & Injectivity Diagnostics:**
   - Absolute residual $\|E(F(x)) - g(E(x))\|_2$ and centered-spread-normalized residual $\|E(F(x)) - g(E(x))\|_2 / \sigma_{\text{lat}}$.
   - Local Jacobian singular values $(\sigma_{\min}(\partial E/\partial x), \sigma_{\max}(\partial E/\partial x))$, condition number $\kappa(\partial E/\partial x)$, determinant sign consistency on $\mathcal{M}$, and held-out pairwise collision search $\min_{x \neq x'} \|E(x) - E(x')\|_2 / \|x - x'\|_2$ (explicitly noted as local/finite diagnostics rather than global injectivity certificates, paired with an **injective-by-construction RealNVP/coupling encoder control**).

---

## 5. Four-Stage Execution Order & Completion Criteria

We will execute the project in the exact four stages prescribed in Section 14 of `skipnorm_author_revision_guide.md`:

### Stage 1: Lock Protocol, Analytic Counterexamples & Unit Tests
- Implement `experiments/stage1_protocol_and_counterexamples.py` verifying:
  1. The $2\times 2$ twist counterexample $F(x) = R_{\log\|x\|_2}x$ ($\sigma \approx \{1.618, 0.618\}$), the $1\to 2$ spiral $F(t) = t(\cos\log t, \sin\log t)$ ($\sigma = \sqrt{2}$), and the $2\times 2$ rank counterexample $J_f = \begin{psmallmatrix}1 & 1 \\ 1 & 0\end{psmallmatrix}$ ($\operatorname{rank}(A)=1$ but $A|_T=0 \implies \operatorname{rank}(J)=1$).
  2. Exact vs. stabilized ($\varepsilon > 0$) singular value spectra of RMSNorm, LayerNorm, and SkipNorm across $d \in \{2,3,4,6,8\}$ and 50-layer cascades (comparing 1-homogeneous LeakyReLU vs. non-homogeneous SiLU).
  3. The **Analytic Resonant Harmonic Oscillator Control** ($\Delta t = T/2$, $F(x)=-x$, $E(q,p)=(2q,p)$, $g(z)=-z$), showing zero commutation error at sampled points alongside large off-grid radial variation.
  4. The **Translated Isometric Encoder Control** ($E_a(x) = a e_1 + x$), demonstrating vanishing relative error and $\operatorname{CV}$ alongside large offset ratio $\|\bar{E}\|_2/\sigma_{\text{lat}}$ and failed physical prediction.
  5. Automated split generator and unit tests for metric edge cases (collapse flags, disjoint invariant levels).

### Stage 2: Decisive Core Pilots (Lotka–Volterra, Double-Well, 4D Canonical Torus)
- **Pilot 2A (Lotka–Volterra Scalar Identification):** 48 train / 16 val / 24 test orbits (interpolation + extrapolation levels, plus near-equilibrium levels). Compare Unconstrained ResNet ($d_z=2$), Matched GSN ($d_z=2$, MLP encoder), Matched GSN ($d_z=2$, **bijective coupling encoder control**), Overparameterized GSN ($d_z=8$), State-Dependent Rotation $U(\rho(z))$ ($d_z=2$), and a faithful **COMET** orthogonalized-gradient baseline. Run the horizon sweep $H \in \{1, 2, 4, 6, 12\}$ and alignment-loss ablation $\lambda_{\text{align}} \in \{0, 0.25\}$ with matched gradient updates.
- **Pilot 2B (Double-Well Multi-Basin Obstruction):** $H(q,p) = \frac{1}{2}p^2 - \frac{1}{2}q^2 + \frac{1}{4}q^4$ with verified $p_0 = 0, |q_0| \in (1.05, 1.38) \implies H \in (-0.247, -0.049) < 0$ for Left and Right wells, near-separatrix orbits approaching $H = 0$, and Super-Barrier orbits $H > 0$. Test (i) balanced 50/50 sampling, (ii) imbalanced 80/20 sampling, and (iii) a genuinely tilted potential $H_\gamma(q,p) = H(q,p) + \gamma q$ ($\gamma = 0.08$). Report Left Well, Right Well, and Super-Barrier metrics (prediction MSE, reconstruction MSE, normalized intertwining residual, true-orbit $\operatorname{CV}$, $R_{\text{test}}^2$, and collision diagnostics) for every seed without cherry-picking.
- **Pilot 2C (4D Canonical Torus Action Recovery):** 64 train / 16 val / 32 test action vectors $(I_1, I_2) \in [0.15, 1.20]^2$ under an explicit invertible polynomial/trig observation map $\Psi: \mathbb{R}^4 \to \mathbb{R}^4$. Compare Unconstrained ResNet (4D), Single-Sphere GSN ($K=1$) with **1-scalar total-radius readout $\|z\|_2$**, Single-Sphere GSN ($K=1$) with **2-coordinate-block readout $(\|z^{(1)}\|_2, \|z^{(2)}\|_2)$**, and 2-Block Toroidal GSN ($K=2$) with **2-block readout**, calibrated against true actions $I_k = r_k^2/2$.

### Stage 3: Full 5-Seed Empirical Comparisons & Converged 1D DNLS Benchmark
- **1D DNLS Wave Propagation:** Generate reference trajectories using a converged 6th-order split-step Fourier / RK4 solver (verified by step-size refinement against an independent solver, with **no** artificial post-step norm projection) across 48 train / 16 val / 32 test trajectories (evaluating both in-distribution wavepackets and out-of-distribution amplitude/width wavepackets).
- Compare Matched Nonlinear Baselines: (1) Unconstrained ResNet, (2) Pre-RMSNorm & Post-RMSNorm ResNet, (3) Learned Linear Orthogonal ($U^\top U = I$), (4) State-Dependent Rotation $U(\rho)$, (5) **Projected Residual SkipNorm** ($\|x\|_2 (x + \alpha f(x))/\|x + \alpha f(x)\|_2$, without tangent projection), and (6) **Tangent-Projected Geodesic SkipNorm** (isolating the exact effect of tangent projection). Report paired differences across common test trajectories, rollout MSE, amplitude-weighted wrapped phase error, wave-norm drift, and DNLS Hamiltonian drift across 5 initialization seeds.

### Stage 4: Figure Generation & TMLR Manuscript Assembly
- Generate all 5 figures directly from versioned `.json` / `.npz` records using the installed **K-Dense `scientific-visualization`** pipeline (`style_context`, `palette_audit.py`, `image_metadata.py`, `figure_export.py` with JSON provenance manifests), with neutral measurement-based captions and zero hardcoded annotations:
  - **Figure 1:** Differential Geometry, Tangential Rank Counterexample, and Stabilized ($\varepsilon = 0$ vs. $\varepsilon > 0$) Singular Value Spectra.
  - **Figure 2:** 1D DNLS Held-Out Wave Propagation (Trajectory MSE, Phase Error, Hamiltonian Drift, and Wave-Norm Conservation across Projected vs. Tangent-Projected Updates and Baselines).
  - **Figure 3:** Lotka–Volterra Scalar Identification (Phase-Plane Radius Field, Held-Out Interpolation/Extrapolation Calibration, Bijective Encoder & Resonant Sampling Controls, and Horizon/Loss Ablations).
  - **Figure 4:** Double-Well Multi-Basin Obstruction (Phase Plane & Encoded Image Domains $E(D_1), E(D_2)$, Per-Orbit Radius vs. Energy $H$, and Per-Basin Metrics across Balanced, 80/20 Imbalanced, and Tilted Potentials).
  - **Figure 5:** 4D Canonical Torus Identification (Action-Grid Image under $\boldsymbol{\rho}(I_1, I_2)$, 1-Scalar Total-Radius Readout vs. 2-Block Readout Comparison, and Intertwining/Conditioning Diagnostics).
- Format the manuscript using the official **TMLR LaTeX style (`tmlr.sty`)**, rewrite the abstract strictly around the completed empirical numbers, and verify 100% consistency between code, saved JSON records, tables, figures, and text.
