# Mathematical support for the rotation-coupling experiment

This note establishes the properties used by the Stage 5 experiment. It makes
no priority claim for conditional rotations, invertible coupling layers, or
spectral constraints. The statements concern real arithmetic and derivatives
with respect to the state, at fixed model parameters. Numerical tests check
implementation agreement; predictive value remains an empirical question.

## 1. Simultaneous pair rotations

Write

\[
x=(a,b_1,\ldots,b_K)\in\mathbb R^{m+2K},\qquad b_j\in\mathbb R^2,
\]

with \(K\ge1\) and possibly \(m=0\). Fix a state-independent \(\delta>0\), a
real scalar \(\alpha\), and a \(C^1\) map
\(h:\mathbb R^{m+1}\to\mathbb R^K\) satisfying
\(\sup_c\|Dh(c)\|_2\le L\). The norm is the **joint Euclidean operator norm**,
not a separate bound on each output coordinate. Define

\[
s(x)=\sqrt{\delta^2+\|x\|_2^2},\qquad
c(x)=\frac{(a,\delta)}{s(x)},\qquad
\theta(x)=\alpha h(c(x)),
\]

\[
R_t=\begin{pmatrix}\cos t&-\sin t\\\sin t&\cos t\end{pmatrix},\qquad
F(x)=(a,R_{\theta_1(x)}b_1,\ldots,R_{\theta_K(x)}b_K).
\]

The conditioning variables \(a\), every pair radius \(\|b_j\|\), and the total
radius are unchanged. Consequently \(c(F(x))=c(x)\), and the exact inverse is

\[
F^{-1}(y)=(a_y,R_{-\theta_1(y)}y_1,\ldots,R_{-\theta_K(y)}y_K).
\]

Both maps are \(C^1\) on all of Euclidean space, including the origin. Thus
\(F\) is a global diffeomorphism and \(\|F(x)\|=\|x\|\).

## 2. The full Jacobian is an orthogonal factor times a shear

Let \(J_2=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)\),
\(A=\operatorname{diag}(I_m,R_{\theta_1},\ldots,R_{\theta_K})\), and form a
\((m+2K)\times K\) matrix \(U\) whose \(j\)-th column is zero except for
\(J_2b_j\) in pair \(j\). Put \(V=D_x\theta\). Differentiation gives

\[
DF=A(I+N),\qquad N=UV.
\]

The conditioner is invariant under rotation of each pair independently, so
\(Dc\,U=0\) and therefore \(VU=0\). In particular,

\[
N^2=0,\qquad \det DF=\det A\det(I+N)=1.
\]

The square-zero property also gives the **complete singular spectrum**.
If \(t_1,\ldots,t_r>0\) are the nonzero singular values of \(N\), then
\(DF\) has the singular values

\[
\sigma_{j,\pm}=\frac{\sqrt{t_j^2+4}\pm t_j}{2}
=\exp\!\left(\pm\operatorname{asinh}(t_j/2)\right),\quad j=1,\ldots,r,
\]

and \(m+2K-2r\) additional singular values equal to one. To see this, use
\(\operatorname{range}N\subseteq\ker N\): the nonzero left singular vectors
of \(N\) are orthogonal to its nonzero right singular vectors. On each paired
two-dimensional subspace, \(I+N\) is the shear
\(\left(\begin{smallmatrix}1&t_j\\0&1\end{smallmatrix}\right)\); it is the
identity on the remaining subspace. Left multiplication by \(A\) leaves
singular values unchanged. This argument includes all conditioner derivatives.

Norm preservation separately implies the invariant-covector identity

\[
DF(x)^T F(x)=x.
\]

It does not generally make the input and output radial directions a forward
unit singular pair. The full Jacobian is orthogonal precisely when \(N=0\).

## 3. Global distortion bound and composition

View \(c\) as a coordinate projection of
\(y/\|y\|\), where \(y=(x,\delta)\). Since

\[
D_y(y/\|y\|)=\frac{I-\widehat y\widehat y^T}{s},
\]

restriction to derivatives in \(x\), followed by coordinate projection, gives
\(\|Dc\|_2\le1/s\). The columns of \(U\) have disjoint supports, hence

\[
\|N\|_2\le\|U\|_2\|V\|_2
\le |\alpha|L\,\frac{\max_j\|b_j\|_2}{s}
\le |\alpha|L.
\]

Every singular value therefore lies in \([e^{-b},e^b]\), with

\[
b=\operatorname{asinh}(|\alpha|L/2).
\]

Both \(F\) and \(F^{-1}\) are globally \(e^b\)-Lipschitz. For a composition
of \(M\) layers, every singular value lies in \([e^{-B},e^B]\), where
\(B=\sum_{\ell=1}^M b_\ell\); its condition number is at most \(e^{2B}\).
State-independent orthogonal mixing adds no distortion. General orthogonal
mixing has absolute determinant one; determinant \(+1\) requires an
orientation-preserving overall mixing map. The implemented permutation
conjugation \(P^{-1}FP\) retains determinant \(+1\), regardless of the
permutation's sign. A learned linear orthogonal map is state-independent for
these purposes once its parameters are fixed.

The inverse of a composition applies individual inverses **in reverse order**.
Negating all angles while keeping their original order is generally incorrect.
If the composition represents one time step, its \(n\)-step rollout has
worst-case budget \(nB\), not \(B\). For step-size-scaled angle multipliers,
the accumulated budget can remain bounded at fixed physical time under step
refinement; it need not remain bounded as physical time increases. A small
budget may exclude genuine dynamical expansion.

For eight layers with \(L_\ell\le1\), the choice
\(\alpha_\ell=2\sinh(0.5/8)\approx0.1250814\) gives a whole-step budget of
\(B=0.5\). These are global worst-case bounds, not claims that the bound is
attained by each trained model or trajectory.

## 4. What spectral projection and origin regularity actually guarantee

For a feedforward conditioner with 1-Lipschitz activations,
\(L\le\prod_q\|W_q\|_2\). Projecting **every** matrix, including the output
matrix, onto the operator-norm unit ball gives \(L\le1\). The implementation
uses SVD singular-value clipping and tanh activations. This constraint must hold
at initialization and after every optimizer update. Unrestricted biases do
not change the bound. Unprojected normalized controls can still be assigned
a valid, potentially much larger, certificate using their actual product of
weight norms. A coordinatewise output bound would instead permit a factor of
\(\sqrt K\); GELU/SiLU, residual paths, and approximate spectral estimates
require their own additional bounds.

Because \(\delta>0\), no state-dependent denominator vanishes. At the origin,
\(U=0\) and \(DF(0)=A(0)\), so the derivative is orthogonal. A learned
\(\delta\) is also allowed if it remains positive and state-independent.
Uniform input-Jacobian bounds do not bound parameter gradients as
\(\delta\to0\), nor do they prevent numerical overflow or inaccurate
trigonometric evaluation. Use stable norm evaluation and appropriate
floating-point tolerances. ReLU would give almost-everywhere derivative
statements rather than an everywhere \(C^1\) theorem. Automatic derivatives
must describe the actual forward map; detached conditioning quantities do not
provide such a check.

## 5. Scope, restrictions, and prior work

- Norm and volume preservation together preserve surface measure on each
  sphere: in polar coordinates the map is \((r,\omega)\mapsto(r,\varphi_r(\omega))\),
  and unit ambient Jacobian forces unit surface Jacobian. Thus every radial
  probability density is invariant. The architecture cannot normalize the
  distribution of radii or represent arbitrary norm-preserving angular
  compression. In two dimensions, an orientation-preserving norm- and
  area-preserving map is a radius-dependent rotation.
- Generic real coordinate mixing and conditioning do not ensure canonical
  symplecticity or DNLS global phase equivariance. These, physical energy
  conservation, forecasting accuracy, and invariant identification are
  separate properties. Pairing real coordinates does not by itself enforce
  the physical complex structure. Restricting all rotations and mixing to
  fixed blocks additionally conserves each block norm and can overconstrain
  a target system.
- Exact norm preservation and invertibility do not imply an isometric
  Jacobian. The shear spectrum explicitly quantifies the permitted distortion.
  No universality assertion is established here.

There is a direct physical obstruction to combining a fixed finite global
budget with unrestricted amplitude expressivity. The self-phase flow

\[
T_t(z)=e^{it|z|^2}z,\qquad z\in\mathbb C\simeq\mathbb R^2,
\]

preserves norm, area, and the canonical symplectic form, but its shear magnitude
at radius \(r\) is \(2|t|r^2\). Its singular values are
\(\exp(\pm\operatorname{asinh}(|t|r^2))\). For \(t\ne0\), no model with a
finite global budget \(B\) can equal this flow at all amplitudes. Exact
\(C^1\) agreement on a ball of radius \(R\) requires at least
\(B\ge\operatorname{asinh}(|t|R^2)\). This is the nonlinear self-phase
component relevant to DNLS, not an argument that the full DNLS step has this
same closed-form spectrum. The practical target must therefore be a specified
bounded physical domain with a suitable budget; the global regularity guarantee
does not establish accuracy outside that domain.

The feature map also makes this restriction visible directly. Along a fixed
unit direction \(\omega\),
\(c(r\omega)\to(\omega_a,0)\) and
\(\|\partial_r c(r\omega)\|\le\delta/(\delta^2+r^2)\). Each angle thus
approaches a direction-dependent limit, whereas the self-phase angle has radial
derivative \(2tr\). A fixed finite composition retains an asymptotically
direction-dependent angular map; it cannot reproduce unlimited amplitude
dependence merely by increasing the input radius.

Conditional rotations already occur in the rigid-body G-LPNets construction;
see Eldred et al., Section 6, especially Eqs. 72–73 of
[Lie-Poisson Neural Networks](https://arxiv.org/html/2308.15349#S6).
Broader invertible constructions on compact manifolds appear in Rezende et al.,
[Normalizing Flows on Tori and Spheres](https://proceedings.mlr.press/v119/rezende20a.html).
Lipschitz-constrained invertible residual architectures are established by
Behrmann et al.,
[Invertible Residual Networks](https://proceedings.mlr.press/v97/behrmann19a.html).
These precedents motivate the comparison; this note supplies a derivation for
the specific implemented normalized conditioner, not a claim that the general
ideas are new or that these methods have identical guarantees.

## 6. Sharp analytic stress case: raw conditioning has unbounded distortion

Take \(m=1\), \(K=1\), \(\alpha\ne0\), and the bounded, 1-Lipschitz
conditioner \(h(u,v)=\tanh u\). Compare

\[
\theta_{\rm raw}(a,b)=\alpha\tanh a,
\qquad
\theta_{\rm norm}(a,b)=\alpha\tanh\!\left(
\frac{a}{\sqrt{\delta^2+a^2+\|b\|^2}}\right).
\]

At \(x_R=(0,R,0)\), \(R\ge0\), both maps fix the state exactly and both
preserve norm, volume, and an explicit inverse. Nevertheless,

\[
DF_{\rm raw}(x_R)=
\begin{pmatrix}1&0&0\\0&1&0\\\alpha R&0&1\end{pmatrix},\qquad
DF_{\rm norm}(x_R)=
\begin{pmatrix}1&0&0\\0&1&0\\
\alpha R/\sqrt{\delta^2+R^2}&0&1\end{pmatrix}.
\]

Their shear magnitudes are

\[
t_{\rm raw}(R)=|\alpha|R,\qquad
t_{\rm norm}(R)=\frac{|\alpha|R}{\sqrt{\delta^2+R^2}}\le|\alpha|.
\]

For either map the exact nontrivial singular values are
\(\sigma_\pm(R)=(\sqrt{4+t(R)^2}\pm t(R))/2\), with the third equal to one.
As \(R\to\infty\), the raw map has
\(\sigma_+\sim|\alpha|R\), \(\sigma_-\sim1/(|\alpha|R)\), and condition
number \(\sim\alpha^2R^2\), despite bounded angles and a 1-Lipschitz
conditioner. The normalized map approaches
\(\sigma_\pm=e^{\pm\operatorname{asinh}(|\alpha|/2)}\), demonstrating that
the radius-independent bound is sharp as a supremum.

A decisive plot needs no fitting: set \(\delta=1\), use the experiment's
\(\alpha=2\sinh(0.5/8)\), sweep \(R\) logarithmically from \(10^{-6}\) to
\(10^6\), and plot both exact \(\sigma_+\) and \(\sigma_-\), together with
the normalized bound. This example embeds unchanged in 32 dimensions by using
one conditioning coordinate and one rotated pair, leaving every other angle
zero. It isolates normalized conditioning from spectral training and shows why
norm conservation, exact inversion, bounded angles, and volume preservation
alone do not establish a global Jacobian bound.
