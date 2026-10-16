"""Rebuild the research report from saved Stage 5, 6 and 7 records."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
NAMES={'hamiltonian_bounded':'Certified Hamiltonian',
       'hamiltonian_unconstrained':'Projection removed',
       'naive_phase':'Omitted common phase', 'learned_kerr':'Learned Kerr split',
       'learned_unitary':'Learned complex unitary'}
ISOLATED_NAMES={'perspective_bounded':'Projected perspective',
                'perspective_unconstrained':'Projection removed',
                'log_intensity_bounded':'Bounded log-intensity',
                'scalar_kerr':'Scalar Kerr (known form)'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--theory-notes',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    run=ROOT/'results/stage6_hamiltonian_normalization/pilot_v1'
    old=ROOT/'results/stage5_rotation_coupling/pilot_v1'
    isolated=ROOT/'results/stage7_isolated_kerr/pilot_v1'
    isolated_summary=json.loads((isolated/'summary.json').read_text())['summary']
    isolated_rows=[json.loads((isolated/f'{m}_seed{s}.json').read_text())
                   for m in ISOLATED_NAMES for s in range(3)]
    new=json.loads((run/'summary.json').read_text())['summary']
    previous=json.loads((old/'summary.json').read_text())['summary']
    post=json.loads((run/'posthoc_diagnostics.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted(run.glob('*_seed*.json'))]
    def pm(record,key):
        v=record[key]
        return f"{v['mean']:.6f} ± {v['sample_sd']:.6f}"
    isolated_table=[]
    certificate_table=[]
    for mode,label in ISOLATED_NAMES.items():
        s=isolated_summary[mode]
        isolated_table.append(f"| {label} | {s['n_parameters']:,} | "
            f"{pm(s['test_id'],'mse_200')} | {pm(s['test_ood'],'mse_200')} | "
            f"{s['test_id']['aligned_mse_200']['mean']:.6f} | "
            f"{s['test_ood']['aligned_mse_200']['mean']:.6f} | "
            f"{s['jacobian']['200']['relative_frobenius_error_mean']['mean']:.3f} |")
        values=[r['certified_log_bound_one_step_global'] for r in isolated_rows if r['mode']==mode]
        bound=f"{min(values):.3f}–{max(values):.3f}" if values[0] is not None else 'No global bound'
        certificate_table.append(f"| {label} | {bound} |")
    table=[]
    for mode,label in [('geodesic_skipnorm','Geodesic SkipNorm (previous)'),
                       ('coupling_bounded','Certified rotation coupling (previous)'),
                       ('linear_orthogonal','Real orthogonal linear (previous)'),
                       *NAMES.items()]:
        s=previous[mode] if mode in previous else new[mode]
        table.append(f"| {label} | {s['n_parameters']:,} | {pm(s,'mse_20')} | {pm(s,'mse_200')} | {s['rollout_ms']['mean']:.2f} |")
    metrics=[]
    for mode,label in NAMES.items():
        rr=[r for r in post['metrics'] if r['mode']==mode]
        metrics.append(f"| {label} | {np.mean([r['aligned_mse_200'] for r in rr]):.6f} | {np.mean([r['intensity_mse_200'] for r in rr]):.6f} | {new[mode]['mean_scaled_final_energy_error']['mean']:.4f} |")
    selected=[('geodesic_skipnorm','SkipNorm',old,'#b65e38'),
              ('coupling_bounded','Previous coupling',old,'#a28ca8'),
              ('hamiltonian_bounded','Certified Hamiltonian',run,'#007f73'),
              ('learned_kerr','Learned Kerr split',run,'#2e5691'),
              ('learned_unitary','Complex linear',run,'#bd963f')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'grid.alpha':.18})
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for mode,label,directory,color in selected:
        traces=np.stack([np.load(directory/f'{mode}_seed{s}.npz')['mse_trace'] for s in range(3)])
        cumulative=np.cumsum(traces,axis=1)/np.arange(1,250)
        mean=cumulative.mean(axis=0);sd=cumulative.std(axis=0,ddof=1)
        ax[0].plot(np.arange(1,250),mean,label=label,color=color,lw=2)
        ax[0].fill_between(np.arange(1,250),np.maximum(mean-sd,1e-12),mean+sd,color=color,alpha=.1)
    ax[0].set(yscale='log',xlabel='Predicted steps',ylabel='Cumulative state MSE',title='Forecast accuracy: three seeds')
    ax[0].legend(fontsize=8);ax[0].grid()
    short=['Certified\nHamiltonian','Projection\nremoved','Omitted\ncommon phase','Learned\nKerr','Complex\nlinear']
    positions=np.arange(5)
    raw=np.array([new[m]['mse_200']['mean'] for m in NAMES])
    aligned=np.array([np.mean([r['aligned_mse_200'] for r in post['metrics'] if r['mode']==m]) for m in NAMES])
    ax[1].bar(positions-.18,raw,.36,color='#007f73',label='Original MSE')
    ax[1].bar(positions+.18,aligned,.36,color='#87b8b1',label='Phase-aligned MSE')
    ax[1].set_xticks(positions,short,fontsize=8)
    ax[1].set(ylabel='200-step MSE',title='Global phase changes the metric')
    ax[1].legend(fontsize=8);ax[1].grid(axis='y')
    values=np.array([[r['repeated_time_step_jacobian']['200']['maximum_condition_number']
                     for r in rows if r['mode']==m] for m in NAMES])
    ax[2].bar(positions,values.mean(axis=1),yerr=values.std(axis=1,ddof=1),
              capsize=3,color=['#007f73','#63a29c','#a28ca8','#2e5691','#bd963f'])
    reference=json.loads((old/'diagnostics.json').read_text())['reference_jacobians']['200']['maximum_condition_number']
    ax[2].axhline(reference,color='#333333',ls='--',label=f'Reference dynamics: {reference:.1f}')
    ax[2].set_xticks(positions,short,fontsize=8)
    ax[2].set(ylabel='Largest sampled condition number',title='Sensitivity: two initial states')
    ax[2].legend(fontsize=8);ax[2].grid(axis='y')
    fig.savefig(out/'hamiltonian_research_results.png',dpi=180)
    fig.savefig(out/'hamiltonian_research_results.pdf')
    plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    positions=np.arange(4)
    short=['Projected\nperspective','Projection\nremoved','Bounded\nlog-intensity','Scalar Kerr\n(known form)']
    for a,split,title in zip(ax[:2],['test_id','test_ood'],
            ['Fresh states: power 0.5–3.4','Power extrapolation: 3.4–4.0']):
        for offset,key,label,color in [(-.18,'mse_200','Original MSE','#007f73'),
                                      (.18,'aligned_mse_200','Phase-aligned MSE','#87b8b1')]:
            vals=[isolated_summary[m][split][key]['mean'] for m in ISOLATED_NAMES]
            sd=[isolated_summary[m][split][key]['sample_sd'] for m in ISOLATED_NAMES]
            a.bar(positions+offset,vals,.36,yerr=sd,capsize=3,color=color,label=label)
        a.set_xticks(positions,short,fontsize=8)
        a.set(yscale='log',ylabel='Cumulative MSE through 200 steps',title=title)
        a.grid(axis='y');a.legend(fontsize=8)
    vals=[isolated_summary[m]['jacobian']['200']['relative_frobenius_error_mean']['mean'] for m in ISOLATED_NAMES]
    sd=[isolated_summary[m]['jacobian']['200']['relative_frobenius_error_mean']['sample_sd'] for m in ISOLATED_NAMES]
    ax[2].bar(positions,vals,yerr=sd,capsize=3,color=['#007f73','#87b8b1','#bd963f','#2e5691'])
    ax[2].set_xticks(positions,short,fontsize=8)
    ax[2].set(ylabel='Relative Frobenius error',title='Jacobian fidelity at step 200')
    ax[2].grid(axis='y')
    fig.suptitle('Isolated Kerr phase dynamics • Three seeds • Mean ± sample SD',fontsize=12)
    fig.savefig(out/'isolated_phase_results.png',dpi=180)
    fig.savefig(out/'isolated_phase_results.pdf')
    plt.close(fig)
    report=rf'''# What survived the deeper theoretical search

15 October 2026. Three independent agents reviewed the mathematics, prior art and experiments through several rounds of proposals and counterexamples. This report combines their conclusions with two completed new screens: 27 trained models and two known-equation controls. The publication target remains TMLR.

**Assessment.** The strongest direction is to control the physical distortion of conservative maps. A normalization can preserve norm exactly while unnecessarily restricting the dynamics it can learn. We now have a concrete theorem separating a convenient normalized-coordinate certificate from the actual geometric requirement, plus experiments testing alternatives.

The first new construction normalizes invariant coordinates inside a scalar Hamiltonian. It gives norm preservation, canonical symplecticity, phase symmetry, volume preservation, explicit inversion, smoothness at zero and a global two-sided state-Jacobian certificate. It improves on the previous coupling experiment, but a much smaller, established learned Kerr splitting model performs better. A simpler bounded log-intensity model then improves raw accuracy on fresh isolated phase data, with an important phase-aligned extrapolation failure. We have a useful mathematical distinction and a testable research direction; this is not yet a demonstrated novel, generally superior method or a submission-ready paper.

## 1. A general conservative construction

For complex coordinates $z_j=q_j+i p_j$, let $I_j=|z_j|^2/2$ and define

$$\rho=\delta+\sum_j I_j,\qquad u=I/\rho,\qquad K(I)=\rho h(u),\quad\delta>0.$$

The exact action-Hamiltonian update is

$$z'_j=e^{{-it\,\partial_jK(I)}}z_j,\qquad
\nabla_I K=\nabla h(u)+[h(u)-u^\top\nabla h(u)]\mathbf1.$$

Every action stays fixed within this phase update, so its inverse changes $t$ to $-t$. Complex-unitary changes of coordinates mix actions between updates while retaining the geometric guarantees. General real orthogonal maps need not preserve the canonical complex structure.

If $\|\nabla^2h\|_2\le L$ on the closed simplex, the agents verified the sharp, dimension-independent inequality

$$\left\|\operatorname{{diag}}(\sqrt{{2I}})\,\nabla_I^2K\,
\operatorname{{diag}}(\sqrt{{2I}})\right\|_2\le L.$$

It gives the global certificate

$$\boxed{{e^{{-b}}\le\sigma_j(D\Phi_t)\le e^b,\qquad
b=\operatorname{{arsinh}}(|t|L/2).}}$$

Budgets add under composition; unitary mixing adds zero. This bounds state derivatives, not parameter gradients or optimization success.

The implementation uses signed softplus features with their affine Taylor part at zero removed. For $h(u)=a^\top\operatorname{{softplus}}(Wu+b)$, affine terms do not affect

$$L\le\tfrac14\|W\|_2^2\|a\|_\infty.$$

Full-SVD projection of $W$ to operator norm one and clipping $a$ to $[-4,4]$ enforce $L\le1$. The forward pass uses explicit first derivatives without a Hessian evaluation. A separate learned unitary step supplies linear dynamics because centered phase updates have identity derivative at zero.

## 2. Corrections that matter

**Norm plus canonical symplecticity already imply global phase symmetry.** A symplectic map preserving $N=\|z\|^2/2$ commutes with the Hamiltonian flow of $N$, namely global phase rotation. These are not three independent achievements.

**The common-phase ablation changes only global phase at matched parameters.** Dropping $h-u^\top\nabla h$ preserves intensities and phase-aligned trajectories through unitary mixing and repeated forecasts, although full-space canonical symplecticity is lost. At trained matched weights, the maximum phase-aligned state MSE was below $4.6\times10^{{-29}}$ over 249 steps, and energy differences were below $1.7\times10^{{-14}}$. Separately trained controls have different weights: their accuracy differences cannot establish a direct benefit from breaking symplecticity.

**The earlier linear-baseline win did not establish near-linearity.** The median ratio $\||\psi|^2\psi\|/\|\Delta\psi\|$ is 1.191 on training states and 0.974 on long test trajectories. The nonlinearity is substantial. A fitted linear model can compensate for nonlinear behavior over a restricted data distribution.

**Smaller derivatives are not automatically more faithful.** The certified Hamiltonian model has sampled 200-step condition numbers of 15–19, compared with 85.2 for the numerical reference. Its largest singular values are 3.93–4.31 versus 9.23 for the reference. It is under-sensitive on these sampled states. The unconstrained Hamiltonian and learned Kerr models are closer on this diagnostic. The bounded model's actual certificates are only 0.110–0.116, so it has not exhausted the advertised $B=0.5$ budget.

## 3. Completed experiments

All **27 repository tests passed**, including seven new Hamiltonian tests covering the gradient/Hessian identities, sharp bound, finite differences, symplecticity, inverses, norm, phase symmetry, projection, composition and matched-parameter equivalence. Numerical checks include zero and finite scales through $10^8$; the real-arithmetic global theorem does not promise floating-point safety at arbitrary magnitudes.

We trained **15 new models**: five variants × three optimization seeds. Two known-equation Strang solvers were evaluated separately. The fixed protocol uses the existing DNLS split, 900 Adam updates, batch 64, eight-step loss, learning rate .003 and final checkpoints. Validation was recorded without selection. This is adaptive exploratory research on an already inspected split, not fresh confirmation. Parameter counts are not matched, and no candidate was retuned after seeing these results.

The generic model has four phase stages, width 64 and prescribed budget $B=0.5$. Controls remove projection after initialization or omit the common phase. The learned Kerr model uses an arbitrary learned complex-unitary half-step, one shared learned nonlinear coefficient and the same half-step again. It does not receive the correct Laplacian matrix. Its phase equals Kerr inside the declared total-power ball of 4 and uses a globally bounded $C^1$ continuation outside. All saved trajectories lie inside that ball. The complex-linear control is nested in this model by setting its nonlinear coefficient to zero.

Values are mean ± sample SD over optimization seeds. MSE averages predicted transitions through the stated horizon, trajectories and real state coordinates. Timing is the warm median of ten synchronized 249-step forecasts of all 12 test trajectories, averaged over seeds. New timings were repeated after training stopped; earlier timings are contextual measurements on the same machine.

| Model | Parameters | 20-step MSE | 200-step MSE | Rollout ms |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

The certified Hamiltonian model has **53.5% lower** 200-step MSE than the previous certified coupling, with fewer parameters. But the learned Kerr model has **4.60 times lower** error than the Hamiltonian model, uses 513 parameters and is substantially faster. The prescribed bound does not establish accuracy superiority over the unconstrained or omitted-phase versions.

![Second research screen](hamiltonian_research_results.png)

The bounded Hamiltonian model's worst relative norm-squared drift is $3.23\times10^{{-13}}$ over 249 steps, with inverse recovery below $8.4\times10^{{-13}}$. Symplectic defects and phase-equivariance errors are around $10^{{-15}}$.

**Additional diagnostics.** Phase-aligned and intensity metrics were added after training started, when the matched-parameter equivalence was established. They do not replace the original metric or select models. Alignment chooses one optimal common phase per trajectory and time. Intensity MSE averages squared errors in $|\psi_j|^2$. Physical-energy error uses the noncancelling initial scale $S_0=\sum_j|\psi_{{j+1}}-\psi_j|^2+\tfrac12\sum_j|\psi_j|^4$, avoiding division by a nearly zero signed energy.

| Model | Phase-aligned MSE through 200 | Intensity MSE through 200 | Mean $|\Delta H_{{249}}|/S_0$ |
|---|---:|---:|---:|
{chr(10).join(metrics)}

Kerr remains best among these learned candidates on both additional forecast metrics. The budget improves the physical-energy diagnostic relative to the unconstrained Hamiltonian control, but none of the learned models exactly conserves DNLS energy.

**Known-equation controls.** Correct-coefficient Strang splitting gives 200-step MSE $9.38\times10^{{-9}}$ with one substep and $5.87\times10^{{-10}}$ with two. These zero-training numerical controls know the equation; they are not learned competitors. They show that reference precision and unavoidable coarse-step error do not explain the learned models' errors. Learned Kerr coefficients were approximately 0.93–0.97 versus the true value 1; this is not a general equation-discovery result.

## 4. A deeper expressivity result, with precise scope

The agents checked a compact-domain density argument for a broader family: compositions of unitary-conjugated scalar action-phase gates can approximate smooth $U(1)$-invariant Hamiltonian flows in $C^1$ on a fixed power ball over finite time. Approximate the invariant Hamiltonian by balanced polynomials, express these as combinations of $|v^*z|^{{2k}}$, and compose the exact elementary flows. Smooth extensions outside the physical domain give every factor a finite global certificate without changing its behavior inside.

This shows why exact conservation need not restrict the broader family to linear or integrable dynamics. However, stages and distortion budgets may grow. It is **not** a universality theorem for the fixed four-stage neural implementation or $B=0.5$. The theory note supplies assumptions and proof; the literature audit describes close prior splitting and reduced-sphere results. No neural $C^2$ approximation theorem is silently assumed.

We also examined conjugate action-angle models $F_t=E^{{-1}}D_tE$. They permit direct-time evaluation and at most linear derivative growth under suitable bounds, but impose integrability and exclude chaos and hyperbolic dynamics. They are useful scoped models, not generic replacements for DNLS.

Global regularity and finite-domain physical accuracy can coexist through domain-exact continuation. A fixed small global budget cannot represent arbitrarily strong shear over every amplitude. These constructions also use equal, even-dimensional states; genuine compression cannot retain a globally invertible full-state map without additional information.

## 5. The deeper finding: a certificate can exclude admissible dynamics

The normalized-potential Hessian bound is sufficient for controlling physical derivatives, but it need not describe the full set of maps with the desired distortion. We found an exact separation.

Consider any finite stack of action phases in one unchanged basis, all sharing one normalizer $\delta>0$, with layer certificates $b_\ell=\operatorname{{arsinh}}(|t_\ell|L_\ell/2)$ and $\sum_\ell b_\ell\le B$. Their effective normalized potential satisfies

$$\|\nabla^2 h_{{\mathrm{{eff}}}}\|
\le 2\sum_\ell\sinh b_\ell\le 2\sinh B.$$

At $B=0.5$, that upper bound is $1.042191$, independently of depth. Exact full-state Kerr evolution $z_j\mapsto e^{{ic|z_j|^2}}z_j$ on an open action region forces, up to affine terms,

$$h_*(u)=-c\delta\,\frac{{\|u\|^2}}{{1-\mathbf1^\top u}}.$$

For $N=\sum I_j$ and $\kappa=\|I\|^2/N^2$, its curvature obeys the lower bound

$$\|\nabla^2h_*\|\ge
2|c|N\left[3+2\sqrt{{2+n\kappa}}\right],\qquad\text{{for every shared }}\delta>0.$$

With $n=16$, $c=.04$, and the strictly positive action vector $I=(1.585,.001,\ldots,.001)$, this lower bound is **1.461073**, greater than 1.042191. The point has total power 3.2, inside the declared power-3.4 ball. Neither adding depth nor retuning the shared $\delta$ removes this exact-representation obstruction.

Yet the true Kerr map on that ball requires only

$$B_{{\mathrm{{physical}}}}=\operatorname{{arsinh}}(.04\times3.4)=.135584<.5.$$

A domain-exact global extension attains this budget with $C^1$ regularity, or an arbitrarily slightly larger budget with $C^\infty$ regularity. Thus the difficulty comes from the particular sufficient certificate. It is not a fundamental incompatibility between conservation and controlled gradients.

**Scope is essential.** This rules out exact full-state matching on an open action domain for a shared-normalizer stack with no intermediate mixing. It does not rule out the mixed Stage 6 architecture, heterogeneous normalizers, another certificate, or accurate finite-sample fitting. Integer phase aliases change the effective potential only by affine terms and cannot evade the proof. In contrast, adding a function of total power changes only global phase and can change the curvature: the same obstruction does not automatically apply to phase-aligned prediction. The theorem alone gives no numerical error floor for the finite experiments.

## 6. A simpler alternative, tested on fresh data

For each mode, use a learned scalar phase

$$z'_j=e^{{i\theta(|z_j|^2)}}z_j,\qquad
\theta(s)=\alpha g(\log(1+s/\delta)),\qquad |g'|\le1.$$

It remains canonical symplectic, norm preserving, smooth at zero and explicitly invertible. Its physical shear and distortion satisfy

$$2s|\theta'(s)|\le2|\alpha|,\qquad
b\le\operatorname{{arsinh}}|\alpha|.$$

Parallel modes use the maximum bound, not their sum. Choosing $\alpha=\sinh(.5)$ gives a whole-layer global budget of .5. This controls a first derivative of a scalar function and avoids the shared normalized-potential Hessian restriction. The scalar smooth-function family can match Kerr throughout the power-3.4 ball with this budget. That does not imply exact representation by our finite neural network. Its certificate also has slack: this is a useful alternative, not an optimal intrinsic parameterization.

**Fresh fixed protocol.** We generated 1,024 training states, 256 validation states, 512 in-distribution test states and 512 states at larger power. Training and ID power range from .5 to 3.4; the new range is 3.4 to 4.0. Half are diffuse and half concentrated. The target is the exact isolated Kerr step $z_j\mapsto e^{{i.04|z_j|^2}}z_j$, with no spatial coupling. All choices were declared before fitting; no Stage 7 test-directed tuning occurred.

The 12 runs use four models, three seeds, 900 updates and the same eight-step loss and optimization schedule. The bounded log model uses one shared scalar $1\to32\to32\to1$ tanh network with spectrally projected weights. The perspective models use four width-64 potentials in a fixed basis. The scalar Kerr control learns one coefficient and knows the law's form. Analytical repeated phases are exact evaluations of these maps, not shortened DNLS simulations.

The two raw-error columns give mean ± sample SD. The phase-aligned columns give means. Jacobian error is mean relative Frobenius error against the exact 200-step reference derivative on 32 predeclared ID states, then averaged over seeds.

| Model | Parameters | ID raw MSE200 | Larger-power raw MSE200 | ID aligned MSE200 | Larger-power aligned MSE200 | Jacobian error |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(isolated_table)}

![Fresh isolated phase experiment](isolated_phase_results.png)

The bounded log-intensity model has **4.17 times lower ID raw error** and **5.59 times lower larger-power raw error** than the projected perspective model. Its reference-Jacobian error also decreases. However, it has **worse phase-aligned error at larger power**: .013064 versus .003916. Much of the perspective model's raw extrapolation error is global phase error. Both metrics must remain visible.

Changing to a shared scalar function also changes parameter count and inductive bias. This is not a clean coordinate-only ablation. Removing projection helps the perspective model but does not close the gap, and every fitted neural model's actual-weight certificate remains below .5:

| Model | Actual global one-step log certificate, range across seeds |
|---|---:|
{chr(10).join(certificate_table)}

The scalar Kerr control has a finite certificate on the declared power ball, but its uncapped cubic phase has no global bound. It wins decisively, as expected for a model containing the exact target. Its learned coefficients are .9833, .9878 and .9874; residual error after 900 updates is not a representational impossibility.

All 12 saved checkpoints reproduce their predictions and final validation losses. Dataset/source/protocol hashes, split uniqueness, means and sample SDs were independently audited. Recomputed metrics agree to $5.6\times10^{{-17}}$. This is a fresh mechanism experiment, not a new full-DNLS benchmark or proof that the obstruction explains the entire learning gap.

## 7. A more general next candidate: interacting action phases

The scalar log model treats each mode separately. A further theoretical round produced a construction with genuine cross-action interactions and a direct physical-Hessian certificate:

$$K(I)=\sum_r f_r(w_r^\top I),\qquad
w_r\ge0,\qquad
a_r=\sup_{{s\ge0}}s|f_r''(s)|<\infty.$$

Assume the scalar functions are smooth at zero as well as away from it. With $D=\operatorname{{diag}}(\sqrt{{2I}})$, weighted Cauchy–Schwarz gives

$$\boxed{{\|D\nabla^2K(I)D\|_2
\le2\max_j\sum_r a_r w_{{rj}},\qquad
b\le\operatorname{{arsinh}}\!\left(|t|\max_j\sum_r a_r w_{{rj}}\right).}}$$

Its exact update is

$$z'_j=\exp\!\left[-it\sum_r w_{{rj}}f_r'(w_r^\top I)\right]z_j.$$

It preserves actions, norm and symplectic structure, has an explicit inverse, and is smooth at the origin. A ridge such as $w=(1,2)$ couples relative phases to both actions; this is more than a common phase. Interleaving complex-unitary maps allows actions to change between stages.

A trainable realization can use

$$f_r'(s)=\beta_r g_r(\log(1+s/\delta_r)),\qquad |g_r'|\le1,$$

which gives $a_r\le|\beta_r|$. Positive directions span homogeneous action polynomials when signed scalar functions are allowed. Smooth extensions then give compact-domain approximation of smooth action Hamiltonians, with width and certificate allowed to grow. This is a function-family result, not a theorem about a particular finite neural network or a fixed budget.

This family can represent the independent Kerr map on the declared domain at its intrinsic budget using a $C^1$ domain-exact extension, or with arbitrarily small extra budget for a smooth extension. It avoids the shared-normalizer obstruction. The single-ridge bound is sharp.

**A second counterexample prevents overclaiming.** The inexpensive bound on a sum of ridges still pays for cancellations. For $K=I_1I_2$ on $I_1+I_2\le P$, the true maximum weighted-Hessian norm is $P$, but this certificate for a signed positive-ridge representation must be at least $2P$. It therefore does not solve optimal certification or fixed-budget approximation. Positivity of the ridge weights is essential to the bound.

The construction and proof were independently checked. An untrained four-mode numerical example on 101 states verified the derivative bound, norm, inverse and symplecticity; finite differences agreed within $1.3\times10^{{-11}}$. **No forecasting model from this interacting family has been trained.** The [separate derivation](action_ridge_candidate.md) includes the approximation scope and cancellation example. This is the most useful new candidate for the next controlled comparison, with no claim that ridge Hamiltonians themselves are new.

## 8. Prior art and the research decision

Normalized-action Hamiltonians and the common phase are explicit in [Abrams–Lloyd (1998), Eqs. 15–20](https://arxiv.org/pdf/quant-ph/9801041). Learned phase/linear splitting appears in [Häger–Pfister](https://arxiv.org/html/2010.14258); nonlinear unitary circuits in [Yu–Piao–Park](https://arxiv.org/pdf/2412.04112); curvature-controlled bi-Lipschitz symplectic networks in [Kevrekidis et al., Appendix B.4](https://arxiv.org/html/2406.19957). Exact-flow composition and universality have substantial prior art, including [Tapley](https://arxiv.org/html/2408.09821).

Logarithmic nonlinear phases and their exact splitting updates also have direct predecessors in [Bao et al., Eq. 2.8](https://arxiv.org/pdf/1803.10068) and [their splitting analysis, Eq. 2.3 and Lemma 2.1](https://arxiv.org/pdf/1811.12711). The new log-profile experiment does not make the underlying phase formula new.

The plausible paper is about **the difference between physical distortion and restrictions introduced by a normalization certificate**, with a constructive alternative and an honest expressivity–accuracy tradeoff. The strongest package would combine the sharp perspective bound, the shared-normalizer separation theorem, the compact-domain approximation theorem, and a method that improves useful tasks at comparable physical distortion. The interacting action-ridge construction provides a next candidate, while its cancellation example keeps the optimal-certification problem open. The search has not established that this complete result is new. We should not claim invention of Hamiltonian normalization, nonlinear unitary networks, certified symplectic networks or logarithmic phase maps.

**Next decisive study.** The isolated-Kerr test is now complete. The next comparison should hold sharing, parameter count and optimization effort as closely matched as possible while changing the certificate. Use fresh data with at least one non-Kerr scalar phase and one coupled-action Hamiltonian, followed by a full nonlinear propagation problem. Include the inexpensive learned scalar-phase split and established symplectic baselines. Fix the physical power domain and required derivative range first, report full-state and phase-aligned fidelity together, and compare learned Jacobians with the target Jacobian. Report the whole accuracy–distortion curve instead of selecting a favorable budget.

Do not expand the original manuscript's claims yet. The promising result is a precise explanation of why a reasonable normalization can be too restrictive, together with a route around that restriction. A generic method still has to earn its extra complexity. These results justify continuing the research; they do not yet justify a broad superiority or submission-readiness claim.

## Files and reproducibility

The accompanying [theory note](hamiltonian_theory.md) gives proofs, hypotheses and counterexamples. The [prior-art audit](hamiltonian_prior_art.md) records related constructions and novelty limits. Fixed protocols are in [the DNLS protocol](hamiltonian_experiment_protocol.md) and [the isolated-phase protocol](isolated_phase_protocol.md). Per-seed metrics are exported as CSV; the experiment archive includes datasets, checkpoints, predictions, source snapshots, environment lockfile and a SHA256 manifest.
'''
    (out/'hamiltonian_research_report.md').write_text(report)
    for source,dest in [('theory_candidates.md','hamiltonian_theory.md'),('prior_art.md','hamiltonian_prior_art.md')]:
        shutil.copy2(args.theory_notes/source,out/dest)
    shutil.copy2(args.theory_notes/'action_ridge_candidate.md',out/'action_ridge_candidate.md')
    shutil.copy2(run.parent/'PROTOCOL.md',out/'hamiltonian_experiment_protocol.md')
    shutil.copy2(isolated.parent/'PROTOCOL.md',out/'isolated_phase_protocol.md')
    with (out/'isolated_phase_metrics.csv').open('w',newline='') as f:
        fields=['mode','seed','n_parameters','id_mse_200','ood_mse_200','id_aligned_mse_200',
                'ood_aligned_mse_200','relative_jacobian_error_200',
                'certified_log_bound_one_step_global','certified_log_bound_power_ball_4']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in isolated_rows:
            writer.writerow({'mode':r['mode'],'seed':r['seed'],'n_parameters':r['n_parameters'],
                'id_mse_200':r['test_id']['mse_200'],'ood_mse_200':r['test_ood']['mse_200'],
                'id_aligned_mse_200':r['test_id']['aligned_mse_200'],
                'ood_aligned_mse_200':r['test_ood']['aligned_mse_200'],
                'relative_jacobian_error_200':r['jacobian']['200']['relative_frobenius_error_mean'],
                'certified_log_bound_one_step_global':r['certified_log_bound_one_step_global'],
                'certified_log_bound_power_ball_4':r['certified_log_bound_power_ball_4']})
    with (out/'hamiltonian_experiment_metrics.csv').open('w',newline='') as f:
        fields=['mode','seed','n_parameters','mse_20','mse_200','rollout_ms','max_relative_power_drift',
                'mean_symplectic_defect','relative_phase_equivariance_error','certified_log_bound_one_step',
                'mean_scaled_final_energy_error']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(out/'hamiltonian_research_report.md')


if __name__=='__main__':
    main()
