"""Build the user-facing Markdown report and scientific plots from saved runs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ORDER=["coupling_bounded","coupling_unbounded","coupling_raw",
       "geodesic_skipnorm","projected_residual","linear_orthogonal"]
NAMES={"coupling_bounded":"Bounded coupling", "coupling_unbounded":"Unconstrained coupling",
       "coupling_raw":"Raw conditioner", "geodesic_skipnorm":"Geodesic SkipNorm",
       "projected_residual":"Projected residual", "linear_orthogonal":"Linear orthogonal"}
COLORS=dict(zip(ORDER,["#007f73","#4aa8a1","#9c7d9b","#cd5d2b","#d59b44","#335b8e"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"results/stage5_rotation_coupling/pilot_v1")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir; out=args.output
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads((run/"summary.json").read_text())
    summary=data["summary"]; diagnostics=json.loads((run/"diagnostics.json").read_text())
    records={m:[json.loads(p.read_text()) for p in sorted(run.glob(m+"_seed*.json"))] for m in ORDER}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","axes.labelcolor":"#26323a",
        "text.color":"#26323a","axes.edgecolor":"#8a969d","grid.alpha":.18})
    fig,axes=plt.subplots(1,3,figsize=(15.5,4.5),layout="constrained")
    for m in ORDER:
        curves=np.stack([np.load(run/f"{m}_seed{r['seed']}.npz")["mse_trace"] for r in records[m]])
        curves=np.cumsum(curves,axis=1)/np.arange(1,250)
        mean=curves.mean(axis=0); sd=curves.std(axis=0,ddof=1)
        axes[0].plot(np.arange(1,250),mean,color=COLORS[m],label=NAMES[m],lw=2 if m==ORDER[0] else 1.4)
        axes[0].fill_between(np.arange(1,250),np.maximum(mean-sd,1e-12),mean+sd,color=COLORS[m],alpha=.10)
    axes[0].set(xscale="log",yscale="log",xlabel="Forecast transitions",ylabel="Cumulative state MSE",title="Forecast accuracy")
    axes[0].legend(fontsize=7.5,loc="upper left"); axes[0].grid(True,which="both")
    depths=np.array([1,10,50,200])
    for m in ["coupling_bounded","coupling_unbounded","geodesic_skipnorm","linear_orthogonal"]:
        ys=np.array([[r["repeated_time_step_jacobian"][str(n)]["maximum_condition_number"] for n in depths] for r in records[m]])
        axes[1].plot(depths,np.median(ys,axis=0),marker="o",ms=4,color=COLORS[m],label=NAMES[m])
        axes[1].fill_between(depths,ys.min(axis=0),ys.max(axis=0),color=COLORS[m],alpha=.12)
    truth=[diagnostics["reference_jacobians"][str(n)]["maximum_condition_number"] for n in depths]
    axes[1].plot(depths,truth,"k--",lw=1.7,label="Reference dynamics")
    axes[1].set(xscale="log",yscale="log",xlabel="Repeated transitions",ylabel="Jacobian condition number",title="Sensitivity along two test states")
    axes[1].legend(fontsize=7.5,loc="upper left"); axes[1].grid(True,which="both")
    axes[1].text(.98,.02,"Median and range across seeds",ha="right",transform=axes[1].transAxes,fontsize=7.5)
    vals=[summary[m]["rollout_ms"]["mean"] for m in ORDER]
    errs=[summary[m]["rollout_ms"]["sample_sd"] for m in ORDER]
    axes[2].barh(np.arange(len(ORDER)),vals,xerr=errs,color=[COLORS[m] for m in ORDER],height=.64,capsize=3)
    axes[2].set_yticks(np.arange(len(ORDER)),[NAMES[m] for m in ORDER],fontsize=8)
    axes[2].invert_yaxis(); axes[2].set(xlabel="Warm rollout time (ms)",title="Inference cost")
    axes[2].grid(True,axis="x"); axes[2].set_axisbelow(True)
    for i,v in enumerate(vals): axes[2].text(v+.2,i,f"{v:.2f}",va="center",fontsize=8)
    axes[2].set_xlim(0,max(vals)*1.18)
    fig.suptitle("DNLS pilot: three optimization seeds on one fixed split",fontsize=14,fontweight="bold")
    fig.savefig(out/"rotation_coupling_results.png",dpi=180)
    fig.savefig(out/"rotation_coupling_results.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout="constrained")
    stress=diagnostics["analytical_scale_stress"]
    axes[0].loglog(stress["radius"],stress["raw_sigma_max"],color=COLORS["coupling_raw"],label="Raw conditioner")
    axes[0].loglog(stress["radius"],stress["normalized_sigma_max"],color=COLORS["coupling_bounded"],label="Normalized conditioner")
    axes[0].axhline(stress["normalized_global_sigma_max"],color="#45515b",ls="--",lw=1,label="Global normalized bound")
    axes[0].set(xlabel="Input radius",ylabel="Largest Jacobian singular value",title="Exact adverse scale example")
    axes[0].legend(fontsize=8); axes[0].grid(True,which="both")
    stress=diagnostics["analytical_depth_stress"]
    axes[1].plot(stress["depth"],stress["fixed_stage_step_sigma_max"],color=COLORS["coupling_raw"],marker="o",label="Fixed step per layer")
    axes[1].plot(stress["depth"],stress["fixed_total_budget_sigma_max"],color=COLORS["coupling_bounded"],marker="o",label="Fixed total budget B = 0.5")
    axes[1].axhline(stress["fixed_total_budget_bound"],color="#45515b",ls="--",lw=1,label="Fixed-budget bound")
    axes[1].set(xscale="log",yscale="log",xlabel="Number of layers",ylabel="Largest Jacobian singular value",title="Exact repeated-shear example")
    axes[1].legend(fontsize=8); axes[1].grid(True,which="both")
    fig.suptitle("Analytical stress cases — no training or fitted curves",fontsize=13,fontweight="bold")
    fig.savefig(out/"rotation_coupling_guarantees.png",dpi=180)
    fig.savefig(out/"rotation_coupling_guarantees.pdf"); plt.close(fig)

    rows=[]
    for m in ORDER:
        s=summary[m]
        rows.append(f"| {NAMES[m]} | {s['n_parameters']:,} | {s['mse_20']['mean']:.6f} ± {s['mse_20']['sample_sd']:.6f} | {s['mse_200']['mean']:.6f} ± {s['mse_200']['sample_sd']:.6f} | {s['rollout_ms']['mean']:.2f} ± {s['rollout_ms']['sample_sd']:.2f} |")
    table="\n".join(rows)
    b=records["coupling_bounded"]
    improvement=100*(1-summary['coupling_bounded']['mse_200']['mean']/summary['geodesic_skipnorm']['mse_200']['mean'])
    vslinear=100*(summary['coupling_bounded']['mse_200']['mean']/summary['linear_orthogonal']['mse_200']['mean']-1)
    speed=summary['coupling_bounded']['rollout_ms']['mean']/summary['geodesic_skipnorm']['rollout_ms']['mean']
    lin=[r for r in diagnostics['origin_linearization'] if r['mode']=='coupling_bounded']
    linearized=np.mean([r['linearized_mse_200'] for r in lin])
    measured=np.array([r['repeated_time_step_jacobian']['200']['maximum_condition_number'] for r in b])
    gsn=np.array([r['repeated_time_step_jacobian']['200']['maximum_condition_number'] for r in records['geodesic_skipnorm']])
    unc=np.array([r['repeated_time_step_jacobian']['200']['maximum_condition_number'] for r in records['coupling_unbounded']])
    energyrows=[]
    for m in ORDER:
        er=[r for r in diagnostics['energy_diagnostics'] if r['mode']==m]
        energyrows.append(f"| {NAMES[m]} | {np.mean([r['mean_absolute_energy_error'] for r in er]):.4f} | {np.mean([r['mean_energy_error_over_noncancelling_initial_scale'] for r in er]):.4f} |")
    report=rf"""# Reversible norm-preserving couplings: checked construction and first experiments

15 October 2026. Exploratory screen; three optimization seeds, one existing DNLS split.

**Assessment.** The mathematical construction survived the deeper audit and is now implemented. Its inverse, conservation, volume and Jacobian guarantees pass the numerical checks. The first training experiment shows a useful stability–accuracy tradeoff, but does not establish a superior forecasting method or a submission-ready new paper.

**What the deeper check confirmed.** In each layer, split the state into unchanged coordinates $a$ and rotated planar pairs $b_j$. With a positive fixed $\delta$, use

$$c(x)=\frac{{(a,\delta)}}{{\sqrt{{\delta^2+\|x\|^2}}}},\qquad
\theta=\alpha h(c(x)),\qquad
F(x)=(a,R_{{\theta_1}}b_1,\ldots,R_{{\theta_K}}b_K).$$

The whole vector-valued conditioner must be $L$-Lipschitz in Euclidean norm. Separate scalar-output bounds would not suffice. The implementation projects **every** weight matrix, including the output matrix, onto the operator-norm unit ball with a full SVD, at initialization and after every update of the bounded variant. Biases are unrestricted and activations are tanh.

The unchanged conditioner gives an explicit inverse. The Jacobian is an orthogonal factor times $I+N$, where $N^2=0$ and $\|N\|\le|\alpha|L$. This gives norm preservation, determinant one and full rank, with each nontrivial singular pair

$$\sigma_\pm=\frac{{\sqrt{{t^2+4}}\pm t}}{{2}},\qquad t\le|\alpha|L.$$

For a composition,

$$e^{{-B}}\le\sigma_i(DF)\le e^B,\qquad
B=\sum_\ell\operatorname{{arsinh}}(|\alpha_\ell|L_\ell/2).$$

The eight-stage bounded propagator uses $B=0.5$, hence a whole-step interval $[0.6065,1.6487]$. A learned, state-independent orthogonal mixing matrix adds no distortion. Positive $\delta$ makes conditioning smooth at zero without changing conservation.

**Qualifications that matter.** The bound controls state gradients, not parameter gradients or optimization success. An $n$-step forecast has worst-case budget $nB$; the bound is not constant over unlimited physical time. Norm and volume preservation imply preservation of surface measure on every sphere and every radial probability density. This is a restricted family of conservative maps. It does not automatically preserve physical energy, the canonical symplectic form, or DNLS phase symmetry. Exact Euclidean isometry in every direction would preclude nonlinear expressivity in equal dimensions.

There is also a precise amplitude limitation. The simple physical map $z\mapsto e^{{it|z|^2}}z$ has shear magnitude $2|t|r^2$. Exact first-order agreement on a radius-$R$ ball requires $B\ge\operatorname{{arsinh}}(|t|R^2)$. A fixed global budget cannot represent this dynamics over all amplitudes. The budget is therefore an explicit modeling restriction, to be chosen for the physical domain.

**Validation.** All 20 repository tests passed, including six new tests: finite-difference Jacobians, positive/negative angle multipliers, zero pairs, inputs from zero through $10^{{200}}$, inverse derivatives, exact singular spectra, vector conditioning, spectral projection, and composition. These checks support the algebra; they are not a substitute for the proof in the accompanying theory note.

For the three trained bounded models, the worst relative power drift over 249 steps was {max(r['max_relative_power_drift'] for r in b):.2e}; the worst forward–inverse recovery error after those steps was {max(r['full_rollout_inverse_relative_error'] for r in b):.2e}. Actual-weight log certificates ranged from {min(r['certified_log_bound_one_step'] for r in b):.4f} to {max(r['certified_log_bound_one_step'] for r in b):.4f}, below the prescribed 0.5.

![Analytical guarantee checks](rotation_coupling_guarantees.png)

The analytical stress example fixes the input state exactly while its derivative changes: at $(a,b)=(0,(r,0))$, a raw angle $\alpha a$ has shear $|\alpha|r$, whereas the normalized angle has shear $|\alpha|r/\sqrt{{\delta^2+r^2}}$. Bounded tanh angles give the same derivative at this point. Thus bounded angle values alone do not imply bounded gradients. Random directions need not expose this failure; the plotted case is deliberately adversarial.

**Experimental design.** Nine models were newly trained: three coupling variants × seeds 0, 1, 2. Nine existing baseline checkpoints were reevaluated after checking source, split and training-configuration hashes. Every model used the same locked split (28 training, 8 validation, 12 test trajectories), 900 updates, Adam, batch 64, eight-step rollout loss, and learning rate 0.003 with the same decay and window sampling. Final checkpoints were used; validation curves did not select models. There was no test-directed tuning. New models use 16,192 parameters versus 15,585 for SkipNorm and projected residual; the linear baseline is much smaller. A two-update execution smoke check is excluded.

Each new model has eight coupling stages, width 32 and two hidden layers. All three start from identical projected weights and fixed coordinate permutations. The unconstrained normalized model stops projecting after initialization; the raw-conditioner model also removes invariant input normalization. The normalized control still has a finite global bound from its actual weight norms, but no enforced $B=0.5$ budget. All use the same angle multiplier and linear mixing. Counts include all stored trainable parameters; the Cayley matrix parametrization contains redundancy.

MSE is averaged across predicted transitions, test trajectories and 32 state coordinates. Values below are mean ± sample SD across optimization seeds. They do not quantify uncertainty over fresh datasets. Timing is the warm median of ten synchronized forecasts of all 12 test trajectories for 249 steps, then summarized across seeds; compilation is excluded. Recorded training times include validation and compilation and are not a fair comparison with cached baselines.

| Model | Parameters | MSE through 20 steps | MSE through 200 steps | Rollout ms |
|---|---:|---:|---:|---:|
{table}

![DNLS screen](rotation_coupling_results.png)

**What the numbers support.** Bounded coupling has {improvement:.1f}% lower 200-step MSE than Geodesic SkipNorm, but {vslinear:.1f}% higher MSE than the separately trained linear model. Its 20-step error is also higher than both, and inference is about {speed:.2f} times slower than SkipNorm. The bounded and unconstrained normalized models have similar long-horizon errors; three seeds do not establish a robust accuracy advantage from enforcing the bound. No overall superiority claim is justified.

The sensitivity result is more promising. On the same two held-out initial states (test indices 0 and 3), the largest sampled 200-step condition number was {diagnostics['reference_jacobians']['200']['maximum_condition_number']:.1f} for the numerical reference dynamics. Across seeds it ranged {measured.min():.1f}–{measured.max():.1f} for bounded coupling, {unc.min():.1f}–{unc.max():.1f} for unconstrained normalized coupling, and {gsn.min():.0f}–{gsn.max():.0f} for SkipNorm. These are empirical derivatives of repeated model evolution, not a universal long-rollout guarantee or an accuracy metric. Only two initial states were checked. Reference trajectories reproduced the locked arrays within about $10^{{-15}}$ at these evaluation points.

**Post-hoc nonlinear diagnostic.** After seeing the strong linear baseline, we replaced each trained coupling model by its exact derivative at the origin, without retraining. That derivative is orthogonal. For bounded coupling, mean 200-step MSE increased from {summary['coupling_bounded']['mse_200']['mean']:.5f} to {linearized:.5f}. The trained model depends materially on its state-dependent update. This does not prove that nonlinearity improves on the best linear fit; the independently trained linear model remains better. The origin is outside the training-state range, so this is a diagnostic rather than a matched linear ablation.

**Physical properties that are not solved.** The bounded models' phase-equivariance errors were about 1.4% on sampled states, and their mean canonical symplectic defects were about 0.047. They therefore do not obey those symmetries exactly. Physical energy is not conserved by any tested learned architecture. The initially recorded relative-energy diagnostic divides by $|H(0)|$, which is small for some test trajectories and exaggerates the ratio. The post-hoc diagnostics retain that number and additionally report absolute energy error and a noncancelling reference scale:

$$S_0=\sum_j |\psi_{{j+1}}-\psi_j|^2+\tfrac12\sum_j|\psi_j|^4.$$

At 249 steps:

| Model | Mean absolute energy error | Mean $|\Delta H|/S_0$ |
|---|---:|---:|
{chr(10).join(energyrows)}

**Research decision.** Keep this construction and its proofs. The pilot justifies a focused follow-up on useful conservative representations with controlled sensitivity; it does not justify replacing the current paper's claims with a method-superiority claim. The next experiment should use fresh, predeclared trajectories spanning stronger nonlinear regimes, and compare against both linear and structure-preserving nonlinear baselines. If DNLS remains the main application, canonical symplecticity and phase symmetry should be addressed explicitly, because the current real-coordinate couplings do not enforce them. A future budget sweep must use validation data and report its accuracy–conditioning tradeoff rather than selecting the best test result.

**Prior work and novelty.** Conditional rotations are established in [G-LPNets, Section 6](https://arxiv.org/html/2308.15349#S6); invertible sphere/torus transformations appear in [Rezende et al. (2020)](https://proceedings.mlr.press/v119/rezende20a.html); composing explicit Lipschitz bounds is standard in [invertible residual networks](https://proceedings.mlr.press/v97/behrmann19a.html). The candidate contribution is the smooth invariant-normalized conditioner and enforceable global two-sided Jacobian bound, together with useful empirical consequences. The targeted literature search found no exact match, but did not establish exhaustive novelty.

**Saved materials.** The accompanying package contains the implementation, tests, fixed protocol, source snapshots, dataset, per-seed checkpoints, predictions, metrics and plot/report generators. The separate mathematical note gives the proof and limits in detail. Existing manuscript experiments were not replaced by these exploratory results.
"""
    (out/"rotation_coupling_experiment_report.md").write_text(report)
    shutil.copy2(ROOT/"results/stage5_rotation_coupling/THEORY.md",out/"rotation_coupling_theory.md")
    shutil.copy2(ROOT/"results/stage5_rotation_coupling/PROTOCOL.md",out/"rotation_coupling_protocol.md")
    with (out/"rotation_coupling_metrics.csv").open("w",newline="") as handle:
        fields=["mode","seed","n_parameters","mse_20","mse_50","mse_200","rollout_ms","max_relative_power_drift","full_rollout_inverse_relative_error","certified_log_bound_one_step"]
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore"); writer.writeheader()
        for m in ORDER:
            writer.writerows(records[m])
    print(out/"rotation_coupling_experiment_report.md")


if __name__=="__main__":
    main()
