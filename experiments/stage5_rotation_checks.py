"""Post-screen diagnostics, with no retraining or model selection.

Analytical stress checks were proposed before full training. Origin linearization
was added after the first screen showed a strong linear baseline; it is explicitly
a post-hoc diagnostic, not an additional trained competitor.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from experiments.stage5_rotation_coupling import make_model, digest, singular_summary, dnls_energy


def exact_shear_max(beta):
    return (np.sqrt(beta*beta + 4) + beta)/2


def dnls_step(x):
    def rhs(z):
        q, p = z[:16], z[16:]
        amp = q*q+p*p
        lapq = jnp.roll(q, -1)-2*q+jnp.roll(q, 1)
        lapp = jnp.roll(p, -1)-2*p+jnp.roll(p, 1)
        return jnp.concatenate([-lapp-amp*p, lapq+amp*q])
    def substep(_, z):
        dt=.004
        a=rhs(z); b=rhs(z+.5*dt*a); c=rhs(z+.5*dt*b); d=rhs(z+dt*c)
        return z+dt*(a+2*b+2*c+d)/6
    return jax.lax.fori_loop(0, 10, substep, x)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"results/stage5_rotation_coupling/pilot_v1")
    args=parser.parse_args()
    out=args.run_dir
    config=json.loads((out/"config.json").read_text())
    split=np.load(ROOT/"data/splits/dnls_1d.npz")
    test=split["test_orbits_long"]
    records=[]
    for p in sorted(out.glob("coupling*_seed*.json")):
        record=json.loads(p.read_text())
        model=make_model(record["mode"], record["seed"], config["coupling"]["budget"])
        model=eqx.tree_deserialise_leaves(p.with_suffix(".eqx"), model)
        matrix=eqx.filter_jit(lambda m: jax.jacfwd(m.step)(jnp.zeros(32)))(model)
        def linear_forecast(q, initial):
            def step(x, _):
                y=x@q.T
                return y,y
            return jnp.swapaxes(jax.lax.scan(step, initial, None, length=249)[1],0,1)
        pred=np.asarray(jax.jit(linear_forecast)(matrix,jnp.asarray(test[:,0])))
        mse=np.mean((pred-test[:,1:])**2,axis=(0,2))
        records.append({"mode":record["mode"],"seed":record["seed"],
            "origin_linearization_orthogonality_error":float(jnp.linalg.norm(matrix.T@matrix-jnp.eye(32))),
            "linearized_mse_20":float(mse[:20].mean()),
            "linearized_mse_200":float(mse[:200].mean()),
            "full_mse_20":record["mse_20"],"full_mse_200":record["mse_200"]})

    # Relative error divided by |H(0)| is ill-conditioned near zero energy.
    # Add an explicit noncancelling initial-energy scale, without deleting the
    # originally recorded diagnostic or changing any model.
    q,p=test[:,0,:16],test[:,0,16:]
    energy0=dnls_energy(test[:,0])
    energy_scale=np.sum((np.roll(q,-1,axis=-1)-q)**2+
        (np.roll(p,-1,axis=-1)-p)**2+.5*(q*q+p*p)**2,axis=-1)
    energy=[]
    for file in sorted(out.glob("*_seed*.json")):
        record=json.loads(file.read_text())
        predicted=np.load(file.with_suffix(".npz"))["prediction"]
        error=np.abs(dnls_energy(predicted[:,-1])-energy0)
        energy.append({"mode":record["mode"],"seed":record["seed"],
            "mean_absolute_energy_error":float(error.mean()),
            "mean_energy_error_over_noncancelling_initial_scale":float(np.mean(error/energy_scale))})

    truth={}
    # Match the exact states used by the main driver's repeated-step Jacobians.
    reference_indices=np.array([0,3])
    points=jnp.asarray(test[reference_indices,0])
    for n in [1,10,50,200]:
        def terminal(x):
            return jax.lax.fori_loop(0,n,lambda _,z:dnls_step(z),x)
        jac=jax.jit(jax.vmap(jax.jacfwd(terminal)))(points)
        predicted=np.asarray(jax.jit(jax.vmap(terminal))(points))
        truth[str(n)]={**singular_summary(jac),
            "maximum_reference_state_difference":float(np.max(np.abs(predicted-test[reference_indices,n])))}

    # At x=(0,r,0), theta=alpha*a/s versus theta=alpha*a has zero
    # rotation but nonzero derivative. This is an exact, adverse example.
    radii=np.logspace(-6,6,97)
    alpha=.5; delta=1.
    normalized_beta=alpha*radii/np.sqrt(delta*delta+radii*radii)
    raw_beta=alpha*radii
    scale={"radius":radii.tolist(),"normalized_sigma_max":exact_shear_max(normalized_beta).tolist(),
        "raw_sigma_max":exact_shear_max(raw_beta).tolist(),
        "normalized_global_sigma_max":float(np.exp(np.arcsinh(alpha/2)))}
    depths=np.array([1,2,4,8,16,32,64,128])
    budget=.5; radius=10.
    factor=radius/np.sqrt(1+radius*radius)
    fixed_budget_beta=depths*2*np.sinh(budget/depths)*factor
    fixed_stage_beta=depths*.1250814*factor
    depth={"depth":depths.tolist(),"fixed_total_budget_sigma_max":exact_shear_max(fixed_budget_beta).tolist(),
        "fixed_stage_step_sigma_max":exact_shear_max(fixed_stage_beta).tolist(),
        "fixed_total_budget_bound":float(np.exp(budget)),
        "note":"Repeated zero-angle shears at a fixed state; an analytical stress example, not training."}
    output={"config_sha256":digest(out/"config.json"),"source_sha256":digest(__file__),
        "status":"post-hoc diagnostic only; no training or model selection",
        "origin_linearization":records,"reference_jacobians":truth,
        "reference_trajectory_indices":reference_indices.tolist(),
        "energy_diagnostics":energy,"initial_test_energies":energy0.tolist(),
        "initial_noncancelling_energy_scales":energy_scale.tolist(),
        "analytical_scale_stress":scale,"analytical_depth_stress":depth}
    (out/"diagnostics.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"origin_linearization":records,"reference_jacobians":truth},indent=2))


if __name__=="__main__":
    main()
