"""Fresh fixed-basis Kerr identification; see results/stage7_isolated_kerr/PROTOCOL.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

from skipnorm.hamiltonian_normalization import PerspectivePhase

BASE = ROOT / "results/stage7_isolated_kerr"
MODES = ["perspective_bounded", "perspective_unconstrained", "scalar_kerr", "log_intensity_bounded"]
SPLITS = {"train": (1024, 817001, .5, 3.4), "validation": (256, 817002, .5, 3.4),
          "test_id": (512, 817003, .5, 3.4), "test_ood": (512, 817004, 3.4, 4.)}
JAC_INDICES = list(range(16)) + list(range(256, 272))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PotentialModel(eqx.Module):
    phases: tuple[PerspectivePhase, ...]
    bounded: bool = eqx.field(static=True)

    def __init__(self, bounded, *, key):
        self.phases = tuple(PerspectivePhase(dim=16, width=64, delta=1.,
            time=2*math.sinh(.5/4), project=True, key=k) for k in jax.random.split(key, 4))
        self.bounded = bool(bounded)

    def angles(self, x):
        actions = .5*(x[:16]**2+x[16:]**2)
        return -sum(p.time*p.phase_frequencies(actions) for p in self.phases)

    def step(self, x):
        return rotate(x, self.angles(x))

    def inverse(self, x):
        return rotate(x, -self.angles(x))

    def project_parameters(self):
        if not self.bounded:
            return self
        return eqx.tree_at(lambda m:m.phases, self, tuple(p.project_lipschitz() for p in self.phases))

    def certified_log_bound(self):
        return sum(p.certified_log_bound() for p in self.phases)


class ScalarKerr(eqx.Module):
    raw_gamma: jax.Array

    def __init__(self):
        self.raw_gamma = jnp.asarray(0.)

    def angles(self, x):
        return .04*2*jnp.tanh(self.raw_gamma)*(x[:16]**2+x[16:]**2)

    def step(self, x):
        return rotate(x, self.angles(x))

    def inverse(self, x):
        return rotate(x, -self.angles(x))

    def project_parameters(self):
        return self

    def certified_log_bound(self):
        return None


class LogIntensityPhase(eqx.Module):
    """Shared scalar phase with a global Cartesian log bound <= 0.5."""
    network: eqx.nn.MLP

    def __init__(self, *, key):
        self.network = eqx.nn.MLP(1,1,32,2,activation=jnp.tanh,key=key)
        self.network = self.project_parameters().network

    def angles(self, x):
        intensity = x[:16]**2+x[16:]**2
        return math.sinh(.5)*jax.vmap(lambda s:self.network(jnp.log1p(s)[None])[0])(intensity)

    def step(self, x):
        return rotate(x,self.angles(x))

    def inverse(self, x):
        return rotate(x,-self.angles(x))

    def project_parameters(self):
        def project(weight):
            u,s,vh = jnp.linalg.svd(weight,full_matrices=False)
            return (u*jnp.minimum(s,1.))@vh
        layers = tuple(eqx.tree_at(lambda l:l.weight,layer,project(layer.weight))
                       for layer in self.network.layers)
        return eqx.tree_at(lambda m:m.network.layers,self,layers)

    def certified_log_bound(self):
        lipschitz = math.prod(float(np.linalg.norm(np.asarray(layer.weight),ord=2))
                              for layer in self.network.layers)
        return math.asinh(math.sinh(.5)*lipschitz)


def rotate(x, angles):
    q, p = x[:16], x[16:]
    return jnp.concatenate((jnp.cos(angles)*q-jnp.sin(angles)*p,
                            jnp.sin(angles)*q+jnp.cos(angles)*p))


def target_angles(x):
    return .04*(x[:16]**2+x[16:]**2)


def make_model(mode, seed):
    if mode == "scalar_kerr":
        return ScalarKerr()
    key = jax.random.PRNGKey(12000+89*seed)
    if mode == "log_intensity_bounded":
        return LogIntensityPhase(key=key)
    return PotentialModel(mode == "perspective_bounded",key=key)


def dataset():
    result = {}
    for name, (count, seed, low, high) in SPLITS.items():
        rng = np.random.default_rng(seed)
        power = rng.uniform(low, high, count)
        half = count//2
        diffuse = rng.normal(size=(half,16))+1j*rng.normal(size=(half,16))
        diffuse *= np.sqrt(power[:half]/np.sum(abs(diffuse)**2, axis=-1))[:,None]
        intensity = rng.dirichlet(np.full(16,.2), size=half)*power[half:,None]
        concentrated = np.sqrt(intensity)*np.exp(1j*rng.uniform(-np.pi,np.pi,(half,16)))
        z = np.concatenate((diffuse,concentrated))
        result[name] = np.concatenate((z.real,z.imag),axis=-1)
        result[name+"_power"] = power
        result[name+"_component"] = np.concatenate((np.zeros(half,dtype=int),np.ones(half,dtype=int)))
    result["jacobian_test_id_indices"] = np.asarray(JAC_INDICES)
    return result


def numerical_checks():
    x = jnp.linspace(-.7,.9,32)
    jac = np.asarray(jax.jacfwd(lambda y:rotate(y,target_angles(y)))(x))
    eps = 1e-5
    finite = np.column_stack([np.asarray(rotate(x+eps*e,target_angles(x+eps*e))-
                               rotate(x-eps*e,target_angles(x-eps*e)))/(2*eps) for e in jnp.eye(32)])
    fd_error = float(np.max(abs(jac-finite)))
    if fd_error > 2e-9:
        raise AssertionError(("target finite differences", fd_error))
    model = make_model("perspective_bounded",0)
    repeated = x
    for _ in range(8):
        repeated = model.step(repeated)
    analytical = rotate(x,8*model.angles(x))
    repeat_error = float(jnp.max(jnp.abs(repeated-analytical)))
    if repeat_error > 1e-12:
        raise AssertionError(("repeated phase", repeat_error))
    seq = x
    for phase in model.phases:
        seq = phase(seq)
    if not np.allclose(seq,model.step(x),rtol=1e-12,atol=1e-12):
        raise AssertionError("phase sum differs from layer composition")
    gamma_one = eqx.tree_at(lambda m:m.raw_gamma,ScalarKerr(),jnp.arctanh(.5))
    if not np.allclose(gamma_one.step(x),rotate(x,target_angles(x)),atol=1e-14):
        raise AssertionError("Kerr coefficient/sign control")
    log_model = make_model("log_intensity_bounded",0)
    log_jac = np.asarray(jax.jacfwd(log_model.step)(x))
    log_finite = np.column_stack([np.asarray(log_model.step(x+eps*e)-log_model.step(x-eps*e))/(2*eps)
                                 for e in jnp.eye(32)])
    log_fd = float(np.max(abs(log_jac-log_finite)))
    inverse_error = float(jnp.max(jnp.abs(log_model.inverse(log_model.step(x))-x)))
    singular = np.linalg.svd(log_jac,compute_uv=False)
    budget = log_model.certified_log_bound()
    if log_fd>2e-9 or inverse_error>1e-12 or budget>.5+1e-12:
        raise AssertionError(("log-intensity checks",log_fd,inverse_error,budget))
    if singular.max()>math.exp(budget)+1e-12 or singular.min()<math.exp(-budget)-1e-12:
        raise AssertionError("log-intensity Jacobian violates certificate")
    return {"target_max_finite_difference_error":fd_error,"phase_power_max_difference":repeat_error,
            "log_intensity_finite_difference_error":log_fd,"log_intensity_inverse_error":inverse_error,
            "log_intensity_initial_certificate":budget}


def train(model, data, seed):
    optimizer = optax.adam(optax.cosine_decay_schedule(.003,900,alpha=.05))
    state = optimizer.init(eqx.filter(model,eqx.is_inexact_array))
    @eqx.filter_jit
    def loss(m,xs):
        predicted = jax.vmap(m.angles)(xs)
        intensity = xs[:,:16]**2+xs[:,16:]**2
        difference = predicted-.04*intensity
        arguments = .5*jnp.arange(1,9)[None,:,None]*difference[:,None,:]
        return jnp.mean(2*intensity[:,None,:]*jnp.sin(arguments)**2)
    @eqx.filter_jit
    def update(m,opt_state,xs):
        value,gradient = eqx.filter_value_and_grad(loss)(m,xs)
        change,opt_state = optimizer.update(gradient,opt_state,m)
        return eqx.apply_updates(m,change).project_parameters(),opt_state,value
    validation = jnp.asarray(data["validation"])
    curve = [{"update":0,"validation_mse":float(loss(model,validation))}]
    rng = np.random.default_rng(52000+seed)
    started = time.perf_counter()
    for n in range(1,901):
        batch = jnp.asarray(data["train"][rng.integers(0,len(data["train"]),64)])
        model,state,value = update(model,state,batch)
        if n == 1 or n%100 == 0:
            record = {"update":n,"training_mse":float(value),
                "validation_mse":float(loss(model,validation)),"elapsed_seconds":time.perf_counter()-started}
            if not np.isfinite(record["validation_mse"]+record["training_mse"]):
                raise FloatingPointError(record)
            curve.append(record)
            print(json.dumps(record),flush=True)
    return model,curve,time.perf_counter()-started


def evaluate(model, data):
    result,arrays = {},{}
    for name in ("test_id","test_ood"):
        x = data[name]
        z = x[:,:16]+1j*x[:,16:]
        intensity = abs(z)**2
        angles = np.asarray(eqx.filter_jit(lambda m,xs:jax.vmap(m.angles)(xs))(model,jnp.asarray(x)))
        times = np.arange(1,201)[None,:,None]
        pred = z[:,None,:]*np.exp(1j*times*angles[:,None,:])
        truth = z[:,None,:]*np.exp(1j*times*(.04*intensity)[:,None,:])
        raw = np.mean(abs(pred-truth)**2,axis=-1)/2
        overlap = np.sum(np.conj(pred)*truth,axis=-1)
        unit_phase = np.ones_like(overlap)
        np.divide(overlap,abs(overlap),out=unit_phase,where=abs(overlap)>0)
        aligned = np.mean(abs(pred*unit_phase[...,None]-truth)**2,axis=-1)/2
        error = angles-.04*intensity
        wrapped = np.angle(np.exp(1j*error))
        metrics = {"mse_1":float(raw[:,0].mean()),"mse_200":float(raw.mean()),
            "endpoint_mse_200":float(raw[:,-1].mean()),"aligned_mse_1":float(aligned[:,0].mean()),
            "aligned_mse_200":float(aligned.mean()),"aligned_endpoint_mse_200":float(aligned[:,-1].mean()),
            "phase_increment_rmse":float(np.sqrt(np.mean(error**2))),
            "wrapped_phase_increment_rmse":float(np.sqrt(np.mean(wrapped**2))),
            "power_weighted_phase_increment_rmse":float(np.sqrt(np.sum(intensity*error**2)/np.sum(intensity))),
            "max_absolute_phase_increment_error":float(abs(error).max())}
        for code,mask in (("diffuse",data[name+"_component"]==0),("concentrated",data[name+"_component"]==1)):
            metrics[code+"_mse_200"] = float(raw[mask].mean())
            metrics[code+"_aligned_mse_200"] = float(aligned[mask].mean())
        result[name] = metrics
        arrays.update({name+"_angles":angles,name+"_endpoint_prediction_200":np.concatenate((pred[:,-1].real,pred[:,-1].imag),axis=-1),
            name+"_raw_mse_trace":raw.mean(0),name+"_aligned_mse_trace":aligned.mean(0),
            name+"_per_state_mse_200":raw.mean(1),name+"_per_state_aligned_mse_200":aligned.mean(1)})
    points = jnp.asarray(data["test_id"][JAC_INDICES])
    jacobians = {}
    for steps in (1,200):
        model_j = np.asarray(eqx.filter_jit(lambda m,xs:jax.vmap(jax.jacfwd(
            lambda x:rotate(x,steps*m.angles(x))))(xs))(model,points))
        target_j = np.asarray(jax.jit(jax.vmap(jax.jacfwd(lambda x:rotate(x,steps*target_angles(x)))))(points))
        relative = np.linalg.norm(model_j-target_j,axis=(-2,-1))/np.linalg.norm(target_j,axis=(-2,-1))
        sv = np.linalg.svd(model_j,compute_uv=False)
        target_sv = np.linalg.svd(target_j,compute_uv=False)
        jacobians[str(steps)] = {"relative_frobenius_error_mean":float(relative.mean()),
            "relative_frobenius_error_max":float(relative.max()),"minimum_singular_value":float(sv[:,-1].min()),
            "maximum_singular_value":float(sv[:,0].max()),"maximum_condition_number":float((sv[:,0]/sv[:,-1]).max()),
            "reference_minimum_singular_value":float(target_sv[:,-1].min()),
            "reference_maximum_singular_value":float(target_sv[:,0].max())}
        arrays.update({f"model_jacobians_{steps}":model_j,f"target_jacobians_{steps}":target_j,
            f"relative_jacobian_errors_{steps}":relative,f"model_singular_values_{steps}":sv})
    result["jacobian"] = jacobians
    result["certified_log_bound_one_step_global"] = model.certified_log_bound()
    if isinstance(model,ScalarKerr):
        gamma = float(2*jnp.tanh(model.raw_gamma))
        result["learned_gamma"] = gamma
        result["certified_log_bound_power_ball_4"] = math.asinh(.04*abs(gamma)*4)
    else:
        result["certified_log_bound_power_ball_4"] = model.certified_log_bound()
    result["n_parameters"] = sum(int(a.size) for a in jax.tree_util.tree_leaves(eqx.filter(model,eqx.is_inexact_array)))
    return result,arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes",nargs="+",choices=MODES,default=MODES)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--output",type=Path,default=BASE/"pilot_v1")
    parser.add_argument("--check-only",action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(numerical_checks(),indent=2));return
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    datafile = out/"dataset.npz"
    if not datafile.exists():
        np.savez_compressed(datafile,**dataset())
    data = dict(np.load(datafile))
    sources = ["experiments/stage7_isolated_kerr.py","skipnorm/hamiltonian_normalization.py"]
    config = {"protocol_sha256":digest(BASE/"PROTOCOL.md"),"dataset_sha256":digest(datafile),
        "source_sha256":{p:digest(ROOT/p) for p in sources},"splits":SPLITS,
        "updates":900,"horizon":8,"batch_size":64,"learning_rate":.003,
        "cosine_final_fraction":.05,"seeds":[0,1,2],"model_selection":"final parameters",
        "phase":{"stages":4,"width":64,"delta":1.,"time":2*math.sinh(.5/4),"budget":.5},
        "log_intensity_phase":{"width":32,"hidden_layers":2,"alpha":math.sinh(.5),
            "shared_across_modes":True,"spectral_cap":1.,"activation":"tanh","budget":.5},
        "jacobian_test_id_indices":JAC_INDICES,"metrics":"mse_200 cumulative; endpoint separately recorded"}
    cfg = out/"config.json"
    # Compare the JSON representation, including tuple-to-list normalization.
    encoded = json.dumps(config,indent=2)+"\n"
    if cfg.exists() and json.loads(cfg.read_text()) != json.loads(encoded):
        raise RuntimeError("Source/configuration changed: preserve this run and use a new output directory")
    cfg.write_text(encoded)
    for source in sources:
        dest = out/"source_snapshot"/source;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,dest)
    checks = numerical_checks()
    (out/"numerical_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    for mode in args.modes:
        for seed in args.seeds:
            stem = out/f"{mode}_seed{seed}"
            record = stem.with_suffix(".json")
            if record.exists():
                print(f"Resume complete: {mode} seed{seed}",flush=True);continue
            model = make_model(mode,seed)
            training_file = out/f"{mode}_seed{seed}.training.json"
            if stem.with_suffix(".eqx").exists():
                model = eqx.tree_deserialise_leaves(stem.with_suffix(".eqx"),model)
                training = json.loads(training_file.read_text())
            else:
                print(f"Training {mode} seed{seed}",flush=True)
                model,curve,seconds = train(model,data,seed)
                training = {"training_curve":curve,"training_seconds":seconds}
                training_file.write_text(json.dumps(training,indent=2,allow_nan=False)+"\n")
                eqx.tree_serialise_leaves(stem.with_suffix(".eqx"),model)
            metrics,arrays = evaluate(model,data)
            metrics.update(training)
            metrics.update({"mode":mode,"seed":seed,"config_sha256":digest(cfg),
                            "target":"exact isolated Kerr phase, dt=.04; no full DNLS"})
            np.savez_compressed(stem.with_suffix(".npz"),**arrays)
            record.write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
            print(json.dumps({"mode":mode,"seed":seed,"id_mse200":metrics["test_id"]["mse_200"],
                "ood_mse200":metrics["test_ood"]["mse_200"],"jacobian_error":metrics["jacobian"]["1"]["relative_frobenius_error_mean"]}),flush=True)
    records = [json.loads(p.read_text()) for p in out.glob("*_seed*.json") if not p.name.endswith(".training.json")]
    summary = {}
    def stats(values):
        return {"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)) if len(values)>1 else None}
    for mode in MODES:
        runs = sorted((r for r in records if r["mode"]==mode),key=lambda r:r["seed"])
        if not runs:continue
        summary[mode] = {"seeds":[r["seed"] for r in runs],"n_parameters":runs[0]["n_parameters"],
            "test_id":{key:stats([r["test_id"][key] for r in runs]) for key in runs[0]["test_id"]},
            "test_ood":{key:stats([r["test_ood"][key] for r in runs]) for key in runs[0]["test_ood"]},
            "jacobian":{step:{key:stats([r["jacobian"][step][key] for r in runs]) for key in runs[0]["jacobian"][step]} for step in ("1","200")}}
    result = {"config":config,"status":"fresh exact-subflow mechanism experiment; no tuning/model selection",
        "complete":len(records)==12,"summary":summary,"numerical_checks":checks,
        "hardware":{"platform":platform.platform(),"jax":jax.__version__,"equinox":eqx.__version__,
                    "numpy":np.__version__,"optax":optax.__version__,"devices":[str(d) for d in jax.devices()]}}
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
