"""Fixed exploratory screen of certified Hamiltonian normalization.

See results/stage6_hamiltonian_normalization/PROTOCOL.md. This is an adaptive
follow-up on an already inspected split, not a fresh confirmatory experiment.
"""
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
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax

from skipnorm.hamiltonian_normalization import PerspectivePhase, UnitaryMix
from experiments.stage5_rotation_coupling import evaluate, forecast, dnls_energy

MODES = ['hamiltonian_bounded', 'hamiltonian_unconstrained', 'naive_phase',
         'learned_kerr', 'learned_unitary', 'oracle_strang_1', 'oracle_strang_2']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HamiltonianPropagator(eqx.Module):
    phases: tuple[PerspectivePhase, ...]
    mixers: tuple[UnitaryMix, ...]
    global_mix: UnitaryMix
    project: bool = eqx.field(static=True)

    def __init__(self, mode, *, key):
        keys = jax.random.split(key, 9)
        self.phases = tuple(PerspectivePhase(dim=16, width=64, delta=1.,
            time=2*math.sinh(.5/4), omit_common_phase=mode=='naive_phase',
            project=True, key=keys[i]) for i in range(4))
        self.mixers = tuple(UnitaryMix(dim=16, key=keys[4+i]) for i in range(4))
        self.global_mix = UnitaryMix(dim=16, key=keys[8])
        self.project = mode != 'hamiltonian_unconstrained'

    def step(self, x):
        x = self.global_mix(x)
        for mix, phase in zip(self.mixers, self.phases):
            x = mix.inverse(phase(mix(x)))
        return x

    def inverse(self, x):
        for mix, phase in zip(reversed(self.mixers), reversed(self.phases)):
            x = mix.inverse(phase.inverse(mix(x)))
        return self.global_mix.inverse(x)

    def project_parameters(self):
        if not self.project:
            return self
        return eqx.tree_at(lambda m:m.phases, self,
            tuple(p.project_lipschitz() for p in self.phases))

    def certified_log_bound(self):
        values = [p.certified_log_bound() for p in self.phases]
        return None if any(v is None for v in values) else sum(values)


class LearnedSplitting(eqx.Module):
    half_step: UnitaryMix
    raw_gamma: jax.Array | None

    def __init__(self, nonlinear, *, key):
        self.half_step = UnitaryMix(dim=16, key=key)
        self.raw_gamma = jnp.asarray(0.) if nonlinear else None

    def _phase(self, x, sign):
        if self.raw_gamma is None:
            return x
        z = x[:16]+1j*x[16:]
        intensity = jnp.real(z*jnp.conj(z))
        cutoff = 4.
        safe = jnp.maximum(intensity, cutoff)
        continued = jnp.where(intensity<=cutoff, intensity,
                              cutoff*(1+jnp.log(safe/cutoff)))
        angle = sign*.04*2*jnp.tanh(self.raw_gamma)*continued
        y = z*jnp.exp(1j*angle)
        return jnp.concatenate([y.real, y.imag])

    def step(self, x):
        return self.half_step(self._phase(self.half_step(x), 1.))

    def inverse(self, x):
        return self.half_step.inverse(self._phase(self.half_step.inverse(x), -1.))

    def project_parameters(self):
        return self

    def certified_log_bound(self):
        if self.raw_gamma is None:
            return 0.
        return math.asinh(.04*abs(float(2*jnp.tanh(self.raw_gamma)))*4.)


class OracleStrang(eqx.Module):
    substeps: int = eqx.field(static=True)

    def _apply(self, x, sign):
        z = x[:16]+1j*x[16:]
        dt = sign*.04/self.substeps
        spectrum = -4*jnp.sin(jnp.pi*jnp.arange(16)/16)**2
        multiplier = jnp.exp(.5j*dt*spectrum)
        def one(_, state):
            state = jnp.fft.ifft(multiplier*jnp.fft.fft(state))
            state = state*jnp.exp(1j*dt*jnp.real(state*jnp.conj(state)))
            return jnp.fft.ifft(multiplier*jnp.fft.fft(state))
        z = jax.lax.fori_loop(0, self.substeps, one, z)
        return jnp.concatenate([z.real,z.imag])

    def step(self,x):
        return self._apply(x,1.)

    def inverse(self,x):
        return self._apply(x,-1.)

    def certified_log_bound(self):
        return None


def make_model(mode, seed):
    key = jax.random.PRNGKey(9100+89*seed)
    if mode.startswith('oracle'):
        return OracleStrang(int(mode[-1]))
    if mode.startswith('learned_'):
        return LearnedSplitting(mode=='learned_kerr', key=key)
    return HamiltonianPropagator(mode,key=key)


def train(model, split, updates, seed):
    tr, va = split['train_orbits'],split['val_orbits']
    schedule=optax.cosine_decay_schedule(.003, updates, alpha=.05)
    optimizer=optax.adam(schedule)
    state=optimizer.init(eqx.filter(model,eqx.is_inexact_array))
    @eqx.filter_jit
    def loss(m,windows):
        pred=forecast(m,windows[:,0],8)
        return jnp.mean((pred-windows[:,1:])**2)
    @eqx.filter_jit
    def update(m,opt_state,windows):
        value,grad=eqx.filter_value_and_grad(loss)(m,windows)
        changes,opt_state=optimizer.update(grad,opt_state,m)
        m=eqx.apply_updates(m,changes).project_parameters()
        return m,opt_state,value
    rng=np.random.default_rng(40000+seed)
    val=jnp.asarray(np.stack([o[t:t+9] for o in va for t in range(0,len(o)-8,8)]))
    curve=[{'update':0,'validation_mse':float(loss(model,val))}]
    started=time.perf_counter()
    for n in range(1,updates+1):
        mi=rng.integers(0,len(tr),size=64)
        ti=rng.integers(0,tr.shape[1]-9,size=64)
        windows=jnp.asarray(np.stack([tr[m,t:t+9] for m,t in zip(mi,ti)]))
        model,state,value=update(model,state,windows)
        if n==1 or n%100==0 or n==updates:
            record={'update':n,'training_mse':float(value),
                    'validation_mse':float(loss(model,val)),
                    'elapsed_seconds':time.perf_counter()-started}
            if not np.isfinite(record['training_mse']):
                raise FloatingPointError(record)
            curve.append(record)
            print(json.dumps(record),flush=True)
    return model,curve,time.perf_counter()-started


def extra_diagnostics(model,test,prediction):
    initial=jnp.asarray(test[:,0])
    back=eqx.filter_jit(lambda m,y:jax.lax.fori_loop(0,249,
        lambda _,z:jax.vmap(m.inverse)(z),y))(model,jnp.asarray(prediction[:,-1]))
    recovery=float(jnp.max(jnp.linalg.norm(back-initial,axis=-1)/jnp.linalg.norm(initial,axis=-1)))
    q,p=test[:,0,:16],test[:,0,16:]
    scale=np.sum((np.roll(q,-1,axis=-1)-q)**2+(np.roll(p,-1,axis=-1)-p)**2
                 +.5*(q*q+p*p)**2,axis=-1)
    error=np.abs(dnls_energy(prediction[:,-1])-dnls_energy(test[:,0]))
    result={'full_rollout_inverse_relative_error':recovery,
            'certified_log_bound_one_step':model.certified_log_bound(),
            'mean_absolute_final_energy_error':float(error.mean()),
            'mean_scaled_final_energy_error':float(np.mean(error/scale))}
    if isinstance(model,LearnedSplitting) and model.raw_gamma is not None:
        result['learned_gamma']=float(2*jnp.tanh(model.raw_gamma))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--updates',type=int,default=900)
    parser.add_argument('--output',type=Path,default=ROOT/'results/stage6_hamiltonian_normalization/pilot_v1')
    parser.add_argument('--timing-only',action='store_true')
    args=parser.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sources=['experiments/stage6_hamiltonian_normalization.py','skipnorm/hamiltonian_normalization.py',
             'experiments/stage5_rotation_coupling.py','experiments/stage3_dnls_benchmark.py',
             'skipnorm/jax_layers.py','skipnorm/rotation_coupling.py']
    config={'updates':args.updates,'seeds':[0,1,2],'batch_size':64,'horizon':8,
        'learning_rate':.003,'cosine_final_fraction':.05,'precision':'float64',
        'model_selection':'final; validation recorded only',
        'perspective':{'stages':4,'width':64,'delta':1.,'budget':.5},
        'kerr':{'power_cutoff':4.,'gamma_bound':2.,'dt':.04},
        'split_sha256':digest(ROOT/'data/splits/dnls_1d.npz'),
        'protocol_sha256':digest(ROOT/'results/stage6_hamiltonian_normalization/PROTOCOL.md'),
        'source_sha256':{p:digest(ROOT/p) for p in sources}}
    cfg=out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise RuntimeError('Source/configuration changed; use a fresh run directory.')
    cfg.write_text(json.dumps(config,indent=2)+'\n')
    for p in sources:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/p,dest)
    split=np.load(ROOT/'data/splits/dnls_1d.npz')
    for mode in args.modes:
        seeds=[0] if mode.startswith('oracle') else args.seeds
        for seed in seeds:
            stem=out/f'{mode}_seed{seed}';record_file=stem.with_suffix('.json')
            if record_file.exists() and not args.timing_only:
                print(f'Resume complete: {mode} {seed}',flush=True);continue
            model=make_model(mode,seed)
            curve=[];seconds=None
            if stem.with_suffix('.eqx').exists():
                model=eqx.tree_deserialise_leaves(stem.with_suffix('.eqx'),model)
                if record_file.exists():
                    old=json.loads(record_file.read_text());curve=old['training_curve'];seconds=old['training_seconds']
            elif mode.startswith('oracle'):
                pass
            elif args.timing_only:
                raise RuntimeError(f'Missing checkpoint {stem}')
            else:
                print(f'Training {mode} seed {seed}',flush=True)
                model,curve,seconds=train(model,split,args.updates,seed)
            eqx.tree_serialise_leaves(stem.with_suffix('.eqx'),model)
            result,arrays=evaluate(model,split['test_orbits_long'])
            result.update(extra_diagnostics(model,split['test_orbits_long'],arrays['prediction']))
            result.update({'mode':mode,'seed':seed,'training_curve':curve,
                           'training_seconds':seconds,'config_sha256':digest(cfg),
                           'oracle_knows_equation':mode.startswith('oracle')})
            record_file.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            np.savez_compressed(stem.with_suffix('.npz'),**arrays)
            print(json.dumps({k:result[k] for k in ['mode','seed','mse_20','mse_200','rollout_ms']}),flush=True)
    rows=[json.loads(p.read_text()) for p in out.glob('*_seed*.json')]
    summary={}
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode]
        if not rr:continue
        summary[mode]={'seeds':[r['seed'] for r in rr],'n_parameters':rr[0]['n_parameters']}
        for metric in ['mse_20','mse_200','rollout_ms','mean_scaled_final_energy_error',
                       'mean_symplectic_defect','relative_phase_equivariance_error']:
            values=[r[metric] for r in rr]
            summary[mode][metric]={'mean':float(np.mean(values)),
                'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None}
    hardware={'platform':platform.platform(),'jax':jax.__version__,'equinox':eqx.__version__,
              'devices':[str(x) for x in jax.devices()]}
    (out/'summary.json').write_text(json.dumps({'config':config,'hardware':hardware,
        'status':'adaptive exploratory follow-up; previously inspected split',
        'summary':summary},indent=2)+'\n')


if __name__=='__main__':
    main()
