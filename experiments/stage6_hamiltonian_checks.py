"""Post-hoc gauge-aware diagnostics; no fitting, tuning or model selection."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from experiments.stage6_hamiltonian_normalization import make_model,digest
from experiments.stage5_rotation_coupling import forecast,dnls_energy


def complex_state(x):
    return x[...,:16]+1j*x[...,16:]


def aligned_errors(pred,truth):
    z,w=complex_state(pred),complex_state(truth)
    overlap=np.sum(np.conj(z)*w,axis=-1)
    # At zero overlap every unit phase is optimal; angle(0)=0 chooses one.
    phase=np.exp(1j*np.angle(overlap))
    aligned=z*phase[...,None]
    state=np.mean(np.abs(aligned-w)**2,axis=-1)/2
    intensity=np.mean((np.abs(z)**2-np.abs(w)**2)**2,axis=-1)
    return state,intensity


def main():
    run=ROOT/'results/stage6_hamiltonian_normalization/pilot_v1'
    split=np.load(ROOT/'data/splits/dnls_1d.npz')
    test=split['test_orbits_long'];truth=test[:,1:]
    records=[]
    for directory in [ROOT/'results/stage5_rotation_coupling/pilot_v1',run]:
        for p in sorted(directory.glob('*_seed*.json')):
            r=json.loads(p.read_text());pred=np.load(p.with_suffix('.npz'))['prediction']
            state,intensity=aligned_errors(pred,truth)
            records.append({'source_run':directory.name,'stage':directory.parent.name,
                'mode':r['mode'],'seed':r['seed'],
                **{f'aligned_mse_{n}':float(state[:,:n].mean()) for n in [20,200,249]},
                **{f'intensity_mse_{n}':float(intensity[:,:n].mean()) for n in [20,200,249]}})
    matched=[]
    predict=eqx.filter_jit(lambda m,x:forecast(m,x,249))
    for seed in range(3):
        model=make_model('hamiltonian_bounded',seed)
        model=eqx.tree_deserialise_leaves(run/f'hamiltonian_bounded_seed{seed}.eqx',model)
        # Rebuild with changed static metadata, while copying every numerical leaf.
        naive=make_model('naive_phase',seed)
        naive=eqx.tree_deserialise_leaves(run/f'hamiltonian_bounded_seed{seed}.eqx',naive)
        corrected=np.asarray(predict(model,jnp.asarray(test[:,0])))
        omitted=np.asarray(predict(naive,jnp.asarray(test[:,0])))
        aligned,intensity=aligned_errors(omitted,corrected)
        matched.append({'seed':seed,'max_phase_aligned_state_mse':float(aligned.max()),
            'max_intensity_mse':float(intensity.max()),
            'max_absolute_energy_difference':float(np.abs(dnls_energy(omitted)-dnls_energy(corrected)).max()),
            'unaligned_state_mse':float(np.mean((omitted-corrected)**2))})
    nonlinearity={}
    for key in ['train_orbits','test_orbits_long']:
        z=complex_state(split[key])
        lap=np.roll(z,1,axis=-1)-2*z+np.roll(z,-1,axis=-1)
        ratio=np.linalg.norm(np.abs(z)**2*z,axis=-1)/np.linalg.norm(lap,axis=-1)
        nonlinearity[key]={'median_nonlinear_to_laplacian_ratio':float(np.median(ratio)),
            'minimum':float(ratio.min()),'maximum':float(ratio.max()),
            'max_local_intensity':float(np.max(np.abs(z)**2)),
            'max_total_power':float(np.max(np.sum(np.abs(z)**2,axis=-1)))}
    output={'status':'post-hoc diagnostics; no fitting or model selection',
        'source_sha256':digest(__file__),'config_sha256':digest(run/'config.json'),
        'declaration_sha256':digest(run.parent/'POSTHOC_DIAGNOSTICS.md'),
        'metrics':records,'matched_common_phase_omission':matched,
        'nonlinearity':nonlinearity}
    (run/'posthoc_diagnostics.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'matched_control':matched,'nonlinearity':nonlinearity},indent=2))


if __name__=='__main__':
    main()
