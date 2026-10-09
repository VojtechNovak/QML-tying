"""Atomic runners for all predeclared QML study tasks."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from qml_tying.models import Model,ModelSpec
from qml_tying.datasets import cache_data,classical_features
from qml_tying.optimizers import optimize
from qml_tying.metrics import classify_metrics
from qml_tying.classical import fit as fit_mlp
from qml_tying.quench import quench
from qml_tying.diagnostics import landscape,expressivity
from qml_tying.io import write_json,record_id


def _save(path,record,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);npz=path.with_suffix('.npz');tmp=npz.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream,**arrays)
    tmp.replace(npz);write_json(path,record)  # JSON is the final completion marker.


def fit_arrays(fit):
    arrays={k:fit[k] for k in ('phi','train_best_phi','trace','evaluation_seconds') if k in fit}
    c=fit.get('candidates',[])
    if c:
        arrays['candidate_phi']=np.array([a['phi'] for a in c])
        for k in ('nfev','train_loss','val_loss'): arrays['candidate_'+k]=np.array([a[k] for a in c])
    metadata={k:v for k,v in fit.items() if k not in arrays and k!='candidates'}
    return metadata,arrays


def split_metrics(model,data,phi,task):
    arrays={};metrics={}
    for s in ('train','val','test'):
        f=model.predict(getattr(data,'x_'+s),phi);arrays['prediction_'+s]=f
        if task=='infidelity': metrics[s]=dict(mean_fidelity=float(np.mean(f)),mean_infidelity=float(np.mean(1-f)),fidelity_q10=float(np.quantile(f,.1)))
        else: metrics[s]=classify_metrics(getattr(data,'y_'+s),f)
    return metrics,arrays


def run_task(task,root):
    root=Path(root);cfg=task['config'];case=cfg['case'];spec=ModelSpec(**case['spec']);kind=task['kind'];started=time.perf_counter()
    model=Model(spec,cfg['mode'],cfg['map_seed']) if kind!='baseline' else None
    data,data_id=cache_data(root,spec,case['dataset'],cfg['split_seed'],cfg['n_samples'],cfg['teacher_seed'])
    # All optimizers in a map/dataset/run block receive exactly the same x0
    # and stochastic seed. Every first candidate is charged to the budget.
    dimension=model.tying.d if model else cfg['target_parameters']
    seed=int(record_id('optimizer-seed',case['spec'],case['dataset'],cfg['split_seed'],cfg['run_seed'],dimension,cfg['map_seed'])[:8],16)
    record=dict(id=task['id'],kind=kind,block_id=task['block_id'],config=cfg,case=case['name'],family=spec.family,
                tier=case['tier'],dataset=case['dataset'],data_id=data_id,data_ref=f'datasets/{data_id}.npz',
                data_payload_sha256=data.metadata['payload_sha256'],split_seed=cfg['split_seed'],run_seed=cfg['run_seed'],map_seed=cfg['map_seed'],
                spec=case['spec'],complete=True)
    if model: record.update(model=model.metadata(),tie=cfg['mode'],d=model.tying.d,p=model.p)
    if kind=='transfer':
        path=root/'fit'/f'{cfg["source_fit_id"]}.json';parent=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:phi=z['phi'].copy()
        if len(phi)!=model.tying.d:raise ValueError('Transfer cannot change or pad a learned vector.')
        prediction=model.predict(data.x_test,phi);arrays=dict(phi=phi,prediction_test=prediction)
        record.update(optimizer=parent['optimizer'],source_fit_id=parent['id'],source_case=cfg['source_case'],
                      source_data_id=parent['data_id'],source_spec=parent['spec'],source_budget=parent['fit']['budget'],
                      source_selection='unchanged validation-selected vector from smaller circuit',optimization_performed=False,
                      metrics=dict(test=classify_metrics(data.y_test,prediction)),target_training_or_validation_used=False)
    elif kind=='fit':
        fit=optimize(model,data,cfg['optimizer'],cfg['budget'],seed,model.task,checkpoints=cfg['checkpoints'])
        metadata,arrays=fit_arrays(fit);metrics,pred=split_metrics(model,data,fit['phi'],model.task);arrays.update(pred)
        record.update(optimizer=cfg['optimizer'],optimizer_seed=seed,fit=metadata,metrics=metrics,
                      loss=model.task,selection='validation-best of fixed checkpoint training incumbents; test inspected once')
        if spec.family=='graph':
            record['invariant_orbit_bounds']=data.metadata['invariant_orbit_bounds']
            record['orbit_bounds_apply']=cfg['mode'] in ('node_edge','layer_shared')
        if case['dataset']=='spin_orientation_mismatch' and cfg['mode'] in ('su2','su2_spatial'):
            record['constant_readout_certificate']={s:dict(expected_log_loss=float(-np.mean(getattr(data,'y_'+s)*np.log(1e-6)+(1-getattr(data,'y_'+s))*np.log1p(-1e-6))),expected_balanced_accuracy=.5) for s in ('train','val','test')}
        if spec.family=='autoencoder':
            record['pure_branch_multiplier']=1+(1<<(spec.n-spec.latent))
            record['reconstruction']='exact discard/reset/decode channel; unnormalized branches, no density-matrix approximation'
    elif kind=='baseline':
        cd=classical_features(data,spec);mlp,fit=fit_mlp(cd,cfg['target_parameters'],cfg['budget'],seed,cfg['checkpoints'])
        metadata,arrays=fit_arrays(fit);metrics,pred=split_metrics(mlp,cd,fit['phi'],'classification');arrays.update(pred)
        record.update(tie=cfg['mode'],d=mlp.d,p=mlp.d,optimizer='Adam',fit=metadata,metrics=metrics,preprocessing=cd.metadata,
                      baseline_role='approximate parameter-count reference; representation and gradient cost differ from QNN')
        for s in ('train','val','test'): arrays['mlp_input_'+s]=getattr(cd,'x_'+s)
    elif kind=='quench':
        seed=int(record_id('quench-independent-start',data_id,model.tying.d,cfg['map_seed'],cfg['start'])[:8],16)
        q=quench(model,data,cfg['budget'],seed)
        arrays={k:q[k] for k in ('phi','initial_phi','gradient','trace')};metadata={k:v for k,v in q.items() if k not in arrays}
        record.update(quench=metadata,start=cfg['start'],seed=seed,
                      physical_gradient_rms=float(np.linalg.norm(q['gradient']/np.sqrt(model.tying.multiplicity))/np.sqrt(model.tying.d)))
    elif kind=='landscape':
        endpoints=[]
        for tid in cfg['quench_ids']:
            r=json.loads((root/'quench'/f'{tid}.json').read_text());z=np.load(root/'quench'/f'{tid}.npz',allow_pickle=False)
            endpoints.append(dict(r['quench'],phi=z['phi'],gradient=z['gradient']))
        incumbents=[];sources=[]
        for path in sorted((root/'fit').glob('*.json')):
            r=json.loads(path.read_text())
            if r['data_id']==data_id and r['tie']==cfg['mode'] and r['map_seed']==cfg['map_seed']:
                if not path.with_suffix('.npz').exists(): continue
                z=np.load(path.with_suffix('.npz'),allow_pickle=False)
                incumbents.append(dict(phi=z['train_best_phi'],loss=r['fit']['best_train_loss']));sources.append(r['id'])
        diag,arrays=landscape(model,data,endpoints,incumbents,cfg['landscape_cfg'],seed)
        record.update(diagnostics=diag,quench_sources=cfg['quench_ids'],incumbent_sources=sources,
                      optimization_performed=False,source_methods='adapted from vendor/landscape_characterization_fast_supplied.py')
    elif kind=='expressivity':
        diag,arrays=expressivity(model,data,cfg['expressivity_cfg'],seed);record['diagnostics']=diag
    else: raise ValueError(kind)
    record['total_wall_seconds']=time.perf_counter()-started
    _save(root/task['record'],record,arrays)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    run_task(json.loads(Path(a.task).read_text()),Path(a.out))


if __name__=='__main__': main()
