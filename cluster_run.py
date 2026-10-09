"""One serial process, whole comparison blocks, atomic task records and a wall cap."""
import argparse
import contextlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from cluster_plan import make_plan
from qml_tying.models import ModelSpec,Model
from qml_tying.datasets import haar
from qml_tying.io import write_json,environment,record_id
from qml_tying import runtime

PROJECT=Path(__file__).absolute().parent


@lru_cache(maxsize=1024)
def payload_hash(path,size,mtime):
    return __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()


def valid_record(root,task):
    path=root/task['record']
    if not path.is_file() or not path.with_suffix('.npz').is_file():return False
    try:
        r=json.loads(path.read_text())
        if r.get('id')!=task['id'] or not r.get('complete'):return False
        required={'fit':('phi','trace'),'baseline':('phi','trace'),'quench':('phi','gradient','trace'),
                  'landscape':('endpoint_phi','anchor_phi'),'expressivity':('outputs','parameter_phi'),
                  'transfer':('phi','prediction_test')}[task['kind']]
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            if not all(k in z.files for k in required):return False
        if r.get('data_ref'):
            data=root/r['data_ref']
            if not data.is_file():return False
            stat=data.stat()
            if payload_hash(str(data),stat.st_size,stat.st_mtime_ns)!=r.get('data_payload_sha256'):return False
        return True
    except (ValueError,OSError,KeyError):return False


def timing_key(task):
    c=task['config'];return record_id(c['case']['spec'],c['mode'],c['map_seed'])


def calibration(plan):
    models={};tables={};rng=np.random.default_rng(plan['protocol'].get('calibration_seed',58204))
    for b in plan['blocks']:
        for t in b['tasks']:
            if t['kind']=='baseline':continue
            c=t['config'];models[timing_key(t)]=(ModelSpec(**c['case']['spec']),c['mode'],c['map_seed'])
    for key,(spec,mode,map_seed) in sorted(models.items()):
        runtime.check();m=Model(spec,mode,map_seed);phi=rng.uniform(-np.pi,np.pi,m.tying.d)
        if spec.family in ('qcnn','autoencoder','su2'):x=haar(rng,16,1<<spec.n)
        else:
            width=2 if spec.family=='reupload' else spec.n-1 if spec.family=='recurrent' else spec.n*(spec.n-1)//2
            x=rng.uniform(-np.pi,np.pi,(16,width))
            if spec.family=='graph':x=(rng.random((16,width))<.5).astype(float)
        m.predict(x,phi);forward=[]
        for _ in range(2):
            start=time.perf_counter();m.predict(x,phi);forward.append(time.perf_counter()-start)
        start=time.perf_counter();m.predict(x[:8],phi,jac=True);jac=time.perf_counter()-start
        tables[key]=dict(spec=spec.__dict__,mode=mode,map_seed=map_seed,seconds_per_16=float(np.median(forward)),
                         seconds_per_8_jacobian=jac,measurements=forward,d=m.tying.d,p=m.p,calibration_labels_used=False)
        print(f'CALIBRATE {spec.family} n={spec.n} {mode}: forward16={tables[key]["seconds_per_16"]:.4g}s jac8={jac:.4g}s',flush=True)
    return tables


def estimate(task,tables):
    c=task['config'];kind=task['kind']
    if kind=='baseline':return 2.+c['budget']*.0004
    v=tables[timing_key(task)];ntrain=int(c['n_samples']*.6)*(3 if c['case']['spec']['family']=='graph' else 1)
    fw=v['seconds_per_16']*max(1,int(np.ceil(ntrain/16)))+.0001
    gj=v['seconds_per_8_jacobian']*max(1,int(np.ceil(ntrain/8)))+.0001
    if kind=='fit':return c['budget']*fw+4+4*fw
    if kind=='quench':return c['budget']*gj+1
    if kind=='transfer':return fw+4
    if kind=='landscape':
        lc=c['landscape_cfg'];k=v['d'] if v['d']<=lc['full_hessian_limit'] else min(v['d'],lc['hessian_projection'])
        forwards=1+lc['global_samples']+lc['starts']+lc['lines']*lc['line_points']
        jacobians=lc['gradient_samples']+2+8*k
        untied_key=record_id(c['case']['spec'],'untied',c['map_seed'])
        untied=tables[untied_key]['seconds_per_8_jacobian']*max(1,int(np.ceil(ntrain/8)))
        return forwards*fw+jacobians*gj+2*untied+5
    if kind=='expressivity':
        ec=c['expressivity_cfg'];forwards=ec['parameter_samples']+16;jacobians=ec['jacobian_samples']
        if c['case']['spec']['family']=='reupload':
            spec=Model(ModelSpec(**c['case']['spec']),c['mode']).reference.spec
            points=(2*max(spec.bandwidth)+3)**2;forwards+=ec['parameter_samples']*points/max(1,ntrain)
        return forwards*fw+jacobians*gj+5
    raise ValueError(kind)


def freeze_feasibility(plan,tables,seconds):
    factor=plan['protocol'].get('calibration_safety_factor',1.8);costs={}
    for b in plan['blocks']:
        b['estimated_seconds']=sum(estimate(t,tables) for t in b['tasks'])*factor
        costs[b['tier']]=costs.get(b['tier'],0.)+b['estimated_seconds']
    if costs.get('core',0)>seconds:raise ValueError(f'Complete {plan["profile"]} core requires estimated {costs["core"]/3600:.3f} h; only {seconds/3600:.3f} h available.')
    selected=[];used=0.
    for tier in plan['protocol']['tier_order']:
        dependencies=plan['protocol'].get('tier_dependencies',{}).get(tier,[])
        if all(d in selected for d in dependencies) and used+costs.get(tier,0.)<=seconds:selected.append(tier);used+=costs.get(tier,0.)
    for b in plan['blocks']:
        b.update(selected=b['tier'] in selected,selection_reason='complete predeclared coverage tier fits initial timing allowance' if b['tier'] in selected else 'whole tier outside initial timing allowance')
    plan.update(calibration=tables,estimated_selected_seconds=used,experiment_allowance_seconds=seconds,
                selection_frozen_before_training=True,selected_tiers=selected,tier_estimated_seconds=costs)
    return plan


def execute(task,root):
    from run_task import run_task
    write_json(root/'task_inputs'/f'{task["id"]}.json',task)
    run_task(task,root)
    if not valid_record(root,task):raise RuntimeError('Runner did not create the expected valid JSON/NPZ record: '+task['record'])


def provenance(root,expected_hashes=None):
    from collect_results import source_files
    paths=list(source_files(PROJECT));hashes={str(p.relative_to(PROJECT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in paths}
    if expected_hashes is not None and hashes!=expected_hashes:
        raise RuntimeError('Source changed while this Slurm job was queued. Submit the reviewed source in a separate output directory.')
    path=root/'provenance/source_hashes.json'
    if path.exists() and json.loads(path.read_text())!=hashes:
        raise RuntimeError('Source files changed since this output was started. Use a new --out directory; the saved source snapshot remains available.')
    write_json(path,hashes)
    import shutil
    for p in paths:
        dest=root/'provenance/source'/p.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(p,dest)
    names={'SLURM_JOB_ID','SLURM_JOB_NUM_NODES','SLURM_CPUS_PER_TASK','SLURM_JOB_NODELIST','SLURM_JOB_PARTITION',
           'SLURM_JOB_ACCOUNT','SLURM_NTASKS','SLURM_SUBMIT_DIR','SLURM_JOB_QOS',
           'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOADEDMODULES'}
    env=environment();env.update(executable=sys.executable,prefix=sys.prefix,threadpools=threadpool_info(),
                                selected_environment={k:v for k,v in os.environ.items() if k in names})
    write_json(root/'provenance/environment.json',env)
    for name,cmd in [('pip_freeze.txt',[sys.executable,'-m','pip','freeze']),('cpu.txt',['lscpu'])]:
        try:r=subprocess.run(cmd,text=True,capture_output=True,timeout=30);(root/'provenance'/name).write_text(r.stdout+r.stderr)
        except (OSError,subprocess.TimeoutExpired) as exc:(root/'provenance'/name).write_text(str(exc))


def run_tests(root):
    path=root/'provenance/compute_node_tests.txt'
    with path.open('w') as stream:
        r=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    if r.returncode:raise RuntimeError('Preflight checks failed. See provenance/compute_node_tests.txt.')


def run(config,root,seconds,skip_tests=False,analysis=True,expected_hashes=None,analysis_only=False):
    root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process is already running in this output directory.')
    signal.signal(signal.SIGUSR1,lambda s,f:runtime.request_stop('Slurm advance warning'))
    signal.signal(signal.SIGTERM,lambda s,f:runtime.request_stop('termination signal'))
    signal.signal(signal.SIGINT,lambda s,f:runtime.request_stop('interrupt'))
    began=time.time();runtime.configure(seconds)
    status=dict(started_epoch=began,experiment_allowance_seconds=seconds,state='starting',job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(root/'run_state.json',status)
    try:
        with threadpool_limits(limits=1):
            if os.environ.get('SLURM_JOB_NUM_NODES','1')!='1':raise RuntimeError('This serial study requires exactly one allocated node.')
            provenance(root,expected_hashes)
            if not skip_tests:run_tests(root)
            protocol=json.loads(Path(config).read_text());plan_path=root/'manifest.json'
            if analysis_only and not plan_path.exists():raise RuntimeError('Analysis-only mode requires an existing frozen manifest in --out.')
            if plan_path.exists():
                plan=json.loads(plan_path.read_text())
                if plan['protocol_hash']!=record_id(protocol):raise RuntimeError('Protocol changed. Use a separate output directory.')
            else:
                candidate_profiles=['standard','compact'] if protocol.get('profile')=='auto' else [protocol.get('profile','standard')]
                tables=calibration(make_plan(protocol,candidate_profiles[0]));rejections=[]
                for profile in candidate_profiles:
                    try:
                        plan=freeze_feasibility(make_plan(protocol,profile),tables,max(0.,runtime.remaining()-60));break
                    except ValueError as exc:rejections.append(str(exc))
                else:
                    write_json(root/'feasibility_failure.json',dict(reasons=rejections,calibration=tables))
                    raise RuntimeError('Even the complete compact core does not fit the remaining ledger allowance. No fitting was started. See feasibility_failure.json.')
                plan['larger_profile_rejections']=rejections
                write_json(plan_path,plan)
                write_json(root/'frozen_protocol.json',protocol)
            selected=[b for b in plan['blocks'] if b['selected']]
            print(f'Frozen plan: {len(selected)}/{len(plan["blocks"])} blocks, {sum(len(b["tasks"]) for b in selected)} tasks; '
                  f'calibrated estimate {plan["estimated_selected_seconds"]/3600:.2f} h.',flush=True)
            if analysis_only:selected=[]
            failures=0;completed=0;last_archive=time.monotonic()
            for b in selected:
                runtime.check()
                if time.monotonic()-last_archive>=3600:
                    from collect_results import collect
                    collect(root,analysis=False)
                    last_archive=time.monotonic()
                for task in b['tasks']:
                    runtime.check()
                    if valid_record(root,task):continue
                    if task.get('dependencies'):
                        if not all((root/p).exists() and (root/p).with_suffix('.npz').exists() for p in task['dependencies']):
                            write_json(root/'task_status'/f'{task["id"]}.json',dict(task_id=task['id'],state='dependency_incomplete'));continue
                    # A JSON marker with a missing NPZ is not a completed task.
                    (root/task['record']).unlink(missing_ok=True)
                    started=time.perf_counter()
                    write_json(root/'task_status'/f'{task["id"]}.json',dict(task_id=task['id'],state='running',job_id=status['job_id']))
                    try:
                        execute(task,root);completed+=1
                        entry=dict(task_id=task['id'],state='complete',record=task['record'],wall_seconds=time.perf_counter()-started,job_id=status['job_id'])
                        print(f'COMPLETE {task["family"]} {task["kind"]} {task["id"]}',flush=True)
                    except runtime.WorkStopped as exc:
                        if hasattr(exc,'partial_fit'):
                            fit=exc.partial_fit;trace=fit.pop('trace')
                            write_json(root/'partial_fits'/f'{task["id"]}.json',dict(task=task,fit=fit,reason=str(exc),complete=False))
                            np.savez_compressed(root/'partial_fits'/f'{task["id"]}.npz',trace=trace)
                        write_json(root/'task_status'/f'{task["id"]}.json',dict(task_id=task['id'],state='interrupted',reason=str(exc),wall_seconds=time.perf_counter()-started))
                        raise
                    except Exception as exc:
                        failures+=1
                        entry=dict(task_id=task['id'],state='failed',error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
                        print(entry['traceback'],file=sys.stderr,flush=True)
                    write_json(root/'task_status'/f'{task["id"]}.json',entry)
                    if failures>=3:raise RuntimeError('Stopped after three failed tasks; failures are saved for diagnosis.')
                status.update(state='running',last_block=b['id'],new_tasks_completed=completed,failures=failures,updated_epoch=time.time())
                write_json(root/'run_state.json',status)
            status.update(state='analysis_only_complete' if analysis_only else 'selected_plan_complete',failures=failures)
    except runtime.WorkStopped as exc:
        status.update(state='wall_or_signal_stop',reason=str(exc))
    except Exception as exc:
        status.update(state='error',reason=str(exc),traceback=traceback.format_exc())
        print(status['traceback'],file=sys.stderr,flush=True)
    finally:
        runtime.configure(None)
        status.update(experiment_elapsed_seconds=time.time()-began,finished_epoch=time.time())
        write_json(root/'run_state.json',status)
        stamp=status.get('job_id') or str(int(began*1000))
        write_json(root/'run_history'/f'{stamp}.json',status)
        from collect_results import collect
        collect(root,analysis=analysis)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return 1 if status['state']=='error' or status.get('failures',0)>0 else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings');p.add_argument('--config',default='configs/cluster_smoke.json')
    p.add_argument('--out',default='results/local_smoke');p.add_argument('--seconds',type=float,default=3600)
    p.add_argument('--skip-tests',action='store_true');p.add_argument('--no-analysis',action='store_true');a=p.parse_args()
    if a.settings:
        s=json.loads(Path(a.settings).read_text());a.config=s['config'];a.out=s['out'];a.skip_tests=s.get('skip_tests',False)
        # Submission settings include the full Slurm allocation, not just fitting.
        reserve=s['finalization_reserve_seconds'];a.seconds=max(0.,s['wall_seconds']-reserve)
        (Path(a.out)/'submission').mkdir(parents=True,exist_ok=True)
        write_json(Path(a.out)/'submission'/f'running_{os.environ.get("SLURM_JOB_ID","unknown")}.json',dict(allocation_script_start_epoch=time.time(),settings=s))
    sys.exit(run(Path(a.config).absolute(),Path(a.out).absolute(),a.seconds,a.skip_tests,not a.no_analysis,
                 s['source_hashes'] if a.settings else None,s.get('analysis_only',False) if a.settings else False))


if __name__=='__main__':main()
