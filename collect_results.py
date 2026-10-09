"""Package raw, partial and analyzed evidence in one self-contained return zip."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).absolute().parent


def source_files(root):
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in ('.py','.sh','.txt','.md'):yield p
        elif p.is_dir() and p.name in ('qml_tying','configs','tests','manuscript','vendor','validation'):
            for f in sorted(p.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.sh','.json','.tex','.bib','.pdf','.txt','.md'):yield f


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)


def accounting(root):
    job=os.environ.get('SLURM_JOB_ID')
    if not job:return
    for name,cmd in [('scontrol',['scontrol','show','job',job]),
                     ('sacct',['sacct','-j',job,'--parsable2','--noheader','--format=JobIDRaw,State,ElapsedRaw,AllocNodes,AllocCPUS,ExitCode,Start,End,MaxRSS'])]:
        try:r=subprocess.run(cmd,text=True,capture_output=True,timeout=15);text=r.stdout+r.stderr
        except (OSError,subprocess.TimeoutExpired) as exc:text=str(exc)
        path=root/'slurm'/f'{name}_{job}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)


def inventory(root):
    from cluster_run import valid_record
    manifest=root/'manifest.json'
    if not manifest.exists():
        write_json(root/'block_completion.json',dict(complete_block_ids=[],blocks=[],note='No plan was frozen before the run stopped.'))
        return [],[]
    plan=json.loads(manifest.read_text());rows=[];blocks=[]
    for b in plan['blocks']:
        finished=[]
        for t in b['tasks']:
            status_path=root/'task_status'/f'{t["id"]}.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            complete=valid_record(root,t)
            state='complete' if complete else status.get('state','not_started' if b['selected'] else 'timing_deferred')
            if state=='running':state='interrupted_without_final_record'
            rows.append(dict(task_id=t['id'],block_id=b['id'],family=t['family'],kind=t['kind'],
                             selected=b['selected'],state=state,record=t['record'],wall_seconds=status.get('wall_seconds'),
                             error=status.get('error',status.get('reason',''))))
            finished.append(complete)
        blocks.append(dict(id=b['id'],family=b['family'],kind=b['kind'],selected=b['selected'],tasks=len(finished),
                           complete_tasks=sum(finished),complete=all(finished),estimated_seconds=b['estimated_seconds']))
    complete_ids=[b['id'] for b in blocks if b['selected'] and b['complete']]
    write_json(root/'block_completion.json',dict(complete_block_ids=complete_ids,blocks=blocks,
                                              comparison_rule='Primary comparisons exclude all records from an incomplete comparison block. Raw records are retained.'))
    write_csv(root/'task_inventory.csv',rows);write_csv(root/'block_inventory.csv',blocks)
    return rows,blocks


def aggregate(root):
    kinds=('fit','baseline','quench','landscape','expressivity','transfer');records={}
    for kind in kinds:
        records[kind]=[]
        for p in sorted((root/kind).glob('*.json')):
            if p.with_suffix('.npz').exists():records[kind].append(json.loads(p.read_text()))
    def base(r):
        return {k:r[k] for k in ('id','block_id','case','family','tier','dataset','data_id','split_seed','run_seed','map_seed','tie','d','p')} | dict(n=r['spec']['n'],layers=r['spec']['layers'],encoding=r['spec']['encoding'])
    for kind in ('fit','baseline'):
        rows=[]
        for r in records[kind]:
            row=base(r);row['optimizer']=r['optimizer'];row.update(r['fit'])
            for split,metrics in r['metrics'].items():row.update({split+'_'+k:v for k,v in metrics.items()})
            if r.get('orbit_bounds_apply'):
                for split,bounds in r['invariant_orbit_bounds'].items():
                    row.update({split+'_invariant_'+k:v for k,v in bounds.items()})
                    row[split+'_log_loss_excess_over_invariant_floor']=r['metrics'][split]['log_loss']-bounds['cross_entropy_floor']
            for split,bounds in r.get('constant_readout_certificate',{}).items():row.update({split+'_constant_'+k:v for k,v in bounds.items()})
            rows.append(row)
        write_csv(root/('fits.csv' if kind=='fit' else 'baselines.csv'),rows)
    quenches=[]
    for r in records['quench']:
        quenches.append(base(r)|r['quench']|dict(start=r['start'],physical_gradient_rms=r['physical_gradient_rms']))
    write_csv(root/'quenches.csv',quenches)
    landscapes=[];hessians=[];lines=[]
    for r in records['landscape']:
        d=r['diagnostics'];row=base(r)
        row.update({k:v for k,v in d.items() if not isinstance(v,(list,dict))})
        row.update({'cluster_'+k:v for k,v in d['endpoint_cluster'].items() if k not in ('assignment','counts')});landscapes.append(row)
        hessians.extend([base(r)|h for h in d['hessians']]);lines.extend([base(r)|v for v in d['lines']])
    write_csv(root/'landscape.csv',landscapes);write_csv(root/'hessians.csv',hessians);write_csv(root/'landscape_lines.csv',lines)
    expression=[]
    for r in records['expressivity']:
        d=r['diagnostics'];row=base(r);row.update({k:v for k,v in d.items() if not isinstance(v,(list,dict))})
        row['jacobian_ranks']=json.dumps(d['jacobian_ranks'])
        errors=[a['prediction_error_max'] for a in d['symmetry_audits'] if 'prediction_error_max' in a]
        comm=[a['local_generator_commutator_max'] for a in d['symmetry_audits'] if 'local_generator_commutator_max' in a]
        row.update(symmetry_error_max=max(errors) if errors else None,local_generator_commutator_max=max(comm) if comm else None)
        expression.append(row)
    write_csv(root/'expressivity.csv',expression)
    transfer_rows=[]
    for r in records['transfer']:
        transfer_rows.append(base(r)|dict(optimizer=r['optimizer'],source_fit_id=r['source_fit_id'],source_case=r['source_case'],budget=r['source_budget'],
                                        **{'test_'+k:v for k,v in r['metrics']['test'].items()}))
    write_csv(root/'transfers.csv',transfer_rows)
    datasets=[]
    for p in sorted((root/'datasets').glob('*.json')):
        r=json.loads(p.read_text());datasets.append(dict(data_id=r['data_id'],dataset=r['dataset'],seed=r['seed'],family=r['spec']['family'],n=r['spec']['n'],split_rule=r['split_rule'],
                                                       attainable_infidelity=r.get('attainable_mean_infidelity'),attainable_teacher_loss=r.get('attainable_teacher_train_loss')))
    write_csv(root/'dataset_inventory.csv',datasets)
    return {k:len(v) for k,v in records.items()}


def analysis_allowance(root):
    path=root/'submission'/f'running_{os.environ.get("SLURM_JOB_ID","unknown")}.json'
    if not path.exists():return 3600.
    r=json.loads(path.read_text());s=r['settings']
    return max(0.,s['wall_seconds']-(time.time()-r['allocation_script_start_epoch'])-180)


def analyze(root):
    log=root/'analysis.log'
    ran=[];success=True
    with log.open('a') as stream:
        for script in ('analyze.py',):
            seconds=analysis_allowance(root)
            if seconds<120:
                stream.write(f'Skipped {script}: insufficient time before scheduler limit. Submit --analysis-only inside the remaining study budget.\n');break
            try:
                r=subprocess.run([sys.executable,str(PROJECT/script),'--out',str(root)],cwd=PROJECT,
                                 stdout=stream,stderr=subprocess.STDOUT,timeout=seconds)
                ran.append(dict(script=script,exit_code=r.returncode))
                if r.returncode:success=False;stream.write(f'{script} returned {r.returncode}. Raw records are preserved.\n')
            except subprocess.TimeoutExpired:success=False;stream.write(f'{script} exceeded its finalization allowance.\n');break
    blocks=json.loads((root/'block_completion.json').read_text())['complete_block_ids']
    write_json(root/'analysis_status.json',dict(state='complete' if success and len(ran)==1 else 'partial',
                                              complete_block_ids=blocks,scripts=ran,updated_epoch=time.time()))


def report(root,rows,blocks,counts):
    states=Counter(r['state'] for r in rows);families=Counter(r['family'] for r in rows if r['state']=='complete')
    chosen=[b for b in blocks if b['selected']];complete=sum(b['complete'] for b in chosen)
    state=json.loads((root/'run_state.json').read_text()) if (root/'run_state.json').exists() else {}
    analysis=json.loads((root/'analysis_status.json').read_text()) if (root/'analysis_status.json').exists() else dict(state='not_run')
    current_ids=[b['id'] for b in blocks if b['selected'] and b['complete']]
    analysis['stale']=analysis.get('complete_block_ids')!=current_ids
    write_json(root/'analysis_status.json',analysis)
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    text=f'''# QML parameter tying: manuscript evidence

Run state: `{state.get('state','collection only')}`. Complete selected blocks: **{complete}/{len(chosen)}**.
Profile: `{manifest.get('profile','none')}`. Selected whole coverage tiers: `{manifest.get('selected_tiers',[])}`.
Task states: `{dict(states)}`. Complete records: `{counts}`.
Analysis state: `{analysis.get('state')}`; stale: `{analysis['stale']}`.

The frozen `manifest.json` contains the timing calibration, estimates, selected and deferred tasks.
`task_inventory.csv` and `block_completion.json` identify usable complete comparisons.
`run_history/` preserves interruptions. An unrun task is not an optimizer failure.
The profile/tier selection occurred before fitting, using label-free timings.

## Evidence for writing

- `fits.csv`, `baselines.csv`: train/validation/test metrics, dimensions, budgets, consumed calls and costs.
- `summary.csv`, `tying_contrasts.csv`, `optimizer_contrasts.csv`: complete paired blocks only; numerical runs averaged within dataset seed.
- `transfers.csv`, `transfer_summary.csv`: larger-input tests using unchanged smaller-circuit vectors; no target training or validation selection.
- `expressivity.csv`: sampled function/Jacobian ranks, Fourier bounds for reupload models, symmetry and Lie-generator audits.
- `quenches.csv`, `landscape.csv`, `hessians.csv`, `landscape_lines.csv`: independent accepted endpoints, entropy, gradients, spectra, coupling and physical-distance probes.
- `figures/`, `tables/`, `results_for_manuscript.tex`: plots, LaTeX tables and execution disclosure generated from available evidence.
- `datasets/`: exact cached splits, quantum states, graph orbit IDs, latent-space and singlet teacher certificates.
- `fit/`, `baseline/`, `quench/`, `landscape/`, `expressivity/`, `transfer/`: JSON and NPZ records containing angles, traces, predictions, all sampled probes, Hessians, projection bases and transfer source IDs.
- `provenance/source/`: the exact runnable project, manuscript source and PDF, supplied optimizer cores and original landscape donor.
- `provenance/`, `submission/`, `slurm/`: tests, versions, CPU/thread information, frozen settings and shared budget ledger.
- `SHA256SUMS.txt`: hashes of every archived file except itself.

Parameter clusters are gauge-dependent finite-start observations, not basin volumes or a global-minimum count.
Projected Hessians do not certify a full local minimum. Coupling graphs depend on the coordinate basis.
The known teachers are attainable solutions; they do not prove that every tying map can attain them.
Small bootstrap intervals describe variability of this finite protocol, not a quantum advantage.
Classical MLPs use roughly matched free-parameter counts and a fixed feature/PCA representation; their information access and gradient cost differ.
The optimizer is **jSO-derived**, not a certified implementation of published jSO; **iSOMA-AR** uses the unchanged supplied core.
No VQE or QAOA experiments are run; the supplied landscape script is an unexecuted methodological donor.

Slurm accounting during execution is provisional. After the job ends, optional I/O-only refresh:
`bash submit_barbora.sh --collect {root}`.
If analysis was interrupted, use the documented `--analysis-only` Slurm mode within the remaining shared budget.
The return archive is self-contained and contains no SSH credentials.
'''
    (root/'MANUSCRIPT_README.md').write_text(text)
    (root/'results_for_manuscript.tex').write_text(
        '% Generated execution disclosure. Metrics are in the accompanying CSV tables.\n'
        f'The serial simulation run completed {states.get("complete",0)} experiment tasks '
        f'and {complete} of {len(chosen)} timing-selected comparison blocks. '
        'Incomplete blocks were excluded from primary paired comparisons. '
        'The adaptive differential-evolution implementation is labelled jSO-derived; '
        'iSOMA-AR uses the supplied core with an external objective-call cap.\n')


def archive(root):
    destination=root/'qml_manuscript_results.zip';temp=root/'qml_manuscript_results.zip.tmp'
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and p not in (destination,temp) and
           p.name not in ('run.lock','submission.lock','SHA256SUMS.txt') and not p.name.endswith('.tmp') and '__pycache__' not in p.parts]
    sums=[]
    with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
        for p in files:
            rel=p.relative_to(root).as_posix();info=zipfile.ZipInfo('qml_results/'+rel,time.localtime(p.stat().st_mtime)[:6])
            info.compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED
            h=hashlib.sha256()
            with p.open('rb') as source,z.open(info,'w',force_zip64=True) as target:
                while chunk:=source.read(1024*1024):h.update(chunk);target.write(chunk)
            sums.append(f'{h.hexdigest()}  {rel}')
        checksum='\n'.join(sums)+'\n';z.writestr('qml_results/SHA256SUMS.txt',checksum)
    (root/'SHA256SUMS.txt').write_text(checksum);temp.replace(destination)
    return destination


def collect(root,analysis=True):
    root=Path(root).absolute();root.mkdir(parents=True,exist_ok=True)
    accounting(root);rows,blocks=inventory(root);counts=aggregate(root)
    if analysis:analyze(root)
    report(root,rows,blocks,counts);path=archive(root)
    print(f'RETURN ARCHIVE: {path} ({path.stat().st_size/2**20:.1f} MiB)',flush=True)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='results/barbora');p.add_argument('--quick',action='store_true');a=p.parse_args()
    collect(Path(a.out),not a.quick)


if __name__=='__main__':main()
