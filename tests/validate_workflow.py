"""Optional LOCAL validation with fake Slurm commands; never submits a real job.

Run from a local machine with dependencies installed, not a cluster login node:
python tests/validate_workflow.py --out /absolute/local/temporary/directory
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).absolute().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out).absolute()
    if root.exists():raise SystemExit('Use a fresh output directory for this local validation.')
    root.mkdir(parents=True);bin_dir=root/'fake_slurm';bin_dir.mkdir();budget=root/'control/budget_ledger.json';run_out=root/'run'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',QML_FAKE_SLURM_DIR=str(root))
    commands={
      'sbatch': "import sys,json,os\nfrom pathlib import Path\nr=Path(os.environ['QML_FAKE_SLURM_DIR']);(r/'sbatch_arguments.json').write_text(json.dumps(sys.argv[1:],indent=2));print('987654')\n",
      'sacct': "import json,os\nfrom pathlib import Path\nr=Path(os.environ['QML_FAKE_SLURM_DIR']);p=r/'fake_state.json';s=json.loads(p.read_text()) if p.exists() else dict(state='RUNNING',seconds=0);print('987654|'+s['state']+'|'+str(s['seconds'])+'|1|36|0:0')\n",
      'scontrol': "print('JobId=987654 JobState=LOCAL_VALIDATION NumNodes=1 NumCPUs=36')\n"}
    for name,body in commands.items():
        file=bin_dir/name;file.write_text('#!'+sys.executable+'\n'+body);file.chmod(0o755)
    env['PATH']=str(bin_dir)+os.pathsep+env.get('PATH','')
    # A real, already-created venv prefix, with the local runtime dependencies.
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(root/'existing_env')],check=True,env=env)
    python=root/'existing_env/bin/python';launcher=PROJECT/'submit_barbora.sh'
    common=['bash',str(launcher),'--env',str(root/'existing_env'),'--account','LOCAL-VALIDATION','--hours','1',
            '--config',str(PROJECT/'configs/cluster_smoke.json'),'--out',str(run_out),'--budget-ledger',str(budget)]
    with (root/'dry_run.log').open('w') as stream:
        subprocess.run(common+['--dry-run'],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert not budget.exists(),'dry run must not reserve or submit'
    with (root/'submission.log').open('w') as stream:
        subprocess.run(common+['--skip-install'],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    argv=json.loads((root/'sbatch_arguments.json').read_text())
    assert '--nodes=1' in argv and '--ntasks=1' in argv and '--cpus-per-task=36' in argv and '--no-requeue' in argv
    assert '--exclusive' not in argv and not any(v.startswith('--mem=') for v in argv)
    ledger=json.loads(budget.read_text());settings=ledger['jobs'][-1]['settings']
    # The ordinary installation path must reject an active prior reservation
    # before invoking pip, including when the caller omits --skip-install.
    blocked=subprocess.run(common,cwd=PROJECT,env=env,capture_output=True,text=True)
    (root/'active_reservation_guard.log').write_text(blocked.stdout+blocked.stderr)
    assert blocked.returncode!=0 and 'still holds this serial study budget' in blocked.stderr
    assert 'Collecting ' not in blocked.stdout and 'Requirement already satisfied' not in blocked.stdout
    assert len(json.loads(budget.read_text())['jobs'])==1
    env.update(SLURM_JOB_ID='987654',SLURM_JOB_NUM_NODES='1',SLURM_CPUS_PER_TASK='36',SLURM_JOB_PARTITION='qcpu',SLURM_JOB_ACCOUNT='LOCAL-VALIDATION')
    began=time.time()
    with (root/'batch.log').open('w') as stream:
        subprocess.run(['bash',str(launcher),'--batch',str(python),settings],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    elapsed=time.time()-began;(root/'fake_state.json').write_text(json.dumps(dict(state='COMPLETED',seconds=math.ceil(elapsed))))
    # Login-node collection must only refresh accounting and package files.
    with (root/'collection.log').open('w') as stream:
        subprocess.run([str(python),str(PROJECT/'cluster_submit.py'),'--collect',str(run_out),'--budget-ledger',str(budget)],cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    ledger=json.loads(budget.read_text());assert ledger['jobs'][-1]['terminal'];assert ledger['jobs'][-1]['charged_upper_seconds']<=math.ceil(elapsed)+60
    state=json.loads((run_out/'run_state.json').read_text());assert state['state']=='selected_plan_complete' and state['failures']==0
    complete=json.loads((run_out/'block_completion.json').read_text());assert all(b['complete'] for b in complete['blocks'])
    first={};counts={};chain=[]
    for kind in ('fit','baseline','quench','landscape','expressivity','transfer'):
        records=list((run_out/kind).glob('*.json'));counts[kind]=len(records)
        for path in records:
            r=json.loads(path.read_text())
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                if kind=='fit':
                    assert r['fit']['nfev']<=r['fit']['budget'];assert z['candidate_nfev'][0]==1
                    key=(r['case'],r['split_seed'],r['run_seed'],r['map_seed'],r['d'])
                    if key in first:np.testing.assert_array_equal(first[key],z['candidate_phi'][0])
                    else:first[key]=z['candidate_phi'][0]
                if kind=='landscape':
                    assert 'untied_gradient_0' in z.files and 'known_parameter_periods' in z.files
                    chain.extend(v['chain_rule_residual'] for v in r['diagnostics']['release_gradient_diagnostics'])
                if kind=='transfer':
                    assert r['optimization_performed'] is False and r['target_training_or_validation_used'] is False
                    parent=run_out/'fit'/f'{r["source_fit_id"]}.npz'
                    with np.load(parent,allow_pickle=False) as source:np.testing.assert_array_equal(source['phi'],z['phi'])
    assert counts==dict(fit=112,baseline=18,quench=56,landscape=28,expressivity=28,transfer=32);assert max(chain)<1e-8
    integrity=0
    with zipfile.ZipFile(run_out/'qml_manuscript_results.zip') as z:
        assert z.testzip() is None
        for line in z.read('qml_results/SHA256SUMS.txt').decode().splitlines():
            digest,rel=line.split('  ',1);assert hashlib.sha256(z.read('qml_results/'+rel)).hexdigest()==digest;integrity+=1
    report=dict(local_only=True,real_cluster_submission=False,workflow='shell entry -> existing venv -> dry run -> emulated sbatch -> batch trap/controller -> analysis -> terminal collection',
                elapsed_seconds=elapsed,complete_tasks=sum(counts.values()),counts=counts,complete_blocks=len(complete['complete_block_ids']),
                maximum_shared_chain_rule_residual=max(chain),archive_files_hash_verified=integrity,matched_initialization=True,
                all_npz_without_pickle=True,unchanged_zero_shot_transfer=True,active_reservation_blocks_installation=True,
                analysis=json.loads((run_out/'analysis_status.json').read_text()),budget_ledger=ledger)
    (root/'workflow_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('budget_ledger','analysis')},indent=2))


if __name__=='__main__':main()
