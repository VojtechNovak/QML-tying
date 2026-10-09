"""Slurm admission with a cumulative 50 node-hour reservation ledger (stdlib)."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

PROJECT=Path(__file__).absolute().parent
CAP_SECONDS=50*3600
LIMITS={'qcpu':48,'qcpu_biz':48,'qcpu_long':144,'qcpu_exp':1,'qcpu_free':18,'qcpu_preempt':12}
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def read_accounting(job_id):
    try:
        r=subprocess.run(['sacct','-j',str(job_id),'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,AllocNodes'],text=True,capture_output=True,timeout=20)
        if r.returncode:return None
        for line in r.stdout.splitlines():
            fields=line.split('|')
            if fields[0]!=str(job_id) or len(fields)<4:continue
            state=fields[1].split()[0].rstrip('+');elapsed=int(fields[2]);nodes=int(fields[3])
            return dict(state=state,elapsed_seconds=elapsed,allocated_nodes=nodes)
    except (OSError,ValueError,subprocess.TimeoutExpired):pass
    return None


def refresh(ledger,reader=read_accounting):
    for job in ledger['jobs']:
        if job.get('job_id'):
            info=reader(job['job_id'])
            if info:
                job.update(accounting=info,accounting_checked_epoch=time.time())
                if info['state'] in TERMINAL:
                    # ElapsedRaw excludes queueing. Retain one minute for rounding.
                    job.update(terminal=True,charged_upper_seconds=info['elapsed_seconds']*max(1,info['allocated_nodes'])+60)
    return ledger


def ledger_path(explicit=None):
    """Use the SAME ledger/lock as v2 when it is present next to this release."""
    if explicit:return Path(explicit).expanduser().absolute()
    pointer=PROJECT/'cluster_control/budget_pointer.json'
    if pointer.exists():return Path(json.loads(pointer.read_text())['ledger_path'])
    previous=PROJECT.parent/'qml_parameter_tying_barbora/cluster_control/budget_ledger.json'
    return previous if previous.exists() else PROJECT/'cluster_control/budget_ledger.json'


def admission(ledger,requested_hours,partition):
    if partition not in LIMITS:raise ValueError('Use a documented Barbora CPU partition: '+', '.join(LIMITS))
    active=[j for j in ledger['jobs'] if not j.get('terminal',False) and not j.get('submission_failed',False)]
    if active:raise RuntimeError('A submitted, pending or unaccounted job still holds this serial study budget. Wait for terminal sacct accounting before resubmitting.')
    used=sum(j.get('charged_upper_seconds',j['reserved_seconds']) for j in ledger['jobs'] if not j.get('submission_failed',False))
    if requested_hours is None:requested_hours=min(LIMITS[partition],max(0.,(CAP_SECONDS-used-120)/3600))
    if not math.isfinite(requested_hours) or requested_hours<=0:raise ValueError('--hours must be positive and finite; the shared budget may already be exhausted')
    if requested_hours>LIMITS[partition]:raise ValueError(f'{partition} allows at most {LIMITS[partition]} hours; use qcpu_long for a request above 48 h.')
    wall_seconds=int(math.floor(requested_hours*60))*60
    # Include a two-minute scheduler cleanup margin in admission accounting.
    reserve=wall_seconds+120
    if wall_seconds<300:raise ValueError('Request at least five minutes.')
    if used+reserve>CAP_SECONDS:raise ValueError(f'Cumulative cap is 50 node-hours. At most {max(0,(CAP_SECONDS-used-120))/3600:.3f} job hours remain (including cleanup margin).')
    return wall_seconds,reserve,used


def collect(out,python,explicit=None):
    path=ledger_path(explicit)
    if path.exists():
        lock=(path.parent/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=refresh(json.loads(path.read_text()));write_json(path,ledger)
        write_json(out/'submission/budget_ledger.json',ledger)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        for job in ledger['jobs']:
            if job.get('job_id'):
                for tag,cmd in [('sacct',['sacct','-j',job['job_id'],'--parsable2','--format=JobIDRaw,State,ElapsedRaw,AllocNodes,AllocCPUS,ExitCode,Start,End,MaxRSS']),
                                ('scontrol',['scontrol','show','job',job['job_id']])]:
                    try:r=subprocess.run(cmd,text=True,capture_output=True,timeout=20);txt=r.stdout+r.stderr
                    except (OSError,subprocess.TimeoutExpired) as exc:txt=str(exc)
                    dest=out/'slurm'/f'{tag}_{job["job_id"]}_final.txt';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(txt)
    # Login-node collection performs I/O only; heavy analysis remains a Slurm job.
    return subprocess.call([python,str(PROJECT/'collect_results.py'),'--out',str(out),'--quick'],cwd=PROJECT)


def submit(a):
    if not a.account and not os.environ.get('SBATCH_ACCOUNT'):
        raise ValueError('Supply --account PROJECT-ID (your IT4I allocation) or set SBATCH_ACCOUNT.')
    a.account=a.account or os.environ['SBATCH_ACCOUNT'];out=Path(a.out).absolute();config=Path(a.config).absolute()
    if not config.is_file():raise ValueError('Missing configuration: '+str(config))
    path=ledger_path(a.budget_ledger);controller=path.parent;controller.mkdir(parents=True,exist_ok=True)
    lock=(controller/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    ledger=json.loads(path.read_text()) if path.exists() else dict(schema=1,cap_node_hours=50,jobs=[])
    ledger=refresh(ledger);wall,reserve,used=admission(ledger,a.hours,a.partition)
    rid=uuid.uuid4().hex;settings_path=out/'submission'/f'{rid}.json'
    cfg_path=out/'submission'/f'{rid}_protocol.json'
    hours,rem=divmod(wall,3600);minutes,seconds=divmod(rem,60)
    cmd=['sbatch','--parsable','--job-name=qml_tying','--account='+a.account,'--partition='+a.partition,
         '--nodes=1','--ntasks=1','--cpus-per-task=36','--no-requeue','--export=ALL',
         f'--time={hours:02}:{minutes:02}:{seconds:02}',f'--signal=B:USR1@{min(600,wall//10)}',
         '--chdir='+str(PROJECT),'--output='+str(out/'slurm/slurm-%j.out'),'--error='+str(out/'slurm/slurm-%j.err'),
         str(PROJECT/'submit_barbora.sh'),'--batch',a.python,str(settings_path)]
    if a.dry_run:
        import shlex
        print('DRY RUN: '+shlex.join(cmd))
        print('Budget ledger: '+str(path))
        print(f'One serial allocation; request {wall/3600:g} h; cumulative reservation {(used+reserve)/3600:.3f}/50 NH. No installation or submission performed.')
        return 0
    from collect_results import source_files
    source_hashes={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(PROJECT)}
    settings=dict(python=a.python,config=str(cfg_path),out=str(out),account=a.account,partition=a.partition,
                  wall_seconds=wall,finalization_reserve_seconds=min(5400,max(90,int(wall*.1))),
                  skip_tests=a.skip_tests,analysis_only=a.analysis_only,source_hashes=source_hashes,requested_epoch=time.time(),reservation_id=rid,budget_ledger_path=str(path))
    out.mkdir(parents=True,exist_ok=True);(out/'slurm').mkdir(exist_ok=True)
    write_json(cfg_path,json.loads(config.read_text()));write_json(settings_path,settings)
    write_json(PROJECT/'cluster_control/budget_pointer.json',dict(ledger_path=str(path)))
    job=dict(reservation_id=rid,reserved_seconds=reserve,wall_seconds=wall,terminal=False,python=a.python,project=str(PROJECT),
             out=str(out),settings=str(settings_path),requested_epoch=time.time())
    ledger['jobs'].append(job);write_json(path,ledger)
    try:
        r=subprocess.run(cmd,text=True,capture_output=True,timeout=45)
        if r.returncode:
            job.update(submission_failed=True,terminal=True,error=r.stderr)
            raise RuntimeError('sbatch rejected submission: '+r.stderr.strip())
        job_id=r.stdout.strip().split(';')[0]
        if not job_id.isdigit():raise RuntimeError('Unrecognized sbatch response; reservation retained: '+r.stdout)
        job['job_id']=job_id
    finally:
        write_json(path,ledger);write_json(out/'submission/budget_ledger.json',ledger)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(f'Submitted Slurm job {job_id}: one node, one serial process, wall limit {wall/3600:g} h.')
    print(f'Cumulative study reservation: {(used+reserve)/3600:.3f}/50 node-hours. Return zip: {out}/qml_manuscript_results.zip')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--python',default=sys.executable)
    p.add_argument('--account');p.add_argument('--partition',default='qcpu');p.add_argument('--hours',type=float,help='default: min(partition maximum, remaining shared 50-NH allowance)')
    p.add_argument('--config',default=str(PROJECT/'configs/barbora_48h.json'));p.add_argument('--out',default=str(PROJECT/'results/barbora'))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--skip-tests',action='store_true');p.add_argument('--collect')
    p.add_argument('--analysis-only',action='store_true')
    p.add_argument('--budget-ledger',help='existing study ledger elsewhere; shares its lock and cumulative limit')
    p.add_argument('--print-python',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.print_python:
        path=ledger_path(a.budget_ledger)
        if not path.exists():raise SystemExit('No budget ledger found; activate your existing environment or pass --env.')
        jobs=json.loads(path.read_text()).get('jobs',[])
        if not jobs:raise SystemExit('No interpreter recorded; activate your existing environment or pass --env.')
        print(jobs[-1]['python']);return
    try:code=collect(Path(a.collect).absolute(),a.python,a.budget_ledger) if a.collect else submit(a)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);code=1
    sys.exit(code)


if __name__=='__main__':main()
