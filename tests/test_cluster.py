import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cluster_submit
from cluster_plan import make_plan
from cluster_run import freeze_feasibility,valid_record
from qml_tying.models import Model,ModelSpec
from collect_results import archive

ROOT=Path(__file__).absolute().parents[1]


class ClusterTests(unittest.TestCase):
    def test_reserved_allocation_and_cleanup_within_50(self):
        wall,reserve,used=cluster_submit.admission({'jobs':[]},None,'qcpu')
        self.assertEqual(wall,48*3600);self.assertEqual(reserve,wall+120);self.assertLess(reserve,50*3600)

    def test_prior_completed_job_reduces_default_request(self):
        ledger={'jobs':[dict(terminal=True,reserved_seconds=48*3600+120,charged_upper_seconds=48*3600+60)]}
        wall,reserve,used=cluster_submit.admission(ledger,None,'qcpu');self.assertLess(wall,2*3600);self.assertLessEqual(used+reserve,50*3600)
        with self.assertRaises(ValueError):cluster_submit.admission(ledger,48,'qcpu')

    def test_pending_unknown_and_running_jobs_block(self):
        for state in ('PENDING','RUNNING','UNKNOWN'):
            with self.assertRaises(RuntimeError):cluster_submit.admission({'jobs':[dict(reserved_seconds=3600,terminal=False,state=state)]},1,'qcpu')

    def test_terminal_accounting_charges_elapsed_times_nodes(self):
        ledger={'jobs':[dict(job_id='123',reserved_seconds=900,terminal=False)]}
        cluster_submit.refresh(ledger,lambda job:dict(state='COMPLETED',elapsed_seconds=120,allocated_nodes=2))
        self.assertEqual(ledger['jobs'][0]['charged_upper_seconds'],300)

    def test_old_sibling_ledger_is_shared_and_pointer_persists(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as d:
            base=Path(d);new=base/'qml_parameter_tying_barbora_v3';new.mkdir();old=base/'qml_parameter_tying_barbora/cluster_control';old.mkdir(parents=True);ledger=old/'budget_ledger.json';ledger.write_text('{"jobs":[]}')
            with patch.object(cluster_submit,'PROJECT',new):self.assertEqual(cluster_submit.ledger_path(),ledger)
            (new/'cluster_control').mkdir();(new/'cluster_control/budget_pointer.json').write_text(json.dumps(dict(ledger_path=str(ledger))))
            with patch.object(cluster_submit,'PROJECT',new):self.assertEqual(cluster_submit.ledger_path(),ledger)

    def test_plan_has_all_six_core_families_and_only_four_ranked_optimizers(self):
        cfg=json.loads((ROOT/'configs/barbora_48h.json').read_text());p=make_plan(cfg)
        families={b['family'] for b in p['blocks'] if b['tier']=='core'}
        self.assertEqual(families,{'reupload','qcnn','autoencoder','recurrent','graph','su2'})
        self.assertEqual(set(t['config']['optimizer'] for b in p['blocks'] for t in b['tasks'] if t['kind']=='fit'),{'cobyla','de','jSO-derived','iSOMA-AR'})
        ids=[t['id'] for b in p['blocks'] for t in b['tasks']];self.assertEqual(len(ids),len(set(ids)))

    def test_timing_selection_keeps_whole_core_and_whole_tiers(self):
        cfg=json.loads((ROOT/'configs/cluster_smoke.json').read_text());p=make_plan(cfg);tables={}
        from cluster_run import timing_key
        for b in p['blocks']:
            for t in b['tasks']:
                if t['kind']=='baseline':continue
                c=t['config'];m=Model(ModelSpec(**c['case']['spec']),c['mode'],c['map_seed'])
                tables[timing_key(t)]=dict(seconds_per_16=.0001,seconds_per_8_jacobian=.0002,d=m.tying.d)
        p=freeze_feasibility(p,tables,10000);self.assertTrue(all(b['selected'] for b in p['blocks']))
        with self.assertRaises(ValueError):freeze_feasibility(make_plan(cfg),tables,1)

    def test_transfer_tasks_depend_on_same_dimension_source_fit(self):
        cfg=json.loads((ROOT/'configs/cluster_smoke.json').read_text());p=make_plan(cfg)
        tasks={t['id']:t for b in p['blocks'] for t in b['tasks']}
        transfers=[t for t in tasks.values() if t['kind']=='transfer'];self.assertEqual(len(transfers),32)
        for t in transfers:
            c=t['config'];parent=tasks[c['source_fit_id']];pc=parent['config']
            self.assertEqual(t['dependencies'],[parent['record']]);self.assertEqual(c['mode'],pc['mode'])
            self.assertEqual(c['optimizer'],pc['optimizer']);self.assertEqual(c['split_seed'],pc['split_seed'])
            a=Model(ModelSpec(**pc['case']['spec']),pc['mode'],pc['map_seed'])
            b=Model(ModelSpec(**c['case']['spec']),c['mode'],c['map_seed'])
            self.assertEqual(a.tying.d,b.tying.d);self.assertGreater(b.spec.n,a.spec.n)
        self.assertEqual(cfg['tier_dependencies']['transfer'],['core','teacher'])

    def test_transfer_uses_unchanged_source_phi_without_optimization(self):
        from run_task import run_task
        cfg=json.loads((ROOT/'configs/cluster_smoke.json').read_text());p=make_plan(cfg)
        t=next(t for b in p['blocks'] for t in b['tasks'] if t['kind']=='transfer' and t['config']['source_case']=='recurrent_stationary_teacher')
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as d:
            root=Path(d);(root/'fit').mkdir();pid=t['config']['source_fit_id'];phi=np.linspace(-.5,.5,6)
            parent=dict(id=pid,optimizer=t['config']['optimizer'],data_id='smaller-data',spec=dict(family='recurrent',n=4),fit=dict(budget=40))
            (root/'fit'/f'{pid}.json').write_text(json.dumps(parent));np.savez(root/'fit'/f'{pid}.npz',phi=phi)
            with patch('run_task.optimize',side_effect=AssertionError('Transfer must not optimize.')):r=run_task(t,root)
            self.assertFalse(r['optimization_performed']);self.assertFalse(r['target_training_or_validation_used'])
            with np.load(root/t['record'].replace('.json','.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['phi'],phi)
            self.assertTrue(valid_record(root,t))

    def test_atomic_record_needs_payload_and_matching_id(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as d:
            root=Path(d);(root/'fit').mkdir();task=dict(id='a',record='fit/a.json',kind='fit');p=root/task['record'];p.write_text('{"id":"a","complete":true}')
            self.assertFalse(valid_record(root,task));np.savez(p.with_suffix('.npz'),phi=np.zeros(1),trace=np.zeros((1,3)))
            self.assertTrue(valid_record(root,task));p.write_text('{"id":"b","complete":true}');self.assertFalse(valid_record(root,task))

    def test_return_archive_checksums(self):
        import zipfile
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as d:
            root=Path(d);(root/'a.txt').write_text('raw evidence\n');path=archive(root)
            with zipfile.ZipFile(path) as z:
                self.assertIsNone(z.testzip());checksum=z.read('qml_results/SHA256SUMS.txt').decode().split()[0]
                self.assertEqual(checksum,hashlib.sha256(z.read('qml_results/a.txt')).hexdigest())


if __name__=='__main__':unittest.main()
