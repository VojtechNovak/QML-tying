import hashlib
import unittest
from pathlib import Path
import numpy as np
from qml_tying.models import Model,ModelSpec,collective_rotate,relabel_state,pauli
from qml_tying.datasets import haar,generate,graph_relabel,graph_canonical
from qml_tying.core import loss_and_grad,loss
from qml_tying.diagnostics import symmetry,expressivity,hessian,clusters
from qml_tying.datasets import cache_data
import tempfile
import subprocess
import sys
import os
from qml_tying.optimizers import optimize
from qml_tying.quench import quench
from qml_tying.classical import MLP
from qml_tying import runtime

ROOT=Path(__file__).absolute().parents[1]
MODES={'reupload':['untied','depth','random_depth','space'],
       'qcnn':['untied','spatial','random_spatial','scale_shared'],
       'autoencoder':['untied','inverse','inverse_depth','inverse_random_depth'],
       'recurrent':['untied','temporal','random_temporal','block_time'],
       'graph':['untied','node_edge','random_node_edge','layer_shared'],
       'su2':['untied','su2','random_su2','su2_spatial']}


def inputs(rng,spec,k=4):
    if spec.family in ('qcnn','autoencoder','su2'):return haar(rng,k,1<<spec.n)
    width=2 if spec.family=='reupload' else spec.n-1 if spec.family=='recurrent' else spec.n*(spec.n-1)//2
    a=rng.normal(size=(k,width))
    return (a>0).astype(float) if spec.family=='graph' else a


class NumericalTests(unittest.TestCase):
    def tearDown(self):runtime.configure(None)

    def test_every_family_and_tying_directional_gradient(self):
        rng=np.random.default_rng(777)
        for family,modes in MODES.items():
            spec=ModelSpec(family,4);x=inputs(rng,spec);y=np.array([0,1,0,1])
            for mode in modes:
                with self.subTest(family=family,mode=mode):
                    model=Model(spec,mode,17);phi=rng.normal(size=model.tying.d);v=rng.normal(size=model.tying.d);v/=np.linalg.norm(v)
                    value,g=loss_and_grad(model,x,y,phi,model.task);h=1e-5
                    fd=(loss(model.predict(x,phi+h*v),y,model.task)-loss(model.predict(x,phi-h*v),y,model.task))/(2*h)
                    self.assertAlmostEqual(float(g@v),float(fd),places=7)

    def test_all_families_preserve_total_pure_or_branch_norm(self):
        rng=np.random.default_rng(41)
        for family in MODES:
            spec=ModelSpec(family,4);m=Model(spec,MODES[family][1]);x=inputs(rng,spec);phi=rng.normal(size=m.tying.d)
            state=m.predict(x,phi,states=True);axes=tuple(range(1,state.ndim))
            np.testing.assert_allclose(np.sum(abs(state)**2,axis=axes),1,atol=3e-13)

    def test_autoencoder_matches_explicit_mixed_reconstruction(self):
        rng=np.random.default_rng(20);spec=ModelSpec('autoencoder',3,layers=1,latent=1);m=Model(spec,'untied');phi=rng.normal(size=m.tying.d);x=haar(rng,3,8)
        cut=len(m.ops)//2;enc=m.evolve(x,phi,m.ops[:cut]);u=m.evolve(np.eye(8,dtype=complex),phi,m.ops[cut:]).T;reference=[]
        for psi,z in zip(x,enc):
            z=z.reshape(4,2);rho=z.T@z.conj();reset=np.zeros((8,8),complex);reset[:2,:2]=rho
            decoded=u@reset@u.conj().T;reference.append(np.real(psi.conj()@decoded@psi))
            self.assertAlmostEqual(float(np.trace(decoded).real),1,places=12)
        np.testing.assert_allclose(m.predict(x,phi),reference,atol=1e-12)

    def test_inverse_decoder_is_encoder_adjoint(self):
        rng=np.random.default_rng(31);m=Model(ModelSpec('autoencoder',4),'inverse');phi=rng.normal(size=m.tying.d);x=haar(rng,2,16);cut=len(m.ops)//2
        encoded=m.evolve(x,phi,m.ops[:cut]);decoded=m.evolve(encoded,phi,m.ops[cut:]);np.testing.assert_allclose(decoded,x,atol=1e-12)

    def test_compressible_teacher_attains_unit_fidelity(self):
        for dataset,mode in [('compress_depth','inverse_depth'),('compress_untied','inverse')]:
            spec=ModelSpec('autoencoder',4);data=generate(spec,dataset,12,24);m=Model(spec,mode)
            np.testing.assert_allclose(m.predict(data.x_test,data.extras['teacher_phi']),1,atol=1e-12)

    def test_recurrent_zero_angles_compute_parity(self):
        m=Model(ModelSpec('recurrent',4),'temporal');bits=np.array([[0,0,0],[0,1,0],[1,1,1],[1,1,0]])
        np.testing.assert_allclose(m.predict(bits*np.pi,np.zeros(m.tying.d)),2*(bits.sum(1)%2)-1,atol=1e-12)

    def test_nonzero_recurrent_teacher_reuses_rule_across_lengths(self):
        truth=None
        for n in (4,6):
            spec=ModelSpec('recurrent',n);data=generate(spec,'sequence_teacher',11,24)
            phi=data.extras['teacher_phi'];self.assertGreater(np.linalg.norm(phi),.1)
            if truth is not None:np.testing.assert_array_equal(truth,phi)
            truth=phi
            prediction=Model(spec,'temporal').predict(data.x_test,phi)
            np.testing.assert_array_equal((prediction>=0).astype(int),data.y_test)

    def test_random_control_preserves_counts_and_changes_map(self):
        for family,(structured,random) in {'reupload':('depth','random_depth'),'qcnn':('spatial','random_spatial'),
           'autoencoder':('inverse_depth','inverse_random_depth'),'recurrent':('temporal','random_temporal'),
           'graph':('node_edge','random_node_edge'),'su2':('su2','random_su2')}.items():
            a=Model(ModelSpec(family,4),structured);b=Model(a.spec,random,17)
            self.assertEqual(a.tying.d,b.tying.d);np.testing.assert_array_equal(np.sort(a.tying.multiplicity),np.sort(b.tying.multiplicity))
            self.assertFalse(np.array_equal(a.tying.ids,b.tying.ids),family)

    def test_random_maps_reproduce_across_python_hash_seeds(self):
        code="from qml_tying.models import Model,ModelSpec;import json;print(json.dumps([Model(ModelSpec(f,4),m,17).tying.ids.tolist() for f,m in [('qcnn','random_spatial'),('autoencoder','inverse_random_depth'),('graph','random_node_edge'),('recurrent','random_temporal'),('su2','random_su2')]]))"
        values=[]
        for seed in ('2','95'):
            r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=dict(os.environ,PYTHONHASHSEED=seed),capture_output=True,text=True,check=True)
            values.append(r.stdout)
        self.assertEqual(values[0],values[1])

    def test_graph_channel_equivariance_and_invariant_readout(self):
        rng=np.random.default_rng(62);spec=ModelSpec('graph',5);m=Model(spec,'node_edge');x=inputs(rng,spec,5);phi=rng.normal(size=m.tying.d);p=rng.permutation(5)
        np.testing.assert_allclose(m.predict(x,phi),m.predict(graph_relabel(x,5,p),phi),atol=2e-13)
        np.testing.assert_allclose(relabel_state(m.predict(x,phi,states=True),5,p),m.predict(graph_relabel(x,5,p),phi,states=True),atol=2e-13)

    def test_graph_orbit_splits_do_not_leak(self):
        data=generate(ModelSpec('graph',6),'graph_connectivity',11,24)
        sets=[set(data.extras['orbit_'+s].tolist()) for s in ('train','val','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        for s in ('train','val','test'):
            for x,key in zip(getattr(data,'x_'+s),data.extras['orbit_'+s]):self.assertEqual(graph_canonical(x,6),key)
        self.assertAlmostEqual(data.metadata['invariant_orbit_bounds']['train']['accuracy_upper_bound'],1.)
        broken=generate(ModelSpec('graph',6),'graph_edge_mismatch',11,24)
        self.assertLess(broken.metadata['invariant_orbit_bounds']['train']['accuracy_upper_bound'],1.)

    def test_su2_readout_and_generators_commute(self):
        rng=np.random.default_rng(55);spec=ModelSpec('su2',4);m=Model(spec,'su2');x=haar(rng,5,16);phi=rng.normal(size=m.tying.d)
        np.testing.assert_allclose(m.predict(x,phi),m.predict(collective_rotate(x,4,[.4,1.2,-.7]),phi),atol=3e-13)
        audit=symmetry(m,x,phi,rng);self.assertLess(audit['local_generator_commutator_max'],1e-13)

    def test_anisotropic_control_can_break_su2(self):
        rng=np.random.default_rng(56);m=Model(ModelSpec('su2',4),'untied');x=haar(rng,5,16);phi=rng.normal(size=m.tying.d)
        self.assertGreater(np.max(abs(m.predict(x,phi)-m.predict(collective_rotate(x,4,[.4,1.2,-.7]),phi))),1e-3)

    def test_singlet_teacher_and_mismatched_constant(self):
        spec=ModelSpec('su2',4);m=Model(spec,'su2');data=generate(spec,'singlet_teacher',11,24)
        np.testing.assert_allclose(m.predict(data.x_test,data.extras['teacher_phi']),2*data.y_test-1,atol=1e-12)
        broken=generate(spec,'spin_orientation_mismatch',11,24)
        np.testing.assert_allclose(m.predict(broken.x_train,np.ones(m.tying.d)), -1,atol=1e-12)

    def test_fourier_encoding_bounds(self):
        m=Model(ModelSpec('reupload',2,layers=2),'depth');data=generate(m.spec,'moons',11,24)
        r,_=expressivity(m,data,dict(parameter_samples=3,jacobian_samples=1),72)
        self.assertLess(r['maximum_out_of_bound_coefficient'],1e-12)

    def test_full_and_projected_hessian_agree_with_direction(self):
        rng=np.random.default_rng(63);m=Model(ModelSpec('recurrent',3),'temporal');x=inputs(rng,m.spec);y=np.array([0,1,0,1]);phi=rng.normal(size=m.tying.d)
        stats,a=hessian(m,x,y,phi,rng,full_limit=8);self.assertEqual(stats['mode'],'full');self.assertLess(stats['fd_relative_residual'],.01)
        v=rng.normal(size=m.tying.d);v/=np.linalg.norm(v);v/=np.sqrt(m.tying.multiplicity);h=1e-3
        second=(loss(m.predict(x,phi+h*v),y)-2*loss(m.predict(x,phi),y)+loss(m.predict(x,phi-h*v),y))/h**2
        physical=v*np.sqrt(m.tying.multiplicity);self.assertAlmostEqual(float(physical@a['hessian']@physical),second,places=5)
        projected,raw=hessian(m,x,y,phi,rng,full_limit=2,projection=3);self.assertEqual(projected['mode'],'projected');self.assertEqual(raw['hessian'].shape,(3,3))

    def test_hard_ceiling_for_each_primary_optimizer(self):
        spec=ModelSpec('recurrent',3);m=Model(spec,'temporal');data=generate(spec,'sequence_parity',12,24)
        for method in ('cobyla','de','jSO-derived','iSOMA-AR'):
            with self.subTest(method=method):
                result=optimize(m,data,method,57,seed=81,checkpoints=3)
                self.assertLessEqual(result['nfev'],57);self.assertEqual(len(result['trace']),result['nfev']);self.assertTrue(np.isfinite(result['selected_val_loss']))

    def test_quench_uses_accepted_iterate_and_charged_gradient_calls(self):
        spec=ModelSpec('recurrent',3);m=Model(spec,'temporal');data=generate(spec,'sequence_parity',12,24);q=quench(m,data,5,82)
        self.assertLessEqual(q['nfev'],5);self.assertIn('accepted',q['endpoint_definition'])
        v,g=loss_and_grad(m,data.x_train,data.y_train,q['phi']);self.assertAlmostEqual(v,q['accepted_loss'],places=12);np.testing.assert_allclose(g,q['gradient'],atol=1e-10)

    def test_mlp_analytic_gradient(self):
        rng=np.random.default_rng(12);m=MLP(3);t=m.initialize(rng);x=rng.normal(size=(5,2));y=np.array([0,1,0,1,0]);v,g=m.vg(x,y,t);direction=rng.normal(size=m.d);h=1e-5
        fd=(loss(m.predict(x,t+h*direction),y)-loss(m.predict(x,t-h*direction),y))/(2*h)
        self.assertAlmostEqual(fd,float(g@direction),places=7)

    def test_physical_slot_gradient_obeys_shared_chain_rule(self):
        rng=np.random.default_rng(88)
        for family,modes in MODES.items():
            model=Model(ModelSpec(family,4),modes[1]);independent=Model(model.spec,'untied');phi=rng.normal(size=model.tying.d)
            x=inputs(rng,model.spec);y=np.array([0,1,0,1]);v,g=loss_and_grad(model,x,y,phi,model.task)
            vi,gi=loss_and_grad(independent,x,y,model.tying.expand(phi),model.task)
            self.assertAlmostEqual(v,vi,places=12);np.testing.assert_allclose(g,model.tying.matrix().T@gi,atol=1e-11)

    def test_torus_endpoint_clusters_and_entropy(self):
        r=clusters(np.array([[np.pi-.01],[-np.pi+.01],[0]]),np.array([.2,.2,.4]))
        self.assertEqual(r['n_clusters'],2);self.assertAlmostEqual(r['best_observed_cluster_occupancy'],2/3)
        r=clusters(np.array([[0],[np.pi]]),np.array([.2,.2]),periods=np.array([np.pi]))
        self.assertEqual(r['n_clusters'],1)

    def test_cached_payload_integrity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as d:
            spec=ModelSpec('recurrent',3);data,key=cache_data(Path(d),spec,'sequence_parity',11,24)
            again,_=cache_data(Path(d),spec,'sequence_parity',11,24);np.testing.assert_array_equal(data.x_train,again.x_train)
            payload=Path(d)/'datasets'/f'{key}.npz';payload.write_bytes(payload.read_bytes()+b'corrupted')
            with self.assertRaises(IOError):cache_data(Path(d),spec,'sequence_parity',11,24)

    def test_cooperative_stop_inside_each_family(self):
        rng=np.random.default_rng(4);runtime.request_stop('test signal')
        for family in MODES:
            m=Model(ModelSpec(family,4));
            with self.assertRaises(runtime.WorkStopped):m.predict(inputs(rng,m.spec),np.zeros(m.tying.d))

    def test_supplied_optimizer_core_unchanged(self):
        expected='b166e4fe7a574b6221379c1fe3fcd51ed951e8642b40b78e48df5ae4383e8a52'
        for path in (ROOT/'isoma_ar.py',ROOT/'vendor/isoma_ar_original.py'):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)


if __name__=='__main__':unittest.main()
