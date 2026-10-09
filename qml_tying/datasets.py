"""Auditable small QML tasks, paired splits, attainable teachers and orbit splits."""
from dataclasses import dataclass
from itertools import permutations
from functools import lru_cache
from pathlib import Path
import json
import hashlib
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from .models import Model, ModelSpec, pauli, collective_rotate
from .core import loss
from .io import record_id, write_json
from .runtime import check


@dataclass
class Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict
    extras: dict


def haar(rng, count, dimension):
    z=rng.normal(size=(count,dimension))+1j*rng.normal(size=(count,dimension))
    return z/np.linalg.norm(z,axis=1,keepdims=True)


def product_states(rng,count,n):
    state=np.ones((count,1),complex)
    for _ in range(n):
        q=haar(rng,count,2)
        state=(q[:,:,None]*state[:,None,:]).reshape(count,-1)
    return state


def _split(x,y,seed,metadata,extras=None,classification=True):
    ix=np.arange(len(y));strat=y if classification else None
    train,hold=train_test_split(ix,test_size=.4,random_state=seed,stratify=strat)
    val,test=train_test_split(hold,test_size=.5,random_state=seed+1,stratify=y[hold] if classification else None)
    ex=dict(extras or {});ex.update(index_train=train,index_val=val,index_test=test)
    return Data(x[train],y[train],x[val],y[val],x[test],y[test],metadata,ex)


@lru_cache(maxsize=12)
def phase_hamiltonians(n):
    # Small exact diagonalization generates quantum data, not a VQE task.
    eye=np.eye(1<<n,dtype=complex);h0=np.zeros_like(eye);h1=np.zeros_like(eye);h2=np.zeros_like(eye)
    for q in range(n):
        h0-=pauli(eye,n,'zxz',((q-1)%n,q,(q+1)%n)).T
        h1-=pauli(eye,n,'x',(q,)).T
        h2-=pauli(eye,n,'xx',(q,(q+1)%n)).T
    return h0,h1,h2


@lru_cache(maxsize=6)
def graph_permutation_indices(n):
    edges=[(a,b) for a in range(n) for b in range(a+1,n)];lookup={e:j for j,e in enumerate(edges)}
    result=[]
    for p in permutations(range(n)):
        result.append([lookup[tuple(sorted((p[a],p[b])))] for a,b in edges])
    return np.asarray(result,int)


def graph_relabel(x,n,p):
    edges=[(a,b) for a in range(n) for b in range(a+1,n)];lookup={e:j for j,e in enumerate(edges)}
    dest=[lookup[tuple(sorted((p[a],p[b])))] for a,b in edges]
    out=np.empty_like(x);out[...,dest]=x;return out


def graph_canonical(x,n):
    # At n<=7 every permutation is affordable, with no heuristic isomorphism key.
    index=graph_permutation_indices(n);bits=x[index].astype(np.uint64)
    weights=np.left_shift(np.uint64(1),np.arange(bits.shape[1],dtype=np.uint64))
    return int(np.min(bits@weights))


def graph_connected(x,n):
    adj=[set() for _ in range(n)]
    for v,(a,b) in zip(x,((a,b) for a in range(n) for b in range(a+1,n))):
        if v: adj[a].add(b);adj[b].add(a)
    seen={0};todo=[0]
    while todo:
        a=todo.pop()
        for b in adj[a]-seen: seen.add(b);todo.append(b)
    return len(seen)==n


def graph_data(spec,dataset,seed,count,teacher_seed):
    n=spec.n
    if n>7: raise ValueError('Exact graph orbit splits are restricted to n<=7.')
    rng=np.random.default_rng(seed+83001);wanted=count//2;xs=[];ys=[];ids=[];seen=set();counts=[0,0]
    for _ in range(200000):
        check();p=rng.uniform(.12,.8);x=(rng.random(n*(n-1)//2)<p).astype(float)
        label=int(graph_connected(x,n));key=graph_canonical(x,n)
        if key in seen or counts[label]>=wanted: continue
        seen.add(key);counts[label]+=1;xs.append(x);ys.append(label);ids.append(key)
        if min(counts)>=wanted: break
    else: raise RuntimeError('Too many requested distinct graph orbits; reduce n_samples.')
    x=np.asarray(xs);y=np.asarray(ys);ix=np.arange(len(y))
    train,hold=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
    val,test=train_test_split(hold,test_size=.5,random_state=seed+1,stratify=y[hold])
    arrays={};extras={};label_mode=dataset=='graph_edge_mismatch'
    for split,indices in [('train',train),('val',val),('test',test)]:
        vx=[];vy=[];vg=[];vp=[]
        for i in indices:
            # All augmented views of an isomorphism class stay in one partition.
            for _ in range(3):
                perm=rng.permutation(n);v=graph_relabel(x[i],n,perm)
                vx.append(v);vy.append(int(v[0]) if label_mode else y[i]);vg.append(ids[i]);vp.append(perm)
        arrays['x_'+split]=np.asarray(vx);arrays['y_'+split]=np.asarray(vy)
        extras['orbit_'+split]=np.asarray(vg);extras['permutation_'+split]=np.asarray(vp)
    metadata=dict(dataset=dataset,seed=seed,n_base_orbits=len(y),augmentation_views=3,
                  split_rule='stratified whole isomorphism classes; exact canonicalization',
                  teacher_seed=teacher_seed,symmetry_labels='broken edge(0,1)' if label_mode else 'permutation-invariant connectivity')
    bounds={}
    for split in ('train','val','test'):
        labels=arrays['y_'+split];orbits=extras['orbit_'+split];ce=0.;accuracy=0.;balanced=0.;n0=np.sum(labels==0);n1=np.sum(labels==1)
        for orbit in np.unique(orbits):
            ys=labels[orbits==orbit];p=float(ys.mean());q=np.clip(p,1e-6,1-1e-6)
            ce+=len(ys)*(-p*np.log(q)-(1-p)*np.log1p(-q))/len(labels)
            c0=np.sum(ys==0);c1=len(ys)-c0;accuracy+=max(c0,c1)/len(labels)
            if n0 and n1:balanced+=.5*max(c0/n0,c1/n1)
        bounds[split]=dict(cross_entropy_floor=float(ce),accuracy_upper_bound=float(accuracy),balanced_accuracy_upper_bound=float(balanced) if n0 and n1 else None)
    metadata['invariant_orbit_bounds']=bounds
    return Data(**arrays,metadata=metadata,extras=extras)


def generate(spec,dataset,seed,count=120,teacher_seed=77131):
    rng=np.random.default_rng(seed+12317);n=spec.n
    metadata=dict(dataset=dataset,seed=seed,n_samples=count,teacher_seed=teacher_seed,
                  split_rule='60/20/20 stratified for classification; random for reconstruction')
    if spec.family=='graph': return graph_data(spec,dataset,seed,count,teacher_seed)
    if dataset=='moons':
        x,y=make_moons(n_samples=count,noise=.18,random_state=seed+71);x=np.asarray(x)*np.pi/2
        return _split(x,y,seed,metadata)
    if dataset=='sequence_parity':
        bits=rng.integers(0,2,(count,n-1));x=np.pi*bits+rng.normal(0,.10,bits.shape);y=bits.sum(1)%2
        metadata['motivation']='stationary parity update with one-qubit recurrent memory'
        return _split(x,y,seed,metadata,{'bits':bits})
    if dataset=='sequence_teacher':
        teacher=Model(spec,'temporal');truth=np.random.default_rng(teacher_seed).uniform(-.7,.7,6)
        x=rng.uniform(-np.pi,np.pi,(count,n-1));f=teacher.predict(x,truth);y=(f>=0).astype(int)
        metadata.update(motivation='learn a nonzero stationary quantum-memory rule',
                        attainable_teacher_balanced_accuracy=1.,attainable_teacher_log_loss=loss(f,y),
                        certificate_not_given_to_optimizer=True)
        return _split(x,y,seed,metadata,{'teacher_phi':truth,'teacher_expectation':f})
    if dataset=='phase_regime':
        y=np.arange(count)%2;h1=np.where(y==0,rng.uniform(.15,.65,count),rng.uniform(1.5,2.1,count));h2=rng.uniform(-.10,.10,count)
        h0,hx,hxx=phase_hamiltonians(n);x=[];gaps=[]
        for a,b in zip(h1,h2):
            check();values,vectors=np.linalg.eigh(h0+a*hx+b*hxx);x.append(vectors[:,0]);gaps.append(values[1]-values[0])
        metadata['label_definition']='two separated finite-size cluster/field regimes; not a phase-boundary estimate'
        return _split(np.asarray(x),y,seed,metadata,{'coupling_h1':h1,'coupling_h2':h2,'spectral_gap':np.asarray(gaps)})
    if dataset in ('compress_depth','compress_untied'):
        teacher_mode='inverse_depth' if dataset=='compress_depth' else 'inverse'
        teacher=Model(spec,teacher_mode);truth=np.random.default_rng(teacher_seed).uniform(-1.5,1.5,teacher.tying.d)
        target=np.zeros((count,1<<n),complex);target[:,:1<<spec.latent]=haar(rng,count,1<<spec.latent)
        encoder=teacher.ops[:len(teacher.ops)//2];x=teacher.inverse_evolve(target,truth,encoder);y=np.ones(count)
        fidelity=teacher.predict(x,truth)
        metadata.update(motivation='compress a known low-dimensional subspace; decode with the encoder inverse',
                        input_subspace_dimension=1<<spec.latent,teacher_mode=teacher_mode,
                        attainable_mean_infidelity=float(np.mean(1-fidelity)),certificate_not_given_to_optimizer=True)
        return _split(x,y,seed,metadata,{'teacher_phi':truth},classification=False)
    if dataset=='singlet_teacher':
        teacher=Model(spec,'su2');truth=np.random.default_rng(teacher_seed).uniform(-1.2,1.2,teacher.tying.d)
        y=np.arange(count)%2;rest=product_states(rng,count,n-2);base=np.zeros((count,1<<n),complex)
        pair=np.zeros((count,4),complex);pair[y==0,0]=1
        pair[y==1,1]=1/np.sqrt(2);pair[y==1,2]=-1/np.sqrt(2)
        base=(rest[:,:,None]*pair[:,None,:]).reshape(count,-1)
        x=teacher.inverse_evolve(base,truth)
        # Common rotations are nuisance transformations; the singlet readout
        # and every isotropic exchange gate commute with them.
        angles=np.c_[rng.uniform(-np.pi,np.pi,count),np.arccos(rng.uniform(-1,1,count)),rng.uniform(-np.pi,np.pi,count)]
        x=collective_rotate(x,n,angles);f=teacher.predict(x,truth)
        metadata.update(motivation='learn a singlet feature independently of the collective spin reference frame',
                        attainable_teacher_train_loss=loss(f,y),certificate_not_given_to_optimizer=True)
        return _split(x,y,seed,metadata,{'teacher_phi':truth,'global_rotation_angles':angles})
    if dataset=='spin_orientation_mismatch':
        angles=np.c_[rng.uniform(-np.pi,np.pi,count),np.arccos(rng.uniform(-1,1,count)),np.zeros(count)]
        base=np.zeros((count,1<<n),complex);base[:,0]=1;x=collective_rotate(base,n,angles)
        nx=np.sin(angles[:,1])*np.cos(angles[:,0]);nz=np.cos(angles[:,1]);y=(nz*nz>nx*nx).astype(int)
        metadata.update(motivation='a frame-dependent label deliberately contradicts SU(2) invariance',
                        invariant_readout_constant=True,invariant_probability=0.)
        return _split(x,y,seed,metadata,{'global_rotation_angles':angles})
    raise ValueError(dataset)


def cache_data(root,spec,dataset,seed,count,teacher_seed=77131):
    key=record_id('data-v3',asdict_safe(spec),dataset,seed,count,teacher_seed)
    path=Path(root)/'datasets'/key;marker=path.with_suffix('.json');payload=path.with_suffix('.npz')
    if marker.exists() and payload.exists():
        metadata=json.loads(marker.read_text())
        if hashlib.sha256(payload.read_bytes()).hexdigest()!=metadata.get('payload_sha256'):
            raise IOError('Cached dataset payload fails its recorded SHA256: '+str(payload))
        z=np.load(payload,allow_pickle=False)
        arrays={k:z[k] for k in ('x_train','y_train','x_val','y_val','x_test','y_test')}
        data=Data(**arrays,metadata=metadata,extras={k:z[k] for k in z.files if k not in arrays})
    else:
        data=generate(spec,dataset,seed,count,teacher_seed);marker.parent.mkdir(parents=True,exist_ok=True)
        arrays={k:getattr(data,k) for k in ('x_train','y_train','x_val','y_val','x_test','y_test')};arrays.update(data.extras)
        temporary=payload.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
        temporary.replace(payload);data.metadata.update(data_id=key,spec=asdict_safe(spec),payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest());write_json(marker,data.metadata)
    return data,key


def asdict_safe(spec):
    from dataclasses import asdict
    return asdict(spec)


def classical_features(data,spec):
    """Fixed measurements for quantum data; train-only PCA to two MLP inputs."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    def features(x):
        if spec.family not in ('qcnn','su2','autoencoder'): return np.asarray(x,float)
        cols=[]
        for q in range(spec.n):
            for a in 'xyz': cols.append(np.real(np.sum(x.conj()*pauli(x,spec.n,a,(q,)),axis=1)))
        for q in range(spec.n-1):
            for a in ('xx','yy','zz'): cols.append(np.real(np.sum(x.conj()*pauli(x,spec.n,a,(q,q+1)),axis=1)))
        return np.array(cols).T
    raw=[features(getattr(data,'x_'+s)) for s in ('train','val','test')]
    scaler=StandardScaler().fit(raw[0]);scaled=[scaler.transform(v) for v in raw]
    pca=None
    if scaled[0].shape[1]>2:
        pca=PCA(n_components=2,svd_solver='full').fit(scaled[0]);scaled=[pca.transform(v) for v in scaled]
    metadata=dict(input_representation='fixed local Pauli and adjacent correlator measurements' if spec.family in ('qcnn','su2') else 'classical task features',
                  preprocessing='training-only standardization and PCA to two components',
                  scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                  pca_components=pca.components_.tolist() if pca else None,
                  information_access_matched=False)
    arrays={}
    for s,v in zip(('train','val','test'),scaled): arrays['x_'+s]=v;arrays['y_'+s]=getattr(data,'y_'+s)
    return Data(**arrays,metadata=metadata,extras={})
