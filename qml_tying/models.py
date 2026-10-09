"""Six small QML architectures, exact purified statevectors and shared-angle JVPs.

Bit zero is the least significant bit. Discarded subsystems are retained as a
purification and never touched again. No shot noise or full gate matrices.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from .core import Circuit, CircuitSpec, Tying, make_tying
from .runtime import check


@dataclass(frozen=True)
class ModelSpec:
    family: str = 'reupload'
    n: int = 4
    layers: int = 2
    encoding: str = 'reupload'
    latent: int = 2
    batch_size: int = 16

    def __post_init__(self):
        if self.family not in ('reupload', 'qcnn', 'autoencoder', 'recurrent', 'graph', 'su2'):
            raise ValueError(self.family)
        if not 2 <= self.n <= 8 or self.layers < 1 or self.batch_size < 1:
            raise ValueError('Use 2<=n<=8, layers>=1, batch_size>=1.')
        if self.family == 'qcnn' and self.n not in (2, 4, 8):
            raise ValueError('QCNN uses powers of two.')
        if self.family == 'autoencoder' and not 1 <= self.latent < self.n:
            raise ValueError('The latent register must be smaller than the input.')
        if self.family == 'su2' and self.n % 2:
            raise ValueError('The exchange circuit uses an even periodic chain.')


@lru_cache(maxsize=1024)
def pauli_action(n, axes, qubits):
    """Output-basis permutation and phases for a Pauli string."""
    ix = np.arange(1 << n); perm = ix.copy(); phase = np.ones(len(ix), complex)
    for axis, q in zip(axes, qubits):
        bit = (ix >> q) & 1
        if axis in 'xy': perm ^= 1 << q
        if axis == 'z': phase *= 1 - 2 * bit
        elif axis == 'y': phase *= 1j * (2 * bit - 1)
    return perm, phase


def pauli(state, n, axes, qubits):
    perm, phase = pauli_action(n, axes, tuple(qubits))
    return state[..., perm] * phase


@lru_cache(maxsize=256)
def fixed_action(n, kind, a, b):
    ix = np.arange(1 << n)
    if kind == 'cz': return None, 1 - 2 * (((ix >> a) & 1) * ((ix >> b) & 1))
    if kind == 'cx': return ix ^ ((((ix >> a) & 1) << b)), None
    if kind == 'swap':
        mask = ((ix >> a) & 1) ^ ((ix >> b) & 1)
        return ix ^ ((mask << a) | (mask << b)), None
    raise ValueError(kind)


def fixed(state, n, kind, a, b):
    perm, phase = fixed_action(n, kind, a, b)
    return state[..., perm] if perm is not None else state * phase


def operations(spec):
    """Return physical slots and their semantic keys; sharing changes no gates."""
    ops, keys, roles = [], [], []
    def rot(axes, qs, key, role, coef=1., feature=-1):
        slot = len(keys); keys.append(key); roles.append(role)
        ops.append(('rot', axes, tuple(qs), slot, coef, feature))
    n, L = spec.n, spec.layers
    if spec.family == 'qcnn':
        active = list(range(n)); scale = 0
        while len(active) > 1:
            pairs = [(active[j], active[j+1]) for j in range(0, len(active), 2)]
            if len(active) > 2:
                pairs += [(active[j], active[(j+1) % len(active)]) for j in range(1, len(active), 2)]
            for a, b in pairs:
                for k, (q, axis) in enumerate([(a,'y'),(a,'z'),(a,'x'),(b,'y'),(b,'z'),(b,'x')]):
                    rot(axis, (q,), ('conv',scale,k), ('conv',k))
                rot('zz', (a,b), ('conv',scale,6), ('conv',6))
            for j in range(0, len(active), 2):
                a,b = active[j:j+2]; ops.append(('cx', b, a))
                for k,axis in enumerate(('y','z')):
                    rot(axis, (a,), ('pool',scale,k), ('pool',k))
            active = active[::2]; scale += 1
    elif spec.family == 'recurrent':
        for t in range(n-1):
            ops.append(('input', 'y', t+1, t))
            for k, (q,axis) in enumerate([(0,'y'),(0,'z'),(0,'x'),(t+1,'y'),(t+1,'z'),(t+1,'x')]):
                rot(axis,(q,), (k,), (k%3,))
            ops.append(('cx', t+1, 0))
    elif spec.family == 'graph':
        edges = [(a,b) for a in range(n) for b in range(a+1,n)]
        for l in range(L):
            for q in range(n):
                for axis in ('y','z','x'): rot(axis,(q,), (l,'node',axis), ('node',axis))
            for j,(a,b) in enumerate(edges): rot('zz',(a,b), (l,'edge'), ('edge',),feature=j)
    elif spec.family == 'su2':
        edges = [(q,(q+1)%n) for q in range(0,n,2)] + [(q,(q+1)%n) for q in range(1,n,2)]
        if n == 2: edges = edges[:1]
        for l in range(L):
            for j,(a,b) in enumerate(edges):
                for axis in ('xx','yy','zz'): rot(axis,(a,b), (l,j), ('exchange',))
    elif spec.family == 'autoencoder':
        for l in range(L+1):
            for q in range(n):
                for axis in ('y','z','x'): rot(axis,(q,), (q,axis), ('encoder',axis))
            for q in range(n-1): ops.append(('cz',q,q+1))
            if n > 2: ops.append(('cz',n-1,0))
        encoder = list(ops); count = len(keys)
        # Independent decoder has the adjoint topology; inverse tying equates
        # corresponding encoder/decoder angles, including their negative sign.
        for op in reversed(encoder):
            if op[0] == 'rot':
                _,axis,qs,slot,coef,feature = op
                ops.append(('rot',axis,qs,count+slot,-coef,feature))
            else: ops.append(op)
        keys += list(keys); roles += [('decoder',r[1]) for r in roles]
    else: raise ValueError('The reupload family uses the reference engine.')
    return ops, keys, roles


def _number(keys):
    lookup = {}; return np.array([lookup.setdefault(k,len(lookup)) for k in keys],int)


def tying_for(spec, mode, seed=0):
    if spec.family == 'reupload':
        s = CircuitSpec(spec.n,spec.layers,spec.encoding,batch_size=spec.batch_size)
        return make_tying(s,mode,seed)
    ops, keys, roles = operations(spec); p = len(keys)
    if mode == 'untied': ids = np.arange(p)
    elif spec.family == 'autoencoder':
        half = p//2
        if mode == 'inverse': ids = np.tile(np.arange(half),2)
        elif mode in ('inverse_depth','inverse_random_depth'):
            base = _number(keys[:half])
            if mode == 'inverse_random_depth':
                rng = np.random.default_rng(seed)
                for role in sorted(set(roles[:half]),key=repr):
                    ix = np.array([i for i,r in enumerate(roles[:half]) if r==role])
                    base[ix] = rng.permutation(base[ix])
            ids = np.tile(base,2)
        else: raise ValueError(mode)
    else:
        allowed = {'qcnn':('spatial','random_spatial','scale_shared'),
                   'recurrent':('temporal','random_temporal','block_time'),
                   'graph':('node_edge','random_node_edge','layer_shared'),
                   'su2':('su2','random_su2','su2_spatial')}[spec.family]
        if mode not in allowed: raise ValueError(mode)
        if mode == 'scale_shared': keys = roles
        elif mode == 'block_time': keys = [(i//6//2,i%6) for i in range(p)]
        elif mode == 'layer_shared': keys = [k[1:] for k in keys]
        elif mode == 'su2_spatial': keys = [(k[0],) for k in keys]
        ids = _number(keys)
        if mode.startswith('random_'):
            rng = np.random.default_rng(seed)
            for role in sorted(set(roles),key=repr):
                ix = np.array([i for i,r in enumerate(roles) if r==role])
                ids[ix] = rng.permutation(ids[ix])
    return Tying(ids,np.zeros(p),mode)


class Model:
    def __init__(self, spec, mode='untied', map_seed=0):
        self.spec = spec; self.tying = tying_for(spec,mode,map_seed)
        self.ops = operations(spec)[0] if spec.family != 'reupload' else []
        self.reference = (Circuit(CircuitSpec(spec.n,spec.layers,spec.encoding,batch_size=spec.batch_size),self.tying)
                          if spec.family == 'reupload' else None)
        self.p = len(self.tying.ids); self.task = 'infidelity' if spec.family=='autoencoder' else 'classification'
        ix = np.arange(1<<spec.n)
        self.signs = np.array([1-2*((ix>>q)&1) for q in range(spec.n)])

    def metadata(self):
        stages=(int(np.log2(self.spec.n)) if self.spec.family=='qcnn' else self.spec.n-1 if self.spec.family=='recurrent'
                else self.spec.layers+1 if self.spec.family in ('reupload','autoencoder') else self.spec.layers)
        return dict(spec=asdict(self.spec),physical_parameters=self.p,free_parameters=self.tying.d,
                    effective_stages=stages,stage_role='encoder blocks' if self.spec.family=='autoencoder' else 'scales' if self.spec.family=='qcnn' else 'time steps' if self.spec.family=='recurrent' else 'trainable blocks',
                    tying=self.tying.to_dict(),gate_occurrences=len(self.ops) if self.ops else self.reference.spec.gate_counts['total'])

    def evolve(self, psi, phi, ops=None, features=None, jac=False, tangent=None):
        """Forward-mode derivatives, adding every occurrence of each shared slot."""
        theta = self.tying.expand(phi); d = self.tying.d; n = self.spec.n
        if jac and tangent is None: tangent = np.zeros((d,*psi.shape),complex)
        for op in self.ops if ops is None else ops:
            if op[0] in ('cz','cx'):
                psi = fixed(psi,n,*op)
                if jac: tangent = fixed(tangent,n,*op)
                continue
            if op[0]=='input':
                _,axes,q,j = op; qs=(q,); angle=features[:,j]; group=-1; gain=0.
            else:
                _,axes,qs,slot,coef,feature=op; group=self.tying.ids[slot]
                gain = coef if feature<0 else coef*features[:,feature]
                angle = theta[slot]*gain
            c = np.cos(np.asarray(angle)/2)[...,None]; s = np.sin(np.asarray(angle)/2)[...,None]
            psi = c*psi - 1j*s*pauli(psi,n,axes,qs)
            if jac:
                tangent = c*tangent - 1j*s*pauli(tangent,n,axes,qs)
                if group>=0: tangent[group] += -.5j*np.asarray(gain)[...,None]*pauli(psi,n,axes,qs)
        return (psi,tangent) if jac else psi

    def inverse_evolve(self, psi, phi, ops=None):
        reverse = []
        for op in reversed(self.ops if ops is None else ops):
            if op[0]=='rot': reverse.append((*op[:4],-op[4],op[5]))
            else: reverse.append(op)
        return self.evolve(psi,phi,reverse)

    def _batch(self, x, phi, jac=False, states=False):
        n,d = self.spec.n,self.tying.d; family=self.spec.family
        if family in ('qcnn','autoencoder','su2'): psi=np.array(x,complex,copy=True)
        elif family=='recurrent':
            psi=np.zeros((len(x),1<<n),complex);psi[:,0]=1
        elif family=='graph':
            psi=np.ones((len(x),1<<n),complex)/np.sqrt(1<<n)
            for j,(a,b) in enumerate((a,b) for a in range(n) for b in range(a+1,n)):
                _,phase=fixed_action(n,'cz',a,b);psi *= np.where(x[:,j,None]>0.5,phase,1)
        else: raise ValueError(family)
        if family=='autoencoder':
            cut=len(self.ops)//2
            if jac: enc,dt=self.evolve(psi,phi,self.ops[:cut],jac=True)
            else: enc=self.evolve(psi,phi,self.ops[:cut])
            k=self.spec.latent; latent=1<<k; trash=1<<(n-k)
            branches=np.zeros((len(x),trash,1<<n),complex)
            branches[:,:,:latent]=enc.reshape(len(x),trash,latent)
            db=None
            if jac:
                db=np.zeros((d,len(x),trash,1<<n),complex)
                db[:,:,:,:latent]=dt.reshape(d,len(x),trash,latent)
            if jac: decoded,der=self.evolve(branches,phi,self.ops[cut:],jac=True,tangent=db)
            else: decoded=self.evolve(branches,phi,self.ops[cut:])
            amplitude=np.sum(psi.conj()[:,None,:]*decoded,axis=-1)
            fidelity=np.sum(abs(amplitude)**2,axis=-1).real
            if states: return decoded
            if jac:
                da=np.sum(psi.conj()[None,:,None,:]*der,axis=-1)
                j=2*np.real(np.sum(amplitude.conj()[None]*da,axis=-1)).T
                return fidelity,j
            return fidelity
        if jac: psi,dt=self.evolve(psi,phi,features=x,jac=True)
        else: psi=self.evolve(psi,phi,features=x)
        if states: return psi
        if family=='su2':
            observed=-fixed(psi,n,'swap',0,1)
            value=np.real(np.sum(psi.conj()*observed,axis=-1))
            if jac: j=2*np.real(np.sum(dt.conj()*observed[None],axis=-1)).T
        else:
            obs=self.signs.mean(0) if family=='graph' else self.signs[0]
            if family=='recurrent': obs=-obs
            value=np.sum(abs(psi)**2*obs,axis=-1).real
            if jac: j=2*np.real(np.sum(psi.conj()[None]*dt*obs,axis=-1)).T
        return (value,j) if jac else value

    def predict(self, x, phi, jac=False, states=False):
        if self.reference: return self.reference.predict(x,phi,jac,states)
        x=np.asarray(x); out=[]
        if x.ndim!=2 or len(x)==0: raise ValueError('Nonempty two-dimensional input required.')
        width=(1<<self.spec.n) if self.spec.family in ('qcnn','autoencoder','su2') else (self.spec.n-1 if self.spec.family=='recurrent' else self.spec.n*(self.spec.n-1)//2)
        if x.shape[1]!=width: raise ValueError(f'Expected input width {width}, got {x.shape[1]}.')
        for i in range(0,len(x),self.spec.batch_size):
            check();out.append(self._batch(x[i:i+self.spec.batch_size],phi,jac,states))
        if jac and not states: return np.concatenate([v[0] for v in out]),np.concatenate([v[1] for v in out])
        return np.concatenate(out)


def collective_rotate(psi,n,angles):
    """A common Euler SU(2) action on every qubit, batched angles supported."""
    result=np.array(psi,complex,copy=True)
    for axis,angle in zip(('z','y','z'),np.asarray(angles).T):
        c=np.cos(angle/2)[...,None];s=np.sin(angle/2)[...,None]
        for q in range(n): result=c*result-1j*s*pauli(result,n,axis,(q,))
    return result


def relabel_state(psi,n,permutation):
    """P|bits> where old qubit q is moved to permutation[q]."""
    ix=np.arange(1<<n);dest=np.zeros_like(ix)
    for q,r in enumerate(permutation): dest |= ((ix>>q)&1)<<r
    out=np.empty_like(psi);out[...,dest]=psi;return out
