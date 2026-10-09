"""Batched NumPy statevectors; qubit 0 is the least significant bit.

No full 2**n by 2**n gates, shots, Qiskit, or autodiff dependency are used.
Forward sensitivities implement the chain rule at EVERY shared gate occurrence.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .runtime import check

PAULI = {
    'x': np.array([[0, 1], [1, 0]], complex),
    'y': np.array([[0, -1j], [1j, 0]], complex),
    'z': np.array([[1, 0], [0, -1]], complex),
}
AXES = ('y', 'z', 'x')  # temporal order in each trainable block


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def rotation(axis, angle):
    a = np.asarray(angle)
    return (np.cos(a / 2)[..., None, None] * np.eye(2)
            - 1j * np.sin(a / 2)[..., None, None] * PAULI[axis])


def apply_one(state, matrix, qubit, n):
    shape = state.shape
    a = state.reshape(*shape[:-1], 2 ** (n - qubit - 1), 2, 2 ** qubit)
    out = np.empty_like(a)
    for i in (0, 1):
        out[..., i, :] = (matrix[..., i, 0][..., None, None] * a[..., 0, :]
                          + matrix[..., i, 1][..., None, None] * a[..., 1, :])
    return out.reshape(shape)


@dataclass(frozen=True)
class CircuitSpec:
    n: int = 2
    layers: int = 3  # L encoding opportunities and L+1 trainable blocks
    encoding: str = 'reupload'
    entangler: str = 'ring'
    readout: str = 'mean_z'
    batch_size: int = 64

    def __post_init__(self):
        if self.n < 2 or self.layers < 1 or self.batch_size < 1:
            raise ValueError('Use n>=2, layers>=1, batch_size>=1.')
        if self.encoding not in ('single', 'reupload', 'multi_axis'):
            raise ValueError(self.encoding)
        if self.entangler not in ('ring', 'line', 'none'):
            raise ValueError(self.entangler)
        if self.readout not in ('mean_z', 'z0', 'parity'):
            raise ValueError(self.readout)

    @property
    def p(self):
        return 3 * self.n * (self.layers + 1)

    @property
    def edges(self):
        if self.entangler == 'none':
            return []
        edges = [(q, q + 1) for q in range(self.n - 1)]
        if self.entangler == 'ring' and self.n > 2:
            edges.append((self.n - 1, 0))
        return edges

    @property
    def bandwidth(self):
        rounds = 1 if self.encoding == 'single' else self.layers
        if self.encoding == 'multi_axis':
            return (self.n * rounds, self.n * rounds)
        return (rounds * ((self.n + 1) // 2), rounds * (self.n // 2))

    @property
    def gate_counts(self):
        rounds=1 if self.encoding=='single' else self.layers
        data=self.n*rounds*(2 if self.encoding=='multi_axis' else 1)
        cz=(self.layers+1)*len(self.edges)
        return dict(trainable_rotations=self.p,data_rotations=data,cz_gates=cz,total=self.p+data+cz)


@dataclass
class Tying:
    ids: np.ndarray  # physical angle i = offsets[i] + phi[ids[i]]; -1 is frozen
    offsets: np.ndarray
    name: str

    @property
    def d(self):
        return int(self.ids.max()) + 1

    @property
    def multiplicity(self):
        return np.bincount(self.ids[self.ids >= 0], minlength=self.d)

    def expand(self, phi):
        if len(phi) != self.d:
            raise ValueError('Wrong reduced parameter dimension.')
        theta = self.offsets.copy()
        active = self.ids >= 0
        theta[active] += np.asarray(phi)[self.ids[active]]
        return theta

    def matrix(self):
        a = np.zeros((len(self.ids), self.d))
        active = np.flatnonzero(self.ids >= 0)
        a[active, self.ids[active]] = 1
        return a

    def to_dict(self):
        return dict(name=self.name, d=self.d, ids=self.ids.tolist(),
                    offsets=self.offsets.tolist(), multiplicity=self.multiplicity.tolist())


def make_tying(spec, name='untied', seed=0):
    w, n = spec.layers + 1, spec.n
    if name.startswith('block:'):
        k = int(name.split(':')[1])
        if k < 1:
            raise ValueError('Block size must be positive.')
    keys = []
    for l in range(w):
        for q in range(n):
            for a in range(3):
                if name == 'untied': key = (l, q, a)
                elif name in ('depth', 'random_depth', 'masked_depth'): key = (q, a)
                elif name in ('space', 'random_space'): key = (l, a)
                elif name == 'global_axis': key = (a,)
                elif name.startswith('block:'): key = (l // k, q, a)
                else: raise ValueError('Unknown tying: ' + name)
                keys.append(key)
    lookup = {}
    ids = np.array([lookup.setdefault(k, len(lookup)) for k in keys], int)
    offsets = np.zeros(spec.p)
    rng = np.random.default_rng(seed)
    if name.startswith('random_'):
        # Preserve both axis and the exact structured group's size distribution.
        for a in range(3):
            ix = np.arange(a, spec.p, 3)
            ids[ix] = rng.permutation(ids[ix])
    if name == 'masked_depth':
        offsets = rng.uniform(-np.pi, np.pi, spec.p)
        ids[:] = -1
        for a in range(3):
            ix = rng.choice(np.arange(a, spec.p, 3), n, replace=False)
            ids[ix] = np.arange(n) * 3 + a
            offsets[ix] = 0
    return Tying(ids, offsets, name)


class Circuit:
    def __init__(self, spec, tying):
        self.spec, self.tying = spec, tying
        if len(tying.ids) != spec.p:
            raise ValueError('Tying does not match the circuit.')
        ix = np.arange(2 ** spec.n)
        signs = np.array([1 - 2 * ((ix >> q) & 1) for q in range(spec.n)])
        self.observable = (signs.mean(0) if spec.readout == 'mean_z' else
                           signs[0] if spec.readout == 'z0' else signs.prod(0))
        self.cz_phase = np.ones(len(ix))
        for a, b in spec.edges:
            self.cz_phase *= 1 - 2 * (((ix >> a) & 1) * ((ix >> b) & 1))

    def _batch(self, x, phi, jac=False, return_state=False):
        n, d = self.spec.n, self.tying.d
        psi = np.zeros((len(x), 2 ** n), complex)
        psi[:, 0] = 1
        tangent = np.zeros((d, *psi.shape), complex) if jac else None
        theta = self.tying.expand(phi)
        for l in range(self.spec.layers + 1):
            if l > 0 and (self.spec.encoding != 'single' or l == 1):
                for q in range(n):
                    enc = [('x', q % 2)] if self.spec.encoding != 'multi_axis' else [('x', 0), ('y', 1)]
                    for axis, feature in enc:
                        gate = rotation(axis, x[:, feature])
                        psi = apply_one(psi, gate, q, n)
                        if jac: tangent = apply_one(tangent, gate, q, n)
            for q in range(n):
                for a, axis in enumerate(AXES):
                    i = (l * n + q) * 3 + a
                    gate = rotation(axis, theta[i])
                    if jac:
                        tangent = apply_one(tangent, gate, q, n)
                        g = self.tying.ids[i]
                        if g >= 0:
                            derivative = -0.5j * PAULI[axis] @ gate
                            tangent[g] += apply_one(psi, derivative, q, n)
                    psi = apply_one(psi, gate, q, n)
            psi *= self.cz_phase
            if jac: tangent *= self.cz_phase
        if return_state: return psi
        value = np.sum(np.abs(psi) ** 2 * self.observable, axis=1).real
        if jac:
            j = 2 * np.real(np.sum(psi.conj()[None] * tangent * self.observable, axis=-1)).T
            return value, j
        return value

    def predict(self, x, phi, jac=False, states=False):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('Input must have shape (N,2).')
        out = []
        for i in range(0, len(x), self.spec.batch_size):
            check()
            out.append(self._batch(x[i:i + self.spec.batch_size], phi, jac, states))
        if jac and not states:
            return np.concatenate([a[0] for a in out]), np.concatenate([a[1] for a in out])
        return np.concatenate(out)


def probability(f, epsilon=1e-6):
    # Smooth affine smoothing, rather than clipping: differentiation stays valid.
    return epsilon + (1 - 2 * epsilon) * (np.asarray(f) + 1) / 2


def loss(f, y, task='classification'):
    if task == 'infidelity': return float(np.mean(1 - np.asarray(f)))
    if task == 'regression': return float(np.mean((f - y) ** 2))
    p = probability(f)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def loss_and_grad(circuit, x, y, phi, task='classification'):
    f, j = circuit.predict(x, phi, jac=True)
    if task == 'regression': weights = 2 * (f - y) / len(y)
    elif task == 'infidelity': weights = -np.ones(len(y)) / len(y)
    else:
        p = probability(f)
        weights = 0.5 * (1 - 2e-6) * (p - y) / (p * (1 - p) * len(y))
    return loss(f, y, task), j.T @ weights
