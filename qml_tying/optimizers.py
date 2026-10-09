"""Hard objective-call budgets, including initialization and finite differences.

SciPy local gradients are numerical and their calls are charged by the wrapper.
The attached JSO implementation is adapted as `jSO-derived`, NOT certified jSO.
Validation is observed at a fixed checkpoint schedule outside optimizer decisions.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from .core import loss, wrap
from .runtime import WorkStopped


class BudgetExceeded(Exception):
    pass


class Objective:
    def __init__(self, circuit, data, budget, task='classification', checkpoints=10):
        if budget < 1: raise ValueError('budget must be positive')
        self.circuit, self.data, self.budget, self.task = circuit, data, int(budget), task
        self.nfev, self.local_cap = 0, None
        self.best, self.best_phi = np.inf, None
        self.trace, self.candidates, self.terminations = [], [], []
        self.evaluation_seconds = []
        self.checkpoints = set(np.ceil(np.linspace(1, budget, checkpoints)).astype(int))
        self.started = time.perf_counter()

    @property
    def remaining(self): return self.budget - self.nfev

    def __call__(self, phi):
        if self.remaining <= 0 or (self.local_cap is not None and self.nfev >= self.local_cap):
            raise BudgetExceeded()
        phi = wrap(phi)
        started = time.perf_counter()
        value = loss(self.circuit.predict(self.data.x_train, phi), self.data.y_train, self.task)
        self.evaluation_seconds.append(time.perf_counter()-started)
        if not np.isfinite(value): raise FloatingPointError('Nonfinite objective.')
        self.nfev += 1
        if value < self.best: self.best, self.best_phi = value, phi.copy()
        self.trace.append((self.nfev, value, self.best))
        if self.nfev in self.checkpoints: self.validate()
        return value

    def validate(self):
        if self.candidates and self.candidates[-1]['nfev'] == self.nfev: return
        v = loss(self.circuit.predict(self.data.x_val, self.best_phi), self.data.y_val, self.task)
        self.candidates.append(dict(nfev=self.nfev, train_loss=self.best, val_loss=v,
                                    phi=self.best_phi.copy()))

    def partial(self):
        return dict(nfev=self.nfev,budget=self.budget,task=self.task,
                    best_train_loss=self.best if self.best_phi is not None else None,
                    train_best_phi=self.best_phi,trace=np.array(self.trace),
                    candidates=self.candidates,terminations=self.terminations,
                    wall_seconds=time.perf_counter()-self.started)

    def finish(self):
        self.validate()
        selected = min(self.candidates, key=lambda c: (c['val_loss'], c['nfev']))
        return dict(phi=selected['phi'], train_best_phi=self.best_phi.copy(),
                    best_train_loss=self.best, selected_train_loss=selected['train_loss'],
                    selected_val_loss=selected['val_loss'], selected_nfev=selected['nfev'],
                    nfev=self.nfev, budget=self.budget, trace=np.array(self.trace),
                    wall_seconds=time.perf_counter() - self.started,
                    training_statevectors=self.nfev * len(self.data.y_train),
                    validation_statevectors=len(self.candidates) * len(self.data.y_val),
                    candidates=self.candidates, terminations=self.terminations,
                    evaluation_seconds=np.array(self.evaluation_seconds))


def supplied_adaptive_de(f, d, rng, x0):
    """Supplied code's current-to-pbest/1, success history, and linear reduction.

    Fixes hard-budget initialization and best-vector aliasing. The supplied code
    lacks published jSO's Fw schedule, CR floors and last-memory-entry treatment.
    Boundary clipping and its arithmetic CR update are intentionally retained.
    """
    init = max(30, int(25 * np.sqrt(d) * np.log10(max(d, 1))))
    size = min(init, f.remaining)
    if size < 4:
        for _ in range(size): f(rng.uniform(-np.pi, np.pi, d))
        return
    pop = rng.uniform(-np.pi, np.pi, (size, d)); pop[0] = x0
    fit = np.array([f(p) for p in pop])
    initial = size; archive = []; mf = np.full(5, .5); mcr = np.full(5, .8); mem = 0
    while f.remaining:
        progress = f.nfev / f.budget
        size = max(4, int(round(initial + (4 - initial) * progress)))
        keep = np.argsort(fit)[:min(size, len(pop))]; pop, fit = pop[keep].copy(), fit[keep].copy()
        size = len(pop); pool = np.vstack([pop, archive]) if archive else pop.copy()
        top = np.argsort(fit)[:max(2, round((.25 - .125 * progress) * size))]
        ri = rng.integers(0, 5, size); ff = mf[ri] + .1 * rng.standard_cauchy(size)
        while np.any(ff <= 0):
            bad = ff <= 0; ff[bad] = mf[ri[bad]] + .1 * rng.standard_cauchy(bad.sum())
        ff = np.minimum(ff, 1); cr = np.clip(rng.normal(mcr[ri], .1), 0, 1)
        old = pop.copy(); trials = []
        for i in range(size):
            pb = rng.choice(top); r1 = rng.choice(np.delete(np.arange(size), i))
            r2 = rng.choice(np.setdiff1d(np.arange(len(pool)), [i, r1]))
            mutant = old[i] + ff[i] * (old[pb] - old[i]) + ff[i] * (old[r1] - pool[r2])
            cross = rng.random(d) < cr[i]; cross[rng.integers(d)] = True
            trials.append(np.clip(np.where(cross, mutant, old[i]), -np.pi, np.pi))
        sf, sc, improvements = [], [], []
        for i, trial in enumerate(trials):
            if not f.remaining: break
            value = f(trial)
            if value < fit[i]:
                archive.append(old[i].copy()); sf.append(ff[i]); sc.append(cr[i]); improvements.append(fit[i] - value)
            if value <= fit[i]: pop[i], fit[i] = trial, value
        if len(archive) > size:
            archive = [archive[i] for i in rng.choice(len(archive), size, replace=False)]
        if improvements:
            weights = np.asarray(improvements) / np.sum(improvements); sf = np.asarray(sf)
            mf[mem] = .5 * (mf[mem] + np.sum(weights * sf ** 2) / np.sum(weights * sf))
            mcr[mem] = .5 * (mcr[mem] + np.sum(weights * np.asarray(sc)))
            mem = (mem + 1) % 5


def optimize(circuit, data, method, budget=2000, seed=0, task='classification', x0=None,
             checkpoints=10, restart_chunk=None):
    rng = np.random.default_rng(seed); d = circuit.tying.d
    x0 = rng.uniform(-np.pi, np.pi, d) if x0 is None else np.array(x0, float)
    f = Objective(circuit, data, budget, task, checkpoints)
    try:
        f(x0)  # identical mandatory first candidate in every method
        if method in ('cobyla', 'bfgs', 'powell', 'ms_bfgs', 'ms_cobyla', 'ms_powell'):
            multistart = method.startswith('ms_'); base = method.removeprefix('ms_')
            scipy_method = dict(cobyla='COBYLA', bfgs='BFGS', powell='Powell')[base]
            chunk = restart_chunk or max(d + 2, budget // 4)
            first = True
            while f.remaining:
                start = x0 if first else rng.uniform(-np.pi, np.pi, d); first = False
                f.local_cap = min(budget, f.nfev + chunk) if multistart else budget
                try:
                    opts = dict(maxiter=max(1, f.local_cap - f.nfev))
                    if base == 'cobyla': opts.update(maxiter=max(d + 2, opts['maxiter']), rhobeg=.8, tol=1e-6)
                    if base == 'bfgs': opts.update(gtol=1e-6)
                    if base == 'powell': opts.update(maxfev=opts['maxiter'], xtol=1e-5, ftol=1e-8)
                    res = minimize(f, start, method=scipy_method, options=opts)
                    f.terminations.append(dict(success=bool(res.success), message=str(res.message), end_nfev=f.nfev))
                except BudgetExceeded:
                    f.terminations.append(dict(success=False, message='call budget reached', end_nfev=f.nfev))
                f.local_cap = None
                if not multistart: break
        elif method == 'de':
            # Explicit population prevents SciPy popsize*d from silently dominating small budgets.
            size = min(max(5, 5 * d), max(5, f.remaining))
            init = rng.uniform(-np.pi, np.pi, (size, d)); init[0] = x0
            differential_evolution(f, [(-np.pi, np.pi)] * d, init=init, seed=seed,
                                   maxiter=budget, tol=0, atol=0, polish=False, workers=1)
        elif method == 'dual_annealing':
            dual_annealing(f, [(-np.pi, np.pi)] * d, seed=seed, x0=x0,
                           maxfun=max(1, f.remaining), no_local_search=True)
        elif method == 'jSO-derived': supplied_adaptive_de(f, d, rng, x0)
        elif method == 'random':
            while f.remaining: f(rng.uniform(-np.pi, np.pi, d))
        elif method in ('iSOMA-AR', 'isoma'):
            from isoma_ar import isoma_ar
            isoma_ar(f, x0, bounds=[(-np.pi, np.pi)] * d, max_evals=f.remaining,
                     max_migrations=100000, seed=seed)
        else: raise ValueError('Unknown optimizer: ' + method)
    except BudgetExceeded:
        pass
    except WorkStopped as exc:
        # No further evaluations after the wall deadline. Preserve what was
        # observed; incomplete fits never become benchmark records.
        exc.partial_fit = f.partial()
        raise
    try:return f.finish()
    except WorkStopped as exc:
        exc.partial_fit=f.partial();raise
