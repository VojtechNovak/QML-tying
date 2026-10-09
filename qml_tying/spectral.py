"""Fourier coefficients on a Nyquist-safe two-dimensional angular torus."""
import numpy as np


def torus_grid(bandwidth, points=None):
    minimum = 2 * max(bandwidth) + 1
    k = minimum + 2 if points is None else int(points)
    if k < minimum: raise ValueError(f'Fourier grid must have at least {minimum} points per axis.')
    t = 2 * np.pi * np.arange(k) / k
    x = np.stack(np.meshgrid(t, t, indexing='ij'), axis=-1).reshape(-1, 2)
    frequencies = np.rint(np.fft.fftshift(np.fft.fftfreq(k) * k)).astype(int)
    return x, k, frequencies


def coefficients(values, k):
    return np.fft.fftshift(np.fft.fft2(np.asarray(values).reshape(k, k))) / k**2


def jacobian_summary(jac, multiplicity, rtol=1e-6):
    # Unit columns in physical-angle space remove a trivial sharing factor.
    normalized = jac / np.sqrt(multiplicity)[None, :]
    s = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(s > max(1e-12, rtol * s[0]))) if len(s) else 0
    power = s**2
    participation = float(power.sum()**2 / np.sum(power**2)) if power.sum() > 1e-24 else 0.
    condition = float(s[0] / s[rank-1]) if rank else None
    return dict(rank=rank, relative_tolerance=rtol, absolute_tolerance=1e-12,
                singular_values=s.tolist(), participation_ratio=participation, nonzero_condition=condition)


def swap_invariant(spec, tie):
    return (spec.n % 2 == 0 and spec.entangler == 'ring' and spec.readout == 'mean_z'
            and spec.encoding in ('single', 'reupload') and tie in ('space', 'global_axis'))


def analytic_target(name, x):
    if name == 'antisymmetric': return .45 * (np.sin(x[:, 0]) - np.sin(x[:, 1]))
    if name == 'symmetric': return .35 * (np.cos(x[:, 0]) + np.cos(x[:, 1]))
    if name == 'high_frequency': return .4 * np.sin(4*x[:, 0]) + .25 * np.cos(3*x[:, 1])
    raise ValueError(name)
