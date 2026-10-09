#!/usr/bin/env python3
"""
Fast post-hoc landscape characterization for the three VQE benchmark families.

IMPORTANT
---------
This script NEVER calls an optimizer.  It only:
  * reloads the exact disorder instances already used by the benchmarks,
  * reloads archived optimizer endpoints where available,
  * evaluates objectives/gradients/Hessians at selected points,
  * enumerates the small classical diagonal state spaces (N <= 16).

Default benchmark coverage (read from the supplied benchmark design):
  Q1 diagonal Ising spin glass:       N = 12, 14, 16
  Q2 transverse-field spin glass:     N = 10, 12, Gamma = 0.3
  Q3 Heisenberg-glass VQE:            N = 8, 10, 12, lambda = 0.5

The default wall-time guard is 540 s (9 min), leaving margin below the requested
10-minute ceiling.  If the deadline is reached, already-computed tables are
saved and the manifest records that the run is partial.

Outputs are plain CSV + JSON (and no pickles), designed for easy later analysis.

Example
-------
python landscape_characterization_fast.py \
    --q1-results spin_glass_benchmark_results \
    --q2-results tf_spin_glass_benchmark_results \
    --q3-results heisenberg_variational_benchmark_results \
    --adapter heisenberg_model_adapter.py
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# -----------------------------------------------------------------------------
# Defaults: intentionally modest so all conditions fit comfortably in < 10 min.
# -----------------------------------------------------------------------------
Q1_SIZES = (12, 14, 16)
Q2_SIZES = (10, 12)
Q3_SIZES = (8, 10, 12)
GAMMA = 0.3

DEFAULT_TIME_LIMIT_S = 540.0
DEFAULT_ANALYSIS_SEED = 20260823
DEFAULT_GRAD_SAMPLES = 32
DEFAULT_GLOBAL_SAMPLES = 64
DEFAULT_HESSIAN_MINIMA = 2

# Endpoint clustering is only an empirical basin proxy for Q2/Q3.
ENDPOINT_ENERGY_ABS_TOL = 1e-5
ENDPOINT_ENERGY_REL_TOL = 1e-4
ENDPOINT_TORUS_RMS_TOL = 0.50  # radians

# Numerical thresholds used only to summarize spectra/couplings.
HESSIAN_ACTIVE_REL_TOL = 1e-6
HESSIAN_ACTIVE_ABS_TOL = 1e-8
COUPLING_REL_TOL = 1e-3
COUPLING_ABS_TOL = 1e-8
DISCRETE_TOL = 1e-12


class DeadlineReached(RuntimeError):
    pass


class TimeGuard:
    def __init__(self, seconds: float):
        self.start = time.perf_counter()
        self.seconds = float(seconds)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        return self.seconds - self.elapsed

    def check(self, label: str = "analysis") -> None:
        if self.remaining <= 0:
            raise DeadlineReached(
                f"Wall-time guard reached during {label} after {self.elapsed:.1f} s"
            )


@dataclass
class Condition:
    model: str
    n: int
    dim: int
    energy: Callable[[np.ndarray], float]
    h_diag: np.ndarray
    instance_seed: Optional[int]
    fingerprint: str
    metadata: Dict[str, Any]
    analytic_gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None
    analytic_hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None
    anchor_theta: Optional[np.ndarray] = None
    anchor_source: str = ""
    endpoints: Optional[List[Dict[str, Any]]] = None


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def wrap(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def torus_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def torus_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = torus_delta(a, b)
    return float(np.sqrt(np.mean(d * d)))


def torus_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(torus_delta(a, b)))


def jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return str(x)


def stable_condition_seed(base: int, model: str, n: int, salt: int = 0) -> int:
    model_code = {"Q1": 101, "Q2": 202, "Q3": 303}[model]
    return int((base + 1009 * model_code + 9176 * n + 7919 * salt) % (2**32 - 1))


def bit_matrix(n: int) -> np.ndarray:
    states = np.arange(2**n, dtype=np.uint32)
    return np.stack(
        [((states >> (n - 1 - q)) & 1).astype(np.int8) for q in range(n)],
        axis=1,
    )


def basis_hamming_summary(h_diag: np.ndarray, n: int) -> pd.DataFrame:
    h_diag = np.asarray(h_diag, dtype=float)
    states = np.arange(2**n, dtype=np.uint32)
    ground = int(np.argmin(h_diag))
    xor = states ^ np.uint32(ground)
    # N <= 16, so a byte lookup is simple and fast.
    pc = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    dist = pc[(xor & 255).astype(np.uint8)].astype(np.int16)
    if n > 8:
        dist += pc[((xor >> 8) & 255).astype(np.uint8)].astype(np.int16)
    rows = []
    e0 = float(h_diag[ground])
    for d in range(n + 1):
        vals = h_diag[dist == d]
        if vals.size == 0:
            continue
        rows.append(
            {
                "hamming_distance": d,
                "count": int(vals.size),
                "energy_mean": float(np.mean(vals)),
                "energy_std": float(np.std(vals)),
                "energy_min": float(np.min(vals)),
                "energy_max": float(np.max(vals)),
                "delta_from_diagonal_ground_mean": float(np.mean(vals - e0)),
                "delta_from_diagonal_ground_min": float(np.min(vals - e0)),
            }
        )
    return pd.DataFrame(rows)


def entropy_stats(counts: Sequence[int]) -> Tuple[float, float, float]:
    c = np.asarray(counts, dtype=float)
    if c.size == 0 or np.sum(c) <= 0:
        return np.nan, np.nan, np.nan
    p = c / np.sum(c)
    p = p[p > 0]
    h = float(-np.sum(p * np.log(p)))
    hnorm = float(h / np.log(len(c))) if len(c) > 1 else 0.0
    eff = float(np.exp(h))
    return h, hnorm, eff


# -----------------------------------------------------------------------------
# Q1: exact diagonal instance and analytic objective derivatives.
# -----------------------------------------------------------------------------
def q1_instance_fingerprint(edges: np.ndarray, J: np.ndarray, h: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(edges, dtype=np.int64).tobytes())
    digest.update(np.asarray(J, dtype=np.float64).tobytes())
    digest.update(np.asarray(h, dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_q1_instance(root: Path, n: int) -> Dict[str, Any]:
    npz_path = root / "instances" / f"N{n}_instance.npz"
    json_path = root / "instances" / f"N{n}_metadata.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing saved Q1 instance for N={n}: expected {npz_path} and {json_path}. "
            "This script will not generate a replacement instance."
        )
    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(json_path.read_text())
    J = np.asarray(data["J"], dtype=float)
    h = np.asarray(data["h"], dtype=float)
    edges = np.asarray(data["edges"], dtype=int)
    h_diag = np.asarray(data["H_diag"], dtype=float)
    if h_diag.shape != (2**n,):
        raise ValueError(f"Malformed Q1 H_diag for N={n}: {h_diag.shape}")
    return {
        "J": J,
        "h": h,
        "edges": edges,
        "H_diag": h_diag,
        "metadata": meta,
        "instance_seed": int(meta.get("instance_seed", -1)),
        "fingerprint": q1_instance_fingerprint(edges, J, h),
    }


class Q1AnalyticObjective:
    def __init__(self, J: np.ndarray, h: np.ndarray, edges: np.ndarray):
        self.J = np.asarray(J, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.edges = np.asarray(edges, dtype=int)
        self.n = len(h)

    def energy(self, theta: np.ndarray) -> float:
        c = np.cos(wrap(theta))
        e = float(np.dot(self.h, c))
        for k, (i, j) in enumerate(self.edges):
            e += float(self.J[k] * c[i] * c[j])
        return e

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        t = wrap(theta)
        c = np.cos(t)
        s = np.sin(t)
        local = self.h.copy()
        for k, (i, j) in enumerate(self.edges):
            local[i] += self.J[k] * c[j]
            local[j] += self.J[k] * c[i]
        return -s * local

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        t = wrap(theta)
        c = np.cos(t)
        s = np.sin(t)
        H = np.zeros((self.n, self.n), dtype=float)
        local = self.h.copy()
        for k, (i, j) in enumerate(self.edges):
            J = self.J[k]
            local[i] += J * c[j]
            local[j] += J * c[i]
            hij = J * s[i] * s[j]
            H[i, j] = hij
            H[j, i] = hij
        H[np.diag_indices(self.n)] = -c * local
        return H


def q1_local_minima_and_basins(
    h_diag: np.ndarray, n: int, guard: TimeGuard
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact one-flip minima and deterministic steepest-descent basin map.

    At each state, move to the lowest-energy one-flip neighbour if it is strictly
    lower.  Ties are deterministic by qubit iteration order.  Since energy
    strictly decreases on every move, pointer jumping yields all basin roots.
    """
    E = np.asarray(h_diag, dtype=float)
    states = np.arange(2**n, dtype=np.uint32)

    is_local = np.ones(E.size, dtype=bool)
    for q in range(n):
        guard.check("Q1 local-minimum enumeration")
        mask = np.uint32(1 << (n - 1 - q))
        nb = states ^ mask
        is_local &= E <= E[nb] + DISCRETE_TOL
    minima = states[is_local]

    nxt = states.copy()
    best = E.copy()
    for q in range(n):
        guard.check("Q1 basin next-state map")
        mask = np.uint32(1 << (n - 1 - q))
        nb = states ^ mask
        en = E[nb]
        improve = en < best - DISCRETE_TOL
        best[improve] = en[improve]
        nxt[improve] = nb[improve]

    roots = nxt.copy()
    # Pointer jumping converges in O(log path length).
    for _ in range(n + 3):
        guard.check("Q1 basin pointer jumping")
        new_roots = roots[roots]
        if np.array_equal(new_roots, roots):
            break
        roots = new_roots

    counts = np.bincount(roots.astype(np.int64), minlength=E.size)
    return minima.astype(np.int64), roots.astype(np.int64), counts.astype(np.int64)


def energy_level_ids(energies: Sequence[float], tol: float) -> Tuple[np.ndarray, List[float], List[int]]:
    e = np.asarray(energies, dtype=float)
    if e.size == 0:
        return np.array([], dtype=int), [], []
    order = np.argsort(e)
    ids = np.empty(e.size, dtype=int)
    centers: List[float] = []
    counts: List[int] = []
    current_members: List[int] = []
    current_center = None
    level = -1
    for idx in order:
        val = float(e[idx])
        if current_center is None or abs(val - current_center) > tol:
            if current_members:
                centers.append(float(np.mean(e[current_members])))
                counts.append(len(current_members))
            level += 1
            current_members = [int(idx)]
            current_center = val
        else:
            current_members.append(int(idx))
            current_center = float(np.mean(e[current_members]))
        ids[idx] = level
    if current_members:
        centers.append(float(np.mean(e[current_members])))
        counts.append(len(current_members))
    return ids, centers, counts


# -----------------------------------------------------------------------------
# Q2 objective, reconstructed from the exact saved Q1 instance.
# -----------------------------------------------------------------------------
def make_x_flips(n: int) -> List[np.ndarray]:
    basis = np.arange(2**n, dtype=np.uint32)
    return [basis ^ np.uint32(1 << (n - 1 - q)) for q in range(n)]


def make_cnot_chain_permutation(n: int) -> np.ndarray:
    destination = np.arange(2**n, dtype=np.uint32)
    for control in range(n - 1):
        target = control + 1
        cmask = np.uint32(1 << (n - 1 - control))
        tmask = np.uint32(1 << (n - 1 - target))
        active = (destination & cmask) != 0
        destination[active] ^= tmask
    return destination


class TFObjective:
    def __init__(self, n: int, h_diag: np.ndarray, gamma: float = GAMMA):
        self.n = int(n)
        self.dim = 2 * self.n
        self.h_diag = np.asarray(h_diag, dtype=float)
        self.gamma = float(gamma)
        self.x_flips = make_x_flips(n)
        self.cnot_dest = make_cnot_chain_permutation(n)

    def _apply_ry(self, state: np.ndarray, angle: float, q: int) -> np.ndarray:
        step = 1 << (self.n - 1 - q)
        view = state.reshape(-1, 2, step)
        a0 = view[:, 0, :].copy()
        a1 = view[:, 1, :].copy()
        c = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
        view[:, 0, :] = c * a0 - s * a1
        view[:, 1, :] = s * a0 + c * a1
        return state

    def statevector(self, theta: np.ndarray) -> np.ndarray:
        theta = wrap(theta)
        alpha = theta[: self.n]
        beta = theta[self.n :]
        state = np.array([1.0], dtype=np.float64)
        for angle in alpha:
            local = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0)])
            state = np.kron(state, local)
        tmp = np.empty_like(state)
        tmp[self.cnot_dest] = state
        state = tmp
        for q, angle in enumerate(beta):
            state = self._apply_ry(state, float(angle), q)
        return state

    def energy(self, theta: np.ndarray) -> float:
        psi = self.statevector(theta)
        e = float(np.dot(psi * psi, self.h_diag))
        for flip in self.x_flips:
            e -= self.gamma * float(np.dot(psi, psi[flip]))
        return e


# -----------------------------------------------------------------------------
# Archived endpoint readers (no optimization is performed).
# -----------------------------------------------------------------------------
def load_q1_bfgs_energies(root: Path, n: int) -> List[Dict[str, Any]]:
    base = root / "raw" / f"N{n}"
    if not base.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for json_path in base.glob("B*/BFGS/*.json"):
        try:
            rec = json.loads(json_path.read_text())
            if rec.get("status") != "ok":
                continue
            rows.append(
                {
                    "energy": float(rec["best_energy"]),
                    "budget": int(rec["budget"]),
                    "optimizer": "BFGS",
                    "run_id": int(rec["run_id"]),
                    "seed": int(rec["seed"]),
                    "source_file": str(json_path),
                }
            )
        except Exception:
            continue
    if not rows:
        return []
    max_budget = max(r["budget"] for r in rows)
    return [r for r in rows if r["budget"] == max_budget]


def load_q2_endpoints(root: Path, n: int, dim: int, expected_fingerprint: Optional[str] = None) -> List[Dict[str, Any]]:
    base = root / "raw" / f"N{n}"
    if not base.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for json_path in base.glob("B*/*/*.json"):
        try:
            rec = json.loads(json_path.read_text())
            if rec.get("status") != "ok":
                continue
            if expected_fingerprint is not None:
                saved_fp = rec.get("q1_instance_fingerprint_sha256")
                if saved_fp is not None and str(saved_fp) != str(expected_fingerprint):
                    continue
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            data = np.load(npz_path, allow_pickle=False)
            theta = np.asarray(data["best_theta"], dtype=float)
            if theta.shape != (dim,):
                continue
            rows.append(
                {
                    "energy": float(rec["best_energy"]),
                    "theta": wrap(theta),
                    "budget": int(rec["budget"]),
                    "optimizer": str(rec["optimizer"]),
                    "run_id": int(rec["run_id"]),
                    "seed": int(rec["seed"]),
                    "source_file": str(npz_path),
                }
            )
        except Exception:
            continue
    if not rows:
        return []
    max_budget = max(r["budget"] for r in rows)
    return [r for r in rows if r["budget"] == max_budget]


def load_q3_endpoints(root: Path, n: int, dim: int, expected_fingerprint: Optional[str] = None) -> List[Dict[str, Any]]:
    base = root / "raw" / f"N{n}"
    if not base.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for npz_path in base.glob("B*/*.npz"):
        try:
            data = np.load(npz_path, allow_pickle=False)
            status = scalar_str(data["status"])
            if status != "ok":
                continue
            if expected_fingerprint is not None and "instance_fingerprint" in data.files:
                if scalar_str(data["instance_fingerprint"]) != str(expected_fingerprint):
                    continue
            theta = np.asarray(data["best_x"], dtype=float)
            if theta.shape != (dim,):
                continue
            rows.append(
                {
                    "energy": float(data["best_energy"]),
                    "theta": wrap(theta),
                    "budget": int(data["budget"]),
                    "optimizer": scalar_str(data["optimizer"]),
                    "run_id": int(data["run_id"]),
                    "seed": int(data["seed"]),
                    "source_file": str(npz_path),
                }
            )
        except Exception:
            continue
    if not rows:
        return []
    max_budget = max(r["budget"] for r in rows)
    return [r for r in rows if r["budget"] == max_budget]


def load_q2_anchor(root: Path, n: int) -> Tuple[Optional[np.ndarray], str]:
    path = root / "instances" / f"N{n}_gamma_{GAMMA:.3f}_continuation_reference.npz"
    if path.exists():
        try:
            d = np.load(path, allow_pickle=False)
            theta = np.asarray(d["theta"], dtype=float)
            return wrap(theta), "saved_continuation_reference"
        except Exception:
            pass
    return None, ""


def load_q3_anchor(root: Path, n: int) -> Tuple[Optional[np.ndarray], str]:
    path = root / "references" / f"N{n}_variational_reference.npz"
    if path.exists():
        try:
            d = np.load(path, allow_pickle=False)
            theta = np.asarray(d["theta"], dtype=float)
            if theta.size:
                return wrap(theta), "saved_variational_reference"
        except Exception:
            pass
    return None, ""


def endpoint_energy_tol(endpoints: Sequence[Dict[str, Any]]) -> float:
    if not endpoints:
        return ENDPOINT_ENERGY_ABS_TOL
    best = min(float(x["energy"]) for x in endpoints)
    return max(
        ENDPOINT_ENERGY_ABS_TOL,
        ENDPOINT_ENERGY_REL_TOL * max(1.0, abs(best)),
    )


def cluster_endpoints(
    endpoints: Sequence[Dict[str, Any]], energy_tol: float, param_tol: float
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Greedy joint energy + periodic-parameter clustering of archived endpoints."""
    if not endpoints:
        return [], []
    order = sorted(range(len(endpoints)), key=lambda i: endpoints[i]["energy"])
    clusters: List[Dict[str, Any]] = []
    assignment = [-1] * len(endpoints)

    for idx in order:
        e = float(endpoints[idx]["energy"])
        th = np.asarray(endpoints[idx]["theta"], dtype=float)
        chosen = None
        for cid, cl in enumerate(clusters):
            if abs(e - cl["energy_rep"]) > energy_tol:
                continue
            # Compare against several members so a slightly broad flat basin does
            # not fragment merely because the first representative is off-centre.
            member_ids = cl["members"][: min(8, len(cl["members"]))]
            if min(torus_rms(th, endpoints[j]["theta"]) for j in member_ids) <= param_tol:
                chosen = cid
                break
        if chosen is None:
            chosen = len(clusters)
            clusters.append({"energy_rep": e, "members": []})
        clusters[chosen]["members"].append(idx)
        assignment[idx] = chosen

    out: List[Dict[str, Any]] = []
    total = len(endpoints)
    for cid, cl in enumerate(clusters):
        mids = cl["members"]
        ens = np.array([endpoints[i]["energy"] for i in mids], dtype=float)
        best_i = min(mids, key=lambda i: endpoints[i]["energy"])
        out.append(
            {
                "basin_id": cid,
                "count": len(mids),
                "frequency": len(mids) / total,
                "energy_min": float(np.min(ens)),
                "energy_mean": float(np.mean(ens)),
                "energy_std": float(np.std(ens)),
                "representative_theta": np.asarray(endpoints[best_i]["theta"], dtype=float),
                "representative_source": endpoints[best_i]["source_file"],
                "optimizers": sorted({str(endpoints[i]["optimizer"]) for i in mids}),
                "members": mids,
            }
        )
    out.sort(key=lambda c: c["energy_min"])
    # Re-label after sorting.
    old_to_new: Dict[int, int] = {}
    for new_id, cl in enumerate(out):
        old_to_new[cl["basin_id"]] = new_id
        cl["basin_id"] = new_id
    assignment = [old_to_new[a] for a in assignment]
    return out, assignment


# -----------------------------------------------------------------------------
# Q3 adapter loader.  Environment variables are set before module execution so
# N=10,12 necessarily reuse the saved Q1/Q2 paired disorder files.
# -----------------------------------------------------------------------------
def resolve_adapter_path(requested: Optional[Path]) -> Path:
    if requested is not None:
        if requested.exists():
            return requested.resolve()
        raise FileNotFoundError(f"Requested adapter does not exist: {requested}")
    candidates = [
        Path("heisenberg_model_adapter.py"),
        Path("heisenberg_model_adapter(1).py"),
    ]
    candidates.extend(sorted(Path.cwd().glob("heisenberg_model_adapter*.py")))
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not locate heisenberg_model_adapter.py. Pass --adapter PATH."
    )


def load_heisenberg_adapter(path: Path, q1_root: Path, q2_root: Path):
    os.environ["Q1_INSTANCE_DIR"] = str((q1_root / "instances").resolve())
    os.environ["Q2_INSTANCE_DIR"] = str((q2_root / "instances").resolve())
    name = "_landscape_heisenberg_adapter"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "make_problem"):
        raise AttributeError(f"Adapter {path} does not export make_problem(n)")
    return module


# -----------------------------------------------------------------------------
# Exact parameter-shift derivatives for Q2/Q3.
# Each Ry parameter appears once, so shifts by +/- pi/2 give exact first
# derivatives and the standard four-point mixed derivative gives exact Hessian
# off-diagonal entries.
# -----------------------------------------------------------------------------
class ShiftEvaluator:
    def __init__(self, energy: Callable[[np.ndarray], float], theta: np.ndarray, guard: TimeGuard):
        self.energy_fn = energy
        self.theta = wrap(theta)
        self.guard = guard
        self.cache: Dict[bytes, float] = {}

    def f(self, x: np.ndarray) -> float:
        self.guard.check("parameter-shift objective evaluation")
        y = wrap(x)
        key = np.round(y, 13).tobytes()
        if key not in self.cache:
            self.cache[key] = float(self.energy_fn(y))
        return self.cache[key]

    def gradient(self) -> np.ndarray:
        d = self.theta.size
        g = np.empty(d, dtype=float)
        s = np.pi / 2.0
        for i in range(d):
            xp = self.theta.copy(); xp[i] += s
            xm = self.theta.copy(); xm[i] -= s
            g[i] = 0.5 * (self.f(xp) - self.f(xm))
        return g

    def hessian(self) -> np.ndarray:
        d = self.theta.size
        H = np.empty((d, d), dtype=float)
        f0 = self.f(self.theta)
        # Exact diagonal second derivative for first-harmonic dependence.
        for i in range(d):
            xp = self.theta.copy(); xp[i] += np.pi
            xm = self.theta.copy(); xm[i] -= np.pi
            H[i, i] = 0.25 * (self.f(xp) - 2.0 * f0 + self.f(xm))
        s = np.pi / 2.0
        for i in range(d):
            for j in range(i + 1, d):
                xpp = self.theta.copy(); xpp[i] += s; xpp[j] += s
                xpm = self.theta.copy(); xpm[i] += s; xpm[j] -= s
                xmp = self.theta.copy(); xmp[i] -= s; xmp[j] += s
                xmm = self.theta.copy(); xmm[i] -= s; xmm[j] -= s
                hij = 0.25 * (self.f(xpp) - self.f(xpm) - self.f(xmp) + self.f(xmm))
                H[i, j] = hij
                H[j, i] = hij
        return 0.5 * (H + H.T)


def get_gradient(cond: Condition, theta: np.ndarray, guard: TimeGuard) -> np.ndarray:
    if cond.analytic_gradient is not None:
        guard.check("analytic gradient")
        return np.asarray(cond.analytic_gradient(theta), dtype=float)
    return ShiftEvaluator(cond.energy, theta, guard).gradient()


def get_hessian(cond: Condition, theta: np.ndarray, guard: TimeGuard) -> np.ndarray:
    if cond.analytic_hessian is not None:
        guard.check("analytic Hessian")
        return np.asarray(cond.analytic_hessian(theta), dtype=float)
    return ShiftEvaluator(cond.energy, theta, guard).hessian()


def get_grad_and_hessian(cond: Condition, theta: np.ndarray, guard: TimeGuard) -> Tuple[np.ndarray, np.ndarray]:
    if cond.analytic_gradient is not None and cond.analytic_hessian is not None:
        return get_gradient(cond, theta, guard), get_hessian(cond, theta, guard)
    ev = ShiftEvaluator(cond.energy, theta, guard)
    return ev.gradient(), ev.hessian()


# -----------------------------------------------------------------------------
# Spectrum/coupling summaries
# -----------------------------------------------------------------------------
def summarize_hessian(H: np.ndarray) -> Dict[str, float]:
    H = np.asarray(H, dtype=float)
    eig = np.linalg.eigvalsh(H)
    scale = max(float(np.max(np.abs(eig))), 1.0)
    tol = max(HESSIAN_ACTIVE_ABS_TOL, HESSIAN_ACTIVE_REL_TOL * scale)
    pos = eig[eig > tol]
    neg = eig[eig < -tol]
    zero = eig[np.abs(eig) <= tol]
    if pos.size >= 2:
        cond = float(np.max(pos) / np.min(pos))
    elif pos.size == 1:
        cond = 1.0
    else:
        cond = np.nan
    diag = np.diag(np.diag(H))
    off = H - diag
    norm_h = float(np.linalg.norm(H, ord="fro"))
    off_ratio = float(np.linalg.norm(off, ord="fro") / norm_h) if norm_h > 0 else 0.0
    return {
        "eig_min": float(np.min(eig)),
        "eig_max": float(np.max(eig)),
        "eig_median": float(np.median(eig)),
        "active_positive_count": int(pos.size),
        "negative_count": int(neg.size),
        "near_zero_count": int(zero.size),
        "active_condition_number": cond,
        "offdiag_frobenius_ratio": off_ratio,
        "spectral_tolerance": tol,
    }


def coupling_rows_from_hessian(
    model: str, n: int, point_id: str, H: np.ndarray
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    H = np.asarray(H, dtype=float)
    d = H.shape[0]
    max_abs = float(np.max(np.abs(H))) if H.size else 0.0
    tol = max(COUPLING_ABS_TOL, COUPLING_REL_TOL * max_abs)
    rows: List[Dict[str, Any]] = []
    active = 0
    possible = d * (d - 1) // 2
    abs_sum = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            w = float(H[i, j])
            aw = abs(w)
            is_active = aw > tol
            active += int(is_active)
            abs_sum += aw
            if is_active:
                rows.append(
                    {
                        "model": model,
                        "n_qubits": n,
                        "point_id": point_id,
                        "i": i,
                        "j": j,
                        "hessian_coupling": w,
                        "abs_hessian_coupling": aw,
                        "threshold": tol,
                    }
                )
    return rows, {
        "coupling_threshold": tol,
        "active_coupling_edges": active,
        "possible_coupling_edges": possible,
        "active_coupling_density": (active / possible) if possible else 0.0,
        "mean_abs_offdiag": (abs_sum / possible) if possible else 0.0,
    }


# -----------------------------------------------------------------------------
# Condition construction
# -----------------------------------------------------------------------------
def build_conditions(args: argparse.Namespace, guard: TimeGuard) -> List[Condition]:
    conditions: List[Condition] = []
    q1_cache: Dict[int, Dict[str, Any]] = {}

    # Q1
    for n in Q1_SIZES:
        guard.check("building Q1 conditions")
        inst = load_q1_instance(args.q1_results, n)
        q1_cache[n] = inst
        obj = Q1AnalyticObjective(inst["J"], inst["h"], inst["edges"])
        ground_idx = int(np.argmin(inst["H_diag"]))
        bits = np.array(
            [(ground_idx >> (n - 1 - q)) & 1 for q in range(n)], dtype=float
        )
        anchor = np.pi * bits
        conditions.append(
            Condition(
                model="Q1",
                n=n,
                dim=n,
                energy=obj.energy,
                h_diag=inst["H_diag"],
                instance_seed=inst["instance_seed"],
                fingerprint=inst["fingerprint"],
                metadata=inst["metadata"],
                analytic_gradient=obj.gradient,
                analytic_hessian=obj.hessian,
                anchor_theta=anchor,
                anchor_source="exact_diagonal_ground_bitstring",
                endpoints=[],
            )
        )

    # Q2 reuses saved Q1 J,h; never generates a new instance.
    for n in Q2_SIZES:
        guard.check("building Q2 conditions")
        inst = q1_cache.get(n) or load_q1_instance(args.q1_results, n)
        q1_cache[n] = inst
        obj = TFObjective(n, inst["H_diag"], GAMMA)
        endpoints = load_q2_endpoints(args.q2_results, n, 2 * n, inst["fingerprint"])
        anchor, anchor_source = load_q2_anchor(args.q2_results, n)
        if anchor is None and endpoints:
            best = min(endpoints, key=lambda x: x["energy"])
            anchor = np.asarray(best["theta"], dtype=float)
            anchor_source = "best_archived_endpoint"
        conditions.append(
            Condition(
                model="Q2",
                n=n,
                dim=2 * n,
                energy=obj.energy,
                h_diag=inst["H_diag"],
                instance_seed=inst["instance_seed"],
                fingerprint=inst["fingerprint"],
                metadata={**inst["metadata"], "gamma": GAMMA},
                anchor_theta=anchor,
                anchor_source=anchor_source,
                endpoints=endpoints,
            )
        )

    # Q3 uses its supplied adapter so the physics and paired-instance policy are
    # exactly the same as the benchmark.
    adapter_path = resolve_adapter_path(args.adapter)
    adapter = load_heisenberg_adapter(adapter_path, args.q1_results, args.q2_results)
    for n in Q3_SIZES:
        guard.check("building Q3 conditions")
        problem = adapter.make_problem(n)
        vqa = problem.make_vqa()
        endpoints = load_q3_endpoints(
            args.q3_results, n, int(problem.dim), str(problem.instance_fingerprint)
        )
        anchor, anchor_source = load_q3_anchor(args.q3_results, n)
        if anchor is None and endpoints:
            best = min(endpoints, key=lambda x: x["energy"])
            anchor = np.asarray(best["theta"], dtype=float)
            anchor_source = "best_archived_endpoint"
        metadata = problem.metadata() if hasattr(problem, "metadata") else {}
        conditions.append(
            Condition(
                model="Q3",
                n=n,
                dim=int(problem.dim),
                energy=vqa.energy,
                h_diag=np.asarray(problem.H_diag, dtype=float),
                instance_seed=int(problem.instance_seed),
                fingerprint=str(problem.instance_fingerprint),
                metadata=metadata,
                anchor_theta=anchor,
                anchor_source=anchor_source,
                endpoints=endpoints,
            )
        )

    return conditions


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    guard = TimeGuard(args.time_limit)

    summary_rows: List[Dict[str, Any]] = []
    minima_rows: List[Dict[str, Any]] = []
    basin_rows: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []
    degeneracy_rows: List[Dict[str, Any]] = []
    hessian_rows: List[Dict[str, Any]] = []
    hessian_summary_rows: List[Dict[str, Any]] = []
    coupling_rows: List[Dict[str, Any]] = []
    coupling_summary_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    global_parameter_rows: List[Dict[str, Any]] = []
    global_hamming_frames: List[pd.DataFrame] = []
    warnings: List[str] = []
    complete = True
    deadline_message = ""

    conditions = build_conditions(args, guard)

    for cond in conditions:
        try:
            guard.check(f"starting {cond.model} N={cond.n}")
            print(
                f"[{cond.model} N={cond.n}] dim={cond.dim} "
                f"remaining={guard.remaining:.1f}s",
                flush=True,
            )
            row: Dict[str, Any] = {
                "model": cond.model,
                "n_qubits": cond.n,
                "dimension": cond.dim,
                "instance_seed": cond.instance_seed,
                "instance_fingerprint": cond.fingerprint,
                "anchor_source": cond.anchor_source,
                "analysis_complete": True,
            }

            # ----- Hamming/global discrete backbone for every model -----
            gh = basis_hamming_summary(cond.h_diag, cond.n)
            gh.insert(0, "n_qubits", cond.n)
            gh.insert(0, "model", cond.model)
            gh.insert(2, "anchor", "classical_diagonal_ground")
            global_hamming_frames.append(gh)
            # Exact discrete backbone complexity for every model.  For Q2/Q3 this
            # characterizes the shared computational-basis classical backbone;
            # non-commuting effects are captured separately by continuous Hessians/gradients.
            states_backbone = np.arange(2**cond.n, dtype=np.uint32)
            is_local_backbone = np.ones(2**cond.n, dtype=bool)
            for q in range(cond.n):
                mask = np.uint32(1 << (cond.n - 1 - q))
                nb = states_backbone ^ mask
                is_local_backbone &= cond.h_diag <= cond.h_diag[nb] + DISCRETE_TOL
            row["diagonal_backbone_one_flip_minima"] = int(np.sum(is_local_backbone))
            diag_e0 = float(np.min(cond.h_diag))
            row["diagonal_backbone_ground_degeneracy"] = int(
                np.sum(np.isclose(cond.h_diag, diag_e0, rtol=0.0, atol=1e-10))
            )

            # A simple monotonicity summary of the diagonal backbone.
            if len(gh) > 1:
                rr = spearmanr(
                    gh["hamming_distance"].to_numpy(),
                    gh["energy_mean"].to_numpy(),
                )
                row["hamming_energy_spearman"] = float(rr.statistic)
            else:
                row["hamming_energy_spearman"] = np.nan

            # ----- Multimodality, basins, degeneracy -----
            local_min_thetas: List[Tuple[str, np.ndarray, float]] = []
            if cond.model == "Q1":
                minima, roots, counts = q1_local_minima_and_basins(cond.h_diag, cond.n, guard)
                e_min = cond.h_diag[minima]
                order = np.argsort(e_min)
                minima = minima[order]
                e_min = e_min[order]
                level_ids, level_centers, level_counts = energy_level_ids(e_min, tol=1e-10)
                ground_idx = int(np.argmin(cond.h_diag))
                geometric_ground_basin_fraction = float(counts[ground_idx] / (2**cond.n))
                basin_counts = counts[minima]
                geom_h, geom_hnorm, geom_eff = entropy_stats(basin_counts)

                # Reuse the already-completed BFGS runs as the multi-start basin sample.
                # Q1 did not save theta, but its converged energy can be mapped to the
                # nearest exact one-flip minimum energy without any optimizer rerun.
                bfgs_records = load_q1_bfgs_energies(args.q1_results, cond.n)
                bfgs_counts = np.zeros(len(minima), dtype=int)
                bfgs_unassigned = 0
                assign_tol = max(1e-5, 1e-5 * max(1.0, abs(float(e_min[0]))))
                for br in bfgs_records:
                    j = int(np.argmin(np.abs(e_min - float(br["energy"]))))
                    if abs(float(e_min[j]) - float(br["energy"])) <= assign_tol:
                        bfgs_counts[j] += 1
                    else:
                        bfgs_unassigned += 1
                if len(bfgs_records) > 0:
                    ground_pos = int(np.where(minima == ground_idx)[0][0])
                    ground_basin_frequency = float(bfgs_counts[ground_pos] / len(bfgs_records))
                    bh, bhnorm, beff = entropy_stats(bfgs_counts[bfgs_counts > 0])
                    basin_source = "archived_BFGS_max_budget_energy_to_exact_minimum_mapping"
                else:
                    ground_basin_frequency = geometric_ground_basin_fraction
                    bh, bhnorm, beff = geom_h, geom_hnorm, geom_eff
                    basin_source = "exact_deterministic_steepest_one_flip_descent_fallback"
                    warnings.append(
                        f"Q1 N={cond.n}: no archived BFGS run records found; basin difficulty uses exact steepest-descent geometry instead."
                    )
                row.update(
                    {
                        "multimodality_count": int(len(minima)),
                        "multimodality_source": "exact_one_flip_minima",
                        "basin_source": basin_source,
                        "ground_basin_frequency": ground_basin_frequency,
                        "basin_difficulty_1_minus_ground_frequency": 1.0 - ground_basin_frequency,
                        "basin_entropy": bh,
                        "basin_entropy_normalized": bhnorm,
                        "effective_basin_count": beff,
                        "geometric_ground_basin_fraction": geometric_ground_basin_fraction,
                        "geometric_basin_entropy_normalized": geom_hnorm,
                        "geometric_effective_basin_count": geom_eff,
                        "bfgs_endpoint_count_used": int(len(bfgs_records)),
                        "bfgs_unassigned_count": int(bfgs_unassigned),
                        "endpoint_count_used": int(len(bfgs_records)),
                        "energy_cluster_tolerance": 1e-10,
                    }
                )
                for rank, (state, en, lid) in enumerate(zip(minima, e_min, level_ids)):
                    b = np.array(
                        [(int(state) >> (cond.n - 1 - q)) & 1 for q in range(cond.n)],
                        dtype=float,
                    )
                    theta = np.pi * b
                    freq = float(counts[int(state)] / (2**cond.n))
                    minima_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "minimum_id": rank,
                            "source": "exact_one_flip_minimum",
                            "state_index": int(state),
                            "bitstring": "".join(str(int(x)) for x in b),
                            "energy": float(en),
                            "delta_from_best_minimum": float(en - e_min[0]),
                            "energy_level_id": int(lid),
                            "basin_frequency": freq,
                            "representative_theta_json": json.dumps(theta.tolist()),
                        }
                    )
                    bfgs_count_here = int(bfgs_counts[rank]) if 'bfgs_counts' in locals() else 0
                    bfgs_freq_here = (
                        bfgs_count_here / len(bfgs_records) if 'bfgs_records' in locals() and len(bfgs_records) else np.nan
                    )
                    basin_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "basin_id": rank,
                            "source": basin_source,
                            "energy_min": float(en),
                            "count": bfgs_count_here if len(bfgs_records) else int(counts[int(state)]),
                            "frequency": bfgs_freq_here if len(bfgs_records) else freq,
                            "geometric_count": int(counts[int(state)]),
                            "geometric_frequency": freq,
                            "bfgs_count": bfgs_count_here,
                            "bfgs_frequency": bfgs_freq_here,
                            "all_endpoint_count": np.nan,
                            "all_endpoint_frequency": np.nan,
                            "representative_theta_json": json.dumps(theta.tolist()),
                        }
                    )
                    if rank < args.hessian_minima:
                        local_min_thetas.append((f"minimum_{rank}", theta, float(en)))
                for lid, (ec, mult) in enumerate(zip(level_centers, level_counts)):
                    degeneracy_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "energy_level_id": lid,
                            "energy": ec,
                            "multiplicity": mult,
                            "source": "exact_one_flip_minimum_energy_degeneracy",
                        }
                    )
                row["ground_level_degeneracy"] = int(level_counts[0]) if level_counts else np.nan
                row["max_excited_level_degeneracy"] = int(max(level_counts[1:], default=0))
                row["minimum_energy_mean"] = float(np.mean(e_min))
                row["minimum_energy_std"] = float(np.std(e_min))
                row["minimum_energy_gap_first_excited"] = (
                    float(level_centers[1] - level_centers[0]) if len(level_centers) > 1 else np.nan
                )
            else:
                endpoints = cond.endpoints or []
                etol = endpoint_energy_tol(endpoints)
                basins, assignments = cluster_endpoints(
                    endpoints, etol, ENDPOINT_TORUS_RMS_TOL
                )
                counts_all = np.asarray([b["count"] for b in basins], dtype=int)
                bfgs_indices = [i for i, ep in enumerate(endpoints) if str(ep["optimizer"]).upper() == "BFGS"]
                bfgs_counts = np.zeros(len(basins), dtype=int)
                for i in bfgs_indices:
                    if assignments:
                        bfgs_counts[assignments[i]] += 1
                if len(bfgs_indices) > 0 and len(basins) > 0:
                    basin_counts_for_metric = bfgs_counts
                    ground_freq = float(bfgs_counts[0] / len(bfgs_indices))
                    basin_source = "archived_BFGS_max_budget_endpoint_frequency"
                else:
                    basin_counts_for_metric = counts_all
                    ground_freq = basins[0]["frequency"] if basins else np.nan
                    basin_source = "archived_all_optimizer_max_budget_endpoint_occupancy_fallback"
                    warnings.append(
                        f"{cond.model} N={cond.n}: no archived BFGS endpoints found; basin frequency falls back to all optimizer endpoints."
                    )
                h, hnorm, eff = entropy_stats(basin_counts_for_metric[basin_counts_for_metric > 0])
                row.update(
                    {
                        "multimodality_count": int(len(basins)),
                        "multimodality_source": "archived_all_optimizer_endpoint_parameter_energy_clusters",
                        "basin_source": basin_source,
                        "ground_basin_frequency": ground_freq,
                        "basin_difficulty_1_minus_ground_frequency": (
                            float(1.0 - ground_freq) if np.isfinite(ground_freq) else np.nan
                        ),
                        "basin_entropy": h,
                        "basin_entropy_normalized": hnorm,
                        "effective_basin_count": eff,
                        "bfgs_endpoint_count_used": int(len(bfgs_indices)),
                        "endpoint_count_used": int(len(endpoints)),
                        "energy_cluster_tolerance": etol,
                    }
                )
                if len(endpoints) < 10:
                    warnings.append(
                        f"{cond.model} N={cond.n}: only {len(endpoints)} archived max-budget endpoints; "
                        "empirical basin/multimodality estimates are weak."
                    )
                # Endpoint-level table.
                for idx, ep in enumerate(endpoints):
                    endpoint_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "endpoint_id": idx,
                            "basin_id": assignments[idx] if assignments else np.nan,
                            "energy": float(ep["energy"]),
                            "budget": int(ep["budget"]),
                            "optimizer": ep["optimizer"],
                            "run_id": int(ep["run_id"]),
                            "seed": int(ep["seed"]),
                            "theta_json": json.dumps(np.asarray(ep["theta"]).tolist()),
                            "source_file": ep["source_file"],
                        }
                    )
                # Basin rows and empirical energy-level multiplicities.
                basin_energies = []
                for b in basins:
                    basin_energies.append(float(b["energy_min"]))
                    minima_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "minimum_id": int(b["basin_id"]),
                            "source": "archived_endpoint_basin_cluster",
                            "state_index": np.nan,
                            "bitstring": "",
                            "energy": float(b["energy_min"]),
                            "delta_from_best_minimum": float(b["energy_min"] - basins[0]["energy_min"]),
                            "energy_level_id": np.nan,
                            "basin_frequency": float(b["frequency"]),
                            "representative_theta_json": json.dumps(b["representative_theta"].tolist()),
                        }
                    )
                    bid = int(b["basin_id"])
                    bfgs_count_here = int(bfgs_counts[bid]) if len(bfgs_counts) else 0
                    bfgs_freq_here = bfgs_count_here / len(bfgs_indices) if len(bfgs_indices) else np.nan
                    basin_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "basin_id": bid,
                            "source": basin_source,
                            "energy_min": float(b["energy_min"]),
                            "count": bfgs_count_here if len(bfgs_indices) else int(b["count"]),
                            "frequency": bfgs_freq_here if len(bfgs_indices) else float(b["frequency"]),
                            "geometric_count": np.nan,
                            "geometric_frequency": np.nan,
                            "bfgs_count": bfgs_count_here,
                            "bfgs_frequency": bfgs_freq_here,
                            "all_endpoint_count": int(b["count"]),
                            "all_endpoint_frequency": float(b["frequency"]),
                            "representative_theta_json": json.dumps(b["representative_theta"].tolist()),
                            "optimizers_json": json.dumps(b["optimizers"]),
                        }
                    )
                    if len(local_min_thetas) < args.hessian_minima:
                        bfgs_members = [
                            i for i in b["members"]
                            if str(endpoints[i]["optimizer"]).upper() == "BFGS"
                        ]
                        if bfgs_members:
                            rep_i = min(bfgs_members, key=lambda i: endpoints[i]["energy"])
                            htheta = np.asarray(endpoints[rep_i]["theta"], dtype=float)
                            henergy = float(endpoints[rep_i]["energy"])
                            hname = f"bfgs_basin_{b['basin_id']}"
                        else:
                            htheta = np.asarray(b["representative_theta"], dtype=float)
                            henergy = float(b["energy_min"])
                            hname = f"empirical_basin_{b['basin_id']}"
                        local_min_thetas.append((hname, htheta, henergy))
                if basin_energies:
                    level_ids, level_centers, level_counts = energy_level_ids(basin_energies, tol=etol)
                    # Fill energy-level ids into the minima rows just added for this condition.
                    added = [r for r in minima_rows if r["model"] == cond.model and r["n_qubits"] == cond.n and r["source"] == "archived_endpoint_basin_cluster"]
                    by_id = {int(r["minimum_id"]): r for r in added}
                    for i, lid in enumerate(level_ids):
                        if i in by_id:
                            by_id[i]["energy_level_id"] = int(lid)
                    for lid, (ec, mult) in enumerate(zip(level_centers, level_counts)):
                        degeneracy_rows.append(
                            {
                                "model": cond.model,
                                "n_qubits": cond.n,
                                "energy_level_id": lid,
                                "energy": ec,
                                "multiplicity": mult,
                                "source": "empirical_distinct_parameter_basins_per_energy_level",
                            }
                        )
                    row["ground_level_degeneracy"] = int(level_counts[0])
                    row["max_excited_level_degeneracy"] = int(max(level_counts[1:], default=0))
                    row["minimum_energy_mean"] = float(np.mean(basin_energies))
                    row["minimum_energy_std"] = float(np.std(basin_energies))
                    row["minimum_energy_gap_first_excited"] = (
                        float(level_centers[1] - level_centers[0]) if len(level_centers) > 1 else np.nan
                    )
                else:
                    row.update(
                        {
                            "ground_level_degeneracy": np.nan,
                            "max_excited_level_degeneracy": np.nan,
                            "minimum_energy_mean": np.nan,
                            "minimum_energy_std": np.nan,
                            "minimum_energy_gap_first_excited": np.nan,
                        }
                    )

            # Ensure anchor exists when possible.
            if cond.anchor_theta is None and local_min_thetas:
                cond.anchor_theta = local_min_thetas[0][1].copy()
                cond.anchor_source = "lowest_available_minimum"
                row["anchor_source"] = cond.anchor_source

            # For Q2/Q3, include the saved variational/continuation reference as
            # a conditioning point even if archived endpoints are unavailable.
            if cond.model != "Q1" and cond.anchor_theta is not None:
                if not local_min_thetas or torus_rms(local_min_thetas[0][1], cond.anchor_theta) > 1e-8:
                    anchor_entry = (
                        "saved_or_best_anchor",
                        np.asarray(cond.anchor_theta, dtype=float),
                        float(cond.energy(cond.anchor_theta)),
                    )
                    local_min_thetas = [anchor_entry] + local_min_thetas

            # ----- Conditioning: Hessian at a few lowest minima/endpoints -----
            local_cond_values = []
            stationary_cond_values = []
            local_neg_counts = []
            for point_id, theta, energy_hint in local_min_thetas[: args.hessian_minima]:
                guard.check(f"{cond.model} N={cond.n} Hessian at {point_id}")
                g, H = get_grad_and_hessian(cond, theta, guard)
                hs = summarize_hessian(H)
                grad_norm = float(np.linalg.norm(g))
                grad_rms = grad_norm / math.sqrt(cond.dim)
                local_cond_values.append(hs["active_condition_number"])
                local_neg_counts.append(hs["negative_count"])
                stationary_proxy = bool(grad_rms < 1e-4 and hs["negative_count"] == 0)
                if stationary_proxy and np.isfinite(hs["active_condition_number"]):
                    stationary_cond_values.append(hs["active_condition_number"])
                for k, ev in enumerate(np.linalg.eigvalsh(H)):
                    hessian_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "point_id": point_id,
                            "point_type": "local_minimum_or_archived_basin_representative",
                            "eigenvalue_index": k,
                            "eigenvalue": float(ev),
                        }
                    )
                hessian_summary_rows.append(
                    {
                        "model": cond.model,
                        "n_qubits": cond.n,
                        "point_id": point_id,
                        "point_type": "local_minimum_or_archived_basin_representative",
                        "energy": float(cond.energy(theta)),
                        "energy_hint": float(energy_hint),
                        "gradient_norm": grad_norm,
                        "gradient_rms": grad_rms,
                        "stationary_minimum_proxy": stationary_proxy,
                        **hs,
                    }
                )
            finite_conds = np.asarray([x for x in local_cond_values if np.isfinite(x)], dtype=float)
            stationary_conds = np.asarray(stationary_cond_values, dtype=float)
            row["conditioning_active_kappa_median"] = (
                float(np.median(stationary_conds)) if stationary_conds.size else np.nan
            )
            row["conditioning_active_kappa_median_all_probes"] = (
                float(np.median(finite_conds)) if finite_conds.size else np.nan
            )
            row["conditioning_stationary_minima_count"] = int(stationary_conds.size)
            row["conditioning_negative_eigs_total"] = int(np.sum(local_neg_counts)) if local_neg_counts else np.nan

            # ----- Separability: one deterministic random Hessian probe -----
            rng_c = np.random.default_rng(
                stable_condition_seed(args.analysis_seed, cond.model, cond.n, salt=11)
            )
            probe = rng_c.uniform(-np.pi, np.pi, cond.dim)
            Hprobe = get_hessian(cond, probe, guard)
            hs_probe = summarize_hessian(Hprobe)
            crows, cs = coupling_rows_from_hessian(
                cond.model, cond.n, "random_coupling_probe", Hprobe
            )
            coupling_rows.extend(crows)
            coupling_summary_rows.append(
                {
                    "model": cond.model,
                    "n_qubits": cond.n,
                    "point_id": "random_coupling_probe",
                    **cs,
                    "offdiag_frobenius_ratio": hs_probe["offdiag_frobenius_ratio"],
                }
            )
            row["separability_offdiag_frobenius_ratio"] = hs_probe["offdiag_frobenius_ratio"]
            row["separability_active_coupling_density"] = cs["active_coupling_density"]

            # ----- Trainability: random-parameter gradient-norm distribution -----
            rng_g = np.random.default_rng(
                stable_condition_seed(args.analysis_seed, cond.model, cond.n, salt=23)
            )
            grad_rms_vals = []
            grad_norm_vals = []
            for sidx in range(args.grad_samples):
                guard.check(f"{cond.model} N={cond.n} trainability samples")
                theta = rng_g.uniform(-np.pi, np.pi, cond.dim)
                e = float(cond.energy(theta))
                g = get_gradient(cond, theta, guard)
                gn = float(np.linalg.norm(g))
                gr = gn / math.sqrt(cond.dim)
                grad_norm_vals.append(gn)
                grad_rms_vals.append(gr)
                gradient_rows.append(
                    {
                        "model": cond.model,
                        "n_qubits": cond.n,
                        "sample_id": sidx,
                        "energy": e,
                        "gradient_norm": gn,
                        "gradient_rms": gr,
                        "gradient_max_abs": float(np.max(np.abs(g))),
                    }
                )
            gr = np.asarray(grad_rms_vals, dtype=float)
            gn = np.asarray(grad_norm_vals, dtype=float)
            row.update(
                {
                    "trainability_grad_norm_median": float(np.median(gn)),
                    "trainability_grad_rms_median": float(np.median(gr)),
                    "trainability_grad_rms_p10": float(np.quantile(gr, 0.10)),
                    "trainability_grad_rms_p90": float(np.quantile(gr, 0.90)),
                    "trainability_frac_grad_rms_lt_1e-3": float(np.mean(gr < 1e-3)),
                    "trainability_frac_grad_rms_lt_1e-4": float(np.mean(gr < 1e-4)),
                }
            )

            # ----- Global continuous structure: energy vs torus distance from anchor -----
            if cond.anchor_theta is not None:
                rng_p = np.random.default_rng(
                    stable_condition_seed(args.analysis_seed, cond.model, cond.n, salt=37)
                )
                dists = []
                ens = []
                anchor_e = float(cond.energy(cond.anchor_theta))
                for sidx in range(args.global_samples):
                    guard.check(f"{cond.model} N={cond.n} global structure samples")
                    theta = rng_p.uniform(-np.pi, np.pi, cond.dim)
                    e = float(cond.energy(theta))
                    dist_rms = torus_rms(theta, cond.anchor_theta)
                    dists.append(dist_rms)
                    ens.append(e)
                    global_parameter_rows.append(
                        {
                            "model": cond.model,
                            "n_qubits": cond.n,
                            "sample_id": sidx,
                            "anchor_source": cond.anchor_source,
                            "torus_rms_distance": dist_rms,
                            "torus_l2_distance": torus_l2(theta, cond.anchor_theta),
                            "energy": e,
                            "delta_energy_from_anchor": e - anchor_e,
                        }
                    )
                rr = spearmanr(np.asarray(dists), np.asarray(ens))
                row["global_parameter_distance_energy_spearman"] = float(rr.statistic)
                row["global_anchor_energy"] = anchor_e
            else:
                row["global_parameter_distance_energy_spearman"] = np.nan
                row["global_anchor_energy"] = np.nan
                warnings.append(
                    f"{cond.model} N={cond.n}: no saved/reference theta was available for parameter-distance global structure."
                )

            row["elapsed_seconds_at_condition_end"] = guard.elapsed
            summary_rows.append(row)

        except DeadlineReached as exc:
            complete = False
            deadline_message = str(exc)
            warnings.append(deadline_message)
            # Record a partial condition row if it was not appended yet.
            if not summary_rows or not (
                summary_rows[-1].get("model") == cond.model
                and summary_rows[-1].get("n_qubits") == cond.n
            ):
                summary_rows.append(
                    {
                        "model": cond.model,
                        "n_qubits": cond.n,
                        "dimension": cond.dim,
                        "instance_seed": cond.instance_seed,
                        "instance_fingerprint": cond.fingerprint,
                        "anchor_source": cond.anchor_source,
                        "analysis_complete": False,
                        "elapsed_seconds_at_condition_end": guard.elapsed,
                    }
                )
            break

    # Save everything computed so far, even after a deadline.  Explicit schemas
    # keep empty CSVs machine-readable rather than producing a one-byte blank file.
    schemas: Dict[str, List[str]] = {
        "landscape_summary.csv": ["model", "n_qubits", "dimension", "analysis_complete"],
        "minima.csv": ["model", "n_qubits", "minimum_id", "source", "energy", "energy_level_id", "basin_frequency", "representative_theta_json"],
        "basins.csv": ["model", "n_qubits", "basin_id", "source", "energy_min", "count", "frequency", "bfgs_count", "bfgs_frequency", "all_endpoint_count", "all_endpoint_frequency", "geometric_count", "geometric_frequency", "representative_theta_json"],
        "archived_endpoints.csv": ["model", "n_qubits", "endpoint_id", "basin_id", "energy", "budget", "optimizer", "run_id", "seed", "theta_json", "source_file"],
        "degeneracy_levels.csv": ["model", "n_qubits", "energy_level_id", "energy", "multiplicity", "source"],
        "hessian_spectra.csv": ["model", "n_qubits", "point_id", "point_type", "eigenvalue_index", "eigenvalue"],
        "hessian_summary.csv": ["model", "n_qubits", "point_id", "point_type", "energy", "gradient_norm", "gradient_rms", "stationary_minimum_proxy", "active_condition_number"],
        "coupling_edges.csv": ["model", "n_qubits", "point_id", "i", "j", "hessian_coupling", "abs_hessian_coupling", "threshold"],
        "coupling_summary.csv": ["model", "n_qubits", "point_id", "active_coupling_edges", "possible_coupling_edges", "active_coupling_density", "offdiag_frobenius_ratio"],
        "gradient_samples.csv": ["model", "n_qubits", "sample_id", "energy", "gradient_norm", "gradient_rms", "gradient_max_abs"],
        "global_parameter_samples.csv": ["model", "n_qubits", "sample_id", "anchor_source", "torus_rms_distance", "torus_l2_distance", "energy", "delta_energy_from_anchor"],
        "global_hamming_summary.csv": ["model", "n_qubits", "anchor", "hamming_distance", "count", "energy_mean", "energy_std", "energy_min", "energy_max"],
    }

    def save_csv(name: str, rows_or_df: Any) -> None:
        path = out / name
        if isinstance(rows_or_df, pd.DataFrame):
            df = rows_or_df.copy()
        else:
            df = pd.DataFrame(rows_or_df)
        if df.empty and name in schemas:
            df = pd.DataFrame(columns=schemas[name])
        df.to_csv(path, index=False)

    save_csv("landscape_summary.csv", summary_rows)
    save_csv("minima.csv", minima_rows)
    save_csv("basins.csv", basin_rows)
    save_csv("archived_endpoints.csv", endpoint_rows)
    save_csv("degeneracy_levels.csv", degeneracy_rows)
    save_csv("hessian_spectra.csv", hessian_rows)
    save_csv("hessian_summary.csv", hessian_summary_rows)
    save_csv("coupling_edges.csv", coupling_rows)
    save_csv("coupling_summary.csv", coupling_summary_rows)
    save_csv("gradient_samples.csv", gradient_rows)
    save_csv("global_parameter_samples.csv", global_parameter_rows)
    if global_hamming_frames:
        save_csv("global_hamming_summary.csv", pd.concat(global_hamming_frames, ignore_index=True))
    else:
        save_csv("global_hamming_summary.csv", pd.DataFrame())

    manifest = {
        "complete": complete,
        "deadline_message": deadline_message,
        "elapsed_seconds": guard.elapsed,
        "requested_time_limit_seconds": args.time_limit,
        "analysis_seed": args.analysis_seed,
        "grad_samples_per_condition": args.grad_samples,
        "global_samples_per_condition": args.global_samples,
        "hessian_minima_per_condition": args.hessian_minima,
        "q1_sizes": list(Q1_SIZES),
        "q2_sizes": list(Q2_SIZES),
        "q3_sizes": list(Q3_SIZES),
        "q2_gamma": GAMMA,
        "endpoint_energy_abs_tol": ENDPOINT_ENERGY_ABS_TOL,
        "endpoint_energy_rel_tol": ENDPOINT_ENERGY_REL_TOL,
        "endpoint_torus_rms_tol_radians": ENDPOINT_TORUS_RMS_TOL,
        "no_optimizer_reruns": True,
        "optimizer_modules_called": [],
        "warnings": warnings,
        "files": [
            "landscape_summary.csv",
            "minima.csv",
            "basins.csv",
            "archived_endpoints.csv",
            "degeneracy_levels.csv",
            "hessian_spectra.csv",
            "hessian_summary.csv",
            "coupling_edges.csv",
            "coupling_summary.csv",
            "gradient_samples.csv",
            "global_parameter_samples.csv",
            "global_hamming_summary.csv",
        ],
        "metric_interpretation": {
            "multimodality_Q1": "exact count and energy distribution of one-spin-flip minima",
            "multimodality_Q2_Q3": "number of distinct archived max-budget endpoint clusters in energy + periodic parameter space",
            "basin_Q1": "primary: archived max-budget BFGS final-energy frequencies mapped to exact one-flip minima; geometric exact steepest one-flip basin sizes are also saved",
            "basin_Q2_Q3": "primary: archived max-budget BFGS endpoint frequencies over basins discovered from all archived optimizer endpoints; no BFGS is rerun",
            "conditioning": "active Hessian eigenvalue condition number at a few lowest minima/basin representatives",
            "separability": "off-diagonal Hessian Frobenius ratio and thresholded coupling graph at a deterministic random parameter probe",
            "degeneracy_Q1": "exact multiplicity of one-flip-minimum energy levels",
            "degeneracy_Q2_Q3": "empirical multiplicity of distinct parameter basins sharing an endpoint energy level",
            "global_structure_hamming": "computational-basis diagonal-energy statistics vs Hamming distance from the diagonal ground configuration",
            "global_structure_parameter": "objective energy vs periodic RMS parameter distance from saved/best low-energy anchor",
            "trainability": "full exact parameter-shift gradient-norm distribution (analytic for Q1) over random parameters",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=jsonable))

    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast post-hoc landscape characterization; never reruns optimization."
    )
    p.add_argument(
        "--q1-results",
        type=Path,
        default=Path("spin_glass_benchmark_results"),
        help="Q1 benchmark results root",
    )
    p.add_argument(
        "--q2-results",
        type=Path,
        default=Path("tf_spin_glass_benchmark_results"),
        help="Q2 benchmark results root",
    )
    p.add_argument(
        "--q3-results",
        type=Path,
        default=Path("heisenberg_variational_benchmark_results"),
        help="Q3 benchmark results root",
    )
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to heisenberg_model_adapter.py (auto-detected if omitted)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("landscape_characterization_results"),
        help="Output directory",
    )
    p.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT_S,
        help="Hard analysis wall-time guard in seconds (default 540)",
    )
    p.add_argument(
        "--analysis-seed",
        type=int,
        default=DEFAULT_ANALYSIS_SEED,
        help="Seed only for diagnostic random parameter probes; never used to regenerate disorder",
    )
    p.add_argument(
        "--grad-samples",
        type=int,
        default=DEFAULT_GRAD_SAMPLES,
        help="Random parameter points per condition for gradient-norm distribution",
    )
    p.add_argument(
        "--global-samples",
        type=int,
        default=DEFAULT_GLOBAL_SAMPLES,
        help="Random parameter points per condition for energy-distance structure",
    )
    p.add_argument(
        "--hessian-minima",
        type=int,
        default=DEFAULT_HESSIAN_MINIMA,
        help="Number of low-energy minima/basin representatives per condition for Hessian spectra",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive")
    if args.grad_samples < 1 or args.global_samples < 1 or args.hessian_minima < 1:
        raise ValueError("sample counts and --hessian-minima must be >= 1")

    try:
        manifest = analyze(args)
    except Exception as exc:
        # A setup error (e.g. missing original saved instance) should be explicit;
        # silently regenerating an instance would violate the paired design.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "output_directory": str(args.out),
                "warnings": manifest["warnings"],
            },
            indent=2,
        )
    )
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
