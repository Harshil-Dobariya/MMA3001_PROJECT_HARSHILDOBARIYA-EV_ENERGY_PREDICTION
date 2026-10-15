"""Performance analysis: baseline vs improved code, timing, FLOPs and memory.

The physics model's inner calculation (wheel power and its integration over
a drive cycle, for every vehicle) is the numerical kernel of the project. It
is implemented three ways so their speed, memory use and results can be
compared:

1. :func:`energy_terms_loop`: **baseline**. Pure Python loops over vehicles
   and time steps, one floating-point operation at a time.
2. :func:`energy_terms_per_vehicle`: NumPy vectorised over time, but still a
   Python loop over vehicles.
3. :func:`ev_energy.physics_model.cycle_energy_terms`: **improved**. Fully
   vectorised over vehicles and time with NumPy broadcasting.

All three implement the same equations (central-difference acceleration,
``P_w = (m a + f0 + f1 v + f2 v^2) v``, trapezoidal integration of the
positive and negative parts), so they must give the same answer.

FLOP count per sample (one vehicle, one time step)
--------------------------------------------------
Wheel power ``P = (m a + f0 + f1 v + f2 v^2) v``. The squared speed ``v^2``
depends only on time, so it is computed once per time step and shared by
every vehicle; it is not counted per vehicle. Per vehicle and time step:

* ``m a``: 1 multiply
* ``f1 v``: 1 multiply
* ``f2 v^2``: 1 multiply
* three additions to form the force: 3 adds
* force times ``v``: 1 multiply

giving 7 FLOPs. Splitting into ``max(P, 0)`` and ``max(-P, 0)`` adds one
negation (1 FLOP; comparisons are not counted as FLOPs). The trapezoidal
sums add each part to a running total: 2 adds. Total
:data:`FLOPS_PER_SAMPLE` = 10. (The acceleration and the end-point terms
are computed once per cycle, not per vehicle, so they are negligible.)

Memory traffic of the NumPy version
-----------------------------------
Every NumPy operation writes a new temporary array the full size of the
problem (n_vehicles x n_time), then the next operation reads it back. The
full-size arrays written are: ``m*a``, ``+f0``, ``f1*v``, ``f2*v**2``, two
further sums, ``*v`` (7 for P), then ``-P``, ``max(P,0)`` and ``max(-P,0)``
(3 more): 10 writes. The full-size arrays read are: one for ``+f0``, two
for each of the two sums, one for ``*v``, one each for the two maxima and
the negation, and one for each of the two trapezoidal sums: 11 reads. That
is :data:`NUMPY_ARRAY_PASSES` = 21 passes of 8 bytes per element.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Callable

import numpy as np

from ev_energy import physics_model as pm

#: Floating-point operations per vehicle per time sample (see module notes).
FLOPS_PER_SAMPLE = 10

#: Bytes per double-precision number.
BYTES_PER_DOUBLE = 8

#: Full-size array reads plus writes made by the vectorised NumPy kernel per
#: cycle (10 writes + 11 reads, see module notes).
NUMPY_ARRAY_PASSES = 21


# ---------------------------------------------------------------------------
# Three implementations of the same kernel
# ---------------------------------------------------------------------------


def energy_terms_loop(
    cycle: pm.DriveCycle, mass_kg, f0, f1, f2
) -> dict[str, np.ndarray]:
    """Baseline: compute E+ and E- with pure Python loops.

    Parameters
    ----------
    cycle : DriveCycle
        Drive cycle.
    mass_kg, f0, f1, f2 : array-like
        Vehicle parameters, one value per vehicle.

    Returns
    -------
    dict of str to numpy.ndarray
        ``e_pos_j`` and ``e_neg_j`` per vehicle [J].
    """
    v = [float(x) for x in cycle.speed_mps]
    dt = cycle.dt
    n = len(v)
    # Acceleration: central differences inside, one-sided at the ends.
    a = [0.0] * n
    a[0] = (v[1] - v[0]) / dt
    a[-1] = (v[-1] - v[-2]) / dt
    for i in range(1, n - 1):
        a[i] = (v[i + 1] - v[i - 1]) / (2.0 * dt)

    e_pos, e_neg = [], []
    for m, c0, c1, c2 in zip(mass_kg, f0, f1, f2):
        pos_sum = neg_sum = 0.0
        pos_ends = neg_ends = 0.0
        for i in range(n):
            p = (m * a[i] + c0 + c1 * v[i] + c2 * v[i] * v[i]) * v[i]
            pos = p if p > 0.0 else 0.0
            neg = -p if p < 0.0 else 0.0
            if i == 0 or i == n - 1:
                pos_ends += pos
                neg_ends += neg
            else:
                pos_sum += pos
                neg_sum += neg
        e_pos.append(dt * (pos_sum + 0.5 * pos_ends))
        e_neg.append(dt * (neg_sum + 0.5 * neg_ends))
    return {"e_pos_j": np.array(e_pos), "e_neg_j": np.array(e_neg)}


def energy_terms_per_vehicle(
    cycle: pm.DriveCycle, mass_kg, f0, f1, f2
) -> dict[str, np.ndarray]:
    """Intermediate: NumPy over time, Python loop over vehicles.

    Parameters and returns as for :func:`energy_terms_loop`.
    """
    v = cycle.speed_mps
    a = pm.finite_difference_acceleration(v, cycle.dt)
    e_pos, e_neg = [], []
    for m, c0, c1, c2 in zip(mass_kg, f0, f1, f2):
        p = (m * a + c0 + c1 * v + c2 * v**2) * v
        e_pos.append(float(pm.trapezoid(np.maximum(p, 0.0), cycle.dt)))
        e_neg.append(float(pm.trapezoid(np.maximum(-p, 0.0), cycle.dt)))
    return {"e_pos_j": np.array(e_pos), "e_neg_j": np.array(e_neg)}


def energy_terms_vectorised(
    cycle: pm.DriveCycle, mass_kg, f0, f1, f2
) -> dict[str, np.ndarray]:
    """Improved: fully vectorised (the version used by the project).

    A thin wrapper around :func:`ev_energy.physics_model.cycle_energy_terms`
    with the same signature as the other two implementations.
    """
    terms = pm.cycle_energy_terms(cycle, mass_kg, f0, f1, f2)
    return {"e_pos_j": terms["e_pos_j"], "e_neg_j": terms["e_neg_j"]}


#: The three implementations, in order from baseline to improved.
IMPLEMENTATIONS: dict[str, Callable] = {
    "python_loops": energy_terms_loop,
    "numpy_per_vehicle": energy_terms_per_vehicle,
    "numpy_vectorised": energy_terms_vectorised,
}


# ---------------------------------------------------------------------------
# Measurement tools
# ---------------------------------------------------------------------------


def time_function(func: Callable, repeats: int = 5) -> dict[str, float]:
    """Time a function several times, like ``%timeit``.

    The minimum is the best estimate of the code's own cost, because every
    other run was slowed by something else happening on the computer. The
    median and spread show how much the timing varies.

    Parameters
    ----------
    func : callable
        Function with no arguments.
    repeats : int, optional
        Number of timed runs, at least 1.

    Returns
    -------
    dict of str to float
        ``min_s``, ``median_s`` and ``max_s`` in seconds.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return {"min_s": min(times), "median_s": float(np.median(times)), "max_s": max(times)}


def peak_memory_mb(func: Callable) -> float:
    """Return the peak memory allocated while running ``func``, in MB.

    Uses Python's ``tracemalloc``, which records memory allocated by Python
    and NumPy while the function runs.

    Parameters
    ----------
    func : callable
        Function with no arguments.

    Returns
    -------
    float
        Peak additional memory in megabytes (10^6 bytes).
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def kernel_flops(n_vehicles: int, n_samples: int) -> int:
    """Total floating-point operations for one cycle (see module notes).

    Examples
    --------
    >>> kernel_flops(712, 1370)
    9754400
    """
    return FLOPS_PER_SAMPLE * n_vehicles * n_samples


def numpy_bytes_moved(n_vehicles: int, n_samples: int) -> int:
    """Estimated memory traffic of the vectorised NumPy kernel, in bytes.

    Examples
    --------
    >>> numpy_bytes_moved(1, 1)
    168
    """
    return NUMPY_ARRAY_PASSES * BYTES_PER_DOUBLE * n_vehicles * n_samples


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte of memory traffic (Week 2.1, equation 2.5).

    Examples
    --------
    >>> round(arithmetic_intensity(10, 168), 4)
    0.0595
    """
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive.")
    return flops / bytes_moved


def fused_kernel_arithmetic_intensity() -> float:
    """Arithmetic intensity if the kernel were fused (one pass, no temporaries).

    A compiled loop would keep each vehicle's four parameters and the running
    sums in registers and stream only ``v`` and ``a`` (2 doubles = 16 bytes
    per sample). Because ``v`` and ``a`` are the same for every vehicle, they
    would mostly come from cache after the first vehicle, so this is a
    pessimistic (lowest) figure for the fused version.

    Returns
    -------
    float
        FLOPs per byte.

    Examples
    --------
    >>> round(fused_kernel_arithmetic_intensity(), 4)
    0.625
    """
    return FLOPS_PER_SAMPLE / (2 * BYTES_PER_DOUBLE)


def tile_vehicles(mass_kg, f0, f1, f2, n: int) -> tuple[np.ndarray, ...]:
    """Repeat the vehicle parameters to reach ``n`` vehicles (for scaling tests)."""
    arrays = [np.asarray(x, dtype=float) for x in (mass_kg, f0, f1, f2)]
    reps = int(np.ceil(n / len(arrays[0])))
    return tuple(np.tile(x, reps)[:n] for x in arrays)
