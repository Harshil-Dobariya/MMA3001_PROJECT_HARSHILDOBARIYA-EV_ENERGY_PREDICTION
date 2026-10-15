"""Physics-based EV energy consumption model integrated over EPA drive cycles.

Model (agreed form)
-------------------
Longitudinal dynamics on a flat dynamometer:

    m a(t) = F_t(t) - F_RL(v(t)),   F_RL(v) = f0 + f1 v + f2 v**2

Wheel power:

    P_w(t) = F_t(t) v(t) = [m a(t) + f0 + f1 v(t) + f2 v(t)**2] v(t)

Wall energy over one cycle of duration T:

    E_b = E_pos / eta_d - eta_r * E_neg + P_aux * T

where

    E_pos = integral of max(P_w, 0) dt    (driving energy at the wheels)
    E_neg = integral of max(-P_w, 0) dt   (braking energy available)

and consumption is ``c = E_b / D`` with ``D = integral of v dt``.

Symbols
-------
m      : equivalent test weight [kg]
v, a   : speed [m/s] and acceleration [m/s^2] from the drive-cycle trace
f0     : constant road-load term [N]
f1     : linear road-load term [N/(m/s)]
f2     : quadratic road-load term [N/(m/s)^2]
eta_d  : wall-to-wheel efficiency when driving [-]
eta_r  : fraction of braking energy recovered [-]
P_aux  : constant accessory power [W]

Because ``c`` is linear in ``k1 = 1/eta_d``, ``k2 = eta_r`` and ``k3 = P_aux``,
the three parameters are fitted by bounded linear least squares
(:class:`PhysicsModel`).

Numerical methods
-----------------
* Acceleration: second-order central finite differences in the interior,
  first-order one-sided differences at the two end points
  (:func:`finite_difference_acceleration`).
* Integration: composite trapezoidal rule (:func:`trapezoid`) and composite
  Simpson's rule with a 3/8 correction for an odd number of intervals
  (:func:`simpson`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear

from ev_energy.data_cleaning import MPS_PER_MPH

#: Joules per kWh.
J_PER_KWH = 3.6e6

#: Metres per 100 miles.
M_PER_100MI = 160_934.4

#: Conversion factor from J/m to kWh/100 mi.
J_PER_M_TO_KWH_PER_100MI = M_PER_100MI / J_PER_KWH

#: Supported integration methods.
INTEGRATORS = ("trapezoid", "simpson")

#: Vehicles processed together in :func:`cycle_energy_terms`. Chosen in
#: Stage 5 (scripts/stage5_performance.py). The best value depends on the
#: CPU cache: on an Intel Xeon with a 1 MB L2 cache, chunks of 32 vehicles
#: (about 350 kB per temporary array) were fastest; on the target machine
#: (Apple Silicon, much larger L2 cache) processing about 712 vehicles at
#: once (about 7.8 MB per array) was fastest. 712 is used: on the target
#: machine it matches the speed of processing everything at once, while
#: capping memory use as the dataset grows (at 7,120 vehicles, peak memory
#: is about 23 MB instead of 234 MB).
DEFAULT_CHUNK_SIZE = 712


# ---------------------------------------------------------------------------
# Drive cycles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriveCycle:
    """A speed-time trace sampled at a uniform time step.

    Attributes
    ----------
    name : str
        Short name, e.g. ``"city"`` or ``"hwy"``.
    time_s : numpy.ndarray
        Time stamps in seconds, uniformly spaced, starting at 0.
    speed_mps : numpy.ndarray
        Target speed in m/s at each time stamp.
    """

    name: str
    time_s: np.ndarray
    speed_mps: np.ndarray

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        return float(self.time_s[1] - self.time_s[0])

    @property
    def duration_s(self) -> float:
        """Total cycle duration in seconds."""
        return float(self.time_s[-1] - self.time_s[0])

    def subsample(self, step: int) -> "DriveCycle":
        """Return the cycle keeping every ``step``-th sample.

        Used for the step-size convergence study. The last sample is always
        kept so the cycle duration is unchanged where possible.

        Parameters
        ----------
        step : int
            Keep one sample in every ``step``. Must be at least 1.

        Returns
        -------
        DriveCycle
            Coarser cycle with time step ``step * dt``.
        """
        if step < 1:
            raise ValueError("step must be at least 1.")
        idx = np.arange(0, len(self.time_s), step)
        return DriveCycle(self.name, self.time_s[idx], self.speed_mps[idx])

    def refine(self, factor: int, kind: str = "linear") -> "DriveCycle":
        """Return the cycle resampled at ``dt / factor`` by interpolation.

        Used for the step-size convergence study. EPA only publishes the
        trace at 1 s, so values between samples must be assumed. Two
        interpolants are offered so their effect can be compared:

        * ``"linear"``: straight lines between samples.
        * ``"pchip"``: shape-preserving cubic (PCHIP). Unlike a standard
          cubic spline it cannot overshoot, so speed never goes negative.

        Parameters
        ----------
        factor : int
            Refinement factor, at least 1. ``factor=4`` gives 0.25 s steps.
        kind : {"linear", "pchip"}
            Interpolant used between the original samples.

        Returns
        -------
        DriveCycle
            Finer cycle covering the same time span.
        """
        if factor < 1:
            raise ValueError("factor must be at least 1.")
        n_new = (len(self.time_s) - 1) * factor + 1
        t_new = np.linspace(self.time_s[0], self.time_s[-1], n_new)
        if kind == "linear":
            v_new = np.interp(t_new, self.time_s, self.speed_mps)
        elif kind == "pchip":
            v_new = PchipInterpolator(self.time_s, self.speed_mps)(t_new)
        else:
            raise ValueError("kind must be 'linear' or 'pchip'.")
        return DriveCycle(self.name, t_new, v_new)


def load_drive_cycle(path: str | Path, name: str) -> DriveCycle:
    """Load an EPA drive-cycle text file (time in s, speed in mph).

    The EPA files have two header lines followed by tab-separated columns.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the EPA ``.txt`` file.
    name : str
        Short name for the cycle.

    Returns
    -------
    DriveCycle
        Cycle with speed converted to m/s.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the time step is not uniform, or any speed is negative.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Drive cycle file not found: {path}")
    data = np.loadtxt(path, skiprows=2)
    time_s, speed_mph = data[:, 0], data[:, 1]
    steps = np.diff(time_s)
    if not np.allclose(steps, steps[0]):
        raise ValueError(f"{path.name}: time step is not uniform.")
    if np.any(speed_mph < 0):
        raise ValueError(f"{path.name}: negative speeds found.")
    return DriveCycle(name, time_s, speed_mph * MPS_PER_MPH)


def load_epa_cycles(raw_dir: str | Path = "data/raw") -> dict[str, DriveCycle]:
    """Load the UDDS (city) and HWFET (highway) cycles.

    Parameters
    ----------
    raw_dir : str or pathlib.Path, optional
        Folder holding ``udds.txt`` and ``hwfet.txt``.

    Returns
    -------
    dict of str to DriveCycle
        Keys ``"city"`` and ``"hwy"``.
    """
    raw_dir = Path(raw_dir)
    return {
        "city": load_drive_cycle(raw_dir / "udds.txt", "city"),
        "hwy": load_drive_cycle(raw_dir / "hwfet.txt", "hwy"),
    }


# ---------------------------------------------------------------------------
# Numerical methods
# ---------------------------------------------------------------------------


def finite_difference_acceleration(speed: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate a uniformly sampled speed trace.

    Interior points use the second-order central difference
    ``a_i = (v_{i+1} - v_{i-1}) / (2 dt)``. The first and last points use
    first-order forward and backward differences.

    Parameters
    ----------
    speed : numpy.ndarray
        Speed samples, shape ``(n,)`` with ``n >= 2``.
    dt : float
        Time step.

    Returns
    -------
    numpy.ndarray
        Acceleration at every sample, shape ``(n,)``.

    Examples
    --------
    >>> finite_difference_acceleration(np.array([0.0, 1.0, 4.0, 9.0]), 1.0)
    array([1., 2., 4., 5.])
    """
    speed = np.asarray(speed, dtype=float)
    if speed.size < 2:
        raise ValueError("Need at least two samples to differentiate.")
    accel = np.empty_like(speed)
    accel[1:-1] = (speed[2:] - speed[:-2]) / (2.0 * dt)
    accel[0] = (speed[1] - speed[0]) / dt
    accel[-1] = (speed[-1] - speed[-2]) / dt
    return accel


def trapezoid(y: np.ndarray, dt: float) -> np.ndarray:
    """Composite trapezoidal rule along the last axis.

    ``I = dt * (y_0 / 2 + y_1 + ... + y_{n-2} + y_{n-1} / 2)``

    Parameters
    ----------
    y : numpy.ndarray
        Samples at uniform spacing, shape ``(..., n)``.
    dt : float
        Sample spacing.

    Returns
    -------
    numpy.ndarray
        Integral along the last axis.

    Examples
    --------
    >>> float(trapezoid(np.array([0.0, 1.0, 2.0]), 1.0))
    2.0
    """
    y = np.asarray(y, dtype=float)
    return dt * (y[..., 1:-1].sum(axis=-1) + 0.5 * (y[..., 0] + y[..., -1]))


def _simpson_13(y: np.ndarray, dt: float) -> np.ndarray:
    """Composite Simpson's 1/3 rule for an even number of intervals."""
    return dt / 3.0 * (
        y[..., 0] + y[..., -1]
        + 4.0 * y[..., 1:-1:2].sum(axis=-1)
        + 2.0 * y[..., 2:-1:2].sum(axis=-1)
    )


def _simpson_38(y: np.ndarray, dt: float) -> np.ndarray:
    """Simpson's 3/8 rule over exactly three intervals (four points)."""
    return 3.0 * dt / 8.0 * (y[..., 0] + 3.0 * y[..., 1] + 3.0 * y[..., 2] + y[..., 3])


def simpson(y: np.ndarray, dt: float) -> np.ndarray:
    """Composite Simpson's rule along the last axis.

    With an even number of intervals the 1/3 rule is used throughout. With
    an odd number (as in both EPA cycles: 1369 and 765 intervals), the 1/3
    rule covers all but the last three intervals, and the 3/8 rule covers
    those three. Both rules are fourth-order accurate for smooth integrands.

    Parameters
    ----------
    y : numpy.ndarray
        Samples at uniform spacing, shape ``(..., n)`` with ``n >= 3``.
    dt : float
        Sample spacing.

    Returns
    -------
    numpy.ndarray
        Integral along the last axis.

    Examples
    --------
    >>> x = np.linspace(0.0, 1.0, 5)       # 4 intervals, even
    >>> round(float(simpson(x**3, 0.25)), 12)
    0.25
    >>> x = np.linspace(0.0, 1.0, 6)       # 5 intervals, odd
    >>> round(float(simpson(x**3, 0.2)), 12)
    0.25
    """
    y = np.asarray(y, dtype=float)
    n_intervals = y.shape[-1] - 1
    if n_intervals < 2:
        raise ValueError("Simpson's rule needs at least two intervals.")
    if n_intervals % 2 == 0:
        return _simpson_13(y, dt)
    if n_intervals == 3:
        return _simpson_38(y, dt)
    return _simpson_13(y[..., :-3], dt) + _simpson_38(y[..., -4:], dt)


def integrate(y: np.ndarray, dt: float, method: str = "trapezoid") -> np.ndarray:
    """Integrate uniformly sampled data with the chosen rule.

    Parameters
    ----------
    y : numpy.ndarray
        Samples, shape ``(..., n)``.
    dt : float
        Sample spacing.
    method : {"trapezoid", "simpson"}
        Integration rule.

    Returns
    -------
    numpy.ndarray
        Integral along the last axis.
    """
    if method == "trapezoid":
        return trapezoid(y, dt)
    if method == "simpson":
        return simpson(y, dt)
    raise ValueError(f"Unknown method {method!r}; choose from {INTEGRATORS}.")


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def wheel_power(
    speed: np.ndarray,
    accel: np.ndarray,
    mass_kg: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
) -> np.ndarray:
    """Wheel power ``P_w = (m a + f0 + f1 v + f2 v^2) v`` for many vehicles.

    Parameters
    ----------
    speed, accel : numpy.ndarray
        Cycle speed [m/s] and acceleration [m/s^2], shape ``(n_time,)``.
    mass_kg, f0, f1, f2 : numpy.ndarray
        Vehicle parameters, shape ``(n_vehicles,)``.

    Returns
    -------
    numpy.ndarray
        Wheel power in W, shape ``(n_vehicles, n_time)``.
    """
    v = np.asarray(speed, dtype=float)[None, :]
    a = np.asarray(accel, dtype=float)[None, :]
    m = np.asarray(mass_kg, dtype=float)[:, None]
    f0 = np.asarray(f0, dtype=float)[:, None]
    f1 = np.asarray(f1, dtype=float)[:, None]
    f2 = np.asarray(f2, dtype=float)[:, None]
    return (m * a + f0 + f1 * v + f2 * v**2) * v


def cycle_energy_terms(
    cycle: DriveCycle,
    mass_kg: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    method: str = "trapezoid",
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Compute the physics terms of the energy model for one cycle.

    Parameters
    ----------
    cycle : DriveCycle
        Drive cycle to integrate over.
    mass_kg, f0, f1, f2 : numpy.ndarray
        Vehicle parameters, shape ``(n_vehicles,)``.
    method : {"trapezoid", "simpson"}
        Integration rule.
    chunk_size : int or None, optional
        Number of vehicles processed together. ``None`` processes all at
        once, which makes temporary arrays (and memory use) grow with the
        number of vehicles. Any chunk size gives identical results; the
        fastest depends on the CPU cache. See :data:`DEFAULT_CHUNK_SIZE`.

    Returns
    -------
    dict of str to numpy.ndarray
        ``e_pos_j`` and ``e_neg_j`` (J, per vehicle), plus ``dist_m`` and
        ``duration_s`` (same for every vehicle, broadcast to the same shape).
    """
    mass_kg, f0, f1, f2 = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (mass_kg, f0, f1, f2))
    n = len(mass_kg)
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be at least 1, or None.")
    step = n if chunk_size is None else chunk_size
    accel = finite_difference_acceleration(cycle.speed_mps, cycle.dt)
    e_pos = np.empty(n)
    e_neg = np.empty(n)
    for start in range(0, n, max(step, 1)):
        sl = slice(start, start + step)
        p_w = wheel_power(cycle.speed_mps, accel, mass_kg[sl], f0[sl], f1[sl], f2[sl])
        e_pos[sl] = integrate(np.maximum(p_w, 0.0), cycle.dt, method)
        e_neg[sl] = integrate(np.maximum(-p_w, 0.0), cycle.dt, method)
    return {
        "e_pos_j": e_pos,
        "e_neg_j": e_neg,
        "dist_m": np.full(n, float(integrate(cycle.speed_mps, cycle.dt, method))),
        "duration_s": np.full(n, cycle.duration_s),
    }


def add_physics_features(
    vehicles: pd.DataFrame,
    cycles: dict[str, DriveCycle],
    method: str = "trapezoid",
) -> pd.DataFrame:
    """Add per-cycle physics terms as columns of the vehicle table.

    For each cycle name ``k`` in ``cycles`` the columns ``k_e_pos_j``,
    ``k_e_neg_j``, ``k_dist_m`` and ``k_duration_s`` are added.

    Parameters
    ----------
    vehicles : pandas.DataFrame
        Cleaned table with ``etw_kg``, ``f0_n``, ``f1_n_per_mps`` and
        ``f2_n_per_mps2``.
    cycles : dict of str to DriveCycle
        Typically from :func:`load_epa_cycles`.
    method : {"trapezoid", "simpson"}
        Integration rule.

    Returns
    -------
    pandas.DataFrame
        Copy of ``vehicles`` with the new columns.
    """
    out = vehicles.copy()
    for name, cycle in cycles.items():
        terms = cycle_energy_terms(
            cycle,
            out["etw_kg"].to_numpy(),
            out["f0_n"].to_numpy(),
            out["f1_n_per_mps"].to_numpy(),
            out["f2_n_per_mps2"].to_numpy(),
            method,
        )
        for key, values in terms.items():
            out[f"{name}_{key}"] = values
    return out


def consumption_kwh_per_100mi(
    e_pos_j: np.ndarray,
    e_neg_j: np.ndarray,
    duration_s: np.ndarray,
    dist_m: np.ndarray,
    eta_d: float,
    eta_r: float,
    p_aux_w: float,
) -> np.ndarray:
    """Evaluate ``c = (E_pos/eta_d - eta_r E_neg + P_aux T) / D`` in kWh/100 mi.

    Parameters
    ----------
    e_pos_j, e_neg_j : numpy.ndarray
        Positive and braking wheel energy over the cycle [J].
    duration_s : numpy.ndarray
        Cycle duration [s].
    dist_m : numpy.ndarray
        Cycle distance [m].
    eta_d : float
        Wall-to-wheel driving efficiency, in (0, 1].
    eta_r : float
        Fraction of braking energy recovered, in [0, 1].
    p_aux_w : float
        Accessory power [W].

    Returns
    -------
    numpy.ndarray
        Consumption in kWh/100 mi.

    Examples
    --------
    >>> # 10 MJ positive, no braking, no aux, perfect efficiency, 160.9344 km:
    >>> c = consumption_kwh_per_100mi(1e7, 0.0, 0.0, 160934.4, 1.0, 0.0, 0.0)
    >>> round(float(c), 4)
    2.7778
    """
    if not 0 < eta_d <= 1:
        raise ValueError("eta_d must be in (0, 1].")
    e_b = np.asarray(e_pos_j) / eta_d - eta_r * np.asarray(e_neg_j) + p_aux_w * np.asarray(duration_s)
    return e_b / np.asarray(dist_m) * J_PER_M_TO_KWH_PER_100MI


# ---------------------------------------------------------------------------
# Calibrated physics model
# ---------------------------------------------------------------------------


class PhysicsModel:
    """Physics model with three parameters fitted by bounded least squares.

    The model is fitted jointly on the city and highway results, so one set
    of ``eta_d``, ``eta_r`` and ``P_aux`` describes both cycles.

    Dividing the energy equation by distance gives, per vehicle and cycle,

        c = k1 (E_pos/D) - k2 (E_neg/D) + k3 (T/D)

    with ``k1 = 1/eta_d``, ``k2 = eta_r`` and ``k3 = P_aux``. The bounds
    ``k1 >= 1``, ``0 <= k2 <= 1`` and ``k3 >= 0`` keep every parameter
    physically meaningful.

    Parameters
    ----------
    cycles : tuple of str, optional
        Cycle name prefixes used in the feature columns.

    Attributes
    ----------
    eta_d_, eta_r_, p_aux_w_ : float
        Fitted parameters (available after :meth:`fit`).
    """

    def __init__(self, cycles: tuple[str, ...] = ("city", "hwy")):
        self.cycles = cycles

    def _design(self, df: pd.DataFrame, cycle: str) -> np.ndarray:
        dist = df[f"{cycle}_dist_m"].to_numpy()
        scale = J_PER_M_TO_KWH_PER_100MI
        return np.column_stack(
            [
                df[f"{cycle}_e_pos_j"].to_numpy() / dist * scale,
                -df[f"{cycle}_e_neg_j"].to_numpy() / dist * scale,
                df[f"{cycle}_duration_s"].to_numpy() / dist * scale,
            ]
        )

    def fit(self, df: pd.DataFrame) -> "PhysicsModel":
        """Fit the three parameters to measured consumption.

        Parameters
        ----------
        df : pandas.DataFrame
            Vehicles with physics features (from :func:`add_physics_features`)
            and measured ``city_kwh_per_100mi`` / ``hwy_kwh_per_100mi``.

        Returns
        -------
        PhysicsModel
            The fitted model (``self``).
        """
        X = np.vstack([self._design(df, c) for c in self.cycles])
        y = np.concatenate([df[f"{c}_kwh_per_100mi"].to_numpy() for c in self.cycles])
        result = lsq_linear(X, y, bounds=([1.0, 0.0, 0.0], [np.inf, 1.0, np.inf]))
        k1, k2, k3 = result.x
        self.eta_d_, self.eta_r_, self.p_aux_w_ = 1.0 / k1, k2, k3
        return self

    def predict(self, df: pd.DataFrame, cycle: str) -> np.ndarray:
        """Predict consumption in kWh/100 mi for one cycle.

        Parameters
        ----------
        df : pandas.DataFrame
            Vehicles with physics features.
        cycle : str
            ``"city"`` or ``"hwy"``.

        Returns
        -------
        numpy.ndarray
            Predicted consumption, kWh/100 mi.
        """
        if not hasattr(self, "eta_d_"):
            raise RuntimeError("Call fit() before predict().")
        k = np.array([1.0 / self.eta_d_, self.eta_r_, self.p_aux_w_])
        return self._design(df, cycle) @ k

    def params(self) -> dict[str, float]:
        """Return the fitted parameters as a dictionary."""
        return {"eta_d": self.eta_d_, "eta_r": self.eta_r_, "p_aux_w": self.p_aux_w_}
