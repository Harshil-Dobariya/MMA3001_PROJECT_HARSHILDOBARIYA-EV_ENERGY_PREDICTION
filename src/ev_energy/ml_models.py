"""Machine-learning and hybrid models of EV energy consumption.

Model families
--------------
All use the same design inputs (see :data:`NUMERIC_FEATURES` and
:data:`CATEGORICAL_FEATURES`) and predict one cycle at a time.

* ``mean_baseline``: always predicts the training mean (``DummyRegressor``).
  Any useful model must beat this.
* ``linear``: ordinary least-squares linear regression.
* ``tree``: decision-tree regression.
* ``svr``: support-vector regression with an RBF kernel.
* ``mlp``: multilayer-perceptron neural network.
* ``physics``: the Stage 2 physics model (three parameters, fitted jointly to
  both cycles), from :mod:`ev_energy.physics_model`.
* ``hybrid_<family>``: the physics prediction plus an ML model of the physics
  residual. The physics model captures the known energy balance; the ML
  model learns what physics misses (for example efficiency that varies
  between vehicles).

Every numerical input is standardised inside a ``Pipeline``, so the scaling is
learned from training data only, as in the Week 5 notes.

Leakage control
---------------
Cross-validation folds are built by :func:`ev_energy.splits.group_kfold_indices`
on ``group_id``, so a vehicle model is never in the training and validation
folds at the same time, and the folds are identical on every computer. In
:func:`cross_validate_models` the physics model is refitted inside every fold
using only that fold's training rows, so hybrid models never see validation
targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ev_energy.physics_model import PhysicsModel
from ev_energy.splits import RANDOM_STATE, group_kfold_indices

#: Continuous design inputs (all SI units).
NUMERIC_FEATURES = ["etw_kg", "rated_hp", "f0_n", "f1_n_per_mps", "f2_n_per_mps2"]

#: Categorical design inputs, one-hot encoded.
CATEGORICAL_FEATURES = ["drive", "vehicle_type"]

#: The two prediction targets.
CYCLES = ("city", "hwy")

#: Number of cross-validation folds.
N_FOLDS = 5

#: ML model families that can be tuned and used in hybrids.
ML_FAMILIES = ("linear", "tree", "svr", "mlp")

#: Hyperparameter grids searched for each family (keys use pipeline names).
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "linear": {},
    "tree": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, None],
        "model__min_samples_leaf": [1, 3, 5, 10],
    },
    "svr": {
        "model__C": [1.0, 10.0, 100.0],
        "model__epsilon": [0.1, 0.5, 1.0],
        "model__gamma": ["scale", 0.05, 0.1, 0.3],
    },
    "mlp": {
        "model__regressor__hidden_layer_sizes": [(10,), (30,), (30, 15)],
        "model__regressor__alpha": [1e-3, 1e-2, 1e-1],
    },
}


# ---------------------------------------------------------------------------
# Building models
# ---------------------------------------------------------------------------


def make_preprocessor() -> ColumnTransformer:
    """Standardise numeric inputs and one-hot encode categorical inputs.

    Categories not seen during training are encoded as all zeros rather than
    raising an error, so an unusual drivetrain in new data is handled.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessing step.
    """
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_estimator(family: str):
    """Return an unfitted estimator for one model family.

    Parameters
    ----------
    family : str
        One of ``"mean_baseline"``, ``"linear"``, ``"tree"``, ``"svr"`` or
        ``"mlp"``.

    Returns
    -------
    sklearn estimator
        The bare estimator (without preprocessing).

    Raises
    ------
    ValueError
        If ``family`` is unknown.
    """
    if family == "mean_baseline":
        return DummyRegressor(strategy="mean")
    if family == "linear":
        return LinearRegression()
    if family == "tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE)
    if family == "svr":
        return SVR(kernel="rbf")
    if family == "mlp":
        # The target is also standardised, which helps the optimiser
        # converge; predictions are converted back to kWh/100 mi.
        return TransformedTargetRegressor(
            regressor=MLPRegressor(
                solver="adam",
                activation="relu",
                early_stopping=True,
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            transformer=StandardScaler(),
        )
    raise ValueError(f"Unknown model family {family!r}.")


def make_pipeline(family: str, params: dict | None = None) -> Pipeline:
    """Build a preprocessing-plus-model pipeline.

    Parameters
    ----------
    family : str
        Model family, see :func:`make_estimator`.
    params : dict, optional
        Hyperparameters in pipeline notation (e.g.
        ``{"model__max_depth": 5}``), typically from :func:`tune`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline with steps ``"prep"`` and ``"model"``.
    """
    pipe = Pipeline([("prep", make_preprocessor()), ("model", make_estimator(family))])
    if params:
        pipe.set_params(**params)
    return pipe


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, R^2 and maximum absolute error.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Measured and predicted values.

    Returns
    -------
    dict of str to float
        ``rmse``, ``mae``, ``r2`` and ``max_abs_error`` (same units as y,
        except R^2 which is dimensionless).

    Examples
    --------
    >>> m = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    >>> round(m["rmse"], 4), m["mae"], m["max_abs_error"]
    (1.1547, 0.6666666666666666, 2.0)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
    }


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def physics_predictions(model: PhysicsModel, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the physics prediction for each cycle."""
    return {c: model.predict(df, c) for c in CYCLES}


def tune(
    train: pd.DataFrame,
    family: str,
    cycle: str,
    residual: bool = False,
    n_folds: int = N_FOLDS,
    n_jobs: int | None = 1,
) -> dict:
    """Choose hyperparameters by grouped cross-validation on training data.

    Parameters
    ----------
    train : pandas.DataFrame
        Training vehicles with design inputs, targets and physics features.
    family : str
        Model family in :data:`ML_FAMILIES`.
    cycle : str
        ``"city"`` or ``"hwy"``.
    residual : bool, optional
        If True, tune the model as the residual part of a hybrid, i.e. on
        measured minus physics prediction. The physics model for this is
        fitted on the whole training set. Only the hyperparameters are taken
        from this step; :func:`cross_validate_models` then refits physics
        inside every fold for the actual comparison.
    n_folds : int, optional
        Number of grouped folds.
    n_jobs : int or None, optional
        CPU cores used to fit grid candidates in parallel. ``1`` runs them
        one after another; ``-1`` uses every core. The result is the same
        either way; only the run time changes (compared in Stage 5).

    Returns
    -------
    dict
        Best hyperparameters (empty for ``"linear"``), in pipeline notation.
    """
    grid = PARAM_GRIDS[family]
    if not grid:
        return {}
    y = train[f"{cycle}_kwh_per_100mi"].to_numpy()
    if residual:
        y = y - PhysicsModel().fit(train).predict(train, cycle)
    search = GridSearchCV(
        make_pipeline(family),
        grid,
        cv=group_kfold_indices(train, n_folds),
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    search.fit(train, y)
    return dict(search.best_params_)


# ---------------------------------------------------------------------------
# Cross-validated comparison
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Out-of-fold predictions and timings from :func:`cross_validate_models`.

    Attributes
    ----------
    predictions : pandas.DataFrame
        One row per training vehicle, with its fold number and one prediction
        column per model and cycle (``"<model>__<cycle>"``).
    fit_seconds : dict
        Total fitting time per model and cycle, summed over folds.
    """

    predictions: pd.DataFrame
    fit_seconds: dict = field(default_factory=dict)

    def metrics(self) -> pd.DataFrame:
        """Return RMSE, MAE, R^2 and max error for every model and cycle."""
        rows = []
        for col in self.predictions.columns:
            if "__" not in col:
                continue
            model, cycle = col.split("__")
            y = self.predictions[f"{cycle}_kwh_per_100mi"]
            rows.append(
                {
                    "model": model,
                    "cycle": cycle,
                    **regression_metrics(y, self.predictions[col]),
                    "fit_seconds": self.fit_seconds.get(col, np.nan),
                }
            )
        return pd.DataFrame(rows)


def cross_validate_models(
    train: pd.DataFrame,
    best_params: dict[str, dict[str, dict]],
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> CVResult:
    """Compare all models with the same grouped folds of the training set.

    For each fold, the physics model is fitted on that fold's training rows
    only. Each ML model is then fitted directly to the measured target, and
    each hybrid is fitted to the physics residual. All predictions are made
    on the held-out fold, so every training vehicle receives exactly one
    out-of-fold prediction from every model.

    Parameters
    ----------
    train : pandas.DataFrame
        Training vehicles (never the test set).
    best_params : dict
        ``best_params[name][cycle]`` gives hyperparameters for model ``name``
        (an ML family or ``"hybrid_<family>"``) on that cycle.
    n_folds : int, optional
        Number of grouped folds.
    random_state : int, optional
        Seed for assigning vehicle models to folds. Changing it gives a
        different but equally valid set of folds (see
        :func:`repeated_cv_metrics`).

    Returns
    -------
    CVResult
        Out-of-fold predictions and fit times.
    """
    out = train[["group_id", "city_kwh_per_100mi", "hwy_kwh_per_100mi"]].copy()
    out["fold"] = -1
    fit_seconds: dict[str, float] = {}

    folds = group_kfold_indices(train, n_folds, random_state)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        out.loc[out.index[va_idx], "fold"] = fold

        start = time.perf_counter()
        physics = PhysicsModel().fit(tr)
        physics_time = time.perf_counter() - start
        phys_tr, phys_va = physics_predictions(physics, tr), physics_predictions(physics, va)

        for cycle in CYCLES:
            y_tr = tr[f"{cycle}_kwh_per_100mi"].to_numpy()
            _store(out, va_idx, f"physics__{cycle}", phys_va[cycle])
            _add_time(fit_seconds, f"physics__{cycle}", physics_time)

            for family in ("mean_baseline",) + ML_FAMILIES:
                params = best_params.get(family, {}).get(cycle, {})
                pred, seconds = _fit_predict(family, params, tr, y_tr, va)
                _store(out, va_idx, f"{family}__{cycle}", pred)
                _add_time(fit_seconds, f"{family}__{cycle}", seconds)

            for family in ML_FAMILIES:
                name = f"hybrid_{family}"
                params = best_params.get(name, {}).get(cycle, {})
                resid_pred, seconds = _fit_predict(family, params, tr, y_tr - phys_tr[cycle], va)
                _store(out, va_idx, f"{name}__{cycle}", phys_va[cycle] + resid_pred)
                _add_time(fit_seconds, f"{name}__{cycle}", seconds + physics_time)

    return CVResult(out, fit_seconds)


def repeated_cv_metrics(
    train: pd.DataFrame,
    best_params: dict[str, dict[str, dict]],
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_folds: int = N_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat grouped cross-validation with different fold assignments.

    A single cross-validation run depends on which vehicle models happen to
    share a fold. Repeating it with several seeds shows how much of the
    difference between models is real and how much is luck of the split.

    Parameters
    ----------
    train : pandas.DataFrame
        Training vehicles.
    best_params : dict
        Hyperparameters, as for :func:`cross_validate_models`.
    seeds : tuple of int, optional
        Fold-assignment seeds, one cross-validation run per seed.
    n_folds : int, optional
        Number of grouped folds per run.

    Returns
    -------
    per_run : pandas.DataFrame
        Metrics for every model, cycle and seed.
    summary : pandas.DataFrame
        Mean and standard deviation of RMSE over the runs, the number of
        runs each model had the lowest RMSE, sorted best first per cycle.
    """
    runs = []
    for seed in seeds:
        metrics = cross_validate_models(train, best_params, n_folds, seed).metrics()
        runs.append(metrics.assign(seed=seed))
    per_run = pd.concat(runs, ignore_index=True)
    winners = per_run.loc[per_run.groupby(["cycle", "seed"])["rmse"].idxmin()]
    wins = winners.groupby(["model", "cycle"]).size().rename("runs_won")
    summary = (
        per_run.groupby(["model", "cycle"])["rmse"]
        .agg(rmse_mean="mean", rmse_std="std")
        .join(wins)
        .fillna({"runs_won": 0})
        .reset_index()
        .sort_values(["cycle", "rmse_mean"])
        .reset_index(drop=True)
    )
    summary["runs_won"] = summary["runs_won"].astype(int)
    return per_run, summary


def _fit_predict(family, params, tr, y_tr, va):
    """Fit one pipeline and predict, returning predictions and fit time."""
    pipe = clone(make_pipeline(family, params))
    start = time.perf_counter()
    pipe.fit(tr, y_tr)
    seconds = time.perf_counter() - start
    return pipe.predict(va), seconds


def _store(out, idx, col, values):
    if col not in out.columns:
        out[col] = np.nan
    out.loc[out.index[idx], col] = values


def _add_time(store, key, seconds):
    store[key] = store.get(key, 0.0) + seconds
