"""Final validation: test-set evaluation, uncertainty, sensitivity and range.

This module is used once the model has been chosen (Stage 3). Nothing here
feeds back into model choice.

Contents
--------
* :func:`fit_and_predict`: fit every model on a training set and predict a
  test set, using the hyperparameters chosen by cross-validation.
* :func:`group_bootstrap`: confidence intervals for test RMSE, resampling
  whole vehicle models (not rows) because rows of one model are correlated.
* :func:`out_of_domain`: flag vehicles whose inputs fall outside the range
  seen in training, where data-driven models are extrapolating.
* :func:`sensitivity`: how predictions respond when one physical input is
  changed, as a check that models behave physically.
* :func:`label_mpge`, :func:`combined_consumption`, :func:`label_range_miles`:
  convert unadjusted test consumption to the figures on an EPA window
  sticker, so predictions can be compared with official values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from ev_energy import ml_models as ml
from ev_energy import physics_model as pm
from ev_energy.data_cleaning import KWH_PER_GALLON_EQUIVALENT
from ev_energy.splits import RANDOM_STATE

#: EPA's approximate adjustment from unadjusted (lab) to label values. Label
#: MPGe is about 0.7 times the unadjusted MPGe; equivalently label
#: consumption is about the unadjusted consumption divided by 0.7.
LABEL_ADJUSTMENT = 0.7

#: EPA combines city and highway as 55% city and 45% highway. Because
#: consumption is energy per distance, the weights apply to consumption
#: directly (equivalent to a harmonic average of MPGe).
CITY_WEIGHT = 0.55


# ---------------------------------------------------------------------------
# Loading Stage 3 choices
# ---------------------------------------------------------------------------


def load_best_params(path: str | Path) -> dict:
    """Load hyperparameters saved by Stage 3.

    JSON stores tuples as lists, so neural-network layer sizes are converted
    back to tuples.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``best_params.json``.

    Returns
    -------
    dict
        ``params[model][cycle]`` in pipeline notation.
    """
    with open(path) as f:
        params = json.load(f)
    for per_cycle in params.values():
        for p in per_cycle.values():
            for key, value in p.items():
                if key.endswith("hidden_layer_sizes") and isinstance(value, list):
                    p[key] = tuple(value)
    return params


# ---------------------------------------------------------------------------
# Fitting and predicting
# ---------------------------------------------------------------------------


def fit_and_predict(
    train: pd.DataFrame, test: pd.DataFrame, best_params: dict
) -> pd.DataFrame:
    """Fit every model on ``train`` and predict ``test``.

    The physics model is fitted jointly to both cycles of the training set.
    Hybrids learn the training residual of that physics model.

    Parameters
    ----------
    train, test : pandas.DataFrame
        Vehicles with design inputs, physics features and targets.
    best_params : dict
        Hyperparameters from Stage 3 (``params[model][cycle]``).

    Returns
    -------
    pandas.DataFrame
        ``test`` with one prediction column per model and cycle, named
        ``"<model>__<cycle>"``.
    """
    out = test.copy()
    physics = pm.PhysicsModel().fit(train)
    for cycle in ml.CYCLES:
        y_tr = train[f"{cycle}_kwh_per_100mi"].to_numpy()
        phys_tr = physics.predict(train, cycle)
        phys_te = physics.predict(test, cycle)
        out[f"physics__{cycle}"] = phys_te
        for family in ("mean_baseline",) + ml.ML_FAMILIES:
            pipe = clone(ml.make_pipeline(family, best_params.get(family, {}).get(cycle, {})))
            out[f"{family}__{cycle}"] = pipe.fit(train, y_tr).predict(test)
        for family in ml.ML_FAMILIES:
            name = f"hybrid_{family}"
            pipe = clone(ml.make_pipeline(family, best_params.get(name, {}).get(cycle, {})))
            out[f"{name}__{cycle}"] = phys_te + pipe.fit(train, y_tr - phys_tr).predict(test)
    return out


def metrics_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Return RMSE, MAE, R^2 and max error for every model and cycle."""
    rows = []
    for col in [c for c in preds.columns if "__" in c]:
        model, cycle = col.split("__")
        rows.append(
            {"model": model, "cycle": cycle,
             **ml.regression_metrics(preds[f"{cycle}_kwh_per_100mi"], preds[col])}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def group_bootstrap(
    preds: pd.DataFrame,
    models: list[str],
    cycle: str,
    n_boot: int = 2000,
    random_state: int = RANDOM_STATE,
    reference: str = "physics",
) -> pd.DataFrame:
    """Bootstrap confidence intervals for test RMSE by resampling models.

    Rows of the same vehicle model (e.g. the same car in several years) are
    not independent, so whole groups are resampled with replacement. For
    each resample, every model's RMSE and its difference from ``reference``
    are recorded. Using the same resample for all models makes the
    comparison paired.

    Parameters
    ----------
    preds : pandas.DataFrame
        Output of :func:`fit_and_predict` (needs ``group_id`` and targets).
    models : list of str
        Model names to evaluate.
    cycle : str
        ``"city"`` or ``"hwy"``.
    n_boot : int, optional
        Number of bootstrap resamples.
    random_state : int, optional
        Seed for reproducibility.
    reference : str, optional
        Model to compare against.

    Returns
    -------
    pandas.DataFrame
        Per model: RMSE on the full test set, 95% interval for RMSE, mean
        difference from the reference, its 95% interval, and the fraction of
        resamples in which the model beat the reference.
    """
    rng = np.random.RandomState(random_state)
    groups = preds["group_id"].to_numpy()
    unique = np.array(sorted(set(groups)))
    rows_of = {g: np.flatnonzero(groups == g) for g in unique}
    y = preds[f"{cycle}_kwh_per_100mi"].to_numpy()
    sq_err = {m: (preds[f"{m}__{cycle}"].to_numpy() - y) ** 2 for m in models + [reference]}

    boot = {m: np.empty(n_boot) for m in models + [reference]}
    for b in range(n_boot):
        idx = np.concatenate([rows_of[g] for g in rng.choice(unique, size=len(unique))])
        for m in boot:
            boot[m][b] = np.sqrt(sq_err[m][idx].mean())

    rows = []
    for m in models:
        diff = boot[m] - boot[reference]
        rows.append(
            {
                "model": m,
                "cycle": cycle,
                "rmse": float(np.sqrt(sq_err[m].mean())),
                "rmse_lo": float(np.percentile(boot[m], 2.5)),
                "rmse_hi": float(np.percentile(boot[m], 97.5)),
                f"diff_vs_{reference}": float(diff.mean()),
                "diff_lo": float(np.percentile(diff, 2.5)),
                "diff_hi": float(np.percentile(diff, 97.5)),
                f"p_better_than_{reference}": float((diff < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Input domain
# ---------------------------------------------------------------------------


def out_of_domain(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str] | None = None,
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> pd.Series:
    """Flag test vehicles with any input outside the training range.

    The range for each input is taken between the given quantiles of the
    training data, so a single extreme training vehicle does not stretch the
    range. A flagged vehicle is one where a data-driven model must
    extrapolate, and its prediction should be treated with caution.

    Parameters
    ----------
    train, test : pandas.DataFrame
        Training and test vehicles.
    columns : list of str, optional
        Numeric inputs to check; defaults to the model's numeric features.
    quantiles : tuple of float, optional
        Lower and upper training quantiles defining the supported range.

    Returns
    -------
    pandas.Series
        Boolean, True where the test vehicle is outside the supported range,
        indexed like ``test``.

    Examples
    --------
    >>> tr = pd.DataFrame({"x": np.arange(100.0)})
    >>> te = pd.DataFrame({"x": [50.0, 150.0]})
    >>> out_of_domain(tr, te, ["x"]).tolist()
    [False, True]
    """
    columns = columns or ml.NUMERIC_FEATURES
    flag = pd.Series(False, index=test.index)
    for col in columns:
        lo, hi = train[col].quantile(list(quantiles))
        flag |= (test[col] < lo) | (test[col] > hi)
    return flag


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def perturb(
    df: pd.DataFrame, column: str, factor: float, cycles: dict[str, pm.DriveCycle]
) -> pd.DataFrame:
    """Scale one design input and recompute the physics features.

    Parameters
    ----------
    df : pandas.DataFrame
        Vehicles with design inputs.
    column : str
        Input to scale, e.g. ``"etw_kg"`` or ``"f2_n_per_mps2"``.
    factor : float
        Multiplier, e.g. ``1.1`` for +10%.
    cycles : dict of str to DriveCycle
        Drive cycles for recomputing the physics terms.

    Returns
    -------
    pandas.DataFrame
        Copy with the input scaled and physics features recomputed, so both
        the physics model and the ML models see the change consistently.
    """
    out = df.copy()
    out[column] = out[column] * factor
    return pm.add_physics_features(out, cycles)


def sensitivity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    best_params: dict,
    cycles: dict[str, pm.DriveCycle],
    columns: tuple[str, ...] = ("etw_kg", "f2_n_per_mps2"),
    step: float = 0.10,
) -> pd.DataFrame:
    """Measure each model's response to a +/- ``step`` change in each input.

    Models are fitted once on ``train``. Each test vehicle's input is then
    scaled by ``1 - step`` and ``1 + step`` and the change in prediction is
    recorded as a central-difference elasticity:

        elasticity = (c(+step) - c(-step)) / c(0) / (2 * step)

    An elasticity of 0.5 means a 1% increase in the input raises predicted
    consumption by 0.5%. Physically, both heavier vehicles and higher drag
    must increase consumption (positive elasticity).

    Parameters
    ----------
    train, test : pandas.DataFrame
        Vehicles with physics features.
    best_params : dict
        Hyperparameters from Stage 3.
    cycles : dict of str to DriveCycle
        Drive cycles.
    columns : tuple of str, optional
        Inputs to perturb.
    step : float, optional
        Relative change, e.g. 0.10 for 10%.

    Returns
    -------
    pandas.DataFrame
        For every model, cycle and input: median elasticity over test
        vehicles and the fraction of vehicles with a physically wrong
        (negative) response.
    """
    base = fit_and_predict(train, test, best_params)
    pred_cols = [c for c in base.columns if "__" in c and not c.startswith("mean_baseline")]
    rows = []
    for column in columns:
        up = fit_and_predict(train, perturb(test, column, 1 + step, cycles), best_params)
        down = fit_and_predict(train, perturb(test, column, 1 - step, cycles), best_params)
        for col in pred_cols:
            model, cycle = col.split("__")
            elasticity = (up[col] - down[col]) / base[col] / (2 * step)
            rows.append(
                {
                    "model": model,
                    "cycle": cycle,
                    "input": column,
                    "median_elasticity": float(elasticity.median()),
                    "fraction_negative": float((elasticity < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Window-sticker conversions
# ---------------------------------------------------------------------------


def label_mpge(kwh_per_100mi: np.ndarray | float) -> np.ndarray | float:
    """Convert unadjusted consumption to an approximate label MPGe.

    ``label MPGe = 0.7 * 100 * 33.705 / c``

    Parameters
    ----------
    kwh_per_100mi : float or numpy.ndarray
        Unadjusted (lab) consumption in kWh/100 mi.

    Returns
    -------
    float or numpy.ndarray
        Approximate window-sticker MPGe.

    Examples
    --------
    >>> round(float(label_mpge(16.89)), 1)
    139.7
    """
    return LABEL_ADJUSTMENT * 100.0 * KWH_PER_GALLON_EQUIVALENT / np.asarray(kwh_per_100mi)


def combined_consumption(city: np.ndarray | float, hwy: np.ndarray | float) -> np.ndarray | float:
    """EPA combined consumption: 55% city, 45% highway.

    Examples
    --------
    >>> float(combined_consumption(20.0, 30.0))
    24.5
    """
    return CITY_WEIGHT * np.asarray(city) + (1 - CITY_WEIGHT) * np.asarray(hwy)


def label_range_miles(
    battery_kwh: float,
    combined_kwh_per_100mi: np.ndarray | float,
    charging_efficiency: float,
) -> np.ndarray | float:
    """Estimate EPA label range from battery size and lab consumption.

    The test consumption is measured as AC energy from the wall, which
    includes charging losses. The energy actually used for driving from a
    full battery is about ``battery_kwh``, so the wall-equivalent energy is
    ``battery_kwh / charging_efficiency``. The label adjustment then scales
    the lab range down:

        range = 0.7 * (battery_kwh / charging_efficiency) / (c / 100)

    Parameters
    ----------
    battery_kwh : float
        Battery capacity [kWh] (published nominal capacity).
    combined_kwh_per_100mi : float or numpy.ndarray
        Unadjusted combined consumption [kWh/100 mi].
    charging_efficiency : float
        Fraction of wall energy that reaches the battery, in (0, 1].

    Returns
    -------
    float or numpy.ndarray
        Estimated label range [miles].

    Examples
    --------
    >>> round(float(label_range_miles(77.4, 17.2, 0.9)), 0)
    350.0
    """
    if not 0 < charging_efficiency <= 1:
        raise ValueError("charging_efficiency must be in (0, 1].")
    return LABEL_ADJUSTMENT * (battery_kwh / charging_efficiency) / (
        np.asarray(combined_kwh_per_100mi) / 100.0
    )
