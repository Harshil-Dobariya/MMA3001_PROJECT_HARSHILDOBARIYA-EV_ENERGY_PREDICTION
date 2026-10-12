"""Tests for ev_energy.splits and ev_energy.ml_models.

The most important tests here are the leakage tests: they change the
targets of held-out vehicles and check that predictions for those vehicles
do not change. If any model, scaler or physics fit could see held-out data,
these tests would fail.
"""

import numpy as np
import pandas as pd
import pytest

from ev_energy import ml_models as ml
from ev_energy import splits as sp


def make_vehicles(n_models=30, per_model=3, seed=0):
    """Synthetic vehicle table with every column the models need.

    Consumption is a smooth function of mass and drag plus noise, so the
    models have something real to learn.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_models):
        mass = rng.uniform(1600, 3500)
        f2 = rng.uniform(0.3, 0.6)
        drive = rng.choice(["All Wheel Drive", "2-Wheel Drive, Rear", "2-Wheel Drive, Front"])
        vtype = rng.choice(["Car", "Truck"])
        for year in range(per_model):
            rows.append(
                {
                    "group_id": f"make|model {g}",
                    "model_year": 2022 + year,
                    "etw_kg": mass,
                    "rated_hp": rng.uniform(150, 600),
                    "f0_n": 0.06 * mass,
                    "f1_n_per_mps": rng.uniform(0.5, 4.0),
                    "f2_n_per_mps2": f2,
                    "drive": drive,
                    "vehicle_type": vtype,
                    # Physics terms (J, s, m) roughly consistent with mass and drag.
                    "city_e_pos_j": 2000 * mass + rng.normal(0, 1e5),
                    "city_e_neg_j": 900 * mass + rng.normal(0, 1e5),
                    "city_duration_s": 1369.0,
                    "city_dist_m": 11_990.0,
                    "hwy_e_pos_j": 1800 * mass + 3e6 * f2 + rng.normal(0, 1e5),
                    "hwy_e_neg_j": 200 * mass + rng.normal(0, 5e4),
                    "hwy_duration_s": 765.0,
                    "hwy_dist_m": 16_507.0,
                }
            )
    df = pd.DataFrame(rows)
    df["city_kwh_per_100mi"] = 8 + 0.007 * df["etw_kg"] + rng.normal(0, 0.5, len(df))
    df["hwy_kwh_per_100mi"] = 6 + 0.006 * df["etw_kg"] + 30 * df["f2_n_per_mps2"] + rng.normal(0, 0.5, len(df))
    return df


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_group_split_has_no_shared_models():
    df = make_vehicles()
    train, test = sp.group_holdout_split(df)
    sp.check_no_group_overlap(train, test)
    assert len(train) + len(test) == len(df)


def test_group_split_is_reproducible():
    df = make_vehicles()
    a = sp.group_holdout_split(df)[1]["group_id"].tolist()
    b = sp.group_holdout_split(df)[1]["group_id"].tolist()
    assert a == b


def test_group_split_holds_out_roughly_the_requested_fraction_of_models():
    df = make_vehicles(n_models=50)
    _, test = sp.group_holdout_split(df, test_fraction=0.2)
    assert test["group_id"].nunique() == 10


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_group_split_rejects_bad_fraction(bad):
    with pytest.raises(ValueError):
        sp.group_holdout_split(make_vehicles(), test_fraction=bad)


def test_overlap_check_catches_leakage():
    df = make_vehicles()
    with pytest.raises(AssertionError):
        sp.check_no_group_overlap(df.iloc[:10], df.iloc[5:20])


def test_kfold_folds_are_disjoint_complete_and_group_safe():
    df = make_vehicles()
    folds = sp.group_kfold_indices(df, n_folds=5)
    all_val = np.concatenate([va for _, va in folds])
    assert sorted(all_val) == list(range(len(df)))  # every row validated once
    for tr, va in folds:
        assert set(df["group_id"].iloc[tr]).isdisjoint(df["group_id"].iloc[va])


def test_kfold_assignment_does_not_depend_on_row_order():
    # The same groups must land in the same folds even if rows are shuffled.
    df = make_vehicles()
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

    def fold_map(frame):
        out = {}
        for k, (_, va) in enumerate(sp.group_kfold_indices(frame)):
            for g in frame["group_id"].iloc[va]:
                out[g] = k
        return out

    assert fold_map(df) == fold_map(shuffled)


def test_kfold_rejects_bad_fold_count():
    with pytest.raises(ValueError):
        sp.group_kfold_indices(make_vehicles(n_models=3), n_folds=5)


def test_temporal_split_trains_on_past_and_tests_on_new_models():
    df = make_vehicles(per_model=2)  # years 2022 and 2023 only
    new = make_vehicles(n_models=5, per_model=1, seed=9).assign(model_year=2025)
    new["group_id"] = "new|" + new["group_id"]
    df = pd.concat([df, new], ignore_index=True)
    train, test = sp.temporal_split(df, cutoff_year=2025)
    assert train["model_year"].max() < 2025
    assert (test["model_year"] >= 2025).all()
    sp.check_no_group_overlap(train, test)
    assert len(test) == 5


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["mean_baseline", "linear", "tree", "svr", "mlp"])
def test_every_family_fits_and_predicts(family):
    df = make_vehicles()
    pipe = ml.make_pipeline(family).fit(df, df["city_kwh_per_100mi"])
    pred = pipe.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        ml.make_estimator("random_forest")


def test_mean_baseline_predicts_training_mean():
    df = make_vehicles()
    y = df["city_kwh_per_100mi"]
    pred = ml.make_pipeline("mean_baseline").fit(df, y).predict(df.iloc[:3])
    assert np.allclose(pred, y.mean())


def test_unseen_category_is_handled():
    df = make_vehicles()
    pipe = ml.make_pipeline("linear").fit(df, df["city_kwh_per_100mi"])
    new = df.iloc[[0]].copy()
    new["drive"] = "Part-time 4-Wheel Drive"  # not in the training data
    assert np.isfinite(pipe.predict(new)).all()


def test_linear_model_beats_baseline_on_learnable_data():
    df = make_vehicles()
    y = df["city_kwh_per_100mi"]
    lin = ml.regression_metrics(y, ml.make_pipeline("linear").fit(df, y).predict(df))
    base = ml.regression_metrics(y, ml.make_pipeline("mean_baseline").fit(df, y).predict(df))
    assert lin["rmse"] < base["rmse"]


def test_regression_metrics_perfect_prediction():
    y = np.array([20.0, 25.0, 30.0])
    m = ml.regression_metrics(y, y)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1 and m["max_abs_error"] == 0


def test_params_are_applied():
    pipe = ml.make_pipeline("tree", {"model__max_depth": 2})
    assert pipe.named_steps["model"].max_depth == 2


def test_tune_returns_values_from_the_grid():
    df = make_vehicles()
    params = ml.tune(df, "tree", "city")
    assert params["model__max_depth"] in ml.PARAM_GRIDS["tree"]["model__max_depth"]
    assert ml.tune(df, "linear", "city") == {}


# ---------------------------------------------------------------------------
# Cross-validation and leakage
# ---------------------------------------------------------------------------

FAST_PARAMS = {"tree": {"city": {"model__max_depth": 3}, "hwy": {"model__max_depth": 3}}}


def test_every_vehicle_gets_one_out_of_fold_prediction():
    df = make_vehicles()
    result = ml.cross_validate_models(df, FAST_PARAMS)
    preds = result.predictions
    assert (preds["fold"] >= 0).all()
    for col in [c for c in preds.columns if "__" in c]:
        assert preds[col].notna().all(), col


def test_folds_never_split_a_vehicle_model():
    df = make_vehicles()
    preds = ml.cross_validate_models(df, FAST_PARAMS).predictions
    assert (preds.groupby("group_id")["fold"].nunique() == 1).all()


def test_metrics_table_covers_every_model_and_cycle():
    df = make_vehicles()
    metrics = ml.cross_validate_models(df, FAST_PARAMS).metrics()
    expected_models = {"mean_baseline", "physics", *ml.ML_FAMILIES, *[f"hybrid_{f}" for f in ml.ML_FAMILIES]}
    assert set(metrics["model"]) == expected_models
    assert set(metrics["cycle"]) == set(ml.CYCLES)


def test_no_leakage_of_held_out_targets():
    # Corrupt the targets of the vehicles in fold 0. If no model (including
    # the physics fit inside the hybrids) can see a held-out fold, the
    # predictions for those vehicles must be identical either way.
    df = make_vehicles()
    first = ml.cross_validate_models(df, FAST_PARAMS).predictions
    fold0 = first.index[first["fold"] == 0]
    corrupted = df.copy()
    corrupted.loc[fold0, ["city_kwh_per_100mi", "hwy_kwh_per_100mi"]] += 100.0
    second = ml.cross_validate_models(corrupted, FAST_PARAMS).predictions
    for col in [c for c in first.columns if "__" in c]:
        assert np.allclose(first.loc[fold0, col], second.loc[fold0, col]), col


def test_repeated_cv_summarises_every_model_and_counts_wins():
    df = make_vehicles()
    per_run, summary = ml.repeated_cv_metrics(df, FAST_PARAMS, seeds=(0, 1))
    assert set(per_run["seed"]) == {0, 1}
    assert len(summary) == 10 * len(ml.CYCLES)
    # Each cycle has exactly one winner per run.
    assert summary.groupby("cycle")["runs_won"].sum().tolist() == [2, 2]
    assert (summary["rmse_std"] >= 0).all()


def test_different_seeds_give_different_folds():
    df = make_vehicles()
    a = [tuple(va) for _, va in sp.group_kfold_indices(df, random_state=0)]
    b = [tuple(va) for _, va in sp.group_kfold_indices(df, random_state=1)]
    assert a != b
