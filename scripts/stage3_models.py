"""Stage 3: split the data, tune the ML and hybrid models, and compare them
by grouped cross-validation on the TRAINING set only.

Run from the repository root (after the Stage 1 cleaning):

    python scripts/stage3_models.py

The held-out test set is created and saved here but is NOT used for any
fitting, tuning or model choice. It is opened once, in Stage 4.

Outputs (results/stage3/):

* split_summary.csv      : size and target spread of the train and test sets
* best_params.json       : hyperparameters chosen for each model and cycle
* cv_metrics.csv         : cross-validated RMSE, MAE, R^2, max error, fit time
* repeated_cv.csv        : RMSE for every model over 5 different fold
                           assignments (per run)
* repeated_cv_summary.csv: mean and spread of RMSE over those runs, and how
                           often each model came first
* cv_rmse_comparison.png : mean cross-validated RMSE with error bars
                           (one standard deviation over the 5 runs)
* cv_parity.png          : out-of-fold predictions, physics vs best model
* environment.json       : Python and library versions used for this run
* selected_model.json    : the single model chosen for Stage 4, by the rule
                           stated in SELECTION_RULE

Also writes data/processed/split_assignments.csv (which vehicles are in the
test set), so Stage 4 uses exactly the same split.
"""

import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_energy import ml_models as ml
from ev_energy import physics_model as pm
from ev_energy import splits as sp
from ev_energy.data_cleaning import VEHICLE_KEY

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "stage3"

#: Model-selection rule, fixed BEFORE the test set is opened. One model is
#: used for both cycles: the one with the lowest average of its city and
#: highway mean RMSE over the repeated cross-validation runs. Choosing per
#: cycle was rejected because on the highway the top four models differ by
#: less than their run-to-run spread, so a per-cycle winner would be chosen
#: by noise.
SELECTION_RULE = "lowest average of city and highway mean CV RMSE (5 repeated grouped 5-fold runs)"

#: Display order and labels for the comparison chart.
MODEL_LABELS = {
    "mean_baseline": "Mean baseline",
    "physics": "Physics only",
    "linear": "Linear",
    "tree": "Decision tree",
    "svr": "SVR",
    "mlp": "Neural net (MLP)",
    "hybrid_linear": "Physics + linear",
    "hybrid_tree": "Physics + tree",
    "hybrid_svr": "Physics + SVR",
    "hybrid_mlp": "Physics + MLP",
}


def tune_all(train: pd.DataFrame) -> dict:
    """Tune every ML family, directly and as a hybrid residual model."""
    best: dict = {}
    for family in ml.ML_FAMILIES:
        for residual in (False, True):
            name = f"hybrid_{family}" if residual else family
            for cycle in ml.CYCLES:
                params = ml.tune(train, family, cycle, residual=residual)
                best.setdefault(name, {})[cycle] = params
                print(f"  {name:<14} {cycle:<4} {params}")
    return best


def plot_rmse(summary: pd.DataFrame) -> None:
    order = list(MODEL_LABELS)
    mean = summary.pivot(index="model", columns="cycle", values="rmse_mean").reindex(order)
    std = summary.pivot(index="model", columns="cycle", values="rmse_std").reindex(order)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, mean["city"], 0.4, yerr=std["city"], capsize=3, label="City (UDDS)")
    ax.bar(x + 0.2, mean["hwy"], 0.4, yerr=std["hwy"], capsize=3, label="Highway (HWFET)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=35, ha="right")
    ax.set_ylabel("Cross-validated RMSE (kWh/100 mi)")
    ax.set_title("Grouped 5-fold CV on the training set, repeated with 5 fold assignments")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cv_rmse_comparison.png", dpi=150)
    plt.close(fig)


def plot_parity(preds: pd.DataFrame, best_model: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cycle in zip(axes, ml.CYCLES):
        y = preds[f"{cycle}_kwh_per_100mi"]
        ax.scatter(preds[f"physics__{cycle}"], y, s=10, alpha=0.5, label="Physics only")
        best = best_model[cycle]
        ax.scatter(preds[f"{best}__{cycle}"], y, s=10, alpha=0.5, label=MODEL_LABELS[best])
        lims = [12, 52]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{'UDDS' if cycle == 'city' else 'HWFET'}: out-of-fold predictions")
        ax.set_xlabel("Predicted (kWh/100 mi)")
        ax.set_ylabel("Measured (kWh/100 mi)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cv_parity.png", dpi=150)
    plt.close(fig)


def record_environment() -> dict:
    """Return the Python and key library versions, for reproducibility."""
    import numpy
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = record_environment()
    with open(OUT / "environment.json", "w") as f:
        json.dump(env, f, indent=2)
    print("Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()))
    cycles = pm.load_epa_cycles(RAW)
    data = pm.add_physics_features(pd.read_csv(PROCESSED / "ev_clean.csv"), cycles)

    # 1. Split by vehicle model and save the assignment for Stage 4.
    train, test = sp.group_holdout_split(data)
    sp.check_no_group_overlap(train, test)
    summary = sp.split_summary(train, test)
    summary.to_csv(OUT / "split_summary.csv", index=False)
    assignments = pd.concat(
        [train[VEHICLE_KEY].assign(split="train"), test[VEHICLE_KEY].assign(split="test")]
    )
    assignments.to_csv(PROCESSED / "split_assignments.csv", index=False)
    print("Split (grouped by vehicle model, test set not used in this stage):")
    print(summary.to_string(index=False))

    # 2. Tune hyperparameters with grouped CV on the training set.
    print("\nTuning hyperparameters (grouped 5-fold CV, training set only):")
    best = tune_all(train)
    with open(OUT / "best_params.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    # 3. Compare all models on the same grouped folds.
    print("\nCross-validating all models...")
    result = ml.cross_validate_models(train, best)
    metrics = result.metrics()
    metrics.to_csv(OUT / "cv_metrics.csv", index=False)

    table = metrics.pivot(index="model", columns="cycle", values=["rmse", "mae", "r2"])
    table = table.reindex(list(MODEL_LABELS)).round(3)
    print("\nCross-validated performance (kWh/100 mi for RMSE and MAE):")
    print(table.to_string())

    # 4. Repeat CV with different fold assignments to see how stable the
    #    ranking is. Model choice uses the mean over these runs.
    print("\nRepeating cross-validation with 5 different fold assignments...")
    per_run, rep = ml.repeated_cv_metrics(train, best)
    per_run.to_csv(OUT / "repeated_cv.csv", index=False)
    rep.to_csv(OUT / "repeated_cv_summary.csv", index=False)
    for cycle in ml.CYCLES:
        print(f"\n  {cycle}: mean RMSE over 5 runs (std), runs won")
        for _, r in rep[rep["cycle"] == cycle].iterrows():
            print(f"    {MODEL_LABELS[r['model']]:<18} {r['rmse_mean']:.3f} ({r['rmse_std']:.3f})  {r['runs_won']}")

    combined = rep.groupby("model")["rmse_mean"].mean().sort_values()
    selected = combined.index[0]
    print("\n  Average of city and highway mean RMSE:")
    for model, value in combined.items():
        print(f"    {MODEL_LABELS[model]:<18} {value:.3f}")
    print(f"\nSelected for Stage 4 ({SELECTION_RULE}): {MODEL_LABELS[selected]}")
    with open(OUT / "selected_model.json", "w") as f:
        json.dump({"selected_model": selected, "rule": SELECTION_RULE,
                   "average_cv_rmse": round(float(combined.iloc[0]), 4)}, f, indent=2)
    best_model = {c: selected for c in ml.CYCLES}

    plot_rmse(rep)
    plot_parity(result.predictions, best_model)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
