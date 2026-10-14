"""Stage 4: final validation on the held-out test set.

Run from the repository root (after Stages 1 and 3):

    python scripts/stage4_validation.py

This is the first and only time the test set is used. The model choice and
hyperparameters were fixed in Stage 3 and are NOT changed here.

Outputs (results/stage4/):

* test_metrics.csv           : RMSE, MAE, R^2, max error for every model
* test_bootstrap.csv         : 95% intervals for RMSE and for the difference
                               from physics alone (vehicle-model bootstrap)
* test_rmse_comparison.png   : test RMSE with 95% intervals
* test_parity.png            : measured vs predicted, physics and selected model
* failure_analysis.csv       : the test vehicles with the largest errors
* domain_check.csv           : test RMSE inside vs outside the training range
* residuals.png              : selected-model errors against mass, power and type
* temporal_metrics.csv       : train on 2022-2024, test on new 2025-2026 models
* sensitivity.csv / .png     : response to +/-10% mass and aerodynamic drag
* well_known_evs.csv         : four well-known EVs against official EPA figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_energy import ml_models as ml
from ev_energy import physics_model as pm
from ev_energy import splits as sp
from ev_energy import validation as va
from ev_energy.data_cleaning import VEHICLE_KEY

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
STAGE3 = ROOT / "results" / "stage3"
OUT = ROOT / "results" / "stage4"

LABELS = {
    "mean_baseline": "Mean baseline", "physics": "Physics only", "linear": "Linear",
    "tree": "Decision tree", "svr": "SVR", "mlp": "Neural net (MLP)",
    "hybrid_linear": "Physics + linear", "hybrid_tree": "Physics + tree",
    "hybrid_svr": "Physics + SVR", "hybrid_mlp": "Physics + MLP",
}

#: Official figures for four well-known EVs. Label MPGe looked up on
#: fueleconomy.gov by the author (October 2026). Battery capacities and EPA
#: ranges for Hyundai and Kia from manufacturer releases. Tesla does not
#: publish battery capacity, so no range is computed for the Teslas.
WELL_KNOWN = [
    {"label": "2024 Tesla Model 3 RWD", "model_year": 2024, "make": "Tesla",
     "model": "Model 3 RWD", "city_mpge": 140, "hwy_mpge": 125, "comb_mpge": 132,
     "battery_kwh": None, "epa_range_mi": None},
    {"label": "2024 Tesla Model Y Long Range AWD", "model_year": 2024, "make": "Tesla",
     "model": "Model Y Long Range AWD", "city_mpge": 122, "hwy_mpge": 112, "comb_mpge": 117,
     "battery_kwh": None, "epa_range_mi": None},
    {"label": "2024 Hyundai Ioniq 6 Long Range RWD (18 in)", "model_year": 2024,
     "make": "Hyundai", "model": "Ioniq 6 Long range RWD (18'' Wheels)",
     "city_mpge": 153, "hwy_mpge": 127, "comb_mpge": 140, "battery_kwh": 77.4,
     "epa_range_mi": 361},
    {"label": "2024 Kia EV6 Standard Range RWD", "model_year": 2024, "make": "Kia",
     "model": "EV6 Standard Range RWD", "city_mpge": 136, "hwy_mpge": 100, "comb_mpge": 117,
     "battery_kwh": 58.0, "epa_range_mi": 232},
]

CHARGING_EFFICIENCIES = (0.85, 0.90, 0.95)


def load_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the Stage 3 split and check it matches the saved one."""
    train, test = sp.group_holdout_split(data)
    sp.check_no_group_overlap(train, test)
    saved_path = PROCESSED / "split_assignments.csv"
    if saved_path.exists():
        saved = pd.read_csv(saved_path)
        saved_test = saved[saved["split"] == "test"][VEHICLE_KEY]
        merged = test[VEHICLE_KEY].merge(saved_test, how="outer", indicator=True)
        if (merged["_merge"] != "both").any():
            raise RuntimeError("Test set differs from the one saved in Stage 3.")
    return train, test


def plot_test_rmse(boot: pd.DataFrame) -> None:
    order = [m for m in LABELS if m != "mean_baseline"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order))
    for offset, cycle, name in ((-0.2, "city", "City (UDDS)"), (0.2, "hwy", "Highway (HWFET)")):
        b = boot[boot["cycle"] == cycle].set_index("model").reindex(order)
        err = np.vstack([b["rmse"] - b["rmse_lo"], b["rmse_hi"] - b["rmse"]])
        ax.bar(x + offset, b["rmse"], 0.4, yerr=err, capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in order], rotation=35, ha="right")
    ax.set_ylabel("Test RMSE (kWh/100 mi)")
    ax.set_title("Held-out test set (175 vehicles, 53 models), 95% bootstrap intervals")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "test_rmse_comparison.png", dpi=150)
    plt.close(fig)


def plot_parity(preds: pd.DataFrame, selected: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cycle in zip(axes, ml.CYCLES):
        y = preds[f"{cycle}_kwh_per_100mi"]
        ax.scatter(preds[f"physics__{cycle}"], y, s=12, alpha=0.6, label="Physics only")
        ax.scatter(preds[f"{selected}__{cycle}"], y, s=12, alpha=0.6, label=LABELS[selected])
        lims = [12, 45]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{'UDDS' if cycle == 'city' else 'HWFET'}: test set")
        ax.set_xlabel("Predicted (kWh/100 mi)")
        ax.set_ylabel("Measured (kWh/100 mi)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "test_parity.png", dpi=150)
    plt.close(fig)


def plot_residuals(preds: pd.DataFrame, selected: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, cycle in enumerate(ml.CYCLES):
        resid = preds[f"{cycle}_kwh_per_100mi"] - preds[f"{selected}__{cycle}"]
        name = "UDDS" if cycle == "city" else "HWFET"
        axes[row, 0].scatter(preds["etw_kg"], resid, s=10, alpha=0.6)
        axes[row, 0].set_xlabel("Test mass (kg)")
        axes[row, 1].scatter(preds["rated_hp"], resid, s=10, alpha=0.6)
        axes[row, 1].set_xlabel("Rated power (hp)")
        types = sorted(preds["vehicle_type"].unique())
        axes[row, 2].boxplot([resid[preds["vehicle_type"] == t] for t in types])
        axes[row, 2].set_xticks(range(1, len(types) + 1))
        axes[row, 2].set_xticklabels(types)
        axes[row, 2].set_xlabel("Vehicle type")
        for ax in axes[row]:
            ax.axhline(0, color="k", lw=1)
            ax.set_ylabel(f"{name} error (measured - predicted)")
            ax.grid(alpha=0.3)
    fig.suptitle(f"Test-set errors of {LABELS[selected]} (kWh/100 mi)")
    fig.tight_layout()
    fig.savefig(OUT / "residuals.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame) -> None:
    order = [m for m in LABELS if m != "mean_baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.suptitle("% change in predicted consumption per 1% change in input (positive = physically correct)")
    titles = {"etw_kg": "Test mass +/-10%", "f2_n_per_mps2": "Aerodynamic term f2 +/-10%"}
    x = np.arange(len(order))
    for ax, column in zip(axes, ("etw_kg", "f2_n_per_mps2")):
        for offset, cycle, name in ((-0.2, "city", "City"), (0.2, "hwy", "Highway")):
            s = sens[(sens["input"] == column) & (sens["cycle"] == cycle)].set_index("model").reindex(order)
            ax.bar(x + offset, s["median_elasticity"], 0.4, label=name)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in order], rotation=40, ha="right")
        ax.set_title(titles[column])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Median elasticity (%/%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=150)
    plt.close(fig)


def well_known_table(preds: pd.DataFrame, selected: str) -> pd.DataFrame:
    rows = []
    for car in WELL_KNOWN:
        match = preds[
            (preds["model_year"] == car["model_year"])
            & (preds["make"] == car["make"])
            & (preds["model"] == car["model"])
        ]
        if match.empty:
            print(f"  Not found in test set: {car['label']}")
            continue
        r = match.iloc[0]
        row = {"vehicle": car["label"]}
        for source, prefix in (("measured", None), ("physics", "physics"), ("selected", selected)):
            city = r["city_kwh_per_100mi"] if prefix is None else r[f"{prefix}__city"]
            hwy = r["hwy_kwh_per_100mi"] if prefix is None else r[f"{prefix}__hwy"]
            comb = va.combined_consumption(city, hwy)
            row[f"{source}_city_mpge"] = float(va.label_mpge(city))
            row[f"{source}_hwy_mpge"] = float(va.label_mpge(hwy))
            row[f"{source}_comb_mpge"] = float(va.label_mpge(comb))
            if car["battery_kwh"]:
                for eff in CHARGING_EFFICIENCIES:
                    row[f"{source}_range_eff{int(eff * 100)}"] = float(
                        va.label_range_miles(car["battery_kwh"], comb, eff)
                    )
        row.update({"official_city_mpge": car["city_mpge"], "official_hwy_mpge": car["hwy_mpge"],
                    "official_comb_mpge": car["comb_mpge"], "official_range_mi": car["epa_range_mi"]})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cycles = pm.load_epa_cycles(RAW)
    data = pm.add_physics_features(pd.read_csv(PROCESSED / "ev_clean.csv"), cycles)
    best = va.load_best_params(STAGE3 / "best_params.json")
    with open(STAGE3 / "selected_model.json") as f:
        selected = json.load(f)["selected_model"]
    models = [m for m in LABELS if m != "mean_baseline"]

    # 1. Test set: opened once.
    train, test = load_split(data)
    print(f"Selected in Stage 3 (unchanged here): {LABELS[selected]}")
    print(f"Fitting on {len(train)} training vehicles, predicting {len(test)} test vehicles...")
    preds = va.fit_and_predict(train, test, best)
    metrics = va.metrics_table(preds)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    table = metrics.pivot(index="model", columns="cycle", values=["rmse", "mae", "r2"])
    print("\nTest-set performance (kWh/100 mi):")
    print(table.reindex(list(LABELS)).round(3).to_string())

    # 2. Uncertainty: bootstrap over vehicle models, paired against physics.
    boot = pd.concat([va.group_bootstrap(preds, models, c) for c in ml.CYCLES], ignore_index=True)
    boot.to_csv(OUT / "test_bootstrap.csv", index=False)
    print("\nTest RMSE with 95% intervals, and difference from physics only:")
    for cycle in ml.CYCLES:
        print(f"  {cycle}:")
        for _, r in boot[boot["cycle"] == cycle].iterrows():
            print(
                f"    {LABELS[r['model']]:<18} {r['rmse']:.2f} [{r['rmse_lo']:.2f}, {r['rmse_hi']:.2f}]"
                f"   diff {r['diff_vs_physics']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}]"
                f"   better in {r['p_better_than_physics']:.0%} of resamples"
            )
    plot_test_rmse(boot)
    plot_parity(preds, selected)

    # 3. Failure analysis for the selected model.
    for cycle in ml.CYCLES:
        preds[f"error_{cycle}"] = preds[f"{cycle}_kwh_per_100mi"] - preds[f"{selected}__{cycle}"]
    preds["worst_abs_error"] = preds[["error_city", "error_hwy"]].abs().max(axis=1)
    worst = preds.sort_values("worst_abs_error", ascending=False).drop_duplicates("group_id").head(10)
    cols = ["model_year", "make", "model", "vehicle_type", "etw_kg", "rated_hp",
            "city_kwh_per_100mi", "error_city", "hwy_kwh_per_100mi", "error_hwy"]
    worst[cols].to_csv(OUT / "failure_analysis.csv", index=False)
    print(f"\nLargest test errors for {LABELS[selected]} (one row per vehicle model):")
    print(worst[cols].round(1).to_string(index=False))
    plot_residuals(preds, selected)

    # 3b. Inputs outside the training range (data-driven models extrapolate).
    preds["out_of_domain"] = va.out_of_domain(train, preds)
    flagged = preds[preds["out_of_domain"]]
    print(f"\nVehicles with an input outside the training 1st-99th percentile range: "
          f"{len(flagged)} of {len(preds)} ({flagged['group_id'].nunique()} models)")
    for _, r in flagged.drop_duplicates("group_id").iterrows():
        print(f"    {r['model_year']} {r['make']} {r['model']}")
    rows = []
    for subset, part in (("inside range", preds[~preds["out_of_domain"]]), ("outside range", flagged)):
        if part.empty:
            continue
        for m in models:
            for cycle in ml.CYCLES:
                err = part[f"{cycle}_kwh_per_100mi"] - part[f"{m}__{cycle}"]
                rows.append({"subset": subset, "model": m, "cycle": cycle, "n": len(part),
                             "rmse": float(np.sqrt((err ** 2).mean()))})
    domain = pd.DataFrame(rows)
    domain.to_csv(OUT / "domain_check.csv", index=False)
    print("\n  Test RMSE inside vs outside the training range:")
    d_table = domain.pivot_table(index="model", columns=["subset", "cycle"], values="rmse")
    print(d_table.reindex(models).round(2).to_string())

    # 4. Temporal check: same hyperparameters, no retuning.
    t_train, t_test = sp.temporal_split(data, cutoff_year=2025)
    t_preds = va.fit_and_predict(t_train, t_test, best)
    t_metrics = va.metrics_table(t_preds)
    t_metrics.to_csv(OUT / "temporal_metrics.csv", index=False)
    print(f"\nTemporal check: train on 2022-2024 ({len(t_train)} vehicles), "
          f"test on new 2025-2026 models ({len(t_test)} vehicles, {t_test['group_id'].nunique()} models):")
    t_table = t_metrics.pivot(index="model", columns="cycle", values=["rmse", "r2"])
    print(t_table.reindex(list(LABELS)).round(3).to_string())

    # 5. Sensitivity to mass and aerodynamic drag.
    print("\nSensitivity analysis (+/-10% mass and aerodynamic term)...")
    sens = va.sensitivity(train, test, best, cycles)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    s_table = sens.pivot_table(index="model", columns=["input", "cycle"], values="median_elasticity")
    print(s_table.reindex([m for m in LABELS if m != "mean_baseline"]).round(2).to_string())
    wrong = sens[sens["fraction_negative"] > 0]
    if not wrong.empty:
        print("\n  Models with physically wrong responses (consumption falls as the input rises):")
        for _, r in wrong.iterrows():
            print(f"    {LABELS[r['model']]:<18} {r['cycle']:<4} {r['input']:<14} "
                  f"in {r['fraction_negative']:.0%} of test vehicles")
    plot_sensitivity(sens)

    # 6. Four well-known EVs against official figures.
    wk = well_known_table(preds, selected)
    wk.to_csv(OUT / "well_known_evs.csv", index=False)
    print("\nWell-known EVs: combined label MPGe (measured / physics / selected / official):")
    for _, r in wk.iterrows():
        print(f"  {r['vehicle']:<44} {r['measured_comb_mpge']:6.1f} {r['physics_comb_mpge']:6.1f} "
              f"{r['selected_comb_mpge']:6.1f} {r['official_comb_mpge']:6.0f}")
    print("\nEstimated label range in miles (charging efficiency 85% / 90% / 95%):")
    for _, r in wk.dropna(subset=["official_range_mi"]).iterrows():
        for source in ("measured", "selected"):
            vals = " / ".join(f"{r[f'{source}_range_eff{int(e * 100)}']:.0f}" for e in CHARGING_EFFICIENCIES)
            print(f"  {r['vehicle']:<44} {source:<9} {vals}   (EPA {r['official_range_mi']:.0f})")

    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
