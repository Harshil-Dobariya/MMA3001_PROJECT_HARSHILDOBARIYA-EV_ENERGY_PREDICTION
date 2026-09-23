"""Stage 2: physics model checks, convergence study and exploratory fit.

Run from the repository root (after running the Stage 1 cleaning):

    python scripts/stage2_physics.py

Outputs (all in results/stage2/):

* drive_cycles.png        : the UDDS and HWFET speed traces
* wheel_power_example.png : wheel power over the UDDS for one vehicle
* convergence.csv/.png    : step-size convergence of the cycle integrals
* exploratory_fit.csv     : physics parameters fitted with and without the
                            suspected window-sticker vehicles
* parity_exploratory.png  : measured vs predicted consumption, flagged
                            vehicles highlighted

The fit here uses every vehicle and is for checking the physics only. The
proper train/test evaluation is done in Stage 4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_energy import physics_model as pm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN_ALL = ROOT / "data" / "processed" / "ev_clean_with_flagged.csv"
OUT = ROOT / "results" / "stage2"
M_PER_MILE = 1609.344
REFINEMENT_FACTORS = (1, 2, 4, 8, 16, 32)


def example_vehicle(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Return the vehicle whose test mass is closest to the median."""
    idx = (vehicles["etw_kg"] - vehicles["etw_kg"].median()).abs().idxmin()
    return vehicles.loc[[idx]]


def plot_drive_cycles(cycles: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharey=True)
    titles = {"city": "UDDS (city)", "hwy": "HWFET (highway)"}
    for ax, (name, cycle) in zip(axes, cycles.items()):
        ax.plot(cycle.time_s, cycle.speed_mps / 0.44704, lw=0.9)
        dist = float(pm.trapezoid(cycle.speed_mps, cycle.dt)) / M_PER_MILE
        ax.set_title(f"{titles[name]}: {cycle.duration_s:.0f} s, {dist:.2f} mi")
        ax.set_ylabel("Speed (mph)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "drive_cycles.png", dpi=150)
    plt.close(fig)


def plot_wheel_power(cycle: pm.DriveCycle, veh: pd.DataFrame) -> None:
    accel = pm.finite_difference_acceleration(cycle.speed_mps, cycle.dt)
    p_w = pm.wheel_power(
        cycle.speed_mps, accel, veh["etw_kg"], veh["f0_n"], veh["f1_n_per_mps"], veh["f2_n_per_mps2"]
    )[0] / 1000.0
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(cycle.time_s, p_w, 0, where=p_w >= 0, color="tab:red", alpha=0.6, label="Driving (E+)")
    ax.fill_between(cycle.time_s, p_w, 0, where=p_w < 0, color="tab:blue", alpha=0.6, label="Braking (E-)")
    row = veh.iloc[0]
    ax.set_title(f"Wheel power over the UDDS: {row['make']} {row['model']} ({row['etw_kg']:.0f} kg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheel power (kW)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wheel_power_example.png", dpi=150)
    plt.close(fig)


def convergence_study(cycles: dict, veh: pd.DataFrame, eta_d: float, eta_r: float) -> pd.DataFrame:
    """Refine the time step and record how the cycle integrals change."""
    rows = []
    args = (veh["etw_kg"], veh["f0_n"], veh["f1_n_per_mps"], veh["f2_n_per_mps2"])
    for name, cycle in cycles.items():
        for kind in ("linear", "pchip"):
            for method in pm.INTEGRATORS:
                for factor in REFINEMENT_FACTORS:
                    fine = cycle.refine(factor, kind)
                    t = pm.cycle_energy_terms(fine, *args, method=method)
                    cons = pm.consumption_kwh_per_100mi(
                        t["e_pos_j"], t["e_neg_j"], t["duration_s"], t["dist_m"], eta_d, eta_r, 0.0
                    )[0]
                    rows.append(
                        {
                            "cycle": name, "interpolant": kind, "method": method,
                            "dt_s": fine.dt, "dist_mi": t["dist_m"][0] / M_PER_MILE,
                            "e_pos_mj": t["e_pos_j"][0] / 1e6, "e_neg_mj": t["e_neg_j"][0] / 1e6,
                            "kwh_per_100mi": cons,
                        }
                    )
    df = pd.DataFrame(rows)
    # Observed order of convergence from three successive refinements:
    # p = log2(|I(4h) - I(2h)| / |I(2h) - I(h)|)
    df["observed_order_e_pos"] = np.nan
    for _, grp in df.groupby(["cycle", "interpolant", "method"]):
        e = grp["e_pos_mj"].to_numpy()
        d = np.abs(np.diff(e))
        order = np.full(len(e), np.nan)
        order[2:] = np.log2(d[:-1] / d[1:])
        df.loc[grp.index, "observed_order_e_pos"] = order
    return df


def plot_convergence(conv: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("city", "hwy")):
        for (kind, method), grp in conv[conv["cycle"] == name].groupby(["interpolant", "method"]):
            ref = grp["kwh_per_100mi"].iloc[-1]
            err = np.abs(grp["kwh_per_100mi"].iloc[:-1] - ref) / ref * 100
            ax.loglog(grp["dt_s"].iloc[:-1], err, "o-", label=f"{method}, {kind}")
        ax.set_title(f"{'UDDS' if name == 'city' else 'HWFET'}: change from finest step")
        ax.set_xlabel("Time step (s)")
        ax.set_ylabel("Relative difference in consumption (%)")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)


def plot_parity(feat: pd.DataFrame, model: pm.PhysicsModel) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, ("city", "hwy")):
        y = feat[f"{name}_kwh_per_100mi"]
        p = model.predict(feat, name)
        flag = feat["label_value_suspect"].to_numpy()
        ax.scatter(p[~flag], y[~flag], s=10, alpha=0.6, label="Other vehicles")
        ax.scatter(p[flag], y[flag], s=18, color="tab:red", label="Suspected window-sticker value")
        lims = [10, 80]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{'UDDS' if name == 'city' else 'HWFET'} (exploratory, fitted without flagged)")
        ax.set_xlabel("Physics prediction (kWh/100 mi)")
        ax.set_ylabel("Measured (kWh/100 mi)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "parity_exploratory.png", dpi=150)
    plt.close(fig)


def fit_summary(feat: pd.DataFrame, label: str) -> tuple[pm.PhysicsModel, dict]:
    """Fit the physics model and return it with a row of summary statistics."""
    model = pm.PhysicsModel().fit(feat)
    row = {"dataset": label, "n_vehicles": len(feat), **model.params()}
    for c in ("city", "hwy"):
        y = feat[f"{c}_kwh_per_100mi"]
        r = y - model.predict(feat, c)
        row[f"{c}_rmse"] = float(np.sqrt((r**2).mean()))
        row[f"{c}_r2"] = float(1 - (r**2).sum() / ((y - y.mean()) ** 2).sum())
    return model, row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cycles = pm.load_epa_cycles(RAW)
    feat_all = pm.add_physics_features(pd.read_csv(CLEAN_ALL), cycles)
    feat = feat_all[~feat_all["label_value_suspect"]].reset_index(drop=True)

    _, row_all = fit_summary(feat_all, "with flagged vehicles")
    model, row_clean = fit_summary(feat, "flagged vehicles excluded")
    summary = pd.DataFrame([row_all, row_clean])
    summary.to_csv(OUT / "exploratory_fit.csv", index=False)

    print("Exploratory physics fit (all vehicles, physics check only; Stage 4 does train/test)")
    for _, r in summary.iterrows():
        print(
            f"  {r['dataset']:<26} n={r['n_vehicles']:>3}  eta_d={r['eta_d']:.3f}  "
            f"eta_r={r['eta_r']:.3f}  P_aux={r['p_aux_w']:6.1f} W  "
            f"city RMSE {r['city_rmse']:.2f} (R2 {r['city_r2']:.3f})  "
            f"hwy RMSE {r['hwy_rmse']:.2f} (R2 {r['hwy_r2']:.3f})"
        )

    # Ratio of measured to predicted for each flagged vehicle, using the
    # model fitted without them. Label values should sit near 1/0.7 = 1.43.
    flagged = feat_all[feat_all["label_value_suspect"]].copy()
    flagged["city_ratio"] = flagged["city_kwh_per_100mi"] / model.predict(flagged, "city")
    flagged["hwy_ratio"] = flagged["hwy_kwh_per_100mi"] / model.predict(flagged, "hwy")
    cols = ["model_year", "make", "model", "city_ratio", "hwy_ratio"]
    flagged[cols].to_csv(OUT / "flagged_vehicles.csv", index=False)
    print("\nFlagged vehicles: measured / physics prediction")
    print(flagged[cols].round(2).to_string(index=False))

    veh = example_vehicle(feat)
    plot_drive_cycles(cycles)
    plot_wheel_power(cycles["city"], veh)
    conv = convergence_study(cycles, veh, model.eta_d_, model.eta_r_)
    conv.to_csv(OUT / "convergence.csv", index=False)
    plot_convergence(conv)
    plot_parity(feat_all, model)

    print("\nConvergence (example vehicle, consumption kWh/100 mi at dt = 1 s vs finest):")
    for (c, kind, method), g in conv.groupby(["cycle", "interpolant", "method"]):
        first, last = g["kwh_per_100mi"].iloc[0], g["kwh_per_100mi"].iloc[-1]
        print(f"  {c:>4} {kind:>6} {method:>9}: {first:.3f} -> {last:.3f} ({(first - last) / last * 100:+.2f}%)")
    print(f"\nFigures and tables saved to {OUT}")


if __name__ == "__main__":
    main()
