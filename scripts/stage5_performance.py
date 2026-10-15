"""Stage 5: optimisation and performance analysis.

Run from the repository root (after Stages 1 and 3):

    python scripts/stage5_performance.py

Takes about 3 to 5 minutes. Timings depend on the computer, so every result
is measured on the machine running the script and its details are recorded.

Parts
-----
1. Physics kernel, baseline vs improved: pure Python loops, NumPy per
   vehicle, fully vectorised NumPy, and vectorised in cache-sized chunks.
   Same answers, different speed and memory.
2. FLOPs, achieved FLOP rate and arithmetic intensity of the kernel.
3. Chunk-size sweep to find the fastest chunk for this computer's cache.
4. Time-step trade-off: run time against numerical error.
5. Profile of one full cross-validation run (cProfile) to find the
   bottleneck of the whole pipeline.
6. Cost of each model: training and prediction time against accuracy.
7. Parallel hyperparameter tuning: one core vs all cores.

Outputs (results/stage5/): CSV tables, profile.txt, machine.json and figures.
"""

import cProfile
import io
import json
import os
import platform
import pstats
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_energy import ml_models as ml
from ev_energy import performance as pf
from ev_energy import physics_model as pm
from ev_energy import splits as sp
from ev_energy import validation as va

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
STAGE3 = ROOT / "results" / "stage3"
OUT = ROOT / "results" / "stage5"

VEHICLE_COUNTS = (10, 100, 712, 7120)
LOOP_MAX_VEHICLES = 712  # pure Python is too slow to time at 7,120
CHUNK_SIZES = (1, 4, 8, 16, 32, 64, 128, 256, 712)
REFINEMENT_FACTORS = (1, 2, 4, 8, 16)

LABELS = {
    "mean_baseline": "Mean baseline", "physics": "Physics only", "linear": "Linear",
    "tree": "Decision tree", "svr": "SVR", "mlp": "Neural net (MLP)",
    "hybrid_linear": "Physics + linear", "hybrid_tree": "Physics + tree",
    "hybrid_svr": "Physics + SVR", "hybrid_mlp": "Physics + MLP",
}


def vehicle_arrays(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(df[c].to_numpy() for c in ("etw_kg", "f0_n", "f1_n_per_mps", "f2_n_per_mps2"))


def chunked(chunk):
    """Return a kernel function using a given chunk size."""
    def run(cycle, m, f0, f1, f2):
        return pm.cycle_energy_terms(cycle, m, f0, f1, f2, chunk_size=chunk)
    return run


def machine_info() -> dict:
    import numpy
    import sklearn
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
    }


# ---------------------------------------------------------------------------
# Parts 1 to 3: the physics kernel
# ---------------------------------------------------------------------------


def benchmark_implementations(cycle, base_arrays) -> pd.DataFrame:
    impls = {
        "Python loops (baseline)": pf.energy_terms_loop,
        "NumPy per vehicle": pf.energy_terms_per_vehicle,
        "NumPy fully vectorised": chunked(None),
        f"NumPy chunks of {pm.DEFAULT_CHUNK_SIZE}": chunked(pm.DEFAULT_CHUNK_SIZE),
    }
    rows = []
    n_samples = len(cycle.time_s)
    for n in VEHICLE_COUNTS:
        arrays = pf.tile_vehicles(*base_arrays, n)
        reference = pm.cycle_energy_terms(cycle, *arrays, chunk_size=None)["e_pos_j"]
        for name, func in impls.items():
            if name.startswith("Python") and n > LOOP_MAX_VEHICLES:
                continue
            result = func(cycle, *arrays)["e_pos_j"]
            max_rel_diff = float(np.max(np.abs(result - reference) / np.abs(reference)))
            repeats = 3 if name.startswith("Python") else 7
            t = pf.time_function(lambda: func(cycle, *arrays), repeats=repeats)
            mem = pf.peak_memory_mb(lambda: func(cycle, *arrays))
            flops = pf.kernel_flops(n, n_samples)
            rows.append({
                "implementation": name, "vehicles": n, "min_s": t["min_s"],
                "median_s": t["median_s"], "peak_memory_mb": mem,
                "gflops_per_s": flops / t["min_s"] / 1e9,
                "max_rel_diff_vs_vectorised": max_rel_diff,
            })
    return pd.DataFrame(rows)


def chunk_sweep(cycle, base_arrays) -> pd.DataFrame:
    rows = []
    for n in (712, 7120):
        arrays = pf.tile_vehicles(*base_arrays, n)
        for chunk in CHUNK_SIZES + (None,):
            func = chunked(chunk)
            t = pf.time_function(lambda: func(cycle, *arrays), repeats=7)
            rows.append({
                "vehicles": n, "chunk_size": "all" if chunk is None else chunk,
                "array_kb": (n if chunk is None else min(chunk, n)) * len(cycle.time_s) * 8 / 1e3,
                "min_s": t["min_s"],
            })
    return pd.DataFrame(rows)


def plot_scaling(bench: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, grp in bench.groupby("implementation", sort=False):
        ax.loglog(grp["vehicles"], grp["min_s"], "o-", label=name)
    ax.set_xlabel("Number of vehicles")
    ax.set_ylabel("Run time, UDDS cycle (s)")
    ax.set_title("Physics kernel: baseline vs improved implementations")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "kernel_scaling.png", dpi=150)
    plt.close(fig)


def plot_chunks(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n, grp in sweep.groupby("vehicles"):
        ax.loglog(grp["array_kb"], grp["min_s"], "o-", label=f"{n} vehicles")
    ax.set_xlabel("Size of each temporary array (kB), set by chunk size")
    ax.set_ylabel("Run time, UDDS cycle (s)")
    ax.set_title("Chunk size vs run time (cache effect)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "chunk_sweep.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 4: time step trade-off
# ---------------------------------------------------------------------------


def step_tradeoff(cycles, vehicles, physics) -> pd.DataFrame:
    rows = []
    arrays = vehicle_arrays(vehicles)
    for name, cycle in cycles.items():
        results = {}
        for factor in REFINEMENT_FACTORS:
            fine = cycle.refine(factor, "pchip")
            t = pf.time_function(lambda: pm.cycle_energy_terms(fine, *arrays), repeats=3)
            terms = pm.cycle_energy_terms(fine, *arrays)
            cons = pm.consumption_kwh_per_100mi(
                terms["e_pos_j"], terms["e_neg_j"], terms["duration_s"], terms["dist_m"],
                physics.eta_d_, physics.eta_r_, physics.p_aux_w_,
            )
            results[factor] = (fine.dt, t["min_s"], cons)
        finest = results[REFINEMENT_FACTORS[-1]][2]
        for factor, (dt, seconds, cons) in results.items():
            rows.append({
                "cycle": name, "dt_s": dt, "min_s": seconds,
                "mean_abs_change_pct": float(np.mean(np.abs(cons - finest) / finest) * 100),
            })
    return pd.DataFrame(rows)


def plot_step_tradeoff(trade: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, grp in trade.groupby("cycle"):
        grp = grp[grp["mean_abs_change_pct"] > 0]
        ax.loglog(grp["min_s"], grp["mean_abs_change_pct"], "o-",
                  label="UDDS (city)" if name == "city" else "HWFET (highway)")
        for _, r in grp.iterrows():
            ax.annotate(f"dt={r['dt_s']:g} s", (r["min_s"], r["mean_abs_change_pct"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Run time for all 712 vehicles (s)")
    ax.set_ylabel("Mean change in consumption vs finest step (%)")
    ax.set_title("Time step: accuracy vs cost")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "step_tradeoff.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parts 5 to 7: the full pipeline
# ---------------------------------------------------------------------------


def profile_cv(train, best) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    ml.cross_validate_models(train, best)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(25)
    return stream.getvalue()


def model_costs(train, best) -> pd.DataFrame:
    """Time fitting and predicting each model on the training set."""
    rows = []
    phys_fit = pf.time_function(lambda: pm.PhysicsModel().fit(train), repeats=5)
    physics = pm.PhysicsModel().fit(train)
    phys_pred = pf.time_function(lambda: [physics.predict(train, c) for c in ml.CYCLES], repeats=5)
    rows.append({"model": "physics", "fit_s": phys_fit["min_s"], "predict_s": phys_pred["min_s"]})
    for family in ("mean_baseline",) + ml.ML_FAMILIES:
        for hybrid in (False, True):
            if hybrid and family == "mean_baseline":
                continue
            name = f"hybrid_{family}" if hybrid else family
            fit_s = pred_s = 0.0
            for cycle in ml.CYCLES:
                y = train[f"{cycle}_kwh_per_100mi"].to_numpy()
                if hybrid:
                    y = y - physics.predict(train, cycle)
                pipe = ml.make_pipeline(family, best.get(name, {}).get(cycle, {}))
                fit_s += pf.time_function(lambda: pipe.fit(train, y), repeats=3)["min_s"]
                pred_s += pf.time_function(lambda: pipe.predict(train), repeats=5)["min_s"]
            if hybrid:
                fit_s += phys_fit["min_s"]
                pred_s += phys_pred["min_s"]
            rows.append({"model": name, "fit_s": fit_s, "predict_s": pred_s})
    return pd.DataFrame(rows)


def plot_cost_accuracy(costs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in costs.iterrows():
        if r["model"] == "mean_baseline":
            continue
        ax.scatter(r["fit_s"], r["cv_rmse_avg"], s=40)
        ax.annotate(LABELS[r["model"]], (r["fit_s"], r["cv_rmse_avg"]), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Training time, both cycles (s, log scale)")
    ax.set_ylabel("Repeated-CV RMSE, average of cycles (kWh/100 mi)")
    ax.set_title("Accuracy vs training cost")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "cost_vs_accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = machine_info()
    with open(OUT / "machine.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Machine: " + ", ".join(f"{k} {v}" for k, v in info.items()))

    cycles = pm.load_epa_cycles(RAW)
    vehicles = pd.read_csv(PROCESSED / "ev_clean.csv")
    base_arrays = vehicle_arrays(vehicles)
    city = cycles["city"]
    n_samples = len(city.time_s)

    # Part 1: implementations.
    print("\n1. Physics kernel implementations (UDDS cycle):")
    bench = benchmark_implementations(city, base_arrays)
    bench.to_csv(OUT / "kernel_benchmark.csv", index=False)
    print(bench.round({"min_s": 5, "median_s": 5, "peak_memory_mb": 2, "gflops_per_s": 3})
          .to_string(index=False))
    at712 = bench[bench["vehicles"] == 712].set_index("implementation")["min_s"]
    print(f"\n   Speed-up at 712 vehicles vs Python loops: "
          + ", ".join(f"{k}: {at712.iloc[0] / v:.0f}x" for k, v in at712.items() if not k.startswith("Python")))
    plot_scaling(bench)

    # Part 2: FLOPs and arithmetic intensity.
    flops = pf.kernel_flops(712, n_samples)
    ai_numpy = pf.arithmetic_intensity(flops, pf.numpy_bytes_moved(712, n_samples))
    ai_fused = pf.fused_kernel_arithmetic_intensity()
    print(f"\n2. FLOPs for 712 vehicles x {n_samples} samples: {flops:,} "
          f"({pf.FLOPS_PER_SAMPLE} per vehicle per sample)")
    print(f"   Arithmetic intensity, NumPy as written: {ai_numpy:.4f} FLOP/byte")
    print(f"   Arithmetic intensity, ideal fused loop: {ai_fused:.3f} FLOP/byte")
    print("   Both are far below the machine balance of a modern CPU (several FLOP/byte),")
    print("   so the kernel is memory-bandwidth bound: speed depends on data movement, not arithmetic.")

    # Part 3: chunk sweep.
    print("\n3. Chunk-size sweep (UDDS cycle):")
    sweep = chunk_sweep(city, base_arrays)
    sweep.to_csv(OUT / "chunk_sweep.csv", index=False)
    for n, grp in sweep.groupby("vehicles"):
        best_row = grp.loc[grp["min_s"].idxmin()]
        full = grp[grp["chunk_size"] == "all"]["min_s"].iloc[0]
        print(f"   {n} vehicles: fastest chunk = {best_row['chunk_size']} "
              f"({best_row['min_s']:.4f} s, arrays {best_row['array_kb']:.0f} kB); "
              f"all at once = {full:.4f} s; speed-up {full / best_row['min_s']:.1f}x")
    plot_chunks(sweep)

    # Part 4: time step trade-off.
    data = pm.add_physics_features(vehicles, cycles)
    train, _ = sp.group_holdout_split(data)
    physics = pm.PhysicsModel().fit(train)
    print("\n4. Time step: run time vs numerical change (all 712 vehicles, PCHIP refinement):")
    trade = step_tradeoff(cycles, vehicles, physics)
    trade.to_csv(OUT / "step_tradeoff.csv", index=False)
    print(trade.round({"min_s": 4, "mean_abs_change_pct": 3}).to_string(index=False))
    plot_step_tradeoff(trade)

    # Part 5: profile the full cross-validation.
    best = va.load_best_params(STAGE3 / "best_params.json")
    print("\n5. Profiling one full cross-validation run (cProfile, top functions by cumulative time)...")
    text = profile_cv(train, best)
    (OUT / "profile.txt").write_text(text)
    print("\n".join(text.splitlines()[:40]))

    # Part 6: model costs vs accuracy.
    print("\n6. Model training and prediction cost (training set, both cycles):")
    costs = model_costs(train, best)
    rep = pd.read_csv(STAGE3 / "repeated_cv_summary.csv")
    acc = rep.groupby("model")["rmse_mean"].mean().rename("cv_rmse_avg")
    costs = costs.merge(acc, left_on="model", right_index=True, how="left")
    costs.to_csv(OUT / "model_costs.csv", index=False)
    print(costs.sort_values("fit_s").round(4).to_string(index=False))
    plot_cost_accuracy(costs)

    # Part 7: parallel tuning.
    print(f"\n7. Hyperparameter tuning of the MLP (city), 1 core vs all {os.cpu_count()} cores:")
    serial = pf.time_function(lambda: ml.tune(train, "mlp", "city", n_jobs=1), repeats=1)["min_s"]
    parallel = pf.time_function(lambda: ml.tune(train, "mlp", "city", n_jobs=-1), repeats=1)["min_s"]
    same = ml.tune(train, "mlp", "city", n_jobs=1) == ml.tune(train, "mlp", "city", n_jobs=-1)
    pd.DataFrame([{"cores": 1, "seconds": serial}, {"cores": os.cpu_count(), "seconds": parallel}]).to_csv(
        OUT / "parallel_tuning.csv", index=False)
    print(f"   1 core: {serial:.1f} s; all cores: {parallel:.1f} s; speed-up {serial / parallel:.1f}x; "
          f"same hyperparameters chosen: {same}")

    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
