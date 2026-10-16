"""Predicting battery electric vehicle energy consumption from EPA test data.

MMA3001 Numerical Methods and Machine Learning, Monash University, 2026.
Individual project by Harshil Dobariya.

The package predicts the energy consumption of a battery electric vehicle
(kWh/100 mi) on the EPA city (UDDS) and highway (HWFET) test cycles from its
design parameters, and compares a physics-based model with machine-learning
and hybrid models.

Top-level inputs
----------------
For each vehicle (all from the EPA test car data, converted to SI units):

| Input | Column | Unit | Typical range in the data |
|---|---|---|---|
| Equivalent test mass | ``etw_kg`` | kg | 1,474 to 4,082 |
| Rated power | ``rated_hp`` | hp | 98 to 1,839 |
| Road-load coefficient A | ``f0_n`` | N | 74 to 295 |
| Road-load coefficient B | ``f1_n_per_mps`` | N/(m/s) | -2.6 to 13.7 |
| Road-load coefficient C | ``f2_n_per_mps2`` | N/(m/s)^2 | 0.22 to 0.82 |
| Drivetrain | ``drive`` | category | e.g. "All Wheel Drive" |
| Vehicle type | ``vehicle_type`` | category | "Car", "Truck", "Both" |

plus the EPA UDDS and HWFET speed traces (s, mph).

Top-level outputs
-----------------
Predicted unadjusted (laboratory) energy consumption on each cycle, in
kWh/100 mi, which converts to approximate window-sticker MPGe and, with a
known battery capacity, driving range (:mod:`ev_energy.validation`).

Modules, in pipeline order
--------------------------
* :mod:`ev_energy.data_cleaning`: load the EPA spreadsheets, fix mixed
  units, remove placeholders and window-sticker values, pair city and
  highway results (Stage 1).
* :mod:`ev_energy.physics_model`: road-load physics integrated over the
  drive cycles with finite differences and the trapezoidal or Simpson's rule;
  three-parameter least-squares fit (Stage 2).
* :mod:`ev_energy.splits`: train/test splits and cross-validation folds
  grouped by vehicle model, to prevent data leakage (Stage 3).
* :mod:`ev_energy.ml_models`: linear, decision-tree, SVR and neural-network
  models, and hybrids that learn the physics model's residual (Stage 3).
* :mod:`ev_energy.validation`: test-set evaluation, bootstrap intervals,
  input-domain check, sensitivity analysis and window-sticker conversions
  (Stage 4).
* :mod:`ev_energy.performance`: baseline and improved implementations of
  the physics kernel, timing, memory, FLOP and arithmetic-intensity tools
  (Stage 5).

Each stage is run by a script in ``scripts/``; ``scripts/run_all.py`` runs
the whole pipeline. See the repository README for results and usage.
"""

__version__ = "1.0.0"
