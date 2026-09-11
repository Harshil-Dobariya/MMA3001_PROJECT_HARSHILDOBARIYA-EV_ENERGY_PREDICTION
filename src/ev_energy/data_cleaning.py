"""Load and clean EPA test car data for battery electric vehicles.

This module turns the raw EPA "Data on Cars Used for Testing Fuel Economy"
spreadsheets into one tidy table with one row per tested vehicle
configuration, holding its design inputs and its measured city (UDDS) and
highway (HWFET) energy consumption.

Source data
-----------
US EPA, Data on Cars Used for Testing Fuel Economy, model years 2022-2026.
https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy

Pipeline (see :func:`clean_ev_dataframe`)
-----------------------------------------
1. Keep only electricity-fuelled charge-depleting UDDS and Highway tests.
2. Rename columns to short snake_case names.
3. Standardise manufacturer brand names.
4. Convert the reported efficiency to a single unit (kWh/100 mi).
5. Reject physically invalid inputs (e.g. placeholder horsepower).
6. Average repeated tests of the same vehicle on the same cycle.
7. Pivot so each vehicle has both a city and a highway result.
8. Add SI-unit versions of mass and road-load coefficients.
9. Apply physical plausibility limits.
10. Add a group identifier for leakage-safe train/test splitting.

Every step records how many rows remain, so data losses are traceable.

Notes on the raw efficiency field
---------------------------------
For EVs, ``RND_ADJ_FE`` is labelled "MPG" but manufacturers report it
inconsistently. Most report unadjusted MPGe (typically 70 to 200), while
some (e.g. Hyundai, Kia, Genesis, Rivian) report kWh/100 mi (typically 15
to 40). Values of 999 and other values above 500 are placeholders. The unit
is therefore inferred from magnitude, see :func:`to_kwh_per_100mi`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants and conversion factors
# ---------------------------------------------------------------------------

#: EPA energy equivalence: 1 US gallon of gasoline = 33.705 kWh.
KWH_PER_GALLON_EQUIVALENT = 33.705

#: Exact conversion factors.
N_PER_LBF = 4.4482216152605
MPS_PER_MPH = 0.44704
KG_PER_LB = 0.45359237

# ---------------------------------------------------------------------------
# Cleaning thresholds (documented assumptions, change here only)
# ---------------------------------------------------------------------------

#: Reported efficiency values below this are taken to be kWh/100 mi.
#: Unadjusted MPGe values for EVs in this dataset sit above 40, while
#: values reported in kWh/100 mi sit below 40. The heaviest GM trucks
#: (40 to 60) are reported by GM in MPGe and are treated as MPGe.
KWH_UNIT_THRESHOLD = 40.0

#: Reported values at or above this are treated as placeholders (e.g. 999).
SENTINEL_THRESHOLD = 500.0

#: Rated horsepower below this is treated as a placeholder (e.g. 1 hp).
MIN_PLAUSIBLE_HP = 50.0

#: Plausible range for unadjusted EV consumption in kWh/100 mi.
MIN_PLAUSIBLE_KWH_PER_100MI = 10.0
MAX_PLAUSIBLE_KWH_PER_100MI = 80.0

#: Speed (mph) at which road load must be positive to be physical.
ROAD_LOAD_CHECK_SPEED_MPH = 60.0

# ---------------------------------------------------------------------------
# Raw column names used from the EPA spreadsheets
# ---------------------------------------------------------------------------

EV_FUEL_DESCRIPTION = "Electricity"
CITY_PROCEDURE = "Charge Depleting UDDS"
HIGHWAY_PROCEDURE = "Charge Depleting Highway"

#: Mapping from raw EPA column names to the short names used in this project.
RAW_TO_CLEAN_COLUMNS = {
    "Model Year": "model_year",
    "Vehicle Manufacturer Name": "manufacturer",
    "Represented Test Veh Make": "make",
    "Represented Test Veh Model": "model",
    "Test Vehicle ID": "vehicle_id",
    "Test Veh Configuration #": "config",
    "Vehicle Type": "vehicle_type",
    "Rated Horsepower": "rated_hp",
    "Drive System Description": "drive",
    "Equivalent Test Weight (lbs.)": "etw_lb",
    "Target Coef A (lbf)": "coef_a_lbf",
    "Target Coef B (lbf/mph)": "coef_b_lbf_per_mph",
    "Target Coef C (lbf/mph**2)": "coef_c_lbf_per_mph2",
    "Test Procedure Description": "procedure",
    "Test Fuel Type Description": "fuel",
    "RND_ADJ_FE": "fe_reported",
}

#: Columns that must exist in every raw file.
REQUIRED_RAW_COLUMNS = tuple(RAW_TO_CLEAN_COLUMNS.keys())

#: Columns identifying one tested vehicle configuration.
VEHICLE_KEY = ["model_year", "make", "model", "vehicle_id", "config"]

#: Design inputs carried through to the cleaned dataset.
SPEC_COLUMNS = [
    "manufacturer",
    "vehicle_type",
    "drive",
    "rated_hp",
    "etw_lb",
    "coef_a_lbf",
    "coef_b_lbf_per_mph",
    "coef_c_lbf_per_mph2",
]

# ---------------------------------------------------------------------------
# Brand-name standardisation
# ---------------------------------------------------------------------------

#: Inconsistent brand spellings in the raw data, mapped to one form.
MAKE_ALIASES = {
    "ACURA": "Acura",
    "AUDI": "Audi",
    "CADILLAC": "Cadillac",
    "CHEVROLET": "Chevrolet",
    "CRUISE": "Cruise",
    "HONDA": "Honda",
    "HYUNDAI": "Hyundai",
    "KIA": "Kia",
    "LEXUS": "Lexus",
    "MAZDA": "Mazda",
    "NISSAN": "Nissan",
    "SUBARU": "Subaru",
    "TOYOTA": "Toyota",
    "Rolls Royce": "Rolls-Royce",
    "Polestar 2": "Polestar",
    "Polestar2 Dual Motor": "Polestar",
    "C40 Recharge Twin": "Volvo",
}

#: Used when the make field holds a year instead of a brand (Lucid rows).
MANUFACTURER_TO_MAKE = {
    "Lucid USA, Inc": "Lucid",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate raw EPA test car spreadsheets.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Paths to the EPA ``.xlsx`` (or ``.csv``) files, one per model year.

    Returns
    -------
    pandas.DataFrame
        All rows from all files, with an added ``source_file`` column.

    Raises
    ------
    ValueError
        If no paths are given, or a file is missing a required column.
    FileNotFoundError
        If a path does not exist.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No input files given.")

    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_excel(path)
        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


def select_ev_cd_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only electric charge-depleting city and highway test rows.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw EPA rows with original column names.

    Returns
    -------
    pandas.DataFrame
        Rows whose fuel is electricity and whose procedure is the
        charge-depleting UDDS or Highway test, with columns renamed to the
        project's snake_case names and a ``cycle`` column (``"city"`` or
        ``"hwy"``) added.
    """
    is_ev = raw["Test Fuel Type Description"] == EV_FUEL_DESCRIPTION
    is_cd = raw["Test Procedure Description"].isin([CITY_PROCEDURE, HIGHWAY_PROCEDURE])
    selected = raw.loc[is_ev & is_cd, list(RAW_TO_CLEAN_COLUMNS)].rename(
        columns=RAW_TO_CLEAN_COLUMNS
    )
    selected["cycle"] = np.where(selected["procedure"] == CITY_PROCEDURE, "city", "hwy")
    return selected.reset_index(drop=True)


def standardise_make(make: object, manufacturer: object) -> str | float:
    """Return a consistent brand name for one row.

    Parameters
    ----------
    make : object
        Raw ``Represented Test Veh Make`` value.
    manufacturer : object
        Raw ``Vehicle Manufacturer Name`` value, used as a fallback when the
        make field holds a year (e.g. ``"2024"``) instead of a brand.

    Returns
    -------
    str or float
        Standardised brand name, or ``numpy.nan`` if it cannot be resolved.

    Examples
    --------
    >>> standardise_make("AUDI", "Volkswagen Group of")
    'Audi'
    >>> standardise_make(2024, "Lucid USA, Inc")
    'Lucid'
    """
    if pd.isna(make):
        return np.nan
    make_str = str(make).strip()
    if make_str.isdigit():
        return MANUFACTURER_TO_MAKE.get(str(manufacturer).strip(), np.nan)
    return MAKE_ALIASES.get(make_str, make_str)


def to_kwh_per_100mi(value: float) -> float:
    """Convert one reported EV efficiency value to kWh/100 mi.

    The unit is inferred from magnitude because manufacturers report the
    same field in different units:

    * ``value >= SENTINEL_THRESHOLD``: placeholder, returns NaN.
    * ``value < KWH_UNIT_THRESHOLD``: already kWh/100 mi, returned as is.
    * otherwise: unadjusted MPGe, converted with
      ``kWh/100 mi = 100 * 33.705 / MPGe``.

    Parameters
    ----------
    value : float
        Reported ``RND_ADJ_FE`` value.

    Returns
    -------
    float
        Energy consumption in kWh/100 mi, or ``numpy.nan`` if invalid.

    Examples
    --------
    >>> round(to_kwh_per_100mi(100.0), 3)
    33.705
    >>> to_kwh_per_100mi(25.0)
    25.0
    >>> import math; math.isnan(to_kwh_per_100mi(999.0))
    True
    """
    if value is None or not np.isfinite(value) or value <= 0:
        return np.nan
    if value >= SENTINEL_THRESHOLD:
        return np.nan
    if value < KWH_UNIT_THRESHOLD:
        return float(value)
    return 100.0 * KWH_PER_GALLON_EQUIVALENT / float(value)


def reported_unit(value: float) -> str:
    """Label the unit inferred for one reported efficiency value.

    Parameters
    ----------
    value : float
        Reported ``RND_ADJ_FE`` value.

    Returns
    -------
    str
        ``"kWh/100mi"``, ``"MPGe"`` or ``"invalid"``.
    """
    if value is None or not np.isfinite(value) or value <= 0 or value >= SENTINEL_THRESHOLD:
        return "invalid"
    return "kWh/100mi" if value < KWH_UNIT_THRESHOLD else "MPGe"


def aggregate_repeat_tests(tests: pd.DataFrame) -> pd.DataFrame:
    """Average repeated tests of the same vehicle on the same cycle.

    Some vehicles were tested more than once on a cycle (e.g. the 2022
    Chevrolet Bolt EUV). The mean is kept as the measured value, and the
    count and standard deviation are kept as evidence of test-to-test
    repeatability.

    Parameters
    ----------
    tests : pandas.DataFrame
        One row per individual test, with ``kwh_per_100mi`` filled in.

    Returns
    -------
    pandas.DataFrame
        One row per vehicle configuration and cycle, with columns
        ``kwh_per_100mi`` (mean), ``n_tests`` and ``kwh_std``.
    """
    keys = VEHICLE_KEY + SPEC_COLUMNS + ["cycle"]
    grouped = tests.groupby(keys, dropna=False)["kwh_per_100mi"]
    return grouped.agg(kwh_per_100mi="mean", n_tests="count", kwh_std="std").reset_index()


def pivot_cycles(per_cycle: pd.DataFrame) -> pd.DataFrame:
    """Reshape to one row per vehicle with city and highway columns.

    Vehicles without a valid result on both cycles are dropped.

    Parameters
    ----------
    per_cycle : pandas.DataFrame
        Output of :func:`aggregate_repeat_tests`.

    Returns
    -------
    pandas.DataFrame
        Columns ``city_kwh_per_100mi``, ``hwy_kwh_per_100mi``,
        ``city_n_tests``, ``hwy_n_tests``, ``city_kwh_std`` and
        ``hwy_kwh_std`` alongside the vehicle key and spec columns.
    """
    index = VEHICLE_KEY + SPEC_COLUMNS
    values = ["kwh_per_100mi", "n_tests", "kwh_std"]
    halves = []
    for cycle in ("city", "hwy"):
        half = per_cycle.loc[per_cycle["cycle"] == cycle, index + values]
        halves.append(half.rename(columns={v: f"{cycle}_{v}" for v in values}))
    # An inner merge keeps only vehicles with a result on both cycles.
    wide = halves[0].merge(halves[1], on=index, how="inner")
    wide = wide.dropna(subset=["city_kwh_per_100mi", "hwy_kwh_per_100mi"])
    return wide.reset_index(drop=True)


def add_si_units(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Add SI versions of test mass and road-load coefficients.

    The EPA road-load force is ``F(v) = A + B v + C v**2`` with ``F`` in lbf
    and ``v`` in mph. In SI form, ``F(v) = f0 + f1 v + f2 v**2`` with ``F``
    in N and ``v`` in m/s, where

    * ``f0 = A * N_PER_LBF``                    [N]
    * ``f1 = B * N_PER_LBF / MPS_PER_MPH``      [N/(m/s)]
    * ``f2 = C * N_PER_LBF / MPS_PER_MPH**2``   [N/(m/s)^2]

    Parameters
    ----------
    vehicles : pandas.DataFrame
        Table with ``etw_lb`` and the three ``coef_*`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy with ``etw_kg``, ``f0_n``, ``f1_n_per_mps`` and
        ``f2_n_per_mps2`` added.
    """
    out = vehicles.copy()
    out["etw_kg"] = out["etw_lb"] * KG_PER_LB
    out["f0_n"] = out["coef_a_lbf"] * N_PER_LBF
    out["f1_n_per_mps"] = out["coef_b_lbf_per_mph"] * N_PER_LBF / MPS_PER_MPH
    out["f2_n_per_mps2"] = out["coef_c_lbf_per_mph2"] * N_PER_LBF / MPS_PER_MPH**2
    return out


def road_load_lbf(vehicles: pd.DataFrame, speed_mph: float) -> pd.Series:
    """Evaluate the road-load force ``A + B v + C v**2`` at one speed.

    Parameters
    ----------
    vehicles : pandas.DataFrame
        Table with the three ``coef_*`` columns.
    speed_mph : float
        Vehicle speed in mph.

    Returns
    -------
    pandas.Series
        Road-load force in lbf for each row.
    """
    v = speed_mph
    return (
        vehicles["coef_a_lbf"]
        + vehicles["coef_b_lbf_per_mph"] * v
        + vehicles["coef_c_lbf_per_mph2"] * v**2
    )


def make_group_id(make: str, model: str) -> str:
    """Build a group identifier shared by all years of one vehicle model.

    The same model is often retested with an identical specification in
    later model years. Splitting by this identifier keeps every copy of a
    model on one side of a train/test split, preventing data leakage.

    Parameters
    ----------
    make : str
        Standardised brand name.
    model : str
        Raw model name.

    Returns
    -------
    str
        Lower-case ``"make|model"`` with whitespace collapsed.

    Examples
    --------
    >>> make_group_id("Tesla", "Model 3  Long Range AWD")
    'tesla|model 3 long range awd'
    """
    text = f"{make}|{model}".lower()
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def clean_ev_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the full cleaning pipeline on raw EPA rows.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw EPA rows with original column names (e.g. from
        :func:`load_raw_files`).

    Returns
    -------
    vehicles : pandas.DataFrame
        One row per tested vehicle configuration with both cycle results.
    report : dict of str to int
        Row count after each step, in order, for the documentation.

    Raises
    ------
    ValueError
        If a required raw column is missing.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    report: dict[str, int] = {"raw rows (all fuels and procedures)": len(raw)}

    tests = select_ev_cd_tests(raw)
    report["EV charge-depleting city/highway test rows"] = len(tests)

    tests["make"] = [
        standardise_make(m, f) for m, f in zip(tests["make"], tests["manufacturer"])
    ]
    tests = tests.dropna(subset=["make"])
    report["rows with a resolvable brand"] = len(tests)

    tests["fe_unit_inferred"] = tests["fe_reported"].apply(reported_unit)
    tests["kwh_per_100mi"] = tests["fe_reported"].apply(to_kwh_per_100mi)
    tests = tests.dropna(subset=["kwh_per_100mi"])
    report["rows with a valid efficiency value (sentinels removed)"] = len(tests)

    tests.loc[tests["rated_hp"] < MIN_PLAUSIBLE_HP, "rated_hp"] = np.nan
    tests = tests.dropna(subset=SPEC_COLUMNS[2:])
    report["rows with complete, plausible design inputs"] = len(tests)

    per_cycle = aggregate_repeat_tests(tests)
    report["vehicle-cycle results after averaging repeat tests"] = len(per_cycle)

    vehicles = pivot_cycles(per_cycle)
    report["vehicles with both city and highway results"] = len(vehicles)

    vehicles = add_si_units(vehicles)

    consumption = vehicles[["city_kwh_per_100mi", "hwy_kwh_per_100mi"]]
    in_range = consumption.apply(
        lambda s: s.between(MIN_PLAUSIBLE_KWH_PER_100MI, MAX_PLAUSIBLE_KWH_PER_100MI)
    ).all(axis=1)
    positive_load = road_load_lbf(vehicles, ROAD_LOAD_CHECK_SPEED_MPH) > 0
    vehicles = vehicles[in_range & positive_load].reset_index(drop=True)
    report["vehicles passing physical plausibility checks"] = len(vehicles)

    vehicles["group_id"] = [
        make_group_id(m, n) for m, n in zip(vehicles["make"], vehicles["model"])
    ]
    vehicles["model_year"] = vehicles["model_year"].astype(int)
    vehicles = vehicles.sort_values(VEHICLE_KEY).reset_index(drop=True)
    return vehicles, report


def clean_ev_data(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load raw EPA files and run the full cleaning pipeline.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Paths to the raw EPA spreadsheets.

    Returns
    -------
    vehicles : pandas.DataFrame
        Cleaned dataset, see :func:`clean_ev_dataframe`.
    report : dict of str to int
        Row count after each cleaning step.
    """
    return clean_ev_dataframe(load_raw_files(paths))


def main(
    raw_dir: str | Path = "data/raw",
    out_path: str | Path = "data/processed/ev_clean.csv",
) -> None:
    """Command-line entry point: clean all raw files and save a CSV.

    Run from the repository root with ``python -m ev_energy.data_cleaning``.

    Parameters
    ----------
    raw_dir : str or pathlib.Path, optional
        Folder holding the raw EPA ``*testcar*.xlsx`` files.
    out_path : str or pathlib.Path, optional
        Where to write the cleaned CSV.
    """
    paths = sorted(Path(raw_dir).glob("*testcar*.xlsx"))
    vehicles, report = clean_ev_data(paths)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vehicles.to_csv(out_path, index=False)

    print("Cleaning report")
    for step, rows in report.items():
        print(f"  {rows:>6}  {step}")
    print(
        f"\nSaved {len(vehicles)} vehicles "
        f"({vehicles['group_id'].nunique()} unique models) to {out_path}"
    )


if __name__ == "__main__":
    main()
