"""Tests for ev_energy.data_cleaning.

Unit tests check each cleaning step against hand-calculated answers.
Integration tests run the full pipeline on a small synthetic dataset whose
correct output is known exactly, so no real data is needed to run them.
"""

import math

import numpy as np
import pandas as pd
import pytest

from ev_energy import data_cleaning as dc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_raw_row(**overrides):
    """Return one raw EPA-style row with sensible EV defaults."""
    row = {
        "Model Year": 2024,
        "Vehicle Manufacturer Name": "Tesla",
        "Represented Test Veh Make": "Tesla",
        "Represented Test Veh Model": "Model 3",
        "Test Vehicle ID": "TV1",
        "Test Veh Configuration #": 1,
        "Vehicle Type": "Car",
        "Rated Horsepower": 300,
        "Drive System Description": "2-Wheel Drive, Rear",
        "Equivalent Test Weight (lbs.)": 4000,
        "Target Coef A (lbf)": 30.0,
        "Target Coef B (lbf/mph)": 0.1,
        "Target Coef C (lbf/mph**2)": 0.015,
        "Test Procedure Description": dc.CITY_PROCEDURE,
        "Test Fuel Type Description": dc.EV_FUEL_DESCRIPTION,
        "RND_ADJ_FE": 150.5,
    }
    row.update(overrides)
    return row


def make_vehicle_pair(vehicle_id="TV1", city_fe=150.5, hwy_fe=130.5, **overrides):
    """Return city and highway raw rows for one vehicle.

    Defaults use one decimal place, like real dynamometer results, so the
    vehicle is not flagged as a suspected window-sticker value.
    """
    city = make_raw_row(**{"Test Vehicle ID": vehicle_id, "RND_ADJ_FE": city_fe, **overrides})
    hwy = make_raw_row(
        **{
            "Test Vehicle ID": vehicle_id,
            "RND_ADJ_FE": hwy_fe,
            "Test Procedure Description": dc.HIGHWAY_PROCEDURE,
            **overrides,
        }
    )
    return [city, hwy]


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def test_mpge_converts_with_epa_equivalence():
    # 100 MPGe -> 100 * 33.705 / 100 = 33.705 kWh/100 mi
    assert to_close(dc.to_kwh_per_100mi(100.0), 33.705)


def test_mpge_and_kwh_are_reciprocal():
    # Converting 150 MPGe, then converting back, returns 150.
    kwh = dc.to_kwh_per_100mi(150.0)
    assert to_close(100.0 * dc.KWH_PER_GALLON_EQUIVALENT / kwh, 150.0)


def test_small_values_are_treated_as_kwh():
    assert dc.to_kwh_per_100mi(25.0) == 25.0
    assert dc.reported_unit(25.0) == "kWh/100mi"


@pytest.mark.parametrize("bad", [999.0, 987.5, 0.0, -5.0, np.nan, np.inf])
def test_invalid_efficiency_values_return_nan(bad):
    assert math.isnan(dc.to_kwh_per_100mi(bad))
    assert dc.reported_unit(bad) == "invalid"


def test_threshold_boundary_is_mpge():
    # Exactly 40 is interpreted as MPGe, not kWh/100 mi.
    assert dc.reported_unit(dc.KWH_UNIT_THRESHOLD) == "MPGe"


def test_si_conversion_of_road_load_coefficients():
    vehicles = pd.DataFrame(
        {"etw_lb": [1.0], "coef_a_lbf": [1.0], "coef_b_lbf_per_mph": [1.0],
         "coef_c_lbf_per_mph2": [1.0]}
    )
    out = dc.add_si_units(vehicles).iloc[0]
    assert to_close(out["etw_kg"], 0.45359237)
    assert to_close(out["f0_n"], 4.4482216152605)
    assert to_close(out["f1_n_per_mps"], 4.4482216152605 / 0.44704)
    assert to_close(out["f2_n_per_mps2"], 4.4482216152605 / 0.44704**2)


def test_si_road_load_matches_imperial_road_load():
    # The force at 60 mph must be the same physical force in both unit systems.
    vehicles = dc.add_si_units(
        pd.DataFrame({"etw_lb": [4000], "coef_a_lbf": [30.0], "coef_b_lbf_per_mph": [0.1],
                      "coef_c_lbf_per_mph2": [0.015]})
    )
    force_lbf = dc.road_load_lbf(vehicles, 60.0).iloc[0]
    v = 60.0 * dc.MPS_PER_MPH
    row = vehicles.iloc[0]
    force_n = row["f0_n"] + row["f1_n_per_mps"] * v + row["f2_n_per_mps2"] * v**2
    assert to_close(force_n, force_lbf * dc.N_PER_LBF)


# ---------------------------------------------------------------------------
# Brand names and grouping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make, manufacturer, expected",
    [
        ("AUDI", "Volkswagen Group of", "Audi"),
        ("Audi", "Volkswagen Group of", "Audi"),
        ("Polestar2 Dual Motor", "Volvo", "Polestar"),
        ("2024", "Lucid USA, Inc", "Lucid"),
        (2022, "Lucid USA, Inc", "Lucid"),
        ("Tesla", "Tesla", "Tesla"),
    ],
)
def test_standardise_make(make, manufacturer, expected):
    assert dc.standardise_make(make, manufacturer) == expected


def test_unknown_numeric_make_is_unresolvable():
    assert pd.isna(dc.standardise_make("2024", "Unknown Maker"))


def test_group_id_ignores_case_and_spacing():
    assert dc.make_group_id("Tesla", "Model 3  LR") == dc.make_group_id("TESLA", "model 3 lr")


# ---------------------------------------------------------------------------
# Full pipeline on synthetic data
# ---------------------------------------------------------------------------


def test_pipeline_pairs_city_and_highway():
    raw = pd.DataFrame(make_vehicle_pair())
    vehicles, report = dc.clean_ev_dataframe(raw)
    assert len(vehicles) == 1
    assert to_close(vehicles.loc[0, "city_kwh_per_100mi"], 3370.5 / 150.5)
    assert to_close(vehicles.loc[0, "hwy_kwh_per_100mi"], 3370.5 / 130.5)
    assert report["vehicles passing physical plausibility checks"] == 1


def test_vehicle_missing_one_cycle_is_dropped():
    raw = pd.DataFrame([make_raw_row()])  # city only
    vehicles, _ = dc.clean_ev_dataframe(raw)
    assert vehicles.empty


def test_repeat_tests_are_averaged():
    rows = make_vehicle_pair(city_fe=150.5) + make_vehicle_pair(city_fe=100.5)
    vehicles, _ = dc.clean_ev_dataframe(pd.DataFrame(rows))
    expected = (3370.5 / 150.5 + 3370.5 / 100.5) / 2
    assert len(vehicles) == 1
    assert vehicles.loc[0, "city_n_tests"] == 2
    assert to_close(vehicles.loc[0, "city_kwh_per_100mi"], expected)


def test_sentinel_values_are_removed():
    raw = pd.DataFrame(make_vehicle_pair(city_fe=999.0))
    vehicles, _ = dc.clean_ev_dataframe(raw)
    assert vehicles.empty


def test_placeholder_horsepower_is_removed():
    raw = pd.DataFrame(make_vehicle_pair(**{"Rated Horsepower": 1}))
    vehicles, _ = dc.clean_ev_dataframe(raw)
    assert vehicles.empty


def test_non_ev_and_non_cd_rows_are_ignored():
    rows = make_vehicle_pair()
    rows.append(make_raw_row(**{"Test Fuel Type Description": "Tier 2 Cert Gasoline"}))
    rows.append(make_raw_row(**{"Test Procedure Description": "HWFE"}))
    vehicles, report = dc.clean_ev_dataframe(pd.DataFrame(rows))
    assert report["EV charge-depleting city/highway test rows"] == 2
    assert len(vehicles) == 1


def test_mixed_units_give_same_result():
    # One vehicle reports MPGe, an identical one reports kWh/100 mi.
    mpge_city, mpge_hwy = 150.5, 130.5
    kwh_city, kwh_hwy = 3370.5 / mpge_city, 3370.5 / mpge_hwy
    rows = make_vehicle_pair("A", mpge_city, mpge_hwy) + make_vehicle_pair("B", kwh_city, kwh_hwy)
    vehicles, _ = dc.clean_ev_dataframe(pd.DataFrame(rows))
    assert len(vehicles) == 2
    assert to_close(vehicles.loc[0, "city_kwh_per_100mi"], vehicles.loc[1, "city_kwh_per_100mi"])


def test_implausible_consumption_is_removed():
    # 2 kWh/100 mi is far below any real EV.
    raw = pd.DataFrame(make_vehicle_pair(city_fe=2.0, hwy_fe=2.0))
    vehicles, _ = dc.clean_ev_dataframe(raw)
    assert vehicles.empty


@pytest.mark.parametrize(
    "value, expected",
    [(109.0, True), (131.0, True), (184.5, False), (25.0, False), (999.0, False), (np.nan, False)],
)
def test_is_whole_number_mpge(value, expected):
    assert dc.is_whole_number_mpge(value) is expected


def test_whole_number_mpge_on_both_cycles_is_excluded_by_default():
    rows = make_vehicle_pair("LABEL", 131.0, 109.0) + make_vehicle_pair("RAW", 184.5, 173.4)
    vehicles, report = dc.clean_ev_dataframe(pd.DataFrame(rows))
    assert list(vehicles["vehicle_id"]) == ["RAW"]
    assert report["vehicles after excluding suspected window-sticker values"] == 1


def test_flagged_vehicles_can_be_kept_and_marked():
    rows = make_vehicle_pair("LABEL", 131.0, 109.0) + make_vehicle_pair("RAW", 184.5, 173.4)
    vehicles, report = dc.clean_ev_dataframe(pd.DataFrame(rows), exclude_label_values=False)
    flags = dict(zip(vehicles["vehicle_id"], vehicles["label_value_suspect"]))
    assert flags == {"LABEL": True, "RAW": False}
    assert "vehicles after excluding suspected window-sticker values" not in report


def test_whole_number_on_one_cycle_only_is_not_flagged():
    rows = make_vehicle_pair("MIXED", 131.0, 109.4)
    vehicles, _ = dc.clean_ev_dataframe(pd.DataFrame(rows))
    assert len(vehicles) == 1


def test_whole_number_kwh_values_are_not_flagged():
    # 25 and 28 are kWh/100 mi (below the unit threshold), not MPGe.
    rows = make_vehicle_pair("KWH", 25.0, 28.0)
    vehicles, _ = dc.clean_ev_dataframe(pd.DataFrame(rows))
    assert len(vehicles) == 1


def test_one_decimal_repeat_test_unflags_the_cycle():
    # Two city tests, one whole and one decimal: not all whole, so not flagged.
    rows = make_vehicle_pair("REP", 131.0, 109.0) + make_vehicle_pair("REP", 130.6, 109.0)
    vehicles, _ = dc.clean_ev_dataframe(pd.DataFrame(rows))
    assert len(vehicles) == 1


def test_missing_required_column_raises():
    raw = pd.DataFrame(make_vehicle_pair()).drop(columns=["RND_ADJ_FE"])
    with pytest.raises(ValueError, match="RND_ADJ_FE"):
        dc.clean_ev_dataframe(raw)


def test_load_raw_files_rejects_empty_list():
    with pytest.raises(ValueError):
        dc.load_raw_files([])


def test_load_raw_files_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dc.load_raw_files([tmp_path / "does_not_exist.xlsx"])


def test_load_raw_files_round_trip(tmp_path):
    path = tmp_path / "24-testcar.csv"
    pd.DataFrame(make_vehicle_pair()).to_csv(path, index=False)
    raw = dc.load_raw_files([path])
    assert len(raw) == 2
    assert (raw["source_file"] == "24-testcar.csv").all()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def to_close(a, b, rel=1e-9):
    """Return True if a and b agree to a relative tolerance."""
    return math.isclose(a, b, rel_tol=rel)


def test_command_line_entry_point_writes_both_files(tmp_path):
    # Two vehicles saved as a raw EPA-style spreadsheet: one normal, one
    # with whole-number (window-sticker) values.
    rows = make_vehicle_pair("RAW", 150.5, 130.5) + make_vehicle_pair("LABEL", 131.0, 109.0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(rows).to_excel(raw_dir / "24-testcar-test.xlsx", index=False)
    out_dir = tmp_path / "processed"
    dc.main(raw_dir, out_dir)
    clean = pd.read_csv(out_dir / "ev_clean.csv")
    flagged = pd.read_csv(out_dir / "ev_clean_with_flagged.csv")
    assert list(clean["vehicle_id"]) == ["RAW"]
    assert len(flagged) == 2
