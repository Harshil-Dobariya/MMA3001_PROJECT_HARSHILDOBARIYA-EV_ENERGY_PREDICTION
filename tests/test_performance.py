"""Tests for ev_energy.performance and the chunked physics kernel.

The key check: the baseline (pure Python), intermediate and improved
(vectorised, chunked) implementations must give the same answers. A faster
version is only an improvement if its results are unchanged.
"""

import numpy as np
import pytest

from ev_energy import ml_models as ml
from ev_energy import performance as pf
from ev_energy import physics_model as pm
from test_ml_models import make_vehicles
from test_validation import small_cycles


def random_vehicles(n=25, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(1500, 3500, n), rng.uniform(80, 300, n),
            rng.uniform(-2, 10, n), rng.uniform(0.2, 0.8, n))


@pytest.mark.parametrize("name", list(pf.IMPLEMENTATIONS))
def test_every_implementation_matches_the_vectorised_result(name):
    cycle = small_cycles()["city"]
    args = random_vehicles()
    reference = pm.cycle_energy_terms(cycle, *args, chunk_size=None)
    result = pf.IMPLEMENTATIONS[name](cycle, *args)
    for key in ("e_pos_j", "e_neg_j"):
        assert np.allclose(result[key], reference[key], rtol=1e-12)


@pytest.mark.parametrize("chunk", [1, 3, 7, 25, 100, None])
def test_chunked_kernel_is_identical_for_any_chunk_size(chunk):
    cycle = small_cycles()["hwy"]
    args = random_vehicles()
    reference = pm.cycle_energy_terms(cycle, *args, chunk_size=None)
    result = pm.cycle_energy_terms(cycle, *args, chunk_size=chunk)
    for key in ("e_pos_j", "e_neg_j", "dist_m", "duration_s"):
        assert np.array_equal(result[key], reference[key])


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        pm.cycle_energy_terms(small_cycles()["city"], *random_vehicles(), chunk_size=0)


def test_single_vehicle_scalar_inputs_work():
    terms = pm.cycle_energy_terms(small_cycles()["city"], 2000.0, 150.0, 2.0, 0.4)
    assert terms["e_pos_j"].shape == (1,)


def test_time_function_reports_ordered_statistics():
    t = pf.time_function(lambda: sum(range(1000)), repeats=5)
    assert 0 < t["min_s"] <= t["median_s"] <= t["max_s"]


def test_time_function_rejects_zero_repeats():
    with pytest.raises(ValueError):
        pf.time_function(lambda: None, repeats=0)


def test_peak_memory_detects_a_large_allocation():
    # Allocating 1,000,000 doubles needs about 8 MB.
    assert pf.peak_memory_mb(lambda: np.ones(1_000_000)) == pytest.approx(8.0, rel=0.1)


def test_flop_and_byte_counts():
    assert pf.kernel_flops(2, 3) == 60
    assert pf.numpy_bytes_moved(2, 3) == 21 * 8 * 6
    assert pf.arithmetic_intensity(10, 168) == pytest.approx(10 / 168)


def test_arithmetic_intensity_rejects_zero_bytes():
    with pytest.raises(ValueError):
        pf.arithmetic_intensity(10, 0)


def test_fused_kernel_has_higher_intensity_than_numpy():
    numpy_ai = pf.arithmetic_intensity(pf.kernel_flops(1, 1), pf.numpy_bytes_moved(1, 1))
    assert pf.fused_kernel_arithmetic_intensity() > numpy_ai


def test_tile_vehicles_reaches_requested_size():
    tiled = pf.tile_vehicles([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], 5)
    assert all(len(x) == 5 for x in tiled)
    assert tiled[0].tolist() == [1.0, 2.0, 1.0, 2.0, 1.0]


def test_parallel_tuning_chooses_the_same_hyperparameters():
    df = make_vehicles()
    assert ml.tune(df, "tree", "city", n_jobs=1) == ml.tune(df, "tree", "city", n_jobs=2)
