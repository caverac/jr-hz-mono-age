"""Tests of the checks of the mono-age result on the synthetic annulus of the population tests.

In the exact sample the heights follow the tracer of the linear potential, so every check
finds ratios near 1; stars marked as off the disk are left out of the disk check alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_populations import G_LINEAR, R_MID_KPC, SIGMA_SURF, linear_coeval, synthetic_sample, tracer_height

from experiments.heights import Z_GRID
from experiments.palicio import PalicioWindow
from experiments.populations import AGE_BINS, Method, VerticalModel
from experiments.robustness import (
    CHECK_RANGES,
    DISK_CIRCULARITY,
    FINE_BINS,
    N_MIN_MID,
    Checks,
    measure_checks,
)


@pytest.fixture(name="checks", scope="module")
def fixture_checks() -> Checks:
    """Return the checks of the exact sample, with the first fifth of the stars of every bin off the disk."""
    sample = synthetic_sample(tracer_height)
    eps = np.where(np.arange(sample.z.size) % 5 == 0, 0.2, 0.95)
    phi = G_LINEAR * Z_GRID
    model = VerticalModel(SIGMA_SURF, phi, phi, R_MID_KPC, linear_coeval())
    return measure_checks(sample, eps, model, Method(PalicioWindow(20.0, 1.0), 2.0, 20))


def test_the_sets_cover_their_bins(checks: Checks) -> None:
    """The 1 Gyr sets have the bins of the main measurement, the fine set those of FINE_BINS."""
    assert [(b.age_lo, b.age_hi) for b in checks.all_stars.bins] == list(AGE_BINS)
    assert [(b.age_lo, b.age_hi) for b in checks.disk.bins] == list(AGE_BINS)
    assert [(b.age_lo, b.age_hi) for b in checks.fine.bins] == list(FINE_BINS)
    for check in (checks.all_stars, checks.disk, checks.fine):
        assert [(r.lo, r.hi) for r in check.ranges] == list(CHECK_RANGES)


def test_the_disk_check_keeps_the_disk_stars(checks: Checks) -> None:
    """All stars keep every bin whole; the disk check keeps the four fifths above DISK_CIRCULARITY."""
    assert DISK_CIRCULARITY == pytest.approx(0.7)
    assert all(b.mass_fraction == 1.0 for b in checks.all_stars.bins)
    assert all(b.mass_fraction == pytest.approx(0.8, abs=0.01) for b in checks.disk.bins)


def test_ratios_follow_the_tracer(checks: Checks) -> None:
    """Heights that follow the tracer give ratios near 1 and trends within 3 errors of 0, clipped or not."""
    for check in (checks.all_stars, checks.disk):
        for b in check.bins:
            v = b.values
            assert v.tracer.value == pytest.approx(1.0, abs=4.0 * v.tracer.err)
            assert v.tracer_unclipped.value == pytest.approx(v.tracer.value, rel=0.05)
        for r in check.ranges:
            assert abs(r.tracer.value) < 3.0 * r.tracer.err
            assert abs(r.tracer_unclipped.value) < 3.0 * r.tracer_unclipped.err


def test_fine_bins_without_stars_at_the_plane_have_no_values() -> None:
    """A bin with fewer than N_MIN_MID stars at the plane, or none at all, gives NaN and no trend from it."""
    sample = synthetic_sample(tracer_height)
    sparse = sample.select(sample.age < 1.0)
    phi = G_LINEAR * Z_GRID
    model = VerticalModel(SIGMA_SURF, phi, phi, R_MID_KPC, linear_coeval())
    eps = np.ones(sparse.z.size)
    checks = measure_checks(sparse, eps, model, Method(PalicioWindow(20.0, 1.0), 2.0, 20))
    empty = checks.all_stars.bins[-1]
    assert np.isnan(empty.mass_fraction) and np.isnan(empty.values.tracer.value) and np.isnan(empty.age_median)
    assert N_MIN_MID == 30
    assert all(np.isnan(r.tracer.value) and np.isnan(r.tracer_unclipped.value) for r in checks.all_stars.ranges)
