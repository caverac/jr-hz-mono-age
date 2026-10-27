"""Tests of the injection test of the scale-height estimators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import invoke

from experiments.heights import Z_GRID
from experiments.injection import (
    MIXTURE_FRACTION,
    SCALES_KPC,
    inject,
    profile,
    run_injection,
    sample_heights,
    spread,
)


@pytest.mark.parametrize("family", sorted(SCALES_KPC))
def test_samples_follow_their_profile(family: str) -> None:
    """The mean height of a large sample equals the mean height of the profile it is drawn from."""
    absz = sample_heights(family, 0.5, 200_000, np.random.default_rng(1))
    rho = profile(family, 0.5)
    mean = np.trapezoid(Z_GRID * rho, Z_GRID) / np.trapezoid(rho, Z_GRID)
    assert absz.mean() == pytest.approx(mean, rel=0.01)


def test_mixture_holds_its_thick_fraction() -> None:
    """The mixture is 70 percent thin and 30 percent thick, by mass, in its profile."""
    rho = profile("mixture", 0.5)
    thin = (1.0 - MIXTURE_FRACTION) * np.exp(-Z_GRID / 0.5) / 0.5
    assert np.trapezoid(rho - thin, Z_GRID) == pytest.approx(MIXTURE_FRACTION, rel=1e-3)


def test_unknown_family() -> None:
    """A family that does not exist is an error, for the profile and for the sample."""
    with pytest.raises(ValueError, match="unknown profile family"):
        profile("gaussian", 1.0)
    with pytest.raises(ValueError, match="unknown profile family"):
        sample_heights("gaussian", 1.0, 10, np.random.default_rng(1))


def test_spread() -> None:
    """The spread ignores the draws without a solution, and is NaN when no draw has one."""
    values = np.array([1.0, 2.0, 3.0, np.nan])
    result = spread(values)
    assert result.median == 2.0
    assert result.lo == pytest.approx(1.32)
    assert result.hi == pytest.approx(2.68)
    assert all(np.isnan(v) for v in vars(spread(np.array([np.nan, np.nan]))).values())


def test_exponential_is_recovered_in_both_windows() -> None:
    """An exponential well inside the window comes back at its scale in both windows."""
    point = inject("exponential", 0.3, np.random.default_rng(2), 5_000, 8)
    assert point.reference == pytest.approx(0.3, rel=1e-3)
    assert point.fixed.median == pytest.approx(1.0, abs=0.03)
    assert point.scaled.median == pytest.approx(1.0, abs=0.03)
    assert point.fixed_failed == 0.0


def test_sech2_is_biased_in_the_fixed_window() -> None:
    """A sech^2 layer as thick as the window is overestimated in it, and recovered in the self-consistent window."""
    point = inject("sech2", 1.0, np.random.default_rng(3), 5_000, 8)
    assert point.fixed.median > 1.3
    assert point.scaled.median == pytest.approx(1.0, abs=0.03)


def test_run_injection_covers_every_family_and_scale() -> None:
    """The test has one point per family and scale, in the order of ``SCALES_KPC``."""
    injection = run_injection(n_stars=300, draws=2)
    expected = [(family, scale) for family, scales in SCALES_KPC.items() for scale in scales]
    assert [(p.family, p.scale) for p in injection.points] == expected
    assert (injection.n_stars, injection.draws) == (300, 2)


def test_window_injection_command(tmp_path: Path) -> None:
    """``window-injection`` prints a line per point and writes a record of standard JSON."""
    output = invoke("window-injection", "--n-stars", "300", "--draws", "2", "--out-dir", str(tmp_path))
    record_path = tmp_path / "window-injection.json"
    assert "wrote" in output
    text = record_path.read_text()
    assert "NaN" not in text
    record = json.loads(text)
    assert len(record["points"]) == sum(len(scales) for scales in SCALES_KPC.values())
