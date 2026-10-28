"""Tests of the whole disk as a sum of isothermal tracers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run

from experiments.auriga import HaloSnapshot
from experiments.commands.auriga_tracer_sum import tracer_sum_name
from experiments.heights import Z_GRID, particle_hz, profile_hz, scaled_particle_hz, scaled_profile_hz
from experiments.palicio import PalicioWindow
from experiments.records import load_tracer_sum
from experiments.tracer_sum import MIN_MID_STARS, SumStars, measure_tracer_sum, ring_profiles, tracer_shape
from experiments.whole_disk import K_WINDOW

NU = 60.0  # km/s/kpc: Delta Phi = nu^2 z^2 / 2, in which a tracer of sigma_z is a Gaussian of sigma_z / nu
PHI = 0.5 * NU**2 * Z_GRID**2


def test_tracer_shape_has_unit_mass() -> None:
    """The shape integrates to 1 over both sides of the plane."""
    shape = tracer_shape(PHI, 30.0)
    assert 2.0 * float(np.sum(0.5 * (shape[1:] + shape[:-1]) * np.diff(Z_GRID))) == pytest.approx(1.0)


def two_tracers(rng: np.random.Generator) -> SumStars:
    """Return a cold young and a hot old tracer of the harmonic potential, with masses 3:1."""
    sigma = np.repeat([15.0, 45.0], [300_000, 100_000])
    v_z = rng.normal(0.0, sigma)
    absz = np.abs(rng.normal(0.0, sigma / NU))
    age = np.repeat([1.5, 7.5], [300_000, 100_000])
    return SumStars(np.full(age.size, 8.0), absz, v_z, np.ones(age.size), age)


def test_the_mixture_recovers_the_heights_that_a_single_tracer_misses() -> None:
    """Summed by age, the tracers give h_Z of the stars in any window; one tracer for all of them does not."""
    stars = two_tracers(np.random.default_rng(2))
    ring = ring_profiles(stars, PHI)
    assert ring.unmodeled_single == 0.0 and ring.unmodeled_mixture == 0.0
    measured = particle_hz(stars.absz, stars.mass, 0.5), scaled_particle_hz(stars.absz, stars.mass, K_WINDOW)[0]
    mixture = profile_hz(ring.mixture, 0.5), scaled_profile_hz(ring.mixture, K_WINDOW)[0]
    single = profile_hz(ring.single, 0.5), scaled_profile_hz(ring.single, K_WINDOW)[0]
    np.testing.assert_allclose(mixture, measured, rtol=0.02)
    assert single[1] > 1.05 * measured[1]


def test_cells_without_midplane_stars_and_ages_outside_the_bins_are_unmodeled() -> None:
    """A sparse age bin and stars older than the last edge leave their mass out of the mixture."""
    stars = two_tracers(np.random.default_rng(3))
    sparse = SumStars(
        np.full(MIN_MID_STARS + 10, 8.0),
        np.full(MIN_MID_STARS + 10, 1.0),
        np.zeros(MIN_MID_STARS + 10),
        np.ones(MIN_MID_STARS + 10),
        np.where(np.arange(MIN_MID_STARS + 10) < 10, 20.0, 12.5),
    )
    joined = SumStars(
        *(np.concatenate([getattr(stars, k), getattr(sparse, k)]) for k in ("r", "absz", "v_z", "mass", "age"))
    )
    ring = ring_profiles(joined, PHI)
    assert ring.unmodeled_mixture == MIN_MID_STARS + 10
    assert ring.unmodeled_single == 0.0
    empty = ring_profiles(sparse, PHI)
    assert empty.unmodeled_single == MIN_MID_STARS + 10 and not np.any(empty.single)


def test_measure_tracer_sum_on_the_mock(mock_run: MockRun) -> None:
    """The rings and the whole range are those of the analysis, with its measured h_Z, and both predictions exist."""
    window = mock_run.analysis.whole_disk.window
    result = measure_tracer_sum(mock_run.prepared.galaxy, mock_run.prepared.potential, window, n_draws=2)
    whole_disk = mock_run.analysis.whole_disk
    assert [(r.r_lo, r.r_hi) for r in result.rings] == [(r.r_lo, r.r_hi) for r in whole_disk.rings]
    assert result.whole.measured.palicio.value == whole_disk.whole.heights.palicio
    assert result.whole.measured.scaled.value == whole_disk.whole.heights.scaled
    assert np.isfinite(result.whole.ratio_mixture.scaled.value) and np.isfinite(result.whole.ratio_single.scaled.value)
    assert 0.0 <= result.whole.unmodeled.mixture < 1.0


def test_tracer_sum_command_writes_a_record(mock_run: MockRun, tmp_path: Path) -> None:
    """``tracer-sum`` measures a halo (the mock, linked as Au5) in a given window and writes a record it can read."""
    root = tmp_path / "auriga"
    halo_dir = HaloSnapshot(5).local_dir(root)
    halo_dir.parent.mkdir(parents=True)
    halo_dir.symlink_to(MOCK_SNAPSHOT.local_dir(mock_run.root))
    out_dir = tmp_path / "results"
    output = invoke(
        "auriga", "tracer-sum", "5", "--root", str(root), "--z-lim", "1", "--l-kpc", "20", "--n-draws", "2",
        "--out-dir", str(out_dir),
    )  # fmt: skip
    assert output.startswith("Au5:\n    R (kpc)")
    path = out_dir / tracer_sum_name(HaloSnapshot(5))
    assert json.loads(path.read_text())["tracer_sum"]["window"] == {"l_kpc": 20.0, "z_lim": 1.0}
    assert load_tracer_sum(path).tracer_sum.window == PalicioWindow(20.0, 1.0)


def test_tracer_sum_needs_the_snapshot(tmp_path: Path) -> None:
    """Without a local copy, ``tracer-sum`` says how to get one."""
    result = run("auriga", "tracer-sum", "24", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "download Au24 with `experiments auriga download`" in result.output
