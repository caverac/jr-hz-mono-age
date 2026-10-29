"""Tests of the mean vertical motions of a population."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run

from experiments.auriga import HaloSnapshot
from experiments.commands.auriga_equilibrium import equilibrium_name
from experiments.equilibrium import KMS_PER_KPC_IN_GYR, MotionSample, measure_equilibrium, motions
from experiments.populations import AGE_BINS


def population(n: int, rng: np.random.Generator) -> MotionSample:
    """Return a population with no mean motion, symmetric about its plane."""
    return MotionSample(
        age=np.full(n, 5.0),
        azimuth=rng.uniform(-np.pi, np.pi, n),
        z=rng.exponential(0.4, n) * rng.choice([-1.0, 1.0], n),
        v_z=rng.normal(0.0, 20.0, n),
        initial_mass=np.ones(n),
    )


def test_motions_measure_breathing_bending_asymmetry_and_warp() -> None:
    """Each departure from a steady, symmetric and flat disk shows up in its own measure, and only there."""
    rng = np.random.default_rng(4)
    rows = population(400_000, rng)
    calm = motions(rows, 2.0)
    assert np.all(np.abs(np.array(calm)) < [0.25, 0.1, 0.01, 0.01])  # about three times the noise of each

    breathing = MotionSample(**{**rows.__dict__, "v_z": rows.v_z + 6.0 * rows.z})
    assert motions(breathing, 2.0)[0] == pytest.approx(6.0 * KMS_PER_KPC_IN_GYR, abs=0.1)

    bending = MotionSample(**{**rows.__dict__, "v_z": rows.v_z + 5.0})
    assert motions(bending, 2.0)[1] == pytest.approx(5.0, abs=0.1)

    asymmetry = motions(rows.select((rows.z > 0.0) | (rng.uniform(size=rows.z.size) < 0.5)), 2.0)[2]
    assert asymmetry == pytest.approx(1.0 / 3.0, abs=0.01)

    warped = MotionSample(**{**rows.__dict__, "z": rows.z + 0.1 * np.cos(rows.azimuth - 1.0)})
    assert motions(warped, 4.0)[3] == pytest.approx(0.1 * np.sinc(1.0 / 8.0), abs=0.01)


def test_motions_of_an_empty_window_are_nan() -> None:
    """Without stars inside the window there is nothing to average."""
    rows = population(100, np.random.default_rng(1))
    assert np.all(np.isnan(motions(rows, 0.0)))


def test_measure_equilibrium_on_the_mock_matches_the_analysis(mock_run: MockRun) -> None:
    """Every age bin is measured on the stars of the analysis, and the steady mock has no strong asymmetry."""
    bins = measure_equilibrium(mock_run.prepared.galaxy, n_bootstrap=2)
    assert [(b.age_lo, b.age_hi) for b in bins] == list(AGE_BINS)
    for ours, theirs in zip(bins, mock_run.analysis.mono_age.bins, strict=True):
        np.testing.assert_equal(ours.hz.value, theirs.heights.measured.value)
        np.testing.assert_equal(ours.window, theirs.heights.window)
    measured = [b for b in bins if np.isfinite(b.hz.value)]
    assert measured and all(abs(b.motions.asymmetry.value) < 0.3 for b in measured)


def test_equilibrium_command_writes_a_record(mock_run: MockRun, tmp_path: Path) -> None:
    """``equilibrium`` measures a halo (the mock, linked as Au5) and writes its record."""
    root = tmp_path / "auriga"
    halo_dir = HaloSnapshot(5).local_dir(root)
    halo_dir.parent.mkdir(parents=True)
    halo_dir.symlink_to(MOCK_SNAPSHOT.local_dir(mock_run.root))
    out_dir = tmp_path / "results"
    output = invoke("auriga", "equilibrium", "5", "--root", str(root), "--n-bootstrap", "2", "--out-dir", str(out_dir))
    assert output.startswith("Au5:\n    age  N_stars")
    record = json.loads((out_dir / equilibrium_name(HaloSnapshot(5))).read_text())
    assert record["snapshot"]["halo"] == 5
    assert len(record["bins"]) == len(AGE_BINS)


def test_equilibrium_needs_the_snapshot(tmp_path: Path) -> None:
    """Without a local copy, ``equilibrium`` says how to get one."""
    result = run("auriga", "equilibrium", "19", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "download Au19 with `experiments auriga download`" in result.output
