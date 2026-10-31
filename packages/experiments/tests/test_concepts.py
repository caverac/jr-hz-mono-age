"""Tests of the material of the Concepts pages: the expansions of the potential and the fit of every population."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run

from experiments.auriga import HaloSnapshot
from experiments.commands.auriga_concepts import concepts_name
from experiments.concepts import (
    PROFILE_WINDOWS,
    VERTICAL_Z_KPC,
    ConceptsRecord,
    measure_concepts,
    profile_extent,
)
from experiments.populations import AGE_BINS, Z_SAMPLE_MAX_KPC


@pytest.fixture(name="record", scope="module")
def fixture_record(mock_run: MockRun) -> ConceptsRecord:
    """Return the Concepts record of the mock galaxy."""
    return measure_concepts(mock_run.prepared)


def test_the_expansions_add_up_to_the_potential(record: ConceptsRecord) -> None:
    """Delta Phi, the vertical force and v_c^2 of the two expansions sum to those of the whole fit."""
    v = record.vertical
    assert v.z == pytest.approx(list(VERTICAL_Z_KPC))
    np.testing.assert_allclose(np.add(v.phi_cylspline, v.phi_multipole), v.phi_total, atol=1e-6)
    np.testing.assert_allclose(np.add(v.force_cylspline, v.force_multipole), v.force_total, atol=1e-6)
    assert v.phi_total[0] == 0.0 and min(v.force_total[1:]) > 0.0  # the force pulls toward the plane
    rot = record.rotation
    np.testing.assert_allclose(np.square(rot.cylspline) + np.square(rot.multipole), np.square(rot.total), rtol=1e-6)
    grids = record.grids
    assert grids.cylspline_r[0] == 0.0 and grids.multipole_lmax == 8
    assert len(record.meridional.log_cylspline) == len(record.meridional.r)
    assert len(record.meridional.log_cylspline[0]) == len(record.meridional.z)


def test_the_fits_are_those_of_the_analysis(record: ConceptsRecord, mock_run: MockRun) -> None:
    """Every bin has the h_Z, the window and the sigma_z of ``analyze``, and profiles of unit mass."""
    assert [(b.age_lo, b.age_hi) for b in record.bins] == list(AGE_BINS)
    assert record.r_mid == mock_run.analysis.annulus.r_mid
    for ours, theirs in zip(record.bins, mock_run.analysis.mono_age.bins, strict=True):
        np.testing.assert_equal(ours.hz, theirs.heights.measured.value)
        np.testing.assert_equal(ours.window, theirs.heights.window)
        np.testing.assert_equal(ours.sigma_z, theirs.kinematics.sigma_z.value)
        assert len(ours.density) == len(ours.edges) - 1 and len(ours.heights) == len(ours.windows)
        assert np.sum(np.multiply(ours.density, np.diff(ours.edges))) <= 1.0 + 1e-6
    measured = [b for b in record.bins if np.isfinite(b.hz)]
    assert measured
    for b in measured:
        assert b.z[-1] == pytest.approx(profile_extent(b.window))
        for profile in (b.tracer, b.sheet, b.jeans_sigma, b.jeans):
            values = np.asarray(profile)
            assert np.all(values[np.isfinite(values)] >= 0.0)  # the Jeans profile is NaN without slices
        assert b.tracer[0] == max(b.tracer) and b.sheet[0] == max(b.sheet)


def test_profile_extent_covers_windows_and_falls_back() -> None:
    """The profiles reach a few windows, never beyond the sample, and the whole sample without a window."""
    assert profile_extent(0.5) == pytest.approx(PROFILE_WINDOWS * 0.5)
    assert profile_extent(20.0) == Z_SAMPLE_MAX_KPC
    assert profile_extent(float("nan")) == Z_SAMPLE_MAX_KPC


def test_concepts_command_writes_a_record(mock_run: MockRun, tmp_path: Path) -> None:
    """``concepts`` evaluates a halo (the mock, linked as Au5) and writes its record."""
    root = tmp_path / "auriga"
    halo_dir = HaloSnapshot(5).local_dir(root)
    halo_dir.parent.mkdir(parents=True)
    halo_dir.symlink_to(MOCK_SNAPSHOT.local_dir(mock_run.root))
    out_dir = tmp_path / "results"
    output = invoke("auriga", "concepts", "5", "--root", str(root), "--out-dir", str(out_dir))
    assert output.startswith("Au5: R_mid = ")
    record = json.loads((out_dir / concepts_name(HaloSnapshot(5))).read_text())
    assert record["snapshot"]["halo"] == 5
    assert len(record["bins"]) == len(AGE_BINS)


def test_concepts_needs_the_snapshot(tmp_path: Path) -> None:
    """Without a local copy, ``concepts`` says how to get one."""
    result = run("auriga", "concepts", "19", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "download Au19 with `experiments auriga download`" in result.output
