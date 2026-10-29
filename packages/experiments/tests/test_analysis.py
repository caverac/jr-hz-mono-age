"""End-to-end tests: the analysis of a small mock galaxy against the truth it was built from."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import MOCK_SNAPSHOT, MockRun

from experiments.analysis import CURVE_SIGMA_Z, prepare_halo
from experiments.commands._report import report_lines, write_json
from experiments.heights import Z_GRID
from experiments.mock import BOX, CENTER, COSMOLOGY, POTENTIAL_OFFSET, check_mock, rotation_matrix, to_box
from experiments.palicio import PalicioWindow
from experiments.populations import AGE_BINS, R_MAX_KPC, R_MIN_KPC
from experiments.potential import vertical_potential
from experiments.whole_disk import RING_EDGES_KPC, measure_whole_disk


def test_mock_check_recovers_the_truth(mock_run: MockRun) -> None:
    """The frame, the ages, the potential, h_Z and J_R,med of the mock come back close to the truth."""
    check = check_mock(mock_run.truth, mock_run.prepared, mock_run.analysis)
    assert check.spin_error_deg < 0.3
    assert check.age_error_gyr < 1e-9
    assert check.v_circ == pytest.approx(check.v_circ_true, rel=0.02)
    assert check.delta_phi_error < 0.15
    assert [(b.age_lo, b.age_hi) for b in check.bins] == list(AGE_BINS)
    for b in check.bins:
        assert b.hz == pytest.approx(b.hz_true, rel=0.08)
        assert b.jr_median == pytest.approx(b.jr_median_true, rel=0.3)


def test_analysis_of_the_mock(mock_run: MockRun) -> None:
    """The annulus, the checks against the snapshot potential and the whole disk are filled in."""
    analysis = mock_run.analysis
    assert analysis.snapshot == MOCK_SNAPSHOT
    assert (analysis.annulus.r_lo, analysis.annulus.r_hi) == (R_MIN_KPC, R_MAX_KPC)
    assert R_MIN_KPC < analysis.annulus.r_mid < R_MAX_KPC
    assert analysis.annulus.n_stars > 0
    assert analysis.potential_check[0].z < 0.1
    # The stars carry the true potential plus a constant: the residual is that constant, up to the zero point
    # of the fit, and changes with |z| by less than a tenth of the true Delta Phi.
    residual, z = np.array(analysis.residual.residual), np.array(analysis.residual.z)
    assert residual[0] == pytest.approx(POTENTIAL_OFFSET, rel=0.05)
    true_phi = np.interp(z, Z_GRID, vertical_potential(mock_run.truth.potential, analysis.annulus.r_mid))
    assert np.all(np.abs(residual - residual[0]) < 0.1 * true_phi + 2.0)
    disk = analysis.whole_disk
    assert [(r.r_lo, r.r_hi) for r in disk.rings] == list(zip(RING_EDGES_KPC[:-1], RING_EDGES_KPC[1:]))
    assert (disk.whole.r_lo, disk.whole.r_hi) == (4.0, 20.0)


def test_whole_disk_rings_follow_palicio_range(mock_run: MockRun) -> None:
    """With L_kpc = 15 the rings run from 4 to 12 kpc and the whole range from 3 to 15 kpc."""
    disk = measure_whole_disk(mock_run.prepared.galaxy, mock_run.prepared.potential, PalicioWindow(15.0, 0.5))
    assert [(r.r_lo, r.r_hi) for r in disk.rings] == [(4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 12.0)]
    assert (disk.whole.r_lo, disk.whole.r_hi) == (3.0, 15.0)


def test_report_and_record(mock_run: MockRun, tmp_path: Path) -> None:
    """The report has one row per age bin and per ring; the JSON record holds every measurement."""
    lines = report_lines(mock_run.analysis)
    assert lines[0] == "Au0 level 4 Original snapshot 127"
    assert sum(line.startswith(("   0-1 ", "   9-10")) for line in lines) == 2
    assert any("whole disk, 4 <= R < 20 kpc" in line for line in lines)
    path = write_json(mock_run.analysis, tmp_path / "out" / "record.json")
    record = json.loads(path.read_text())
    assert set(record) == {
        "snapshot", "annulus", "potential_check", "residual", "mono_age", "whole_disk", "tracer_curve", "checks"
    }  # fmt: skip
    assert set(record["checks"]) == {"all_stars", "disk", "fine"}
    assert len(record["mono_age"]["bins"]) == len(AGE_BINS)
    assert len(record["tracer_curve"]["sigma_z"]) == len(CURVE_SIGMA_Z)


def test_tracer_curve_of_the_mock(mock_run: MockRun) -> None:
    """The tracer heights grow with sigma_z, with eta of at least 1 wherever the window finds a scale."""
    curve = mock_run.analysis.tracer_curve
    hz, eta = np.array(curve.hz), np.array(curve.eta)
    finite = np.isfinite(hz)
    assert finite[0] and np.all(np.diff(hz[finite]) > 0.0)
    assert np.all(eta[np.isfinite(eta)] > 0.95)


def test_prepare_halo_needs_z_0(mock_run: MockRun, tmp_path: Path) -> None:
    """A snapshot at z > 0 is refused before its potential is fitted."""
    halo_dir = MOCK_SNAPSHOT.local_dir(tmp_path)
    shutil.copytree(MOCK_SNAPSHOT.local_dir(mock_run.root), halo_dir)
    with h5py.File(halo_dir / "snapdir_127" / "snapshot_127.0.hdf5", "r+") as snap:
        snap["Header"].attrs["Redshift"] = 0.5
    with pytest.raises(ValueError, match="is at z = 0.500"):
        prepare_halo(MOCK_SNAPSHOT, tmp_path)


def test_mock_geometry() -> None:
    """The rotation is proper, and the mock's center sits near two faces of the box."""
    rotation = rotation_matrix(37.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(to_box(np.zeros((1, 3)), rotation)[0], CENTER)
    across = to_box(np.array([[-3.0, 0.0, 0.0]]), np.eye(3))[0, 0]
    assert across == pytest.approx(BOX - (3.0e-3 * COSMOLOGY.hubble - CENTER[0]))
