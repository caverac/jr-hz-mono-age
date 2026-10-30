"""Tests of the proxy J_R,w from a face-on map of the median J_R."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run

from experiments.auriga import HaloSnapshot
from experiments.commands._report import write_json
from experiments.commands.auriga_analyze import result_name
from experiments.commands.auriga_jr_map import jr_map_name
from experiments.commands.auriga_tracer_sum import tracer_sum_name
from experiments.jr_map import (
    CELL_KPC,
    L_SUN_KPC_KMS,
    MIN_PER_CELL,
    PALICIO_INTERCEPT_LSUN,
    PALICIO_SLOPE_LSUN_PER_KPC,
    JrMap,
    JrMapRecord,
    JrPoint,
    cell_medians,
    fit_line,
    jr_relation,
    map_proxy,
    measure_jr_map,
)
from experiments.palicio import PalicioWindow
from experiments.populations import Estimate
from experiments.records import load_jr_map
from experiments.stats import weighted_median
from experiments.tracer_sum import SumHeights, SumRange, TracerSum, TracerSumRecord, Unmodeled


def test_weighted_median() -> None:
    """The weighted median is the value below which half of the weight lies."""
    values = np.array([3.0, 1.0, 2.0, 10.0])
    assert weighted_median(values, np.ones(4)) == 2.0
    assert weighted_median(values, np.array([1.0, 1.0, 1.0, 5.0])) == 10.0


def test_cell_medians_follow_the_cells() -> None:
    """With J_R equal to the radius, each cell's median is near the radius of its center; sparse cells drop out."""
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-4.0, 4.0, (2, 400_000))
    jr = np.hypot(x, y)
    radius, median = cell_medians(x, y, jr, np.ones(x.size), 4.0)
    assert radius.size == median.size == int(np.ceil(8.0 / CELL_KPC)) ** 2
    np.testing.assert_allclose(median, radius, atol=CELL_KPC)
    sparse_radius, _ = cell_medians(x[:MIN_PER_CELL], y[:MIN_PER_CELL], jr[:MIN_PER_CELL], np.ones(MIN_PER_CELL), 4.0)
    assert sparse_radius.size == 0


def test_map_proxy_splits_the_cells_of_the_radial_range() -> None:
    """With J_R equal to R on a uniform disk, J_R,w is the median radius of the cells in 0.2 L <= R < L."""
    rng = np.random.default_rng(3)
    window = PalicioWindow(10.0, 1.0)
    x, y = rng.uniform(-10.0, 10.0, (2, 2_000_000))
    jr = np.hypot(x, y)
    jr[:10] = np.nan
    proxy = map_proxy(x, y, jr, np.ones(x.size), window)
    area_median = np.sqrt(0.5 * (10.0**2 + 2.0**2))
    assert proxy.jr_w == pytest.approx(area_median, rel=0.03)
    assert proxy.jr_cells_lo < proxy.jr_w < proxy.jr_cells_hi
    assert proxy.n_stars == x.size - 10
    empty = map_proxy(x[:5], y[:5], jr[:5], np.ones(5), window)
    assert empty.n_cells == 0 and np.isnan(empty.jr_w)


def test_measure_jr_map_on_the_mock(mock_run: MockRun) -> None:
    """The mock disk gives a finite J_R,w within the spread of its cells."""
    window = PalicioWindow(12.0, 1.0)
    proxy = measure_jr_map(mock_run.prepared.galaxy, mock_run.prepared.potential, window)
    assert proxy.n_cells > 0
    assert proxy.jr_cells_lo <= proxy.jr_w <= proxy.jr_cells_hi


def test_jr_map_command_writes_a_record(mock_run: MockRun, tmp_path: Path) -> None:
    """``jr-map`` measures a halo (the mock, linked as Au5) in a given window and writes its record."""
    root = tmp_path / "auriga"
    halo_dir = HaloSnapshot(5).local_dir(root)
    halo_dir.parent.mkdir(parents=True)
    halo_dir.symlink_to(MOCK_SNAPSHOT.local_dir(mock_run.root))
    out_dir = tmp_path / "results"
    output = invoke(
        "auriga", "jr-map", "5", "--root", str(root), "--z-lim", "1", "--l-kpc", "12", "--out-dir", str(out_dir)
    )
    assert output.startswith("Au5: J_R,w")
    record = json.loads((out_dir / jr_map_name(HaloSnapshot(5))).read_text())
    assert record["jr_map"]["window"] == {"l_kpc": 12.0, "z_lim": 1.0}
    assert record["snapshot"]["halo"] == 5


def test_jr_map_needs_the_snapshot(tmp_path: Path) -> None:
    """Without a local copy, ``jr-map`` says how to get one."""
    result = run("auriga", "jr-map", "24", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "download Au24 with `experiments auriga download`" in result.output


def test_fit_line_recovers_a_line_and_its_scatter() -> None:
    """Points on a line give it with no error; scattered points give errors that grow with the scatter."""
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    exact = fit_line(x, 3.0 + 70.0 * x)
    assert (exact.slope, exact.intercept) == pytest.approx((70.0, 3.0))
    assert exact.slope_err == pytest.approx(0.0, abs=1e-9)
    scattered = fit_line(x, 3.0 + 70.0 * x + np.array([1.0, -1.0, 1.0, -1.0, 1.0]))
    assert scattered.slope_err > 0.0 and scattered.intercept_err > 0.0
    assert fit_line(x[:2], 3.0 + 70.0 * x[:2]).slope_err == 0.0


def test_jr_relation_fits_both_windows() -> None:
    """The lines run through J_R,w against each h_Z, and the slope of Palicio et al. is in km/s."""
    points = [
        JrPoint(h, 10.0 + 50.0 * hz, hz, 0.8 * hz, 0.9 * hz, 0.7 * hz) for h, hz in ((2, 0.5), (3, 1.0), (16, 1.5))
    ]
    relation = jr_relation(points)
    assert relation.lines.line_palicio.slope == pytest.approx(50.0)
    assert relation.lines.line_scaled.slope == pytest.approx(50.0 / 0.8)
    assert relation.lines.line_palicio_mixture.slope == pytest.approx(50.0 / 0.9)
    assert relation.lines.line_scaled_mixture.slope == pytest.approx(50.0 / 0.7)
    assert relation.palicio_slope == pytest.approx(PALICIO_SLOPE_LSUN_PER_KPC * L_SUN_KPC_KMS)
    assert 72.0 < relation.palicio_slope < 73.0
    assert relation.palicio_intercept == pytest.approx(PALICIO_INTERCEPT_LSUN * L_SUN_KPC_KMS)


def test_jr_relation_leaves_each_galaxy_out() -> None:
    """Each refit drops one galaxy, so a galaxy far from the others moves the slope most."""
    points = [JrPoint(h, 10.0 + 50.0 * hz, hz, hz, hz, hz) for h, hz in ((2, 0.5), (3, 1.0), (16, 1.5), (19, 2.0))]
    points.append(JrPoint(24, 400.0, 2.5, 2.5, 2.5, 2.5))
    relation = jr_relation(points)
    assert [left.halo for left in relation.leave_one_out] == [2, 3, 16, 19, 24]
    without = {left.halo: left.lines.line_scaled.slope for left in relation.leave_one_out}
    assert without[24] == pytest.approx(50.0)
    assert relation.lines.line_scaled.slope > 100.0
    assert max(without, key=lambda halo: abs(without[halo] - relation.lines.line_scaled.slope)) == 24


def test_jr_relation_command(mock_run: MockRun, tmp_path: Path) -> None:
    """``jr-relation`` reads the J_R maps, the analyses and the tracer sums of the halos and writes the lines."""
    for halo, jr_w in ((5, 20.0), (6, 30.0), (7, 45.0)):
        snap = HaloSnapshot(halo)
        path = write_json(mock_run.analysis, tmp_path / result_name(snap))
        record = json.loads(path.read_text())
        record["whole_disk"]["whole"]["heights"].update(palicio=jr_w / 40.0, scaled=jr_w / 50.0)
        path.write_text(json.dumps(record))
        proxy = JrMap(PalicioWindow(20.0, 1.0), 100, 10, jr_w, 0.5 * jr_w, 2.0 * jr_w)
        write_json(JrMapRecord(snap, proxy), tmp_path / jr_map_name(snap))
        write_json(TracerSumRecord(snap, predicted(jr_w / 45.0, jr_w / 60.0)), tmp_path / tracer_sum_name(snap))
    assert load_jr_map(tmp_path / jr_map_name(HaloSnapshot(6))).jr_map.jr_w == 30.0
    output = invoke("auriga", "jr-relation", "5", "6", "7", "--data-dir", str(tmp_path))
    assert "Palicio et al.: J_R,w = 5.8 + 72.6 h_Z (kpc km/s)" in output
    assert "self-consistent slope without one galaxy: 50.0 to 50.0 km/s" in output
    record = json.loads((tmp_path / "jr-relation.json").read_text())
    assert [p["halo"] for p in record["points"]] == [5, 6, 7]
    assert record["lines"]["line_scaled_mixture"]["slope"] == pytest.approx(60.0)
    assert [left["halo"] for left in record["leave_one_out"]] == [5, 6, 7]
    missing = run("auriga", "jr-relation", "8", "--data-dir", str(tmp_path))
    assert missing.exit_code == 1 and "experiments auriga jr-map" in missing.output
    (tmp_path / tracer_sum_name(HaloSnapshot(7))).unlink()
    unsummed = run("auriga", "jr-relation", "7", "--data-dir", str(tmp_path))
    assert unsummed.exit_code == 1 and "experiments auriga tracer-sum" in unsummed.output


def predicted(hz_palicio: float, hz_scaled: float) -> TracerSum:
    """Return a tracer sum whose whole range has the mixture heights *hz_palicio* and *hz_scaled*."""
    one = Estimate(1.0, 0.0)
    heights = SumHeights(Estimate(hz_palicio, 0.0), Estimate(hz_scaled, 0.0))
    whole = SumRange(
        4.0, 20.0, Unmodeled(0.0, 0.0), heights, heights, heights, SumHeights(one, one), SumHeights(one, one)
    )
    return TracerSum(PalicioWindow(20.0, 1.0), [], whole)
