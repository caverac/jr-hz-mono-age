"""Tests of the populations of Imig et al. one at a time against the counts of the same cell.

The synthetic world gives the populations two bins of [M/H] and four age bins, and APOGEE
stars whose |z| is exponential with a scale that grows with age; one star in ten lies on the
high-[alpha/M] side of the split.
"""

from __future__ import annotations

import numpy as np
import pytest
from mw_fixtures import write_world

from experiments import mw_imig_cells
from experiments.commands.milky_way import cell_lines
from experiments.heights import Z_GRID
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import Targeting, TargetingBox, read_apogee, read_gaia, read_lamost, with_distmass_ages
from experiments.mw_imig import ImigPopulations, read_imig
from experiments.mw_imig_cells import (
    MIN_CELL_KINEMATICS,
    CellKinematics,
    CellOurs,
    CellRatios,
    CellSettings,
    CellTheirs,
    CellTrends,
    DispersionSlopes,
    ImigCell,
    ImigCells,
    MetallicityCounts,
    MetallicitySlopes,
    RatioTrends,
    imig_low_alpha,
    lamost_metallicity_counts,
    measure_cells,
    metallicity_counts,
    tracer_estimators,
)
from experiments.mw_pipeline import source_path
from experiments.mw_sources import GAIA_PATH
from experiments.populations import Estimate, Trend

MODEL = SolarModel(np.asarray(900.0 * Z_GRID), 50.0)
Metals = tuple[list[ImigPopulations], list[MetallicityCounts]]


@pytest.fixture(name="metals", scope="module")
def fixture_metals(tmp_path_factory: pytest.TempPathFactory) -> Metals:
    """Return the populations of the synthetic world and our counts of their [M/H] bins."""
    root = tmp_path_factory.mktemp("milky-way-cells")
    write_world(root)
    apogee = with_distmass_ages(
        read_apogee(source_path(root, "astronn"), source_path(root, "allstar")), source_path(root, "distmass")
    )
    populations = read_imig(source_path(root, "imig"))
    return populations, metallicity_counts(read_gaia(root / GAIA_PATH), apogee, populations)


def test_imig_low_alpha_follows_their_split() -> None:
    """At [M/H] = 0 the split lies at [alpha/M] = 0.1353 - 0.05; a star without abundances is not low-alpha."""
    m_h = np.array([0.0, 0.0, np.nan, 0.3])
    alpha_m = np.array([0.08, 0.09, 0.0, np.nan])
    assert imig_low_alpha(m_h, alpha_m).tolist() == [True, False, False, False]


def test_each_metallicity_bin_counts_its_low_alpha_stars(metals: Metals) -> None:
    """The two [M/H] bins of the table give two samples of the same Gaia stars, with disjoint populations."""
    _, metallicities = metals
    assert [(m.m_h_lo, m.m_h_hi) for m in metallicities] == [
        pytest.approx((-0.4, -0.1)),
        pytest.approx((-0.1, 0.2)),
    ]
    first, second = (m.counts for m in metallicities)
    assert np.array_equal(first.gaia_absz, second.gaia_absz)
    assert np.array_equal(first.tracer_age, second.tracer_age)
    assert not np.any(first.tracer_in_disk & second.tracer_in_disk)
    assert 0 < first.tracer_in_disk.sum() < first.tracer_in_disk.size
    assert all(m.kinematics.age.size > 0 for m in metallicities)


def test_cells_compare_each_population_with_its_counts(metals: Metals) -> None:
    """Every population gets a cell; the measured ones have finite heights, ratios and a summary."""
    populations, metallicities = metals
    cells = measure_cells(populations, metallicities, MODEL, np.random.default_rng(1), CellSettings(5, 20))
    assert len(cells.cells) == 8 and cells.min_matches == 20
    measured = [c for c in cells.cells if c.n_matched >= 20]
    assert len(measured) == cells.n_measured >= 3
    for cell in measured:
        ours, theirs, ratios = cell.ours, cell.theirs, cell.ratios
        assert ours.windowed.value > 0.0 and ratios.counted.value == pytest.approx(
            ours.windowed.value / theirs.counted.value
        )
        assert ratios.counted.err > 0.0 and cell.n_stars == 500 and np.isfinite(ours.gradient)
        assert theirs.sun.value > 0.0 and theirs.window > 0.0 and theirs.counted.err > 0.0
        assert theirs.exponential.value > 0.0 and ratios.exponential.value == pytest.approx(
            ours.exponential.value / theirs.exponential.value
        )
        assert len(ours.shape) == 12 and all(np.isfinite(ours.shape[:6]))
        kin = cell.kinematics
        if kin.n_kin >= MIN_CELL_KINEMATICS:
            assert kin.sigma_z.value > 0.0 and ratios.tracer.value == pytest.approx(
                kin.tracer.value / ours.windowed.value
            )
        else:
            assert np.isnan(kin.sigma_z.value) and np.isnan(ratios.tracer.value)
    assert any(c.kinematics.n_kin >= MIN_CELL_KINEMATICS for c in measured)
    assert np.isfinite(cells.median_ratio)
    trends = cells.trends
    assert all(np.isfinite([t.age.value, t.metallicity.value]).all() for t in (trends.counted, trends.exponential))
    slopes = cells.slopes
    assert all(np.isfinite(t.value) for t in (slopes.counts, slopes.counts_exponential, slopes.imig, slopes.imig_sun))
    dispersion = cells.dispersion
    assert all(
        np.isfinite(t.value) and t.err > 0.0
        for t in (dispersion.counts_exponential, dispersion.imig_exponential, dispersion.imig_sun)
    )


def test_cells_without_enough_matches_have_no_counts(metals: Metals) -> None:
    """Below the minimum of matches a cell keeps their h_Z at their Sun but has no counts, and the summary is NaN."""
    populations, metallicities = metals
    cells = measure_cells(populations, metallicities, MODEL, np.random.default_rng(1), CellSettings(3, 10**6))
    assert cells.n_measured == 0 and np.isnan(cells.median_ratio) and np.isnan(cells.trends.counted.age.value)
    assert np.isnan(cells.slopes.counts.value) and np.isnan(cells.slopes.imig.value)
    for c in cells.cells:
        assert np.isnan(c.ours.windowed.value) and np.isnan(c.ratios.counted.value) and np.isnan(c.theirs.counted.value)
        assert np.isnan(c.theirs.exponential.value) and np.isnan(c.ratios.exponential.value)
        assert c.theirs.sun.value > 0.0 and c.theirs.sun.err > 0.0 and c.ours.shape == [] and np.isnan(c.ours.gradient)


def test_cells_without_enough_kinematic_stars_have_no_tracer(metals: Metals, monkeypatch: pytest.MonkeyPatch) -> None:
    """A measured cell with too few kinematic stars keeps its heights, without sigma_z or a tracer."""
    monkeypatch.setattr(mw_imig_cells, "MIN_CELL_KINEMATICS", 10**6)
    populations, metallicities = metals
    cells = measure_cells(populations, metallicities, MODEL, np.random.default_rng(1), CellSettings(2, 20))
    measured = [c for c in cells.cells if c.n_matched >= 20]
    assert measured and all(c.ours.windowed.value > 0.0 for c in measured)
    assert all(np.isnan(c.kinematics.sigma_z.value) and np.isnan(c.kinematics.tracer.value) for c in measured)
    assert np.isnan(cells.slopes.tracer.value) and np.isnan(cells.slopes.sigma_z.value)
    assert np.isnan(cells.dispersion.counts_exponential.value) and np.isnan(cells.dispersion.imig_sun.value)
    assert np.isnan(cells.trends.tracer.age.value) and np.isfinite(cells.trends.counted.age.value)


def test_the_report_summarizes_each_metallicity_bin() -> None:
    """A bin with measured cells prints their median ratio, and a bin without them prints none."""

    def cell(m_h_lo: float, n_matched: int, ratio: float) -> ImigCell:
        h = Estimate(0.3, 0.01)
        ratios = CellRatios(Estimate(ratio, 0.1), Estimate(1.0, 0.1), Estimate(float("nan"), float("nan")))
        return ImigCell(
            1.0, 1.26, m_h_lo, m_h_lo + 0.1, 500, n_matched,
            CellTheirs(h, 0.3, h, h), CellOurs(h, h, 0.0, []), CellKinematics(0, h, h), ratios,
        )  # fmt: skip

    trend = Trend(-1.0, 0.1, 1.0)
    slopes = MetallicitySlopes(Trend(0.1, 0.1, 1.0), trend, trend, trend, trend, trend)
    ratio_trends = RatioTrends(Trend(0.2, 0.05, 1.0), Trend(0.5, 0.2, 1.0))
    cells = ImigCells(
        1000,
        100,
        [cell(0.0, 150, 1.2), cell(0.0, 300, 1.4), cell(0.1, 50, 2.0)],
        2,
        1.3,
        CellTrends(ratio_trends, ratio_trends, ratio_trends),
        slopes,
        DispersionSlopes(Trend(1.3, 0.1, 1.0), trend, Trend(0.9, 0.1, 1.0)),
    )
    lines = cell_lines(cells, "box")
    assert lines[0].startswith("box, 1,000 matches")
    assert lines[1].split()[-3:] == ["1.30", "1.00", "none"] and lines[2].split()[-3:] == ["none"] * 3
    assert "2 cells: median 1.30" in lines[-3] and "counted +0.20+/-0.05 (+0.50+/-0.20)" in lines[-3]
    assert "imig -1.00+/-0.10" in lines[-2]
    assert lines[-1].endswith("counts_exponential +1.30+/-0.10, imig_exponential -1.00+/-0.10, imig_sun +0.90+/-0.10")


def test_a_targeting_box_keeps_the_stars_selected_alike_everywhere() -> None:
    """A star belongs to the box if it lies in it and its cohort's limits contain the box."""
    targeting = Targeting(
        h=np.array([9.0, 9.0, 9.0, 12.0, 9.0, np.nan]),
        jk0=np.array([0.6, 0.9, 0.6, 0.6, 0.6, 0.6]),
        min_h=np.full(6, 7.0),
        max_h=np.array([12.2, 12.2, 12.2, 12.2, 11.0, 12.2]),
        min_jk=np.array([0.5, 0.5, 0.3, 0.5, 0.5, 0.5]),
        max_jk=np.array([9999.99, 9999.99, 0.8, 9999.99, 9999.99, 9999.99]),
    )
    assert TargetingBox(7.0, 12.0, 0.5, 99.0).holds(targeting).tolist() == [True, True, False, True, False, False]
    assert TargetingBox(7.0, 11.0, 0.5, 0.8).holds(targeting).tolist() == [True, False, True, False, True, False]


def test_lamost_splits_its_counts_by_its_own_metallicity(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The LAMOST counts of the two [M/H] bins hold the same Gaia stars and disjoint populations of matches."""
    root = tmp_path_factory.mktemp("milky-way-lamost")
    write_world(root)
    populations = read_imig(source_path(root, "imig"))
    lamost = read_lamost(source_path(root, "lamost"))
    metals = lamost_metallicity_counts(read_gaia(root / GAIA_PATH), lamost, populations, np.random.default_rng(2))
    first, second = (m.counts for m in metals)
    assert all(m.kinematics.age.size > 0 for m in metals)
    assert np.array_equal(first.gaia_absz, second.gaia_absz)
    assert not np.any(first.tracer_in_disk & second.tracer_in_disk)
    assert first.tracer_in_disk.any() and second.tracer_in_disk.any()


def test_the_estimators_of_the_tracer() -> None:
    """In a linear potential the tracer is an exponential of scale sigma_z^2 / g, which every estimator recovers."""
    estimators = tracer_estimators(MODEL)
    assert [e.sigma_z for e in estimators] == [10.0, 13.0, 16.0, 19.0, 22.0, 25.0]
    for e in estimators:
        scale = e.sigma_z**2 / 900.0
        assert e.windowed == pytest.approx(scale, rel=0.005) and e.exponential == pytest.approx(scale, rel=0.005)
        assert e.exponential_whole == pytest.approx(scale, rel=0.005)
