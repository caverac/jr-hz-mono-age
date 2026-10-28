"""Tests of the populations of Imig et al. one at a time against the counts of the same cell.

The synthetic world gives the populations two bins of [M/H] and four age bins, and APOGEE
stars whose |z| is exponential with a scale that grows with age; one star in ten lies on the
high-[alpha/M] side of the split.
"""

from __future__ import annotations

import numpy as np
import pytest
from mw_fixtures import write_world

from experiments.commands.milky_way import cell_lines
from experiments.heights import Z_GRID
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import read_apogee, read_gaia, with_distmass_ages
from experiments.mw_imig import ImigPopulations, read_imig
from experiments.mw_imig_cells import (
    CellSettings,
    ImigCell,
    ImigCells,
    MetallicityCounts,
    imig_low_alpha,
    measure_cells,
    metallicity_counts,
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


def test_cells_compare_each_population_with_its_counts(metals: Metals) -> None:
    """Every population gets a cell; the measured ones have finite heights, ratios and a summary."""
    populations, metallicities = metals
    cells = measure_cells(populations, metallicities, MODEL, np.random.default_rng(1), CellSettings(5, 20))
    assert len(cells.cells) == 8 and cells.min_matches == 20
    measured = [c for c in cells.cells if c.n_matched >= 20]
    assert len(measured) == cells.n_measured >= 3
    for cell in measured:
        assert cell.counts.value > 0.0 and cell.ratio.value == pytest.approx(cell.counts.value / cell.imig.value)
        assert cell.ratio.err > 0.0 and cell.n_stars == 500
    assert np.isfinite(cells.median_ratio) and np.isfinite(cells.trend.value)


def test_cells_without_enough_matches_have_no_counts(metals: Metals) -> None:
    """Below the minimum of matches a cell keeps their h_Z but has no counts, and the summary is NaN."""
    populations, metallicities = metals
    cells = measure_cells(populations, metallicities, MODEL, np.random.default_rng(1), CellSettings(3, 10**6))
    assert cells.n_measured == 0 and np.isnan(cells.median_ratio) and np.isnan(cells.trend.value)
    assert all(np.isnan(c.counts.value) and np.isnan(c.ratio.value) for c in cells.cells)
    assert all(c.imig.value > 0.0 and c.imig.err > 0.0 for c in cells.cells)


def test_the_report_summarizes_each_metallicity_bin() -> None:
    """A bin with measured cells prints their median ratio, and a bin without them prints none."""

    def cell(m_h_lo: float, n_matched: int, ratio: float) -> ImigCell:
        h = Estimate(0.3, 0.01)
        return ImigCell(1.0, 1.26, m_h_lo, m_h_lo + 0.1, 500, n_matched, h, h, Estimate(ratio, 0.1))

    cells = ImigCells(100, [cell(0.0, 150, 1.2), cell(0.0, 300, 1.4), cell(0.1, 50, 2.0)], 2, 1.3, Trend(0.5, 0.2, 1.0))
    lines = cell_lines(cells)
    assert lines[1].split()[-1] == "1.30" and lines[2].split()[-1] == "none"
    assert "2 cells: median 1.30" in lines[-1]
