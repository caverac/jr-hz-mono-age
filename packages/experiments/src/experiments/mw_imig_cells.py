"""The populations of Imig et al. (2025) one at a time against our star counts of the same cell.

The heights of :mod:`experiments.mw_imig` pool the populations of each age bin over [M/H],
so their comparison with our counts could reflect how each side weights metallicities.
This module removes the pooling. Every population that
:func:`experiments.mw_imig.read_imig` keeps is compared with our APOGEE counts of the same
cell: the stars of the same ages on the distmass scale and of the same [M/H] bin,
below the [alpha/M] of their Eq. (1) (:func:`imig_low_alpha`), with the ASPCAP abundances
that they bin in. Their h_Z moves to R_0 by its flare. A single exponential returns its
scale height in any window, so their h_Z needs no window, whereas our counts go through the
self-consistent window of :func:`experiments.mw_counts.count_hz`.

A cell with fewer than ``MIN_CELL_MATCHES`` matched stars is listed without our h_Z. The
errors come from ``N_DRAWS`` draws that resample the counts of each [M/H] bin, one draw for
all its age cells, and draw their h_Z from a split normal. The summary is the median ratio
of the measured cells and the slope of ln(counts / Imig) against the center of the [M/H]
bin.

Our counts assume that the targeting does not depend on the population at a fixed position,
whereas APOGEE targets by (J - K_s)_0 and H, with limits that change between cohorts, and
the colors of giants change with metallicity. ``TARGETING_BOXES`` repeat the comparison with
only the matches of a box of H and (J - K_s)_0 that every cohort they come from contains
(:class:`experiments.mw_catalogs.TargetingBox`), so that one selection holds in every field.

The counts of LAMOST (:func:`lamost_metallicity_counts`) test our counts with another
survey: other targeting, other ages and another [Fe/H]. Their ages are not on the scale of
Imig et al., so their ratios to the heights of Imig et al. mix two age scales. The slope of
ln h_Z per dex of [M/H] at fixed age, fitted with one intercept per age bin
(:class:`MetallicitySlopes`), does not need one scale, and every set of cells carries it,
for our counts and for the heights of Imig et al. in the same cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import (
    Apogee,
    GaiaStars,
    TargetingBox,
    apogee_age_tracers,
    lamost_age_tracers,
    lamost_low_alpha,
)
from experiments.mw_counts import CountSample, apogee_count_sample, count_hz, lamost_count_sample
from experiments.mw_dynamics import R0_KPC
from experiments.mw_imig import N_DRAWS, R_SUN_IMIG_KPC, ImigPopulations, split_normal
from experiments.populations import Estimate, Trend, fit_grouped_log_slope, fit_log_slope
from experiments.stats import half_width

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

MIN_CELL_MATCHES = 100
# Eq. (1) of Imig et al. (2025): the [alpha/M] that splits the two sequences at each [M/H],
# as polynomial coefficients from the cubic term, with their offset of -0.05 from [Mg/Fe].
ALPHA_SPLIT = (0.1754, 0.1119, -0.1253, 0.1353 - 0.05)
JK_NO_RED_LIMIT = 99.0  # redder than any star: only cohorts without a red limit (9999.99) contain it
# 7 < H < 11 holds the bright giants of the counts in every main cohort but a few; the first box
# takes the main-survey limit (J - K_s)_0 > 0.5 with no red limit, the second adds the red limit
# (J - K_s)_0 < 0.8 of some cohorts to every star.
TARGETING_BOXES = (TargetingBox(7.0, 11.0, 0.5, JK_NO_RED_LIMIT), TargetingBox(7.0, 11.0, 0.5, 0.8))


@dataclass(frozen=True)
class MetallicityCounts:
    """Our APOGEE counts on the distmass scale whose populations are the low-[alpha/M] stars of one [M/H] bin."""

    m_h_lo: float
    m_h_hi: float
    counts: CountSample


@dataclass(frozen=True)
class ImigCell:  # pylint: disable=too-many-instance-attributes
    """One population of Imig et al. against our counts of its ages and [M/H]: their h_Z at R_0, ours, the ratio."""

    age_lo: float
    age_hi: float
    m_h_lo: float
    m_h_hi: float
    n_stars: int  # their stars in the cell
    n_matched: int  # our matched stars in the cell
    imig: Estimate  # kpc
    counts: Estimate  # kpc; NaN with fewer than the minimum of matches
    ratio: Estimate  # counts / imig


@dataclass(frozen=True)
class MetallicitySlopes:
    """The slope of ln h_Z per dex of [M/H] at fixed age, one intercept per age bin: ours and theirs, same cells."""

    counts: Trend
    imig: Trend


@dataclass(frozen=True)
class ImigCells:
    """Every cell and, over the measured ones, the median ratio, the slope of its logarithm and those of the heights.

    ``n_matches`` is the number of matched stars that split the counts into populations,
    ``trend`` the slope of ln(counts / Imig) per dex of [M/H], and ``slopes`` those of the
    heights at fixed age.
    """

    n_matches: int
    min_matches: int
    cells: list[ImigCell]
    n_measured: int
    median_ratio: float
    trend: Trend
    slopes: MetallicitySlopes


def imig_low_alpha(m_h: FloatArray, alpha_m: FloatArray) -> BoolArray:
    """Return whether stars lie below the [alpha/M] of Eq. (1) of Imig et al. at their [M/H]; False where NaN."""
    return np.asarray(alpha_m < np.polyval(ALPHA_SPLIT, m_h), dtype=np.bool_)


def metallicity_counts(gaia: GaiaStars, apogee: Apogee, populations: list[ImigPopulations]) -> list[MetallicityCounts]:
    """Return our counts of every [M/H] bin of *populations*, with the ages and ASPCAP abundances of *apogee*."""
    low = imig_low_alpha(apogee.m_h, apogee.alpha_m)
    return [
        MetallicityCounts(
            lo, hi, apogee_count_sample(gaia, apogee_age_tracers(apogee, low & (apogee.m_h >= lo) & (apogee.m_h < hi)))
        )
        for lo, hi in _metallicity_edges(populations)
    ]


def _metallicity_edges(populations: list[ImigPopulations]) -> list[tuple[float, float]]:
    """Return the edges of the [M/H] bins of *populations*, in increasing order."""
    return sorted({edge for pops in populations for edge in zip(pops.m_h_lo.tolist(), pops.m_h_hi.tolist())})


def lamost_metallicity_counts(
    gaia: GaiaStars, lamost: dict[str, FloatArray], populations: list[ImigPopulations]
) -> list[MetallicityCounts]:
    """Return the LAMOST counts of every [M/H] bin of *populations*, split by the [Fe/H] and the alpha of LAMOST."""
    low = lamost_low_alpha(lamost["FEH"], lamost["AFE"])
    return [
        MetallicityCounts(
            lo,
            hi,
            lamost_count_sample(gaia, lamost_age_tracers(lamost, low & (lamost["FEH"] >= lo) & (lamost["FEH"] < hi)))[
                0
            ],
        )
        for lo, hi in _metallicity_edges(populations)
    ]


def _in_cell(sample: CountSample, pops: ImigPopulations) -> BoolArray:
    """Return which matches of *sample* belong to its population and to the ages of *pops*."""
    return sample.tracer_in_disk & (sample.tracer_age >= pops.age_lo) & (sample.tracer_age < pops.age_hi)


@dataclass(frozen=True)
class CellSettings:
    """How the cells are measured: the bootstrap draws, and the matched stars a cell needs for our h_Z."""

    n_draws: int = N_DRAWS
    min_matches: int = MIN_CELL_MATCHES


@dataclass(frozen=True)
class _Target:
    """The population *index* of the age bin *pops*."""

    pops: ImigPopulations
    index: int


def _count_heights(
    counts: CountSample, targets: list[_Target], model: SolarModel, rng: np.random.Generator, settings: CellSettings
) -> tuple[list[int], FloatArray, FloatArray]:
    """Return the matches of every target in *counts*, our h_Z of each, and its draws; NaN without enough matches.

    Every draw resamples *counts* once for all the targets.
    """
    n_matched = [int(np.count_nonzero(_in_cell(counts, t.pops))) for t in targets]
    measured = [j for j, n in enumerate(n_matched) if n >= settings.min_matches]
    point = np.full(len(targets), np.nan)
    draws = np.full((settings.n_draws, len(targets)), np.nan)
    for j in measured:
        point[j] = count_hz(counts, targets[j].pops.age_lo, targets[j].pops.age_hi, model.k)[0]
    if measured:
        for d in range(settings.n_draws):
            sample = counts.resampled(rng)
            for j in measured:
                draws[d, j] = count_hz(sample, targets[j].pops.age_lo, targets[j].pops.age_hi, model.k)[0]
    return n_matched, point, draws


def _imig_height(target: _Target, rng: np.random.Generator, n_draws: int) -> tuple[float, FloatArray]:
    """Return their h_Z of *target* moved to R_0 by its flare, and *n_draws* draws of it from its errors."""
    pops, i = target.pops, target.index
    shift = pops.a_flare[i] * (R0_KPC - R_SUN_IMIG_KPC)
    draws = split_normal(rng, np.full(n_draws, pops.h[i]), np.tile(pops.h_err[i], (n_draws, 1))) + shift
    return float(pops.h[i] + shift), draws


def _metallicity_cells(
    populations: list[ImigPopulations],
    metal: MetallicityCounts,
    model: SolarModel,
    rng: np.random.Generator,
    settings: CellSettings,
) -> list[ImigCell]:
    """Return the cells of the [M/H] bin of *metal*, one per age bin of *populations* that holds it."""
    targets = [_Target(pops, int(i)) for pops in populations for i in np.flatnonzero(pops.m_h_lo == metal.m_h_lo)]
    n_matched, point, draws = _count_heights(metal.counts, targets, model, rng, settings)
    cells = []
    for j, target in enumerate(targets):
        h, h_draws = _imig_height(target, rng, settings.n_draws)
        cells.append(
            ImigCell(
                age_lo=target.pops.age_lo,
                age_hi=target.pops.age_hi,
                m_h_lo=metal.m_h_lo,
                m_h_hi=metal.m_h_hi,
                n_stars=int(target.pops.n_stars[target.index]),
                n_matched=n_matched[j],
                imig=Estimate(h, half_width(h_draws)),
                counts=Estimate(float(point[j]), half_width(draws[:, j])),
                ratio=Estimate(float(point[j] / h), half_width(draws[:, j] / h_draws)),
            )
        )
    return cells


def measure_cells(
    populations: list[ImigPopulations],
    metallicities: list[MetallicityCounts],
    model: SolarModel,
    rng: np.random.Generator,
    settings: CellSettings = CellSettings(),
) -> ImigCells:
    """Return every cell of *populations* against the counts of *metallicities*, and the summary of the ratios.

    The summary uses the cells with the minimum of matches whose ratio and error are finite;
    with fewer than three of them it is NaN.
    """
    min_matches = settings.min_matches
    cells = [cell for metal in metallicities for cell in _metallicity_cells(populations, metal, model, rng, settings)]
    ok = [
        c
        for c in cells
        if c.n_matched >= min_matches and np.isfinite(c.ratio.value) and np.isfinite(c.ratio.err) and c.ratio.err > 0.0
    ]
    n_matches = int(metallicities[0].counts.tracer_age.size) if metallicities else 0
    slopes = _metallicity_slopes(ok)
    if len(ok) < 3:
        nan = float("nan")
        return ImigCells(n_matches, min_matches, cells, len(ok), nan, Trend(nan, nan, nan), slopes)
    ratio = np.array([c.ratio.value for c in ok])
    err = np.array([c.ratio.err for c in ok])
    m_h = np.array([0.5 * (c.m_h_lo + c.m_h_hi) for c in ok])
    trend = fit_log_slope(m_h, ratio, err)
    return ImigCells(n_matches, min_matches, cells, len(ok), float(np.median(ratio)), trend, slopes)


def _metallicity_slopes(cells: list[ImigCell]) -> MetallicitySlopes:
    """Return the slopes of ln h_Z per dex of [M/H] of our counts and of their heights in *cells*, at fixed age."""
    m_h = np.array([0.5 * (c.m_h_lo + c.m_h_hi) for c in cells])
    age = np.array([c.age_lo for c in cells])

    def slope(heights: list[Estimate]) -> Trend:
        value = np.array([h.value for h in heights])
        return fit_grouped_log_slope(m_h, value, np.array([h.err for h in heights]), age)

    return MetallicitySlopes(slope([c.counts for c in cells]), slope([c.imig for c in cells]))


@dataclass(frozen=True)
class TargetedCells:
    """The cells measured with only the matches of one targeting box."""

    box: TargetingBox
    cells: ImigCells
