"""The populations of Imig et al. (2025) one at a time against our star counts of the same cell.

The heights of :mod:`experiments.mw_imig` pool the populations of each age bin over [M/H],
so their comparison with our counts could reflect how each side weights metallicities.
This module removes the pooling. Every population that
:func:`experiments.mw_imig.read_imig` keeps is compared with our APOGEE counts of the same
cell: the stars of the same ages on the distmass scale and of the same [M/H] bin,
below the [alpha/M] of their Eq. (1) (:func:`imig_low_alpha`), with the ASPCAP abundances
that they bin in. Each cell carries three heights of theirs, so that the geometry and the
slices can be taken out in turn:

- ``imig_sun``: their h_Z at their Sun, as published;
- ``imig_window``: their model averaged over the disk of our cylinder
  (:func:`experiments.mw_imig.cylinder_profile`), in the continuous self-consistent window;
- ``imig``: the same model through our counts (:func:`experiments.mw_counts.predicted_hz`),
  with the interpolation between slices that our h_Z goes through. It is the one compared
  with ours.

``shape`` is our counts of the cell over that model in each slice of |z|, each normalized
to its sum, which shows where the two profiles differ.

A cell with fewer than ``MIN_CELL_MATCHES`` matched stars is listed without our h_Z and
without their model through our counts. The errors come from ``N_DRAWS`` draws that
resample the counts of each [M/H] bin, one draw for all its age cells, and draw their h_Z
at their Sun from a split normal; in each draw their model goes through the counts of the
draw. The summary is the median ratio of the measured cells and the slope of
ln(counts / Imig) against the center of the [M/H] bin.

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

from experiments.heights import scaled_profile_hz
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import (
    Apogee,
    GaiaStars,
    TargetingBox,
    apogee_age_tracers,
    lamost_age_tracers,
    lamost_low_alpha,
)
from experiments.mw_counts import (
    CountSample,
    apogee_count_sample,
    count_hz,
    lamost_count_sample,
    predicted_hz,
    slice_fraction,
)
from experiments.mw_imig import N_DRAWS, ImigPopulations, cylinder_profile, cylinder_slices, split_normal
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
    """One population of Imig et al. against our counts of its ages and [M/H]: their h_Z, ours, the ratio."""

    age_lo: float
    age_hi: float
    m_h_lo: float
    m_h_hi: float
    n_stars: int  # their stars in the cell
    n_matched: int  # our matched stars in the cell
    imig_sun: Estimate  # kpc, at their Sun
    imig_window: float  # kpc, their model in our cylinder, in the continuous window
    imig: Estimate  # kpc, their model through our counts; NaN with fewer than the minimum of matches
    counts: Estimate  # kpc; NaN with fewer than the minimum of matches
    ratio: Estimate  # counts / imig
    shape: list[float]  # ours over their model in each slice, each normalized to its sum; empty without the minimum


@dataclass(frozen=True)
class MetallicitySlopes:
    """The slope of ln h_Z per dex of [M/H] at fixed age, one intercept per age bin, in the same cells.

    ``counts`` is that of our counts and ``imig`` that of their model through our counts.
    """

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

    def sun_draws(self, rng: np.random.Generator, n_draws: int) -> FloatArray:
        """Return *n_draws* draws of their h_Z at their Sun from its split normal."""
        pops, i = self.pops, self.index
        return split_normal(rng, np.full(n_draws, pops.h[i]), np.tile(pops.h_err[i], (n_draws, 1)))

    def model_slices(self, h: FloatArray) -> FloatArray:
        """Return the mass of their model in each slice of our counts, one row per value *h* of h_Z at their Sun."""
        return cylinder_slices(self.pops, np.full(h.size, self.index), h)

    def heights(self, sample: CountSample, slices: FloatArray, k: float) -> tuple[float, float]:
        """Return our h_Z of the cell in *sample* and that of their model of mass *slices* through the same counts."""
        return count_hz(sample, self.pops.age_lo, self.pops.age_hi, k)[0], predicted_hz(sample, slices, k)[0]


@dataclass(frozen=True)
class _Heights:
    """Ours and their model through our counts for the targets of one [M/H] bin, and their h_Z at their Sun.

    ``point`` holds ours and theirs at the best values (2, targets) and ``draws`` one row per
    draw (2, draws, targets); a target without the minimum of matches has NaN for both.
    ``shapes`` holds the shape of each measured target.
    """

    n_matched: list[int]
    point: FloatArray
    draws: FloatArray
    sun_draws: FloatArray  # (draws, targets)
    shapes: dict[int, list[float]]


def _shape(counts: CountSample, target: _Target, slices: FloatArray) -> list[float]:
    """Return our counts of *target* over their model of mass *slices*, in each slice, each normalized to its sum."""
    ours = counts.moments.per_slice() * slice_fraction(counts, target.pops.age_lo, target.pops.age_hi)
    return [float(v) for v in (ours / ours.sum()) / (slices / slices.sum())]


def _heights(
    counts: CountSample, targets: list[_Target], model: SolarModel, rng: np.random.Generator, settings: CellSettings
) -> _Heights:
    """Return our h_Z and their model through our counts for every target of *counts*, with their draws.

    Their h_Z at their Sun is drawn first for every target. Every draw then resamples *counts*
    once for all the targets, and their model of the draw goes through the same counts.
    """
    n_matched = [int(np.count_nonzero(_in_cell(counts, t.pops))) for t in targets]
    measured = [j for j, n in enumerate(n_matched) if n >= settings.min_matches]
    sun_draws = np.column_stack([t.sun_draws(rng, settings.n_draws) for t in targets])
    best = {j: targets[j].model_slices(targets[j].pops.h[[targets[j].index]])[0] for j in measured}
    drawn = {j: targets[j].model_slices(sun_draws[:, j]) for j in measured}
    point = np.full((2, len(targets)), np.nan)
    draws = np.full((2, settings.n_draws, len(targets)), np.nan)
    for j in measured:
        point[:, j] = targets[j].heights(counts, best[j], model.k)
    for d in range(settings.n_draws if measured else 0):
        sample = counts.resampled(rng)
        for j in measured:
            draws[:, d, j] = targets[j].heights(sample, drawn[j][d], model.k)
    return _Heights(n_matched, point, draws, sun_draws, {j: _shape(counts, targets[j], best[j]) for j in measured})


def _cell(metal: MetallicityCounts, target: _Target, heights: _Heights, j: int, k: float) -> ImigCell:
    """Return the cell of *target*, the target *j* of the [M/H] bin of *metal*, from the *heights* of the bin."""
    pops, i = target.pops, target.index
    (ours, theirs), (ours_draws, theirs_draws) = heights.point[:, j], heights.draws[:, :, j]
    return ImigCell(
        age_lo=pops.age_lo,
        age_hi=pops.age_hi,
        m_h_lo=metal.m_h_lo,
        m_h_hi=metal.m_h_hi,
        n_stars=int(pops.n_stars[i]),
        n_matched=heights.n_matched[j],
        imig_sun=Estimate(float(pops.h[i]), half_width(heights.sun_draws[:, j])),
        imig_window=scaled_profile_hz(cylinder_profile(pops, i), k)[0],
        imig=Estimate(float(theirs), half_width(theirs_draws)),
        counts=Estimate(float(ours), half_width(ours_draws)),
        ratio=Estimate(float(ours / theirs), half_width(ours_draws / theirs_draws)),
        shape=heights.shapes.get(j, []),
    )


def _metallicity_cells(
    populations: list[ImigPopulations],
    metal: MetallicityCounts,
    model: SolarModel,
    rng: np.random.Generator,
    settings: CellSettings,
) -> list[ImigCell]:
    """Return the cells of the [M/H] bin of *metal*, one per age bin of *populations* that holds it."""
    targets = [_Target(pops, int(i)) for pops in populations for i in np.flatnonzero(pops.m_h_lo == metal.m_h_lo)]
    heights = _heights(metal.counts, targets, model, rng, settings)
    return [_cell(metal, target, heights, j, model.k) for j, target in enumerate(targets)]


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
