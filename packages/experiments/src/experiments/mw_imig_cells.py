"""The populations of Imig et al. (2025) one at a time against our star counts of the same cell.

The heights of :mod:`experiments.mw_imig` pool the populations of each age bin over [M/H],
so their comparison with our counts could reflect how each side weights metallicities.
This module removes the pooling. Every population that
:func:`experiments.mw_imig.read_imig` keeps is compared with our APOGEE counts of the same
cell: the stars of the same ages on the distmass scale and of the same [M/H] bin,
below the [alpha/M] of their Eq. (1) (:func:`imig_low_alpha`), with the ASPCAP abundances
that they bin in. Each cell carries four heights of theirs (:class:`CellTheirs`):

- ``sun``: their h_Z at their Sun, as published;
- ``window``: their model averaged over the disk of our cylinder
  (:func:`experiments.mw_imig.cylinder_profile`), in the continuous self-consistent window;
- ``counted``: the same model through our counts (:func:`experiments.mw_counts.predicted_hz`);
- ``exponential``: the same model through our counts, fit with an exponential as ours is
  (:func:`experiments.mw_counts.predicted_exponential_hz`).

Ours (:class:`CellOurs`) are the windowed h_Z of the counts and the exponential that maximum
likelihood fits to them over |z| < ``Z_MAX_KPC`` without a window, which is the kind of fit
their h_Z is. The two differ for a profile that is not exponential: an isothermal
population, flat at the plane and steeper with height, gives an exponential fit 1.3 to 1.5
times thinner than its windowed h_Z, more for colder populations (:func:`tracer_estimators`).
``shape`` is our counts over their model in each slice of |z|, each normalized to its sum.
The ratios of a cell (:class:`CellRatios`) compare like with like: our windowed h_Z with
their ``counted``, and our exponential fit with their ``exponential``.

The kinematics of each cell (:class:`CellKinematics`) are its own stars of the solar annulus
with |z| < ``Z_MID_MAX_KPC``, whose sigma_z gives a tracer that goes through our counts, as
in the main test. A cell with fewer than ``MIN_CELL_KINEMATICS`` of them has no tracer.

A cell with fewer than ``MIN_CELL_MATCHES`` matched stars is listed with their published
h_Z only. The errors come from ``N_DRAWS`` draws that resample the counts of each [M/H] bin,
one draw for all its age cells, resample the kinematic stars of each cell, and draw their h_Z
at their Sun from a split normal; in each draw every model goes through the counts of the
draw. The summary is the median ratio of our windowed h_Z over their model through our
counts in the measured cells, and how each ratio changes (:class:`CellTrends`): the slope of
its logarithm per ln age at fixed [M/H], with the age of a cell the geometric mean of its
edges, and per dex of [M/H] at fixed age.

Our counts assume that the targeting does not depend on the population at a fixed position,
whereas APOGEE targets by (J - K_s)_0 and H, with limits that change between cohorts, and
the colors of giants change with metallicity. ``TARGETING_BOXES`` repeat the comparison with
only the matches of a box of H and (J - K_s)_0 that every cohort they come from contains
(:class:`experiments.mw_catalogs.TargetingBox`), so that one selection holds in every field.

The counts of LAMOST (:func:`lamost_metallicity_counts`) test our counts with another
survey: other targeting, other ages and another [Fe/H]. Their ages are not on the scale of
Imig et al., so their ratios to the heights of Imig et al. mix two age scales. The slope of
ln h_Z per dex of [M/H] at fixed age, fitted with one intercept per age bin
(:class:`MetallicitySlopes`), does not need one scale, and every set of cells carries it for
each height.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.heights import Z_GRID, exponential_profile_hz, scaled_profile_hz, tracer_profile
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import (
    Apogee,
    GaiaStars,
    KinematicStars,
    TargetingBox,
    apogee_age_tracers,
    apogee_kinematics,
    lamost_age_tracers,
    lamost_kinematics,
    lamost_low_alpha,
)
from experiments.mw_counts import (
    Z_MAX_KPC,
    CountSample,
    apogee_count_sample,
    lamost_count_sample,
    population_counts,
    predicted_exponential_hz,
    predicted_hz,
    profile_slices,
)
from experiments.mw_imig import N_DRAWS, ImigPopulations, cylinder_profile, cylinder_slices, split_normal
from experiments.populations import Z_MID_MAX_KPC, Estimate, Trend, fit_grouped_log_slope
from experiments.stats import deconvolved_dispersion, half_width

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

MIN_CELL_MATCHES = 100
MIN_CELL_KINEMATICS = 30
ESTIMATOR_SIGMAS_KMS = (10.0, 13.0, 16.0, 19.0, 22.0, 25.0)
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
    """Our counts and kinematic stars of one [M/H] bin, whose populations are its low-[alpha/M] stars."""

    m_h_lo: float
    m_h_hi: float
    counts: CountSample
    kinematics: KinematicStars


@dataclass(frozen=True)
class CellTheirs:
    """Their h_Z of a cell (kpc): at their Sun, their model over our cylinder in the window, and through our counts.

    ``counted`` is the windowed h_Z of their model through our counts, and ``exponential``
    the exponential fit to it; both NaN with fewer than the minimum of matches.
    """

    sun: Estimate
    window: float
    counted: Estimate
    exponential: Estimate


@dataclass(frozen=True)
class CellOurs:
    """Our counts of a cell: windowed h_Z and exponential fit (kpc), gradient b (1/kpc), shape against their model.

    All NaN, and the shape empty, with fewer than the minimum of matches.
    """

    windowed: Estimate
    exponential: Estimate
    gradient: float
    shape: list[float]


@dataclass(frozen=True)
class CellKinematics:
    """The kinematic stars of a cell near the plane: their number, sigma_z (km/s), and its tracer through our counts."""

    n_kin: int
    sigma_z: Estimate
    tracer: Estimate


@dataclass(frozen=True)
class CellRatios:
    """Ours over theirs through our counts, windowed and as exponential fits, and the tracer over our windowed h_Z."""

    counted: Estimate
    exponential: Estimate
    tracer: Estimate


@dataclass(frozen=True)
class ImigCell:  # pylint: disable=too-many-instance-attributes
    """One population of Imig et al. against our counts and kinematic stars of its ages and [M/H]."""

    age_lo: float
    age_hi: float
    m_h_lo: float
    m_h_hi: float
    n_stars: int  # their stars in the cell
    n_matched: int  # our matched stars in the cell
    theirs: CellTheirs
    ours: CellOurs
    kinematics: CellKinematics
    ratios: CellRatios


Getter = Callable[[ImigCell], Estimate]  # one value of a cell, with its error


@dataclass(frozen=True)
class MetallicitySlopes:
    """The slope of ln h_Z (ln sigma_z) per dex of [M/H] at fixed age, one intercept per age bin, in the same cells."""

    counts: Trend  # our windowed h_Z
    counts_exponential: Trend
    imig: Trend  # their model through our counts
    imig_sun: Trend
    tracer: Trend
    sigma_z: Trend


@dataclass(frozen=True)
class RatioTrends:
    """The slope of the logarithm of a ratio per ln age, one intercept per [M/H] bin, and per dex, one per age bin."""

    age: Trend
    metallicity: Trend


@dataclass(frozen=True)
class CellTrends:
    """How each ratio of :class:`CellRatios` changes with age and [M/H] over the measured cells."""

    counted: RatioTrends
    exponential: RatioTrends
    tracer: RatioTrends


@dataclass(frozen=True)
class ImigCells:
    """Every cell and, over the measured ones, the median ratio, the trends of the ratios and the slopes of the heights.

    ``n_matches`` is the number of matched stars that split the counts into populations,
    ``median_ratio`` the median of our windowed h_Z over their model through our counts,
    and ``slopes`` the slopes of every height per dex of [M/H] at fixed age.
    """

    n_matches: int
    min_matches: int
    cells: list[ImigCell]
    n_measured: int
    median_ratio: float
    trends: CellTrends
    slopes: MetallicitySlopes


@dataclass(frozen=True)
class TracerEstimator:
    """The windowed h_Z (kpc) of the tracer of *sigma_z* in our potential, and the exponential fits to it.

    ``exponential`` is fit over |z| < ``Z_MAX_KPC``, the edge of the counts, and
    ``exponential_whole`` over the whole profile, as a fit without a limit in |z| sees it.
    """

    sigma_z: float
    windowed: float
    exponential: float
    exponential_whole: float


def tracer_estimators(model: SolarModel) -> list[TracerEstimator]:
    """Return the estimators of the tracer for every sigma_z of ``ESTIMATOR_SIGMAS_KMS``, without the counts."""

    def estimators(sigma: float) -> TracerEstimator:
        rho = tracer_profile(model.phi, sigma)
        return TracerEstimator(
            sigma,
            scaled_profile_hz(rho, model.k)[0],
            exponential_profile_hz(rho, Z_MAX_KPC),
            exponential_profile_hz(rho, float(Z_GRID[-1])),
        )

    return [estimators(sigma) for sigma in ESTIMATOR_SIGMAS_KMS]


def imig_low_alpha(m_h: FloatArray, alpha_m: FloatArray) -> BoolArray:
    """Return whether stars lie below the [alpha/M] of Eq. (1) of Imig et al. at their [M/H]; False where NaN."""
    return np.asarray(alpha_m < np.polyval(ALPHA_SPLIT, m_h), dtype=np.bool_)


def metallicity_counts(gaia: GaiaStars, apogee: Apogee, populations: list[ImigPopulations]) -> list[MetallicityCounts]:
    """Return our counts and kinematic stars of each [M/H] bin of *populations*, by the ASPCAP [M/H] of *apogee*."""
    low = imig_low_alpha(apogee.m_h, apogee.alpha_m)
    metals = []
    for lo, hi in _metallicity_edges(populations):
        population = low & (apogee.m_h >= lo) & (apogee.m_h < hi)
        counts = apogee_count_sample(gaia, apogee_age_tracers(apogee, population))
        metals.append(MetallicityCounts(lo, hi, counts, apogee_kinematics(apogee, population)))
    return metals


def _metallicity_edges(populations: list[ImigPopulations]) -> list[tuple[float, float]]:
    """Return the edges of the [M/H] bins of *populations*, in increasing order."""
    return sorted({edge for pops in populations for edge in zip(pops.m_h_lo.tolist(), pops.m_h_hi.tolist())})


def lamost_metallicity_counts(
    gaia: GaiaStars, lamost: dict[str, FloatArray], populations: list[ImigPopulations], rng: np.random.Generator
) -> list[MetallicityCounts]:
    """Return the LAMOST counts and kinematic stars of each [M/H] bin of *populations*, by LAMOST [Fe/H] and alpha."""
    low = lamost_low_alpha(lamost["FEH"], lamost["AFE"])
    metals = []
    for lo, hi in _metallicity_edges(populations):
        population = low & (lamost["FEH"] >= lo) & (lamost["FEH"] < hi)
        counts = lamost_count_sample(gaia, lamost_age_tracers(lamost, population))[0]
        metals.append(MetallicityCounts(lo, hi, counts, lamost_kinematics(lamost, rng, population)))
    return metals


def _in_cell(sample: CountSample, pops: ImigPopulations) -> BoolArray:
    """Return which matches of *sample* belong to its population and to the ages of *pops*."""
    return sample.tracer_in_disk & (sample.tracer_age >= pops.age_lo) & (sample.tracer_age < pops.age_hi)


@dataclass(frozen=True)
class CellSettings:
    """How the cells are measured: the bootstrap draws, and the matched stars a cell needs for our h_Z."""

    n_draws: int = N_DRAWS
    min_matches: int = MIN_CELL_MATCHES


# The rows of _Heights.point and .draws.
VALUES = ("ours", "ours_exponential", "theirs", "theirs_exponential", "sigma_z", "tracer")


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

    def near(self, stars: KinematicStars) -> BoolArray:
        """Return which of *stars* have the ages of the cell and |z| < ``Z_MID_MAX_KPC``."""
        in_age = (stars.age >= self.pops.age_lo) & (stars.age < self.pops.age_hi)
        return np.asarray(in_age & (np.abs(stars.phase.z) < Z_MID_MAX_KPC), dtype=np.bool_)

    def kinematics(self, stars: KinematicStars) -> KinematicStars | None:
        """Return the stars of *stars* near the plane with the ages of the cell; None with too few of them."""
        near = self.near(stars)
        return stars.select(near) if np.count_nonzero(near) >= MIN_CELL_KINEMATICS else None


@dataclass(frozen=True)
class _Measured:
    """A target with the minimum of matches: their model's mass at the best h_Z and per draw, its kinematic stars."""

    target: _Target
    best: FloatArray
    drawn: FloatArray  # (draws, slices)
    stars: KinematicStars | None

    def values(
        self, sample: CountSample, slices: FloatArray, stars: KinematicStars | None, model: SolarModel
    ) -> list[float]:
        """Return the ``VALUES`` of the cell with *sample*, their model of mass *slices* and the kinematic *stars*."""
        pops = self.target.pops
        ours = population_counts(sample, pops.age_lo, pops.age_hi)
        sigma = float("nan") if stars is None else deconvolved_dispersion(stars.phase.v_z, stars.v_z_err)
        tracer = (
            predicted_hz(sample, profile_slices(tracer_profile(model.phi, sigma)), model.k)[0]
            if np.isfinite(sigma) and sigma > 0.0
            else float("nan")
        )
        return [
            ours.hz(model.k)[0],
            ours.exponential_hz(),
            predicted_hz(sample, slices, model.k)[0],
            predicted_exponential_hz(sample, slices),
            sigma,
            tracer,
        ]

    def resampled_stars(self, rng: np.random.Generator) -> KinematicStars | None:
        """Return a bootstrap copy of the kinematic stars."""
        if self.stars is None:
            return None
        return self.stars.select(rng.integers(0, self.stars.age.size, self.stars.age.size))

    def summary(self, counts: CountSample) -> tuple[float, list[float]]:
        """Return the gradient of our counts of the cell and our counts over their model in each slice, normalized."""
        ours = population_counts(counts, self.target.pops.age_lo, self.target.pops.age_hi)
        stars = ours.per_slice()
        return ours.gradient, [float(v) for v in (stars / stars.sum()) / (self.best / self.best.sum())]


@dataclass(frozen=True)
class _Heights:
    """The ``VALUES`` of the targets of one [M/H] bin at the best values (values, targets) and per draw.

    ``draws`` is (values, draws, targets); a target without the minimum of matches has NaN.
    ``summaries`` holds the gradient and the shape of each measured target.
    """

    n_matched: list[int]
    point: FloatArray
    draws: FloatArray
    sun_draws: FloatArray  # (draws, targets)
    n_kin: list[int]
    summaries: dict[int, tuple[float, list[float]]]


def _heights(
    metal: MetallicityCounts,
    targets: list[_Target],
    model: SolarModel,
    rng: np.random.Generator,
    settings: CellSettings,
) -> _Heights:
    """Return the ``VALUES`` of every target of *metal*, with their draws.

    Their h_Z at their Sun is drawn first for every target. Every draw then resamples the
    counts once for all the targets and the kinematic stars of each, and every model of the
    draw goes through the same counts.
    """
    n_matched = [int(np.count_nonzero(_in_cell(metal.counts, t.pops))) for t in targets]
    sun_draws = np.column_stack([t.sun_draws(rng, settings.n_draws) for t in targets])
    measured = {
        j: _Measured(
            t, t.model_slices(t.pops.h[[t.index]])[0], t.model_slices(sun_draws[:, j]), t.kinematics(metal.kinematics)
        )
        for j, t in enumerate(targets)
        if n_matched[j] >= settings.min_matches
    }
    point = np.full((len(VALUES), len(targets)), np.nan)
    draws = np.full((len(VALUES), settings.n_draws, len(targets)), np.nan)
    for j, m in measured.items():
        point[:, j] = m.values(metal.counts, m.best, m.stars, model)
    for d in range(settings.n_draws if measured else 0):
        sample = metal.counts.resampled(rng)
        for j, m in measured.items():
            draws[:, d, j] = m.values(sample, m.drawn[d], m.resampled_stars(rng), model)
    n_kin = [int(np.count_nonzero(t.near(metal.kinematics))) for t in targets]
    return _Heights(
        n_matched, point, draws, sun_draws, n_kin, {j: m.summary(metal.counts) for j, m in measured.items()}
    )


@dataclass(frozen=True)
class _CellValues:
    """The ``VALUES`` of one target at the best values and per draw, by name."""

    point: dict[str, float]
    draws: dict[str, FloatArray]

    @classmethod
    def of(cls, heights: _Heights, j: int) -> _CellValues:
        """Return the values of the target *j* of *heights*."""
        point = {name: float(v) for name, v in zip(VALUES, heights.point[:, j], strict=True)}
        return cls(point, dict(zip(VALUES, heights.draws[:, :, j], strict=True)))

    def estimate(self, name: str) -> Estimate:
        """Return the value *name* with the spread of its draws."""
        return Estimate(self.point[name], half_width(self.draws[name]))

    def over(self, name: str, value: float, draws: FloatArray) -> Estimate:
        """Return the value *name* over *value*, with the spread of the ratio of their *draws*."""
        return Estimate(self.point[name] / value, half_width(self.draws[name] / draws))


def _cell(metal: MetallicityCounts, target: _Target, heights: _Heights, j: int, k: float) -> ImigCell:
    """Return the cell of *target*, the target *j* of the [M/H] bin of *metal*, from the *heights* of the bin."""
    pops, i = target.pops, target.index
    values = _CellValues.of(heights, j)
    sun = Estimate(float(pops.h[i]), half_width(heights.sun_draws[:, j]))
    gradient, shape = heights.summaries.get(j, (float("nan"), []))
    return ImigCell(
        age_lo=pops.age_lo,
        age_hi=pops.age_hi,
        m_h_lo=metal.m_h_lo,
        m_h_hi=metal.m_h_hi,
        n_stars=int(pops.n_stars[i]),
        n_matched=heights.n_matched[j],
        theirs=CellTheirs(
            sun,
            scaled_profile_hz(cylinder_profile(pops, i), k)[0],
            values.estimate("theirs"),
            values.estimate("theirs_exponential"),
        ),
        ours=CellOurs(values.estimate("ours"), values.estimate("ours_exponential"), gradient, shape),
        kinematics=CellKinematics(heights.n_kin[j], values.estimate("sigma_z"), values.estimate("tracer")),
        ratios=CellRatios(
            values.over("ours", values.point["theirs"], values.draws["theirs"]),
            values.over("ours_exponential", values.point["theirs_exponential"], values.draws["theirs_exponential"]),
            values.over("tracer", values.point["ours"], values.draws["ours"]),
        ),
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
    heights = _heights(metal, targets, model, rng, settings)
    return [_cell(metal, target, heights, j, model.k) for j, target in enumerate(targets)]


def measure_cells(
    populations: list[ImigPopulations],
    metallicities: list[MetallicityCounts],
    model: SolarModel,
    rng: np.random.Generator,
    settings: CellSettings = CellSettings(),
) -> ImigCells:
    """Return every cell of *populations* against the counts of *metallicities*, and the summary of the ratios.

    The summary uses the cells with the minimum of matches whose ratio over their model
    through our counts and its error are finite; with fewer than three of them the median
    is NaN, as is every trend or slope without a degree of freedom.
    """
    min_matches = settings.min_matches
    cells = [cell for metal in metallicities for cell in _metallicity_cells(populations, metal, model, rng, settings)]
    ok = [c for c in _finite(cells, lambda c: c.ratios.counted) if c.n_matched >= min_matches]
    n_matches = int(metallicities[0].counts.tracer_age.size) if metallicities else 0
    median = float(np.median([c.ratios.counted.value for c in ok])) if len(ok) >= 3 else float("nan")
    trends = CellTrends(
        *(
            RatioTrends(_per_ln_age(ok, get), _per_dex(ok, get))
            for get in (lambda c: c.ratios.counted, lambda c: c.ratios.exponential, lambda c: c.ratios.tracer)
        )
    )
    return ImigCells(n_matches, min_matches, cells, len(ok), median, trends, _metallicity_slopes(ok))


def _finite(cells: list[ImigCell], get: Getter) -> list[ImigCell]:
    """Return the cells of *cells* whose value *get* and its error are finite, with a positive error."""
    return [c for c in cells if np.isfinite(get(c).value) and np.isfinite(get(c).err) and get(c).err > 0.0]


def _grouped(
    cells: list[ImigCell], get: Getter, x: Callable[[ImigCell], float], group: Callable[[ImigCell], float]
) -> Trend:
    """Return the slope of ln *get* against *x* over the cells where it is finite, one intercept per *group*."""
    kept = _finite(cells, get)
    return fit_grouped_log_slope(
        np.array([x(c) for c in kept]),
        np.array([get(c).value for c in kept]),
        np.array([get(c).err for c in kept]),
        np.array([group(c) for c in kept]),
    )


def _per_dex(cells: list[ImigCell], get: Getter) -> Trend:
    """Return the slope of ln *get* per dex of [M/H] at fixed age, one intercept per age bin."""
    return _grouped(cells, get, lambda c: 0.5 * (c.m_h_lo + c.m_h_hi), lambda c: c.age_lo)


def _per_ln_age(cells: list[ImigCell], get: Getter) -> Trend:
    """Return the slope of ln *get* per ln age at fixed [M/H], one intercept per [M/H] bin."""
    return _grouped(cells, get, lambda c: 0.5 * float(np.log(c.age_lo * c.age_hi)), lambda c: c.m_h_lo)


def _metallicity_slopes(cells: list[ImigCell]) -> MetallicitySlopes:
    """Return the slope of every height of *cells* per dex of [M/H] at fixed age, over the cells where it is finite."""
    return MetallicitySlopes(
        _per_dex(cells, lambda c: c.ours.windowed),
        _per_dex(cells, lambda c: c.ours.exponential),
        _per_dex(cells, lambda c: c.theirs.counted),
        _per_dex(cells, lambda c: c.theirs.sun),
        _per_dex(cells, lambda c: c.kinematics.tracer),
        _per_dex(cells, lambda c: c.kinematics.sigma_z),
    )


@dataclass(frozen=True)
class TargetedCells:
    """The cells measured with only the matches of one targeting box."""

    box: TargetingBox
    cells: ImigCells
