"""The Milky Way test of the isothermal tracer: star-count h_Z of mono-age populations against the tracer and the sheet.

For each survey (:mod:`experiments.mw_catalogs`) and each age bin of ``AGE_EDGES_GYR``:

- h_Z comes from the star counts of the Gaia cylinder split into ages by the survey
  (:mod:`experiments.mw_counts`), in the self-consistent window L = k h_Z;
- sigma_z comes from the kinematic stars of the bin with |z| < ``Z_MID_MAX_KPC``, and
  sigma_R and J_R,med from those with |z| < ``Z_JR_MAX_KPC``, as in the Auriga annulus;
  the dispersions are clipped and their median velocity error is removed in quadrature
  (:func:`experiments.stats.deconvolved_dispersion`); J_R is computed in McMillan (2017);
- the isothermal tracer, exp(-Delta Phi / sigma_z^2) at R_0 in McMillan (2017), goes through
  the same counts: its mass in each slice sets its fraction, which the interpolation and the
  window of the counts then fit (:func:`experiments.mw_counts.predicted_hz`);
- the sheet, sigma_z^2 / (pi G Sigma) with Sigma the mass of the potential within
  |z| < 1.1 kpc at R_0, as for the Auriga disks, goes through the same counts with its
  profile sech^2(|z| / z_0) (:func:`experiments.heights.sheet_profile`), whose windowed h_Z
  at k = 2 is that scale;
- the vertical Jeans equation in McMillan (2017), with sigma_z^2 measured in the slices of
  |z| of ``JEANS_SLICES_KPC`` and no tilt term, goes through the same counts; it drops the
  tracer's assumption that sigma_z does not change with height;
- the ratios h_pred / h_Z carry the errors of one bootstrap that resamples, in each draw,
  the Gaia stars, their spectroscopic matches and the kinematic stars of the bin;
- the tracer in each potential of a family (:mod:`experiments.mw_potentials`) goes through
  the counts once, without resampling, which shows how the ratios move with the potential.

Over each range of ``AGE_RANGES``, the trend of a ratio with age tests whether the
prediction reproduces the age dependence of h_Z (:func:`experiments.populations.fit_trend`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.heights import Z_GRID, sheet_profile, tracer_profile
from experiments.jeans import JeansProfile, jeans_density
from experiments.mw_catalogs import KinematicStars
from experiments.mw_counts import (
    CountBias,
    CountSample,
    count_bias,
    population_counts,
    predicted_hz,
    profile_slices,
)
from experiments.mw_dynamics import R0_KPC, solar_vertical_potential, surface_density
from experiments.populations import (
    K_MAIN,
    Z_JR_MAX_KPC,
    Z_MID_MAX_KPC,
    Estimate,
    Trend,
    fit_eta,
    fit_power_law,
    fit_trend,
)
from experiments.stats import deconvolved_dispersion, half_width

FloatArray = NDArray[np.float64]

AGE_EDGES_GYR = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
JEANS_SLICES_KPC = (0.0, 0.3, 0.6, 1.0, 1.5)  # |z| slices of sigma_z^2 for the Jeans equation
N_MIN_JEANS_SLICE = 30  # kinematic stars in a slice for its sigma_z
AGE_RANGES = ((0.0, 10.0), (2.0, 10.0))
N_BOOTSTRAP = 300
SEED = 20261028


@dataclass(frozen=True)
class SolarModel:
    """What the predictions need: Delta Phi(R_0, z) ((km/s)^2) on the height grid, Sigma (Msun/pc^2) and k."""

    phi: FloatArray
    sigma_surf: float
    k: float = K_MAIN


def solar_model(pot: agama.Potential, k: float = K_MAIN) -> SolarModel:
    """Return the vertical model of *pot* at R_0."""
    return SolarModel(solar_vertical_potential(pot), surface_density(pot, R0_KPC), k)


@dataclass(frozen=True)
class Survey:
    """One survey: its name, its star counts with ages, its kinematic stars and their J_R (kpc km/s)."""

    name: str
    counts: CountSample
    kinematics: KinematicStars
    jr: FloatArray


@dataclass(frozen=True)
class MwKinematics:
    """The kinematics of one age bin and the count of stars with a J_R."""

    n_kin: int
    sigma_r: Estimate
    sigma_z: Estimate
    jr_median: Estimate


@dataclass(frozen=True)
class MwHeights:
    """h_Z (kpc) of one age bin from the counts, its window, the matches, failed draws, and its gradient b (1/kpc)."""

    measured: Estimate
    window: float
    n_matched: int
    failed: float  # the fraction of bootstrap draws without an h_Z
    gradient: float  # of the population fraction exp(b x) across the cylinder


@dataclass(frozen=True)
class MwPredictions:
    """h_Z (kpc) of one age bin predicted by the sheet, the tracer and the Jeans equation, through the counts."""

    sheet: float
    tracer: float
    jeans: float


@dataclass(frozen=True)
class MwRatios:
    """Each predicted h_Z of one age bin over the measured h_Z; ``family`` holds the tracer of each family member."""

    sheet: Estimate
    tracer: Estimate
    jeans: Estimate
    family: list[float]


@dataclass(frozen=True)
class MwBin:
    """Everything measured and predicted for one mono-age population of the Milky Way."""

    age_lo: float
    age_hi: float
    age_median: float  # of the matches in the population
    kinematics: MwKinematics
    heights: MwHeights
    predictions: MwPredictions
    ratios: MwRatios


@dataclass(frozen=True)
class MwRange:
    """Age exponents over the bins inside [lo, hi] Gyr with a measured h_Z, and the trends of h_pred / h_Z."""

    lo: float
    hi: float
    n_bins: int
    exponents: dict[str, Estimate]
    trends: dict[str, Trend]
    eta: Trend  # d ln h_Z / d ln sigma_z across the bins (experiments.populations.fit_eta)


@dataclass(frozen=True)
class MwSurvey:
    """The test with one survey, and what its counts give exponentials of known scales."""

    name: str
    n_gaia: int
    n_matched: int
    bins: list[MwBin]
    ranges: list[MwRange]
    bias: CountBias


@dataclass(frozen=True)
class _Values:  # pylint: disable=too-many-instance-attributes
    """What one sample of the counts and of the kinematic stars of a bin gives."""

    hz: float
    window: float
    gradient: float
    sigma_r: float
    sigma_z: float
    jr_median: float
    sheet: float
    tracer: float
    jeans: float


def jeans_slices(kin: KinematicStars) -> JeansProfile:
    """Return sigma_z^2 of the stars *kin* in each slice of ``JEANS_SLICES_KPC`` with enough of them, at its median |z|.

    The dispersions are clipped and their velocity errors removed, as near the plane; the
    tilt term is not measured.
    """
    absz = np.abs(kin.phase.z)
    z, sigma2 = [], []
    for lo, hi in zip(JEANS_SLICES_KPC[:-1], JEANS_SLICES_KPC[1:], strict=True):
        rows = (absz >= lo) & (absz < hi)
        if np.count_nonzero(rows) >= N_MIN_JEANS_SLICE:
            z.append(float(np.median(absz[rows])))
            sigma2.append(deconvolved_dispersion(kin.phase.v_z[rows], kin.v_z_err[rows]) ** 2)
    return JeansProfile(tuple(z), tuple(sigma2), tuple(float("nan") for _ in z))


def _values(
    counts: CountSample, kin: KinematicStars, jr: FloatArray, bounds: tuple[float, float], model: SolarModel
) -> _Values:
    """Return h_Z, the kinematics and the predictions of the age bin *bounds*, from *counts* and its stars *kin*.

    The predictions go through *counts*, the sample that h_Z comes from.
    """
    population = population_counts(counts, bounds[0], bounds[1])
    hz, window = population.hz(model.k)
    absz = np.abs(kin.phase.z)
    jr_rows = (absz < Z_JR_MAX_KPC) & np.isfinite(jr) & (jr >= 0.0)
    mid = absz < Z_MID_MAX_KPC
    sigma_z = deconvolved_dispersion(kin.phase.v_z[mid], kin.v_z_err[mid])
    return _Values(
        hz=hz,
        window=window,
        gradient=population.gradient,
        sigma_r=deconvolved_dispersion(kin.phase.v_r[jr_rows], kin.v_r_err[jr_rows]),
        sigma_z=sigma_z,
        jr_median=float(np.median(jr[jr_rows])),
        sheet=predicted_hz(counts, profile_slices(sheet_profile(sigma_z, model.sigma_surf)), model.k)[0],
        tracer=predicted_hz(counts, profile_slices(tracer_profile(model.phi, sigma_z)), model.k)[0],
        jeans=predicted_hz(
            counts, profile_slices(jeans_density(model.phi, jeans_slices(kin), Z_GRID, with_tilt=False)), model.k
        )[0],
    )


@dataclass(frozen=True)
class BinSetting:
    """What every bin is measured with: the model, the generator and size of the bootstrap, and the family."""

    model: SolarModel
    rng: np.random.Generator
    n_bootstrap: int
    family: tuple[SolarModel, ...] = ()


def measure_bin(survey: Survey, bounds: tuple[float, float], setting: BinSetting) -> MwBin:
    """Return the measurement and the predictions of the age bin *bounds* (Gyr) of *survey*, with bootstrap errors.

    The tracer in each model of the family of *setting* goes through the counts once, with
    the sigma_z of the point estimate.
    """
    model, rng = setting.model, setting.rng
    in_age = (survey.kinematics.age >= bounds[0]) & (survey.kinematics.age < bounds[1])
    kin, jr = survey.kinematics.select(in_age), survey.jr[in_age]
    point = _values(survey.counts, kin, jr, bounds, model)
    draws = []
    for _ in range(setting.n_bootstrap):
        rows = rng.integers(0, jr.size, jr.size)
        draws.append(_values(survey.counts.resampled(rng), kin.select(rows), jr[rows], bounds, model))

    def spread(get: Callable[[_Values], float]) -> Estimate:
        return Estimate(get(point), half_width(np.array([get(d) for d in draws], dtype=np.float64)))

    counts = survey.counts
    in_pop = counts.tracer_in_disk & (counts.tracer_age >= bounds[0]) & (counts.tracer_age < bounds[1])
    absz = np.abs(kin.phase.z)
    return MwBin(
        age_lo=bounds[0],
        age_hi=bounds[1],
        age_median=float(np.median(counts.tracer_age[in_pop])) if in_pop.any() else float("nan"),
        kinematics=MwKinematics(
            int(np.count_nonzero((absz < Z_JR_MAX_KPC) & np.isfinite(jr) & (jr >= 0.0))),
            spread(lambda v: v.sigma_r),
            spread(lambda v: v.sigma_z),
            spread(lambda v: v.jr_median),
        ),
        heights=MwHeights(
            spread(lambda v: v.hz),
            point.window,
            int(in_pop.sum()),
            float(np.mean([not np.isfinite(d.hz) for d in draws])),
            point.gradient,
        ),
        predictions=MwPredictions(point.sheet, point.tracer, point.jeans),
        ratios=MwRatios(
            spread(lambda v: v.sheet / v.hz),
            spread(lambda v: v.tracer / v.hz),
            spread(lambda v: v.jeans / v.hz),
            _family_ratios(survey.counts, point, setting.family),
        ),
    )


def _family_ratios(counts: CountSample, point: _Values, family: tuple[SolarModel, ...]) -> list[float]:
    """Return the tracer of the sigma_z of *point* in each model of *family* over the h_Z of *point*."""
    return [
        predicted_hz(counts, profile_slices(tracer_profile(m.phi, point.sigma_z)), m.k)[0] / point.hz for m in family
    ]


def measure_range(bins: list[MwBin], lo: float, hi: float) -> MwRange:
    """Return the age exponents and the trends over the bins inside [*lo*, *hi*] Gyr with a measured h_Z."""
    chosen = [
        b
        for b in bins
        if b.age_lo >= lo and b.age_hi <= hi and np.isfinite(b.heights.measured.value) and np.isfinite(b.age_median)
    ]
    age = np.array([b.age_median for b in chosen])

    def column(get: Callable[[MwBin], Estimate]) -> tuple[FloatArray, FloatArray]:
        return np.array([get(b).value for b in chosen]), np.array([get(b).err for b in chosen])

    exponents = {
        "hz": fit_power_law(age, *column(lambda b: b.heights.measured)),
        "sigma_z": fit_power_law(age, *column(lambda b: b.kinematics.sigma_z)),
        "jr_median": fit_power_law(age, *column(lambda b: b.kinematics.jr_median)),
    }
    return MwRange(
        lo=lo,
        hi=hi,
        n_bins=len(chosen),
        exponents=exponents,
        trends={
            "sheet": fit_trend(age, *column(lambda b: b.ratios.sheet)),
            "tracer": fit_trend(age, *column(lambda b: b.ratios.tracer)),
            "jeans": fit_trend(age, *column(lambda b: b.ratios.jeans)),
        },
        eta=fit_eta([b.kinematics.sigma_z for b in chosen], [b.heights.measured for b in chosen]),
    )


def measure_survey(
    survey: Survey,
    model: SolarModel,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    family: tuple[SolarModel, ...] = (),
) -> MwSurvey:
    """Return the test with *survey*: every age bin of ``AGE_EDGES_GYR`` and every range of ``AGE_RANGES``."""
    edges = AGE_EDGES_GYR
    setting = BinSetting(model, rng, n_bootstrap, family)
    bins = [measure_bin(survey, (lo, hi), setting) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return MwSurvey(
        name=survey.name,
        n_gaia=int(survey.counts.gaia_absz.size),
        n_matched=int(survey.counts.tracer_age.size),
        bins=bins,
        ranges=[measure_range(bins, lo, hi) for lo, hi in AGE_RANGES],
        bias=count_bias(survey.counts, model.k),
    )
