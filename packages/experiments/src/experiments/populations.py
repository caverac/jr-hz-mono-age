"""Mono-age populations in an annulus of a disk: kinematics, scale heights and the slope a = J_R,med / h_Z.

The star particles of the annulus ``R_MIN_KPC`` < R < ``R_MAX_KPC``, |z| < ``Z_SAMPLE_MAX_KPC``,
are split into 1 Gyr age bins from 0 to 10 Gyr. For each bin:

- J_R,med and sigma_R come from the stars with |z| < ``Z_JR_MAX_KPC``, and the midplane
  sigma_z from those with |z| < ``Z_MID_MAX_KPC``. Dispersions are clipped at
  ``VEL_CLIP_SIGMA`` MAD sigmas around the median (:mod:`experiments.stats`).
- h_Z is fitted to the heights weighted by initial mass, the analog of star counts,
  because the number of giants of a given age is proportional to the initial mass of
  the population. It is fitted in the self-consistent window k h_Z and in Palicio's
  fixed window Z_lim (:mod:`experiments.heights`).
- The sheet and the isothermal tracer predict h_Z from sigma_z. The tracer is put
  through the same self-consistent window twice: in the fitted potential and in the
  snapshot's. Two diagnostics of the tracer's assumptions go through it too: the vertical
  Jeans equation in the fitted potential with the measured sigma_z(z), without and with
  the tilt term (:mod:`experiments.jeans`).
- a = J_R,med / h_Z, and its predictions with the same J_R,med.
- The ratio of each predicted h_Z to the measured one, in which J_R,med cancels.

Errors are half the 16th-84th percentile interval of ``N_BOOTSTRAP`` resamplings of the
stars of a bin. Each resampling gives every quantity at once, so that ratios of
quantities measured on the same stars, such as a and h_pred / h_Z, carry their
correlations. The age exponent of each quantity is a weighted least-squares fit of ln y
against ln age. The trend of a ratio h_pred / h_Z is the same fit, with its error scaled
by the square root of the chi^2 per degree of freedom when that exceeds 1. A prediction
that reproduces how h_Z, and hence a, changes with age has a trend of 0: the trend is
the measured minus the predicted exponent of a, both fitted with one set of weights and
without the error of J_R,med.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.heights import (
    Z_GRID,
    particle_hz,
    profile_hz,
    scaled_particle_hz,
    scaled_profile_hz,
    sheet_hz,
    tracer_profile,
)
from experiments.jeans import JeansProfile, JeansStars, jeans_density, jeans_profile
from experiments.palicio import PalicioWindow
from experiments.potential import radial_actions
from experiments.snapshot import Galaxy, cylindrical_velocities
from experiments.stats import clipped_dispersion, half_width

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]

R_MIN_KPC, R_MAX_KPC = 7.5, 9.0
Z_SAMPLE_MAX_KPC = 12.0
Z_RMID_MAX_KPC = 2.5  # the stars whose median radius is R_mid
Z_JR_MAX_KPC = 0.5
Z_MID_MAX_KPC = 0.3
Z_SIGMA_KPC = 1.1  # the layer whose mass per area is the Sigma of the sheet
AGE_BINS = tuple((float(t), float(t + 1)) for t in range(10))
AGE_RANGES = ((0.0, 10.0), (2.0, 10.0), (0.0, 8.0), (2.0, 8.0))
K_MAIN = 2.0
K_SCAN = (1.5, 2.0, 3.0)
Z_LIM_SCAN_KPC = (0.5, 1.0, 1.5, 2.5)
N_BOOTSTRAP = 300
SEED = 20261023


@dataclass(frozen=True)
class Estimate:
    """A value and its 1-sigma error."""

    value: float
    err: float


@dataclass(frozen=True)
class Method:
    """The choices of a measurement: Palicio's window of the galaxy, the self-consistent k and the bootstrap size."""

    window: PalicioWindow
    k: float = K_MAIN
    n_bootstrap: int = N_BOOTSTRAP


@dataclass(frozen=True)
class AnnulusSample:
    """The star particles of the annulus: age (Gyr), z and R (kpc), v_R and v_z (km/s), J_R (kpc km/s), initial mass."""

    age: FloatArray
    z: FloatArray
    v_r: FloatArray
    v_z: FloatArray
    jr: FloatArray  # NaN where |z| >= Z_JR_MAX_KPC
    initial_mass: FloatArray
    r: FloatArray  # kpc, the cylindrical radius

    def select(self, keep: BoolArray | IndexArray) -> AnnulusSample:
        """Return the stars where the mask *keep* is true, or at the indices *keep*, which may repeat."""
        return AnnulusSample(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})


@dataclass(frozen=True)
class VerticalModel:
    """What the predictions of h_Z in one annulus need: Sigma, Delta Phi(R_mid, z) on ``Z_GRID`` and R_mid."""

    sigma_surf: float  # Msun/pc^2
    phi_fit: FloatArray  # (km/s)^2, the fitted potential
    phi_snapshot: FloatArray  # (km/s)^2, the snapshot's potential
    r_mid: float  # kpc


@dataclass(frozen=True)
class Kinematics:
    """The counts and the kinematics of one age bin."""

    n_jr: int
    n_mid: int
    sigma_r: Estimate
    sigma_z: Estimate
    jr_median: Estimate


@dataclass(frozen=True)
class Heights:
    """h_Z of one age bin in the self-consistent window, the window, its star count, and h_Z in the fixed window."""

    measured: Estimate
    window: float
    n_window: int
    fixed: float


@dataclass(frozen=True)
class Predictions:
    """h_Z of one age bin predicted by the sheet and by the tracer, with the tracer's window, and the Jeans diagnostics.

    ``jeans_sigma`` integrates the vertical Jeans equation with the measured sigma_z(z) and
    no tilt term, ``jeans`` with both.
    """

    sheet: float
    tracer: float
    tracer_window: float
    tracer_snapshot: float
    jeans_sigma: float
    jeans: float


@dataclass(frozen=True)
class Slopes:
    """a = J_R,med / h_Z of one age bin (km/s), measured and with each predicted h_Z in place of the measured one."""

    measured: Estimate
    sheet: Estimate
    tracer: Estimate
    tracer_snapshot: Estimate


@dataclass(frozen=True)
class Ratios:
    """Each predicted h_Z of one age bin over the measured h_Z."""

    sheet: Estimate
    tracer: Estimate
    tracer_snapshot: Estimate
    jeans_sigma: Estimate
    jeans: Estimate


@dataclass(frozen=True)
class AgeBin:  # pylint: disable=too-many-instance-attributes
    """Everything measured and predicted for one mono-age population."""

    age_lo: float
    age_hi: float
    age_median: float
    kinematics: Kinematics
    heights: Heights
    predictions: Predictions
    slopes: Slopes
    ratios: Ratios
    jeans: JeansProfile


@dataclass(frozen=True)
class Trend:
    """The age exponent of a ratio h_pred / h_Z, d ln(h_pred / h_Z) / d ln age, and its chi^2 per degree of freedom.

    ``err`` is the 1-sigma error of the weighted fit times sqrt(max(1, ``chi2_dof``)), so
    that a scatter of the ratios about a power law beyond their errors widens it.
    """

    value: float
    err: float
    chi2_dof: float


@dataclass(frozen=True)
class AgeRange:
    """Age exponents over the bins inside [lo, hi] Gyr.

    ``trends`` holds the trend of h_pred / h_Z of each prediction, which is 0 for a
    prediction that reproduces the age dependence of h_Z.
    ``eta`` is d ln h_Z / d ln sigma_z, the ratio of the exponents of h_Z and sigma_z.
    ``sheet_exponent`` is 2 (beta_R - beta_z), the exponent of a that the sheet predicts
    from the exponents beta_R and beta_z of the two dispersions alone.
    """

    lo: float
    hi: float
    exponents: dict[str, Estimate]
    trends: dict[str, Trend]
    eta: float
    sheet_exponent: float


@dataclass(frozen=True)
class ScanRow:
    """Measured and tracer exponents of a at one window parameter (k or Z_lim), and the oldest bin's h_Z and window."""

    parameter: float
    measured: float
    tracer: float
    hz_oldest: float
    window_oldest: float


@dataclass(frozen=True)
class MonoAge:
    """The mono-age measurement of one annulus."""

    method: Method
    hz_all_fixed: float
    k_palicio: float
    bins: list[AgeBin]
    ranges: list[AgeRange]
    k_scan: list[ScanRow]
    window_scan: list[ScanRow]


def surface_density(galaxy: Galaxy, r_lo: float, r_hi: float) -> float:
    """Return the mass per area (Msun/pc^2) of every particle with r_lo <= R < r_hi and |z| < ``Z_SIGMA_KPC``."""
    mass = 0.0
    for parts in galaxy.by_type().values():
        cyl_r = parts.cylindrical_radius
        inside = (cyl_r >= r_lo) & (cyl_r < r_hi) & (np.abs(parts.pos[:, 2]) < Z_SIGMA_KPC)
        mass += float(np.sum(parts.mass[inside]))
    return mass / (np.pi * (r_hi**2 - r_lo**2) * 1.0e6)


def annulus_sample(galaxy: Galaxy, pot: agama.Potential, r_lo: float, r_hi: float) -> tuple[AnnulusSample, BoolArray]:
    """Return the star particles with r_lo <= R < r_hi and |z| < ``Z_SAMPLE_MAX_KPC``, and their mask.

    J_R is computed only for the stars with |z| < ``Z_JR_MAX_KPC``, the only ones it
    enters.
    """
    stars = galaxy.stars
    cyl_r = stars.cylindrical_radius
    annulus = (cyl_r >= r_lo) & (cyl_r < r_hi) & (np.abs(stars.pos[:, 2]) < Z_SAMPLE_MAX_KPC)
    pos, vel = stars.pos[annulus], stars.vel[annulus]
    v_r, v_z = cylindrical_velocities(pos, vel)
    jr = np.full(pos.shape[0], np.nan)
    near = np.abs(pos[:, 2]) < Z_JR_MAX_KPC
    jr[near] = radial_actions(pot, pos[near], vel[near])
    age = galaxy.header.cosmology.lookback_gyr(stars.formation_a[annulus], galaxy.header.scale_factor)
    return AnnulusSample(age, pos[:, 2], v_r, v_z, jr, stars.initial_mass[annulus], cyl_r[annulus]), annulus


def median_radius(galaxy: Galaxy, annulus: BoolArray) -> float:
    """Return R_mid, the median radius of the *annulus* stars with |z| < ``Z_RMID_MAX_KPC``."""
    stars = galaxy.stars
    near = np.abs(stars.pos[annulus, 2]) < Z_RMID_MAX_KPC
    return float(np.median(stars.cylindrical_radius[annulus][near]))


@dataclass(frozen=True)
class _BinValues:  # pylint: disable=too-many-instance-attributes
    """What one sample of the stars of a bin gives: kinematics, h_Z, its predictions and their windows."""

    sigma_r: float
    sigma_z: float
    jr_median: float
    hz: float
    window: float
    sheet: float
    tracer: float
    tracer_window: float
    tracer_snapshot: float
    jeans_sigma: float
    jeans: float


def _bin_values(rows: AnnulusSample, model: VerticalModel, k: float) -> _BinValues:
    """Return the kinematics, h_Z and the predicted h_Z of the stars *rows* of one bin."""
    absz = np.abs(rows.z)
    jr_rows = (absz < Z_JR_MAX_KPC) & np.isfinite(rows.jr)
    sigma_z = clipped_dispersion(rows.v_z[absz < Z_MID_MAX_KPC])
    hz, window = scaled_particle_hz(absz, rows.initial_mass, k)
    tracer, tracer_window = scaled_profile_hz(tracer_profile(model.phi_fit, sigma_z), k)
    profile = _jeans(rows, model)
    return _BinValues(
        sigma_r=clipped_dispersion(rows.v_r[jr_rows]),
        sigma_z=sigma_z,
        jr_median=float(np.median(rows.jr[jr_rows])),
        hz=hz,
        window=window,
        sheet=sheet_hz(sigma_z, model.sigma_surf),
        tracer=tracer,
        tracer_window=tracer_window,
        tracer_snapshot=scaled_profile_hz(tracer_profile(model.phi_snapshot, sigma_z), k)[0],
        jeans_sigma=scaled_profile_hz(jeans_density(model.phi_fit, profile, Z_GRID, with_tilt=False), k)[0],
        jeans=scaled_profile_hz(jeans_density(model.phi_fit, profile, Z_GRID, with_tilt=True), k)[0],
    )


def _jeans(rows: AnnulusSample, model: VerticalModel) -> JeansProfile:
    """Return sigma_z^2(z) and the tilt term of the stars *rows* of one bin, in the radial bins of the annulus."""
    stars = JeansStars(rows.r, rows.z, rows.v_r, rows.v_z, rows.initial_mass)
    return jeans_profile(stars, R_MIN_KPC, R_MAX_KPC, model.r_mid)


def measure_bin(
    rows: AnnulusSample, bounds: tuple[float, float], model: VerticalModel, method: Method, rng: np.random.Generator
) -> AgeBin:
    """Return the measurement and the predictions of the age bin *rows*, whose ages lie within *bounds* (Gyr).

    Every error comes from one bootstrap: each draw resamples the stars of the bin with
    replacement and recomputes every quantity from them.
    """
    point = _bin_values(rows, model, method.k)
    n = rows.z.size
    draws = [_bin_values(rows.select(rng.integers(0, n, n)), model, method.k) for _ in range(method.n_bootstrap)]

    def spread(get: Callable[[_BinValues], float]) -> Estimate:
        return Estimate(get(point), half_width(np.array([get(draw) for draw in draws], dtype=np.float64)))

    absz = np.abs(rows.z)
    kinematics = Kinematics(
        n_jr=int(np.count_nonzero((absz < Z_JR_MAX_KPC) & np.isfinite(rows.jr))),
        n_mid=int(np.count_nonzero(absz < Z_MID_MAX_KPC)),
        sigma_r=spread(lambda v: v.sigma_r),
        sigma_z=spread(lambda v: v.sigma_z),
        jr_median=spread(lambda v: v.jr_median),
    )
    heights = Heights(
        measured=spread(lambda v: v.hz),
        window=point.window,
        n_window=int(np.count_nonzero(absz <= point.window)),
        fixed=particle_hz(absz, rows.initial_mass, method.window.z_lim),
    )
    slopes = Slopes(
        measured=spread(lambda v: v.jr_median / v.hz),
        sheet=spread(lambda v: v.jr_median / v.sheet),
        tracer=spread(lambda v: v.jr_median / v.tracer),
        tracer_snapshot=spread(lambda v: v.jr_median / v.tracer_snapshot),
    )
    ratios = Ratios(
        sheet=spread(lambda v: v.sheet / v.hz),
        tracer=spread(lambda v: v.tracer / v.hz),
        tracer_snapshot=spread(lambda v: v.tracer_snapshot / v.hz),
        jeans_sigma=spread(lambda v: v.jeans_sigma / v.hz),
        jeans=spread(lambda v: v.jeans / v.hz),
    )
    return AgeBin(
        *bounds,
        float(np.median(rows.age)),
        kinematics,
        heights,
        Predictions(
            point.sheet, point.tracer, point.tracer_window, point.tracer_snapshot, point.jeans_sigma, point.jeans
        ),
        slopes,
        ratios,
        _jeans(rows, model),
    )


def _log_line(x: FloatArray, y: FloatArray, y_err: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Return the coefficients, their covariance and the chi^2 of the weighted least-squares line of ln y on ln x."""
    lx, ly, w = np.log(x), np.log(y), (y / y_err) ** 2
    design = np.vstack([np.ones_like(lx), lx]).T
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    coef = cov @ (design.T @ (w * ly))
    return coef, cov, float(np.sum(w * (ly - design @ coef) ** 2))


def fit_power_law(x: FloatArray, y: FloatArray, y_err: FloatArray) -> Estimate:
    """Return the exponent p of y = A x^p and its 1-sigma error, by weighted least squares of ln y on ln x."""
    coef, cov, _ = _log_line(x, y, y_err)
    return Estimate(float(coef[1]), float(np.sqrt(cov[1, 1])))


def fit_trend(x: FloatArray, y: FloatArray, y_err: FloatArray) -> Trend:
    """Return the exponent of y = A x^p, its error times sqrt(max(1, chi^2 per degree of freedom)), and that chi^2."""
    coef, cov, chi2 = _log_line(x, y, y_err)
    chi2_dof = chi2 / (x.size - 2)
    return Trend(float(coef[1]), float(np.sqrt(cov[1, 1] * max(1.0, chi2_dof))), chi2_dof)


def range_exponents(bins: list[AgeBin], lo: float, hi: float) -> AgeRange:
    """Return the age exponents of the bins that lie inside [*lo*, *hi*] Gyr."""
    chosen = [b for b in bins if b.age_lo >= lo and b.age_hi <= hi]
    age = np.array([b.age_median for b in chosen])

    def column(get: Callable[[AgeBin], float]) -> FloatArray:
        return np.array([get(b) for b in chosen], dtype=np.float64)

    def fit(get: Callable[[AgeBin], Estimate]) -> Estimate:
        return fit_power_law(age, column(lambda b: get(b).value), column(lambda b: get(b).err))

    def trend(get: Callable[[AgeBin], Estimate]) -> Trend:
        return fit_trend(age, column(lambda b: get(b).value), column(lambda b: get(b).err))

    exponents = {
        "sigma_r": fit(lambda b: b.kinematics.sigma_r),
        "sigma_z": fit(lambda b: b.kinematics.sigma_z),
        "jr_median": fit(lambda b: b.kinematics.jr_median),
        "hz": fit(lambda b: b.heights.measured),
        "a": fit(lambda b: b.slopes.measured),
        "a_sheet": fit(lambda b: b.slopes.sheet),
        "a_tracer": fit(lambda b: b.slopes.tracer),
        "a_tracer_snapshot": fit(lambda b: b.slopes.tracer_snapshot),
    }
    return AgeRange(
        lo=lo,
        hi=hi,
        exponents=exponents,
        trends={
            "sheet": trend(lambda b: b.ratios.sheet),
            "tracer": trend(lambda b: b.ratios.tracer),
            "tracer_snapshot": trend(lambda b: b.ratios.tracer_snapshot),
            "jeans_sigma": trend(lambda b: b.ratios.jeans_sigma),
            "jeans": trend(lambda b: b.ratios.jeans),
        },
        eta=exponents["hz"].value / exponents["sigma_z"].value,
        sheet_exponent=2.0 * (exponents["sigma_r"].value - exponents["sigma_z"].value),
    )


def _scan_row(parameter: float, bins: list[AgeBin], hz: FloatArray, hz_tracer: FloatArray, window: float) -> ScanRow:
    """Return the exponents of a with the heights *hz* and *hz_tracer*, weighted with the errors of the main run."""
    age = np.array([b.age_median for b in bins])
    jr = np.array([b.kinematics.jr_median.value for b in bins])
    rel_a = np.array([b.slopes.measured.err / b.slopes.measured.value for b in bins])
    return ScanRow(
        parameter=parameter,
        measured=fit_power_law(age, jr / hz, rel_a * jr / hz).value,
        tracer=fit_power_law(age, jr / hz_tracer, rel_a * jr / hz_tracer).value,
        hz_oldest=float(hz[-1]),
        window_oldest=window,
    )


def k_scan(
    groups: list[AnnulusSample], bins: list[AgeBin], phi_z: FloatArray, k_values: tuple[float, ...]
) -> list[ScanRow]:
    """Return the exponents of a, measured and from the tracer, with the self-consistent window at each k."""
    rows = []
    for k in k_values:
        hz = np.array([scaled_particle_hz(np.abs(g.z), g.initial_mass, k)[0] for g in groups])
        hz_tracer = np.array([scaled_profile_hz(tracer_profile(phi_z, b.kinematics.sigma_z.value), k)[0] for b in bins])
        rows.append(_scan_row(k, bins, hz, hz_tracer, float(k * hz[-1])))
    return rows


def window_scan(groups: list[AnnulusSample], bins: list[AgeBin], phi_z: FloatArray) -> list[ScanRow]:
    """Return the exponents of a, measured and from the tracer, with h_Z fitted in each fixed window Z_lim."""
    rows = []
    for z_lim in Z_LIM_SCAN_KPC:
        hz = np.array([particle_hz(np.abs(g.z), g.initial_mass, z_lim) for g in groups])
        hz_tracer = np.array([profile_hz(tracer_profile(phi_z, b.kinematics.sigma_z.value), z_lim) for b in bins])
        rows.append(_scan_row(z_lim, bins, hz, hz_tracer, z_lim))
    return rows


def measure_mono_age(sample: AnnulusSample, model: VerticalModel, method: Method) -> MonoAge:
    """Return the mono-age measurement of the annulus *sample*."""
    rng = np.random.default_rng(SEED)
    absz = np.abs(sample.z)
    z_lim = method.window.z_lim
    hz_all = particle_hz(absz, sample.initial_mass, z_lim)
    groups = [sample.select((sample.age >= lo) & (sample.age < hi)) for lo, hi in AGE_BINS]
    bins = [measure_bin(g, bounds, model, method, rng) for g, bounds in zip(groups, AGE_BINS, strict=True)]
    return MonoAge(
        method=method,
        hz_all_fixed=hz_all,
        k_palicio=z_lim / hz_all,
        bins=bins,
        ranges=[range_exponents(bins, lo, hi) for lo, hi in AGE_RANGES],
        k_scan=k_scan(groups, bins, model.phi_fit, (*K_SCAN, z_lim / hz_all)),
        window_scan=window_scan(groups, bins, model.phi_fit),
    )
