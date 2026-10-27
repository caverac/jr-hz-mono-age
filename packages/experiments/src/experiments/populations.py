"""Mono-age populations in an annulus of a disk: kinematics, scale heights and the slope a = J_R,med / h_Z.

The star particles of the annulus ``R_MIN_KPC`` < R < ``R_MAX_KPC``, |z| < ``Z_SAMPLE_MAX_KPC``,
are split into 1 Gyr age bins from 0 to 10 Gyr. For each bin:

- J_R,med and sigma_R come from the stars with |z| < ``Z_JR_MAX_KPC``, and the midplane
  sigma_z from those with |z| < ``Z_MID_MAX_KPC``. Dispersions are clipped at
  ``VEL_CLIP_SIGMA`` MAD sigmas around the median.
- h_Z is fitted to the heights weighted by initial mass, the analog of star counts,
  because the number of giants of a given age is proportional to the initial mass of
  the population. It is fitted in the self-consistent window k h_Z and in Palicio's
  fixed window Z_lim (:mod:`experiments.heights`).
- The sheet and the isothermal tracer predict h_Z from sigma_z. The tracer is put
  through the same self-consistent window twice: in the fitted potential and in the
  snapshot's.
- a = J_R,med / h_Z, and its predictions with the same J_R,med.

Errors are half the 16th-84th percentile interval of ``N_BOOTSTRAP`` resamplings. The
age exponent of each quantity is a weighted least-squares fit of ln y against ln age;
the predictions of a carry the relative error of J_R,med combined with that of sigma_z
times 2 (sheet) or 1 (tracer).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.heights import (
    particle_hz,
    profile_hz,
    scaled_particle_hz,
    scaled_profile_hz,
    sheet_hz,
    tracer_profile,
)
from experiments.palicio import PalicioWindow
from experiments.potential import radial_actions
from experiments.snapshot import Galaxy, cylindrical_velocities

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]

R_MIN_KPC, R_MAX_KPC = 7.5, 9.0
Z_SAMPLE_MAX_KPC = 12.0
Z_RMID_MAX_KPC = 2.5  # the stars whose median radius is R_mid
Z_JR_MAX_KPC = 0.5
Z_MID_MAX_KPC = 0.3
Z_SIGMA_KPC = 1.1  # the layer whose mass per area is the Sigma of the sheet
VEL_CLIP_SIGMA = 4.0
MAD_TO_SIGMA = 1.4826
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
    """The star particles of the annulus: age (Gyr), z (kpc), v_R and v_z (km/s), J_R (kpc km/s), initial mass."""

    age: FloatArray
    z: FloatArray
    v_r: FloatArray
    v_z: FloatArray
    jr: FloatArray  # NaN where |z| >= Z_JR_MAX_KPC
    initial_mass: FloatArray

    def select(self, keep: BoolArray) -> AnnulusSample:
        """Return the stars where *keep* is true."""
        return AnnulusSample(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})


@dataclass(frozen=True)
class VerticalModel:
    """What the predictions of h_Z in one annulus need: Sigma and Delta Phi(R_mid, z) on ``Z_GRID``."""

    sigma_surf: float  # Msun/pc^2
    phi_fit: FloatArray  # (km/s)^2, the fitted potential
    phi_snapshot: FloatArray  # (km/s)^2, the snapshot's potential


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
    """h_Z of one age bin predicted by the sheet and by the tracer, with the tracer's window."""

    sheet: float
    tracer: float
    tracer_window: float
    tracer_snapshot: float


@dataclass(frozen=True)
class Slopes:
    """a = J_R,med / h_Z of one age bin, measured and predicted (km/s)."""

    measured: Estimate
    sheet: float
    tracer: float
    tracer_snapshot: float


@dataclass(frozen=True)
class AgeBin:
    """Everything measured and predicted for one mono-age population."""

    age_lo: float
    age_hi: float
    age_median: float
    kinematics: Kinematics
    heights: Heights
    predictions: Predictions
    slopes: Slopes


@dataclass(frozen=True)
class AgeRange:
    """Age exponents over the bins inside [lo, hi] Gyr.

    ``eta`` is d ln h_Z / d ln sigma_z, the ratio of the exponents of h_Z and sigma_z.
    ``sheet_exponent`` is 2 (beta_R - beta_z), the exponent of a that the sheet predicts
    from the exponents beta_R and beta_z of the two dispersions alone.
    """

    lo: float
    hi: float
    exponents: dict[str, Estimate]
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


def clipped_dispersion(v: FloatArray) -> float:
    """Return the dispersion of *v* within ``VEL_CLIP_SIGMA`` MAD sigmas of its median."""
    center = np.median(v)
    mad = MAD_TO_SIGMA * np.median(np.abs(v - center))
    return float(np.std(v[np.abs(v - center) < VEL_CLIP_SIGMA * mad]))


def bootstrap(rng: np.random.Generator, n: int, stat: Callable[[IndexArray], float], n_draws: int) -> FloatArray:
    """Return *n_draws* values of *stat* on indices resampled from range(*n*)."""
    return np.array([stat(rng.integers(0, n, n)) for _ in range(n_draws)], dtype=np.float64)


def half_width(draws: FloatArray) -> float:
    """Return half the 16th-84th percentile interval of the finite *draws*."""
    lo, hi = np.percentile(draws[np.isfinite(draws)], [16.0, 84.0])
    return float(0.5 * (hi - lo))


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
    return AnnulusSample(age, pos[:, 2], v_r, v_z, jr, stars.initial_mass[annulus]), annulus


def median_radius(galaxy: Galaxy, annulus: BoolArray) -> float:
    """Return R_mid, the median radius of the *annulus* stars with |z| < ``Z_RMID_MAX_KPC``."""
    stars = galaxy.stars
    near = np.abs(stars.pos[annulus, 2]) < Z_RMID_MAX_KPC
    return float(np.median(stars.cylindrical_radius[annulus][near]))


def _kinematics(rows: AnnulusSample, rng: np.random.Generator, n_draws: int) -> tuple[Kinematics, FloatArray]:
    """Return the kinematics of one bin, and the bootstrap draws of J_R,med that the error of a needs."""
    absz = np.abs(rows.z)
    jr_rows = (absz < Z_JR_MAX_KPC) & np.isfinite(rows.jr)
    jr, v_r = rows.jr[jr_rows], rows.v_r[jr_rows]
    v_z = rows.v_z[absz < Z_MID_MAX_KPC]
    jr_draws = bootstrap(rng, jr.size, lambda i: float(np.median(jr[i])), n_draws)
    sigma_r = Estimate(
        clipped_dispersion(v_r), half_width(bootstrap(rng, v_r.size, lambda i: clipped_dispersion(v_r[i]), n_draws))
    )
    sigma_z = Estimate(
        clipped_dispersion(v_z), half_width(bootstrap(rng, v_z.size, lambda i: clipped_dispersion(v_z[i]), n_draws))
    )
    jr_median = Estimate(float(np.median(jr)), half_width(jr_draws))
    return Kinematics(int(jr.size), int(v_z.size), sigma_r, sigma_z, jr_median), jr_draws


def _heights(rows: AnnulusSample, method: Method, rng: np.random.Generator) -> tuple[Heights, FloatArray]:
    """Return the heights of one bin, and the bootstrap draws of h_Z that the error of a needs."""
    absz, weight = np.abs(rows.z), rows.initial_mass
    hz, window = scaled_particle_hz(absz, weight, method.k)
    hz_draws = bootstrap(
        rng, absz.size, lambda i: scaled_particle_hz(absz[i], weight[i], method.k)[0], method.n_bootstrap
    )
    heights = Heights(
        measured=Estimate(hz, half_width(hz_draws)),
        window=window,
        n_window=int(np.count_nonzero(absz <= window)),
        fixed=particle_hz(absz, weight, method.window.z_lim),
    )
    return heights, hz_draws


def measure_bin(
    rows: AnnulusSample, bounds: tuple[float, float], model: VerticalModel, method: Method, rng: np.random.Generator
) -> AgeBin:
    """Return the measurement and the predictions of the age bin *rows*, whose ages lie within *bounds* (Gyr)."""
    kinematics, jr_draws = _kinematics(rows, rng, method.n_bootstrap)
    heights, hz_draws = _heights(rows, method, rng)
    sigma_z = kinematics.sigma_z.value
    tracer, tracer_window = scaled_profile_hz(tracer_profile(model.phi_fit, sigma_z), method.k)
    predictions = Predictions(
        sheet=sheet_hz(sigma_z, model.sigma_surf),
        tracer=tracer,
        tracer_window=tracer_window,
        tracer_snapshot=scaled_profile_hz(tracer_profile(model.phi_snapshot, sigma_z), method.k)[0],
    )
    jr_median = kinematics.jr_median.value
    slopes = Slopes(
        measured=Estimate(jr_median / heights.measured.value, half_width(jr_draws / hz_draws)),
        sheet=jr_median / predictions.sheet,
        tracer=jr_median / predictions.tracer,
        tracer_snapshot=jr_median / predictions.tracer_snapshot,
    )
    return AgeBin(*bounds, float(np.median(rows.age)), kinematics, heights, predictions, slopes)


def fit_power_law(x: FloatArray, y: FloatArray, y_err: FloatArray) -> Estimate:
    """Return the exponent p of y = A x^p and its 1-sigma error, by weighted least squares of ln y on ln x."""
    lx, ly, w = np.log(x), np.log(y), (y / y_err) ** 2
    design = np.vstack([np.ones_like(lx), lx]).T
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    coef = cov @ (design.T @ (w * ly))
    return Estimate(float(coef[1]), float(np.sqrt(cov[1, 1])))


def range_exponents(bins: list[AgeBin], lo: float, hi: float) -> AgeRange:
    """Return the age exponents of the bins that lie inside [*lo*, *hi*] Gyr."""
    chosen = [b for b in bins if b.age_lo >= lo and b.age_hi <= hi]
    age = np.array([b.age_median for b in chosen])

    def column(get: Callable[[AgeBin], float]) -> FloatArray:
        return np.array([get(b) for b in chosen], dtype=np.float64)

    def fit(get: Callable[[AgeBin], Estimate]) -> Estimate:
        return fit_power_law(age, column(lambda b: get(b).value), column(lambda b: get(b).err))

    rel_jr = column(lambda b: b.kinematics.jr_median.err / b.kinematics.jr_median.value)
    rel_sz = column(lambda b: b.kinematics.sigma_z.err / b.kinematics.sigma_z.value)
    sheet, tracer = column(lambda b: b.slopes.sheet), column(lambda b: b.slopes.tracer)
    snapshot = column(lambda b: b.slopes.tracer_snapshot)
    exponents = {
        "sigma_r": fit(lambda b: b.kinematics.sigma_r),
        "sigma_z": fit(lambda b: b.kinematics.sigma_z),
        "jr_median": fit(lambda b: b.kinematics.jr_median),
        "hz": fit(lambda b: b.heights.measured),
        "a": fit(lambda b: b.slopes.measured),
        "a_sheet": fit_power_law(age, sheet, np.hypot(rel_jr, 2.0 * rel_sz) * sheet),
        "a_tracer": fit_power_law(age, tracer, np.hypot(rel_jr, rel_sz) * tracer),
        "a_tracer_snapshot": fit_power_law(age, snapshot, np.hypot(rel_jr, rel_sz) * snapshot),
    }
    return AgeRange(
        lo=lo,
        hi=hi,
        exponents=exponents,
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
