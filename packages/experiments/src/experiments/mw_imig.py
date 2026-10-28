"""The window-free scale heights of Imig et al. (2025) against our star counts and the isothermal tracer.

Imig et al. (2025, ApJ 990, 203) fit every population of APOGEE DR17 in bins of
[alpha/M], [M/H] and age with an exponential vertical profile over the whole survey
footprint, correcting for its selection function, so their heights carry no window of
|z|. Their h_Z is the scale height at the solar radius R_sun = ``R_SUN_IMIG_KPC``, which
flares linearly with radius, their density is a number density of red giants at R_sun,
and their ages come from the distmass catalog.

For each of their age bins up to ``AGE_MAX_GYR``, this module measures four h_Z, all in
the self-consistent window, so that each difference between their heights and our star
counts can be taken out in turn:

- ``imig_mass``: their low-[alpha/M] populations with [M/H] > ``M_H_MIN`` pooled by their
  surface mass densities, rho(z) = sum_i Sigma_i / (2 h_i) exp(-|z| / h_i);
- ``imig_counts``: the same populations pooled by their number densities of red giants,
  n(z) = sum_i nu_i exp(-|z| / h_i), with each h_i moved to R_0 by its flare,
  h_i(R_0) = h_i + A_flare,i (R_0 - R_sun): the analog of our counts of giants at R_0;
- ``counts_astronn`` and ``counts_distmass``: our counts of the Gaia giants split into ages
  by APOGEE (:mod:`experiments.mw_counts`), in the same range of ages, with the ages of
  astroNN and with those of distmass.

The tracer of each bin uses the sigma_z of the APOGEE kinematic stars of the same ages on
the distmass scale, theirs, so that its comparison with their heights holds the age scale
fixed. The errors come from ``N_DRAWS`` draws in which every h_i, Sigma_i and nu_i comes
from a split normal with its published lower and upper errors, and the counts and the
kinematic stars are resampled; the flare parameters keep their best values, whose effect
on h_i at R_0 is at most 0.0175 kpc.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from experiments.heights import Z_GRID, scaled_profile_hz, tracer_profile
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import KinematicStars
from experiments.mw_counts import CountSample, count_hz
from experiments.mw_dynamics import R0_KPC
from experiments.populations import Z_MID_MAX_KPC, Estimate, Trend, fit_trend
from experiments.stats import deconvolved_dispersion, half_width

FloatArray = NDArray[np.float64]

M_H_MIN = -0.6
AGE_MAX_GYR = 10.0
R_SUN_IMIG_KPC = 8.3
N_DRAWS = 300
COLUMNS = ("MIN_AGE", "MAX_AGE", "H_Z", "H_Z_ERR", "SMD_0", "SMD_0_ERRS", "NU_0", "NU_0_ERRS", "A_FLARE")


@dataclass(frozen=True)
class ImigHeights:
    """The four h_Z (kpc) of one age bin, or a ratio per height: theirs by mass and by counts, ours per age scale."""

    imig_mass: Estimate
    imig_counts: Estimate
    counts_astronn: Estimate
    counts_distmass: Estimate


@dataclass(frozen=True)
class ImigKinematics:
    """The APOGEE kinematic stars of one age bin on the distmass scale: their count, sigma_z and the tracer h_Z."""

    n_kin: int
    sigma_z: Estimate
    tracer: float


@dataclass(frozen=True)
class ImigBin:
    """One age bin of Imig et al.: its heights, the kinematics of its ages, and the tracer over each height."""

    age_lo: float
    age_hi: float
    n_populations: int
    heights: ImigHeights
    kinematics: ImigKinematics
    ratios: ImigHeights


@dataclass(frozen=True)
class ImigCheck:
    """The heights of Imig et al. against the counts and the tracer, bin by bin, and the trend of each ratio."""

    bins: list[ImigBin]
    trends: dict[str, Trend]


@dataclass(frozen=True)
class ImigPopulations:  # pylint: disable=too-many-instance-attributes
    """The populations of one age bin: h_Z, Sigma and nu at R_sun with lower and upper errors, and the flare."""

    age_lo: float
    age_hi: float
    h: FloatArray
    h_err: FloatArray  # (n, 2)
    sigma: FloatArray
    sigma_err: FloatArray  # (n, 2)
    nu: FloatArray
    nu_err: FloatArray  # (n, 2)
    a_flare: FloatArray


@dataclass(frozen=True)
class ImigInputs:
    """What the comparison needs besides their table: our APOGEE counts on both age scales, and the kinematics."""

    counts_astronn: CountSample
    counts_distmass: CountSample
    kinematics: KinematicStars  # on the distmass scale


def read_imig(path: Path) -> list[ImigPopulations]:
    """Return the low-[alpha/M] populations with [M/H] > ``M_H_MIN``, by age bin up to ``AGE_MAX_GYR``."""
    table = fits.getdata(path, ext=1)
    group = np.array([str(value).strip() for value in table["ALPHA_M"]])
    keep = (
        (group == "LOW") & (table["M_H"] > M_H_MIN) & (table["MAX_AGE"] <= AGE_MAX_GYR + 0.01)
        & np.isfinite(table["H_Z"]) & np.isfinite(table["SMD_0"]) & (table["SMD_0"] > 0.0)
        & np.isfinite(table["NU_0"]) & (table["NU_0"] > 0.0)
    )  # fmt: skip
    cols = {k: np.asarray(table[k][keep], dtype=np.float64) for k in COLUMNS}
    bins = []
    for lo in np.unique(cols["MIN_AGE"]):
        rows = cols["MIN_AGE"] == lo
        bins.append(
            ImigPopulations(
                float(lo), float(cols["MAX_AGE"][rows][0]),
                *(cols[k][rows] for k in COLUMNS[2:]),
            )  # fmt: skip
        )
    return bins


def pooled_hz(h: FloatArray, midplane: FloatArray, k: float) -> float:
    """Return h_Z (kpc) of the sum of exponentials of scales *h* and midplane densities *midplane*, in k h_Z."""
    rho = np.sum(midplane[:, None] * np.exp(-Z_GRID[None, :] / h[:, None]), axis=0)
    return scaled_profile_hz(np.asarray(rho, dtype=np.float64), k)[0]


def mass_pooled_hz(h: FloatArray, sigma: FloatArray, k: float) -> float:
    """Return h_Z of the populations of scales *h* and surface mass densities *sigma*, pooled by mass."""
    return pooled_hz(h, sigma / (2.0 * h), k)


def count_pooled_hz(h: FloatArray, nu: FloatArray, a_flare: FloatArray, k: float) -> float:
    """Return h_Z of the populations of scales *h* at R_sun and densities of giants *nu*, pooled by counts at R_0."""
    return pooled_hz(h + a_flare * (R0_KPC - R_SUN_IMIG_KPC), nu, k)


def split_normal(rng: np.random.Generator, value: FloatArray, err: FloatArray) -> FloatArray:
    """Return one draw per value from a split normal with lower and upper errors ``err[:, 0]`` and ``err[:, 1]``."""
    z = rng.standard_normal(value.size)
    return np.asarray(np.maximum(value + np.where(z < 0.0, err[:, 0], err[:, 1]) * z, 1e-3 * value), dtype=np.float64)


def _tracer(sigma_z: float, model: SolarModel) -> float:
    """Return h_Z (kpc) of the isothermal tracer of *sigma_z* in *model*, in the window k h_Z."""
    return scaled_profile_hz(tracer_profile(model.phi, sigma_z), model.k)[0]


@dataclass(frozen=True)
class _Values:
    """What one draw of a bin gives: the four heights and sigma_z."""

    heights: tuple[float, float, float, float]
    sigma_z: float


def _values(pops: ImigPopulations, inputs: ImigInputs, model: SolarModel, rng: np.random.Generator | None) -> _Values:
    """Return the heights and sigma_z of one bin, at the best values (*rng* None) or for one draw."""
    kin = inputs.kinematics
    in_age = (kin.age >= pops.age_lo) & (kin.age < pops.age_hi) & (np.abs(kin.phase.z) < Z_MID_MAX_KPC)
    v_z, v_z_err = kin.phase.v_z[in_age], kin.v_z_err[in_age]
    h, sigma, nu = pops.h, pops.sigma, pops.nu
    astronn, distmass = inputs.counts_astronn, inputs.counts_distmass
    if rng is not None:
        h, sigma, nu = (
            split_normal(rng, v, e) for v, e in ((h, pops.h_err), (sigma, pops.sigma_err), (nu, pops.nu_err))
        )
        rows = rng.integers(0, v_z.size, v_z.size)
        v_z, v_z_err = v_z[rows], v_z_err[rows]
        astronn, distmass = astronn.resampled(rng), distmass.resampled(rng)
    heights = (
        mass_pooled_hz(h, sigma, model.k),
        count_pooled_hz(h, nu, pops.a_flare, model.k),
        count_hz(astronn, pops.age_lo, pops.age_hi, model.k)[0],
        count_hz(distmass, pops.age_lo, pops.age_hi, model.k)[0],
    )
    return _Values(heights, deconvolved_dispersion(v_z, v_z_err))


def measure_imig_bin(
    pops: ImigPopulations, inputs: ImigInputs, model: SolarModel, rng: np.random.Generator, n_draws: int
) -> ImigBin:
    """Return the four heights of one age bin, the tracer of the kinematic stars of its ages, and their ratios."""
    point = _values(pops, inputs, model, None)
    draws = [_values(pops, inputs, model, rng) for _ in range(n_draws)]
    tracers = np.array([_tracer(d.sigma_z, model) for d in draws])
    tracer = _tracer(point.sigma_z, model)
    n_heights = len(fields(ImigHeights))

    kin = inputs.kinematics
    in_age = (kin.age >= pops.age_lo) & (kin.age < pops.age_hi) & (np.abs(kin.phase.z) < Z_MID_MAX_KPC)
    return ImigBin(
        age_lo=pops.age_lo,
        age_hi=pops.age_hi,
        n_populations=int(pops.h.size),
        heights=ImigHeights(
            *(Estimate(point.heights[j], half_width(np.array([d.heights[j] for d in draws]))) for j in range(n_heights))
        ),
        kinematics=ImigKinematics(
            int(in_age.sum()), Estimate(point.sigma_z, half_width(np.array([d.sigma_z for d in draws]))), tracer
        ),
        ratios=ImigHeights(
            *(
                Estimate(tracer / point.heights[j], half_width(tracers / np.array([d.heights[j] for d in draws])))
                for j in range(n_heights)
            )
        ),
    )


def measure_imig(
    populations: list[ImigPopulations],
    inputs: ImigInputs,
    model: SolarModel,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
) -> ImigCheck:
    """Return the comparison in every age bin of *populations*, and the trend of each ratio with the mean age.

    A trend uses the bins whose ratio is finite; the age of a bin is the geometric mean of
    its edges.
    """
    bins = [measure_imig_bin(pops, inputs, model, rng, n_draws) for pops in populations]
    age = np.sqrt(np.array([b.age_lo * b.age_hi for b in bins]))
    trends = {}
    for f in fields(ImigHeights):
        ratio = np.array([getattr(b.ratios, f.name).value for b in bins])
        err = np.array([getattr(b.ratios, f.name).err for b in bins])
        ok = np.isfinite(ratio) & np.isfinite(err) & (err > 0.0)
        trends[f.name] = fit_trend(age[ok], ratio[ok], err[ok])
    return ImigCheck(bins, trends)
