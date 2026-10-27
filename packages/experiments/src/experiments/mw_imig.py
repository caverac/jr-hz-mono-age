"""The isothermal tracer against the window-free scale heights of Imig et al. (2025).

Imig et al. (2025, ApJ 990, 203) fit every population of APOGEE DR17 in bins of
[alpha/M], [M/H] and age with an exponential vertical profile over the whole survey
footprint, correcting for its selection function, so their heights carry no window of
|z|. Their h_Z is the scale height at the solar radius, and their ages come from the
distmass catalog, not from astroNN.

For each of their age bins up to ``AGE_MAX_GYR``, this module pools the low-[alpha/M]
populations with [M/H] > ``M_H_MIN`` into the profile

    rho(z) = sum_i Sigma_i / (2 h_i) exp(-|z| / h_i),

weighted by their surface mass densities at the solar radius, and measures its h_Z in the
self-consistent window, as for every other population of this analysis. Its errors come
from ``N_DRAWS`` draws of every h_i and Sigma_i from split normal distributions with their
published lower and upper errors. The tracer of the bin uses the sigma_z of the APOGEE
kinematic stars with astroNN ages in the same range, so the comparison mixes two age
scales; a systematic difference between them shifts the populations in age.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from experiments.heights import Z_GRID, scaled_profile_hz, tracer_profile
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import KinematicStars
from experiments.populations import Z_MID_MAX_KPC, Estimate, Trend, fit_trend
from experiments.stats import deconvolved_dispersion, half_width

FloatArray = NDArray[np.float64]

M_H_MIN = -0.6
AGE_MAX_GYR = 10.0
N_DRAWS = 300


@dataclass(frozen=True)
class ImigBin:  # pylint: disable=too-many-instance-attributes
    """One age bin of Imig et al.: the pooled h_Z (kpc), the sigma_z of APOGEE, the tracer and their ratio."""

    age_lo: float
    age_hi: float
    n_populations: int
    measured: Estimate
    n_kin: int
    sigma_z: Estimate
    tracer: float
    ratio: Estimate


@dataclass(frozen=True)
class ImigCheck:
    """The tracer against the pooled heights of Imig et al., bin by bin, and the trend of their ratio with age."""

    bins: list[ImigBin]
    trend: Trend


@dataclass(frozen=True)
class ImigPopulations:
    """The populations of one age bin: h_Z and Sigma at the solar radius, with lower and upper errors."""

    age_lo: float
    age_hi: float
    h: FloatArray
    h_err: FloatArray  # (n, 2)
    sigma: FloatArray
    sigma_err: FloatArray  # (n, 2)


def read_imig(path: Path) -> list[ImigPopulations]:
    """Return the low-[alpha/M] populations with [M/H] > ``M_H_MIN``, by age bin up to ``AGE_MAX_GYR``."""
    table = fits.getdata(path, ext=1)
    group = np.array([str(value).strip() for value in table["ALPHA_M"]])
    keep = (
        (group == "LOW") & (table["M_H"] > M_H_MIN) & (table["MAX_AGE"] <= AGE_MAX_GYR + 0.01)
        & np.isfinite(table["H_Z"]) & np.isfinite(table["SMD_0"]) & (table["SMD_0"] > 0.0)
    )  # fmt: skip
    cols = {
        k: np.asarray(table[k][keep], dtype=np.float64)
        for k in ("MIN_AGE", "MAX_AGE", "H_Z", "H_Z_ERR", "SMD_0", "SMD_0_ERRS")
    }
    bins = []
    for lo in np.unique(cols["MIN_AGE"]):
        rows = cols["MIN_AGE"] == lo
        bins.append(
            ImigPopulations(
                float(lo), float(cols["MAX_AGE"][rows][0]),
                cols["H_Z"][rows], cols["H_Z_ERR"][rows], cols["SMD_0"][rows], cols["SMD_0_ERRS"][rows],
            )  # fmt: skip
        )
    return bins


def pooled_hz(h: FloatArray, sigma: FloatArray, k: float) -> float:
    """Return h_Z (kpc) of the sum of exponentials of scales *h* and surface densities *sigma*, in the window k h_Z."""
    rho = np.sum((sigma / (2.0 * h))[:, None] * np.exp(-Z_GRID[None, :] / h[:, None]), axis=0)
    return scaled_profile_hz(np.asarray(rho, dtype=np.float64), k)[0]


def split_normal(rng: np.random.Generator, value: FloatArray, err: FloatArray) -> FloatArray:
    """Return one draw per value from a split normal with lower and upper errors ``err[:, 0]`` and ``err[:, 1]``."""
    z = rng.standard_normal(value.size)
    return np.asarray(np.maximum(value + np.where(z < 0.0, err[:, 0], err[:, 1]) * z, 1e-3 * value), dtype=np.float64)


def _tracer(sigma_z: float, model: SolarModel) -> float:
    """Return h_Z (kpc) of the isothermal tracer of *sigma_z* in *model*, in the window k h_Z."""
    return scaled_profile_hz(tracer_profile(model.phi, sigma_z), model.k)[0]


def measure_imig_bin(
    pops: ImigPopulations, kin: KinematicStars, model: SolarModel, rng: np.random.Generator, n_draws: int
) -> ImigBin:
    """Return the pooled h_Z of *pops*, the tracer of the APOGEE stars *kin* of the same ages, and their ratio.

    Each draw takes the heights and densities of the populations from their split normal
    distributions and resamples the kinematic stars.
    """
    in_age = (kin.age >= pops.age_lo) & (kin.age < pops.age_hi) & (np.abs(kin.phase.z) < Z_MID_MAX_KPC)
    v_z, v_z_err = kin.phase.v_z[in_age], kin.v_z_err[in_age]
    hz = pooled_hz(pops.h, pops.sigma, model.k)
    sigma_z = deconvolved_dispersion(v_z, v_z_err)
    draws = []
    for _ in range(n_draws):
        rows = rng.integers(0, v_z.size, v_z.size)
        h = pooled_hz(split_normal(rng, pops.h, pops.h_err), split_normal(rng, pops.sigma, pops.sigma_err), model.k)
        draws.append((h, deconvolved_dispersion(v_z[rows], v_z_err[rows])))
    h_draws, sz_draws = (np.array(column) for column in zip(*draws, strict=True))
    return ImigBin(
        age_lo=pops.age_lo,
        age_hi=pops.age_hi,
        n_populations=int(pops.h.size),
        measured=Estimate(hz, half_width(h_draws)),
        n_kin=int(v_z.size),
        sigma_z=Estimate(sigma_z, half_width(sz_draws)),
        tracer=_tracer(sigma_z, model),
        ratio=Estimate(
            _tracer(sigma_z, model) / hz, half_width(np.array([_tracer(sz, model) for sz in sz_draws]) / h_draws)
        ),
    )


def measure_imig(
    populations: list[ImigPopulations],
    kin: KinematicStars,
    model: SolarModel,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
) -> ImigCheck:
    """Return the check of every age bin of *populations* and the trend of the ratio with the geometric mean age."""
    bins = [measure_imig_bin(pops, kin, model, rng, n_draws) for pops in populations]
    age = np.sqrt(np.array([b.age_lo * b.age_hi for b in bins]))
    ratio = np.array([b.ratio.value for b in bins])
    err = np.array([b.ratio.err for b in bins])
    return ImigCheck(bins, fit_trend(age, ratio, err))
