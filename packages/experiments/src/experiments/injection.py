"""The injection test of the scale-height estimators: profiles of known shape through the windows.

A fixed window measures h_Z of a population thicker than itself from the part of the
profile it contains. Whether that biases the result depends on the shape of the profile,
which this test isolates. Heights are drawn from three families of vertical profiles:

- ``exponential``: rho ~ exp(-|z| / h), for which the truncated maximum-likelihood scale
  is consistent in any window (Deemer & Votaw 1955);
- ``sech2``: rho ~ sech^2(|z| / z_0), flat at the plane and exponential far from it, like
  an isothermal layer;
- ``mixture``: 70 percent of the heights from an exponential of scale h and 30 percent
  from one of scale 3 h, cusped at the plane.

For every family and scale, ``DRAWS`` draws of ``N_STARS`` heights, about the size of one
age bin of the annulus, are fitted in the fixed window ``Z_LIM_KPC`` and in the
self-consistent window at k = ``K_MAIN``. The reference of a profile is its
self-consistent scale computed on the height grid without noise. Every result is a
ratio to that reference, summarized over the draws by its median and its 16th and 84th
percentiles, with the fraction of draws whose fit has no solution. Because the profiles
and the estimators scale with length, the ratio depends only on reference / Z_lim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.heights import Z_GRID, particle_hz, scaled_particle_hz, scaled_profile_hz
from experiments.populations import K_MAIN

FloatArray = NDArray[np.float64]

N_STARS = 5_000
DRAWS = 40
Z_LIM_KPC = 1.0
MIXTURE_FRACTION = 0.3
MIXTURE_RATIO = 3.0
SEED = 20261027
SCALES_KPC = {
    "exponential": (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0),
    "sech2": (0.2, 0.4, 0.6, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0),
    "mixture": (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3),
}


@dataclass(frozen=True)
class Spread:
    """The median and the 16th and 84th percentiles of the finite values of a ratio; NaN if none is finite."""

    median: float
    lo: float
    hi: float


@dataclass(frozen=True)
class InjectionPoint:
    """The ratios to the reference of one profile, in the fixed and the self-consistent window."""

    family: str
    scale: float  # kpc, the parameter of the family
    reference: float  # kpc, the self-consistent scale of the exact profile
    fixed: Spread
    fixed_failed: float  # the fraction of draws whose fixed-window fit has no solution
    scaled: Spread


@dataclass(frozen=True)
class Injection:
    """The injection test: its settings and one point per family and scale."""

    n_stars: int
    draws: int
    z_lim: float
    k: float
    points: list[InjectionPoint]


def profile(family: str, scale: float, z_grid: FloatArray = Z_GRID) -> FloatArray:
    """Return the density of *family* with parameter *scale* (kpc) on *z_grid*, up to a constant."""
    if family == "exponential":
        return np.asarray(np.exp(-z_grid / scale), dtype=np.float64)
    if family == "sech2":
        return np.asarray(np.cosh(z_grid / scale) ** -2, dtype=np.float64)
    if family == "mixture":
        thick = MIXTURE_RATIO * scale
        return np.asarray(
            (1.0 - MIXTURE_FRACTION) / scale * np.exp(-z_grid / scale)
            + MIXTURE_FRACTION / thick * np.exp(-z_grid / thick),
            dtype=np.float64,
        )
    raise ValueError(f"unknown profile family {family!r}")


def sample_heights(family: str, scale: float, n: int, rng: np.random.Generator) -> FloatArray:
    """Return *n* heights |z| (kpc) drawn from *family* with parameter *scale*."""
    if family == "exponential":
        return rng.exponential(scale, n)
    if family == "sech2":
        # The cumulative distribution of sech^2(z / z_0) on z >= 0 is tanh(z / z_0).
        return np.asarray(scale * np.arctanh(rng.uniform(0.0, 1.0, n)), dtype=np.float64)
    if family == "mixture":
        thick = rng.uniform(size=n) < MIXTURE_FRACTION
        return np.where(thick, rng.exponential(MIXTURE_RATIO * scale, n), rng.exponential(scale, n))
    raise ValueError(f"unknown profile family {family!r}")


def spread(values: FloatArray) -> Spread:
    """Return the median and the 16th and 84th percentiles of the finite *values*."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Spread(float("nan"), float("nan"), float("nan"))
    lo, median, hi = np.percentile(finite, [16.0, 50.0, 84.0])
    return Spread(float(median), float(lo), float(hi))


def inject(family: str, scale: float, rng: np.random.Generator, n_stars: int, draws: int) -> InjectionPoint:
    """Return the ratios to the reference of *draws* samples of *n_stars* heights of one profile."""
    reference = scaled_profile_hz(profile(family, scale), K_MAIN)[0]
    fixed, scaled = np.empty(draws), np.empty(draws)
    for i in range(draws):
        absz = sample_heights(family, scale, n_stars, rng)
        weight = np.ones(n_stars)
        fixed[i] = particle_hz(absz, weight, Z_LIM_KPC) / reference
        scaled[i] = scaled_particle_hz(absz, weight, K_MAIN)[0] / reference
    return InjectionPoint(
        family=family,
        scale=scale,
        reference=reference,
        fixed=spread(fixed),
        fixed_failed=float(np.mean(~np.isfinite(fixed))),
        scaled=spread(scaled),
    )


def run_injection(n_stars: int = N_STARS, draws: int = DRAWS) -> Injection:
    """Return the injection test of every family and scale of ``SCALES_KPC``."""
    rng = np.random.default_rng(SEED)
    points = [inject(family, scale, rng, n_stars, draws) for family, scales in SCALES_KPC.items() for scale in scales]
    return Injection(n_stars, draws, Z_LIM_KPC, K_MAIN, points)
