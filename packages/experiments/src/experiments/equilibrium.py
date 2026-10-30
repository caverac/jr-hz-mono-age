"""Mean vertical motions and asymmetries of each mono-age population: a test of the steady state.

The isothermal tracer and the vertical Jeans equation of :mod:`experiments.jeans` assume a
steady, axisymmetric disk that is symmetric about its plane. A disk that a satellite or a
bar has disturbed departs from all three, and can be thicker or thinner than its
kinematics imply. For every population of ``AGE_BINS``, this module measures four
departures in the stars of the annulus inside the self-consistent window |z| < L of their
h_Z, all weighted by initial mass:

- the thickening rate <sign(z) v_z> / <|z|>, which is d ln <|z|> / dt for stars that stay
  in the annulus;
- the bending velocity <v_z>;
- the north-south asymmetry of mass (M_+ - M_-) / (M_+ + M_-);
- the warp, the amplitude of the m = 1 term of the mean z in ``SECTORS`` equal sectors
  of azimuth.

All four vanish in a steady disk that is axisymmetric and symmetric about its plane. A
single snapshot gives rates, not amplitudes: a population caught at the turning point of
an oscillation has no mean motion, so a thickening rate of 0 does not exclude an
oscillating h_Z.

Errors are half the 16th-84th percentile interval of bootstrap resamplings of the star
particles of the population. The warp, a modulus, is positive even without one; its noise
floor is about its error. Averaging over a sector lowers a pure m = 1 term by
sin(pi / 8) / (pi / 8) = 0.97.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from experiments.auriga import HaloSnapshot
from experiments.heights import scaled_particle_hz
from experiments.populations import (
    AGE_BINS,
    K_MAIN,
    N_BOOTSTRAP,
    R_MAX_KPC,
    R_MIN_KPC,
    SEED,
    Z_SAMPLE_MAX_KPC,
    Estimate,
)
from experiments.snapshot import Galaxy, cylindrical_velocities

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]

SECTORS = 8
KMS_PER_KPC_IN_GYR = 1.0227  # 1 km/s/kpc in 1/Gyr


@dataclass(frozen=True)
class MotionSample:
    """Star particles of the annulus: age (Gyr), azimuth (rad), z (kpc), v_z (km/s) and initial mass (Msun)."""

    age: FloatArray
    azimuth: FloatArray
    z: FloatArray
    v_z: FloatArray
    initial_mass: FloatArray

    def select(self, keep: BoolArray | IndexArray) -> MotionSample:
        """Return the stars where the mask *keep* is true, or at the indices *keep*, which may repeat."""
        return MotionSample(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})


@dataclass(frozen=True)
class Motions:
    """Mean motions inside the window: thickening rate (1/Gyr), bending (km/s), asymmetry, warp (kpc)."""

    thickening: Estimate
    bending: Estimate
    asymmetry: Estimate
    warp: Estimate


@dataclass(frozen=True)
class EquilibriumBin:
    """One age bin: its stars in the annulus, h_Z (kpc) and its window, and its mean motions."""

    age_lo: float
    age_hi: float
    n_stars: int
    hz: Estimate
    window: float
    motions: Motions


@dataclass(frozen=True)
class EquilibriumRecord:
    """The mean motions of every population of one snapshot, as ``experiments auriga equilibrium`` writes them."""

    snapshot: HaloSnapshot
    bins: list[EquilibriumBin]


def motion_sample(galaxy: Galaxy) -> MotionSample:
    """Return the star particles with ``R_MIN_KPC`` <= R < ``R_MAX_KPC`` and |z| < ``Z_SAMPLE_MAX_KPC``."""
    stars = galaxy.stars
    cyl_r = stars.cylindrical_radius
    keep = (cyl_r >= R_MIN_KPC) & (cyl_r < R_MAX_KPC) & (np.abs(stars.pos[:, 2]) < Z_SAMPLE_MAX_KPC)
    pos = stars.pos[keep]
    _, v_z = cylindrical_velocities(pos, stars.vel[keep])
    age = galaxy.header.cosmology.lookback_gyr(stars.formation_a[keep], galaxy.header.scale_factor)
    return MotionSample(age, np.arctan2(pos[:, 1], pos[:, 0]), pos[:, 2], v_z, stars.initial_mass[keep])


def motions(rows: MotionSample, window: float) -> tuple[float, float, float, float]:
    """Return the thickening rate, bending, asymmetry and warp of the stars *rows* with |z| < *window*; NaN if none."""
    inside = np.abs(rows.z) < window
    if not np.any(inside):
        return (float("nan"),) * 4
    z, v_z, weight = rows.z[inside], rows.v_z[inside], rows.initial_mass[inside]
    thickening = np.average(np.sign(z) * v_z, weights=weight) / np.average(np.abs(z), weights=weight)
    north = float(np.sum(weight[z > 0.0]))
    south = float(np.sum(weight[z < 0.0]))
    sector = np.minimum(((rows.azimuth[inside] + np.pi) / (2.0 * np.pi) * SECTORS).astype(np.int64), SECTORS - 1)
    mean_z = np.bincount(sector, weights=weight * z, minlength=SECTORS) / np.bincount(
        sector, weights=weight, minlength=SECTORS
    )
    center = -np.pi + (np.arange(SECTORS) + 0.5) * 2.0 * np.pi / SECTORS
    warp = 2.0 / SECTORS * np.hypot(np.sum(mean_z * np.cos(center)), np.sum(mean_z * np.sin(center)))
    return (
        float(thickening) * KMS_PER_KPC_IN_GYR,
        float(np.average(v_z, weights=weight)),
        (north - south) / (north + south),
        float(warp),
    )


def _values(rows: MotionSample, k: float) -> tuple[float, float, tuple[float, float, float, float]]:
    """Return h_Z, its window and the motions of the stars *rows* of one bin."""
    hz, window = scaled_particle_hz(np.abs(rows.z), rows.initial_mass, k)
    return hz, window, motions(rows, window)


def measure_equilibrium_bin(
    rows: MotionSample, bounds: tuple[float, float], n_bootstrap: int, rng: np.random.Generator, k: float = K_MAIN
) -> EquilibriumBin:
    """Return the mean motions of the stars *rows* of the age bin *bounds* (Gyr), inside their window."""
    hz, window, point = _values(rows, k)
    n = rows.z.size
    draws = [_values(rows.select(rng.integers(0, n, n)), k) for _ in range(n_bootstrap)]

    def spread(value: float, samples: list[float]) -> Estimate:
        return Estimate.of(value, np.array(samples, dtype=np.float64))

    moved = [spread(point[j], [d[2][j] for d in draws]) for j in range(len(point))]
    return EquilibriumBin(bounds[0], bounds[1], n, spread(hz, [d[0] for d in draws]), window, Motions(*moved))


def measure_equilibrium(galaxy: Galaxy, n_bootstrap: int = N_BOOTSTRAP) -> list[EquilibriumBin]:
    """Return the mean motions of every age bin of the annulus of *galaxy*."""
    sample = motion_sample(galaxy)
    rng = np.random.default_rng(SEED)
    return [
        measure_equilibrium_bin(sample.select((sample.age >= lo) & (sample.age < hi)), (lo, hi), n_bootstrap, rng)
        for lo, hi in AGE_BINS
    ]
