"""The vertical Jeans equation of one population, with its measured sigma_z(z) and tilt term.

The isothermal tracer, rho ~ exp(-Delta Phi / sigma_z^2), solves the vertical Jeans
equation of an axisymmetric population in a steady state,

    (1 / R) d(R nu sigma_Rz) / dR + d(nu sigma_z^2) / dz = -nu dPhi / dz,

under two assumptions: sigma_z does not change with height, and the tilt term, the
first one, vanishes. This module measures both from the star particles of one
population, in slices of |z| (``JEANS_EDGES_KPC``), and integrates the equation with
them. Writing the tilt term as nu t(z),

    d ln(nu sigma_z^2) / dz = -(dPhi / dz + t) / sigma_z^2,

so that nu(z) = sigma_z^2(0) / sigma_z^2(z) exp(-int_0^z (dPhi / dz' + t) / sigma_z^2 dz').
With t = 0 and a constant sigma_z this is the isothermal tracer. The profile is a
diagnostic of the tracer's two assumptions, because it uses kinematics measured at every
height rather than predicting them from the midplane.

The slices reach |z| = 12 kpc, the height of the samples, because the self-consistent
window of the thickest populations reaches 8 kpc and the equation holds sigma_z^2 and t
constant beyond the last slice. sigma_z^2 comes from the stars of the annulus. t(z) comes
from a wider range of radii, ``TILT_R_MIN_KPC`` <= R < ``TILT_R_MAX_KPC``, centered on the
annulus, in rings of ``TILT_RING_KPC``: straight lines of ln nu and of
sigma_Rz = <v_R v_z sign(z)> against the mean R of the rings, with nu (initial mass per
area) and sigma_Rz weighted by initial mass, give

    t = (1 / (R nu)) d(R nu sigma_Rz) / dR = sigma_Rz (1 / R + d ln nu / dR) + d sigma_Rz / dR

at R_mid. The sign of z folds the two sides of the plane, on which sigma_Rz has opposite
signs. A slice needs ``TILT_MIN_PER_RING`` stars in every ring, so that the lines sample
R_mid symmetrically; otherwise its tilt is NaN, which the equation counts as 0.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]

JEANS_EDGES_KPC = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
JEANS_MIN_PER_SLICE = 100  # stars of the annulus in a |z| slice
TILT_R_MIN_KPC, TILT_R_MAX_KPC = 6.0, 10.5
TILT_RING_KPC = 0.5
TILT_MIN_PER_RING = 50


@dataclass(frozen=True)
class JeansProfile:
    """sigma_z^2 ((km/s)^2) and the tilt term t ((km/s)^2/kpc, NaN if unmeasured) at the median |z| of each slice."""

    z: tuple[float, ...]
    sigma2: tuple[float, ...]
    tilt: tuple[float, ...]


@dataclass(frozen=True)
class JeansStars:
    """The star particles of one population: cylindrical radius and z (kpc), v_R and v_z (km/s), initial mass."""

    r: FloatArray
    z: FloatArray
    v_r: FloatArray
    v_z: FloatArray
    weight: FloatArray

    def select(self, keep: BoolArray | IndexArray) -> JeansStars:
        """Return the stars where the mask *keep* is true, or at the indices *keep*."""
        return JeansStars(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})


def _variance(values: FloatArray, weight: FloatArray) -> float:
    """Return the variance of *values* about their mean, both weighted by *weight*."""
    mean = np.average(values, weights=weight)
    return float(np.average((values - mean) ** 2, weights=weight))


def slice_tilt(radius: FloatArray, nu: FloatArray, sigma_rz: FloatArray, r_mid: float) -> float:
    """Return t at *r_mid* from straight lines of ln *nu* and of *sigma_rz* against the ring radii *radius*."""
    dln_nu = float(np.polyfit(radius, np.log(nu), 1)[0])
    slope, intercept = np.polyfit(radius, sigma_rz, 1)
    return float((intercept + slope * r_mid) * (1.0 / r_mid + dln_nu) + slope)


def cell_sums(stars: JeansStars, z_edges: FloatArray, r_edges: FloatArray) -> FloatArray:
    """Return the count, mass, mass R and mass v_R v_z sign(z) of the stars in each cell, shape (4, slices, rings)."""
    n_z, n_r = z_edges.size - 1, r_edges.size - 1
    in_z = np.searchsorted(z_edges, np.abs(stars.z), side="right") - 1
    in_r = np.searchsorted(r_edges, stars.r, side="right") - 1
    inside = (in_z >= 0) & (in_z < n_z) & (in_r >= 0) & (in_r < n_r)
    cell = (in_z * n_r + in_r)[inside]
    weight = stars.weight[inside]
    moments = (
        np.ones(cell.size),
        weight,
        weight * stars.r[inside],
        weight * (stars.v_r * stars.v_z * np.sign(stars.z))[inside],
    )
    return np.stack([np.bincount(cell, weights=m, minlength=n_z * n_r).reshape(n_z, n_r) for m in moments])


def tilt_profile(stars: JeansStars, r_mid: float) -> FloatArray:
    """Return t ((km/s)^2/kpc) in each slice of ``JEANS_EDGES_KPC``, NaN where a ring holds too few stars."""
    r_edges = np.arange(TILT_R_MIN_KPC, TILT_R_MAX_KPC + 0.5 * TILT_RING_KPC, TILT_RING_KPC)
    count, mass, moment_r, moment_rz = cell_sums(stars, np.array(JEANS_EDGES_KPC), r_edges)
    area = np.pi * np.diff(r_edges**2)
    tilt = np.full(count.shape[0], np.nan)
    for j in range(count.shape[0]):
        if np.all(count[j] >= TILT_MIN_PER_RING):
            tilt[j] = slice_tilt(moment_r[j] / mass[j], mass[j] / area, moment_rz[j] / mass[j], r_mid)
    return tilt


def jeans_profile(stars: JeansStars, r_lo: float, r_hi: float, r_mid: float) -> JeansProfile:
    """Return sigma_z^2 and t of *stars* in the slices of ``JEANS_EDGES_KPC`` up to the first sparse one.

    *stars* cover the range of radii of the tilt term. sigma_z^2 is the variance of v_z of
    those in the annulus [*r_lo*, *r_hi*), weighted by initial mass as the density is, and
    not clipped: the Jeans equation needs the second moment <v_z^2> of all the stars of
    the population, and clipping lowers it for the heavy tails of a population that mixes
    several dispersions. A slice is sparse if it holds fewer than
    ``JEANS_MIN_PER_SLICE`` stars of the annulus.
    """
    annulus = stars.select((stars.r >= r_lo) & (stars.r < r_hi))
    absz = np.abs(annulus.z)
    tilt = tilt_profile(stars, r_mid)
    z_mid, sigma2 = [], []
    for lo, hi in zip(JEANS_EDGES_KPC[:-1], JEANS_EDGES_KPC[1:], strict=True):
        inside = (absz >= lo) & (absz < hi)
        if np.count_nonzero(inside) < JEANS_MIN_PER_SLICE:
            break
        z_mid.append(float(np.median(absz[inside])))
        sigma2.append(_variance(annulus.v_z[inside], annulus.weight[inside]))
    return JeansProfile(tuple(z_mid), tuple(sigma2), tuple(tilt[: len(z_mid)].tolist()))


def jeans_density(phi_z: FloatArray, profile: JeansProfile, z_grid: FloatArray, with_tilt: bool) -> FloatArray:
    """Return nu(z) on *z_grid* from the vertical Jeans equation in Delta Phi *phi_z*, with or without the tilt term.

    sigma_z^2 and t are interpolated linearly between the slices and held constant beyond
    the outer one; sigma_z^2 is also constant below the inner one, whereas t rises from 0 at
    z = 0, where it vanishes by symmetry. A NaN tilt counts as 0. Without slices the
    density is NaN everywhere.
    """
    if not profile.z:
        return np.full(z_grid.size, np.nan)
    z = np.array(profile.z)
    sigma2 = np.interp(z_grid, z, np.array(profile.sigma2))
    tilt = np.interp(z_grid, np.concatenate([[0.0], z]), np.concatenate([[0.0], np.nan_to_num(np.array(profile.tilt))]))
    integrand = (np.gradient(phi_z, z_grid) + (tilt if with_tilt else 0.0)) / sigma2
    steps = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z_grid)
    log_flux = -np.concatenate([[0.0], np.cumsum(steps)])
    return np.asarray(sigma2[0] / sigma2 * np.exp(log_flux), dtype=np.float64)
