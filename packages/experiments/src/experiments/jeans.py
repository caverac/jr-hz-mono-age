"""The vertical Jeans equation of one population, with its measured sigma_z(z) and tilt term.

The isothermal tracer, rho ~ exp(-Delta Phi / sigma_z^2), solves the vertical Jeans
equation of an axisymmetric population in a steady state,

    (1 / R) d(R nu sigma_Rz) / dR + d(nu sigma_z^2) / dz = -nu dPhi / dz,

under two assumptions: sigma_z does not change with height, and the tilt term, the
first one, vanishes. This module measures both from the star particles of one
population in an annulus, in slices of |z| (``JEANS_EDGES_KPC``), and integrates the
equation with them. Writing the tilt term as nu t(z),

    d ln(nu sigma_z^2) / dz = -(dPhi / dz + t) / sigma_z^2,

so that nu(z) = sigma_z^2(0) / sigma_z^2(z) exp(-int_0^z (dPhi / dz' + t) / sigma_z^2 dz').
With t = 0 and a constant sigma_z this is the isothermal tracer. The profile is a
diagnostic of the tracer's two assumptions, because it uses kinematics measured at every
height rather than predicting them from the midplane.

t(z) comes from ``JEANS_RADIAL_BINS`` equal radial bins of the annulus: the density nu
(initial mass per volume) and sigma_Rz = <v_R v_z sign(z)> of each, the slope of
R nu sigma_Rz against R by least squares, divided by R nu at the median radius. The
sign of z folds the two sides of the plane, on which sigma_Rz has opposite signs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.stats import clipped_dispersion

FloatArray = NDArray[np.float64]

JEANS_EDGES_KPC = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0)
JEANS_MIN_PER_SLICE = 100  # stars in a |z| slice, and in each of its radial bins for the tilt
JEANS_RADIAL_BINS = 3


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


def _ring(stars: JeansStars, ring: NDArray[np.bool_], volume: float) -> tuple[float, float, float]:
    """Return the mean R, the density nu and R nu sigma_Rz of the stars *ring* of one radial bin of a slice."""
    radius = float(np.mean(stars.r[ring]))
    nu = float(np.sum(stars.weight[ring])) / volume
    sigma_rz = float(np.mean(stars.v_r[ring] * stars.v_z[ring] * np.sign(stars.z[ring])))
    return radius, nu, radius * nu * sigma_rz


def _tilt(stars: JeansStars, inside: NDArray[np.bool_], r_edges: FloatArray, r_mid: float, dz: float) -> float:
    """Return t = (1 / (R nu)) d(R nu sigma_Rz) / dR at *r_mid* for the stars *inside* one slice; NaN if sparse."""
    rings = []
    for lo, hi in zip(r_edges[:-1], r_edges[1:], strict=True):
        ring = inside & (stars.r >= lo) & (stars.r < hi)
        if np.count_nonzero(ring) < JEANS_MIN_PER_SLICE:
            return float("nan")
        rings.append(_ring(stars, ring, np.pi * (hi**2 - lo**2) * 2.0 * dz))
    radius, density, flux = (np.array(column) for column in zip(*rings, strict=True))
    nu_mid = float(np.polyval(np.polyfit(radius, density, 1), r_mid))
    return float(np.polyfit(radius, flux, 1)[0]) / (r_mid * nu_mid)


def jeans_profile(stars: JeansStars, r_lo: float, r_hi: float, r_mid: float) -> JeansProfile:
    """Return sigma_z^2 and t of *stars* in the slices of ``JEANS_EDGES_KPC`` up to the first sparse one.

    sigma_z^2 is the variance of v_z in the slice, clipped as the midplane sigma_z is
    (:func:`experiments.stats.clipped_dispersion`); t uses the radial bins of the
    annulus [*r_lo*, *r_hi*) and is NaN where a radial bin is sparse.
    """
    absz = np.abs(stars.z)
    r_edges = np.linspace(r_lo, r_hi, JEANS_RADIAL_BINS + 1)
    z_mid, sigma2, tilt = [], [], []
    for lo, hi in zip(JEANS_EDGES_KPC[:-1], JEANS_EDGES_KPC[1:], strict=True):
        inside = (absz >= lo) & (absz < hi)
        if np.count_nonzero(inside) < JEANS_MIN_PER_SLICE:
            break
        z_mid.append(float(np.median(absz[inside])))
        sigma2.append(clipped_dispersion(stars.v_z[inside]) ** 2)
        tilt.append(_tilt(stars, inside, r_edges, r_mid, hi - lo))
    return JeansProfile(tuple(z_mid), tuple(sigma2), tuple(tilt))


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
