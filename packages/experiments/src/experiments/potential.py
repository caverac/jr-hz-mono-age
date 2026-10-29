"""The axisymmetric potential of a galaxy, fitted to its particles with AGAMA, and the radial actions in it.

AGAMA (Vasiliev 2019, MNRAS 482, 1525) expands the potential of the particles in two
parts: a CylSpline for the gas and stars of the disk region, which resolves the thin
disk, and a Multipole for the dark matter and everything else. Both expansions reject a
massive particle at r = 0, where one particle always sits because ``SubhaloPos`` is the
position of the most bound particle, so particles exactly at the origin are left out.

The snapshot also stores the potential each star particle feels, with the softening and
the non-axisymmetric structure that the fit leaves out. Two comparisons use it: the
median Delta Phi of the fit and of the snapshot in |z| slices (:func:`potential_check`),
and the vertical part of their difference at the median radius of an annulus, averaged
over azimuth, which turns the fitted Delta Phi(z) into the snapshot's
(:func:`snapshot_vertical_potential`).
"""

from __future__ import annotations

from dataclasses import dataclass

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.heights import Z_GRID
from experiments.snapshot import Galaxy, Stars

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DISK_TYPES = (0, 4)  # gas and stars, the types that the CylSpline holds
CYL_R_MAX_KPC, CYL_Z_MAX_KPC = 50.0, 10.0
CYL_GRID = {"gridSizeR": 50, "gridSizeZ": 50, "rmin": 0.1, "rmax": CYL_R_MAX_KPC, "zmin": 0.02, "zmax": CYL_Z_MAX_KPC}
MULTIPOLE_GRID = {"gridSizeR": 50, "rmin": 0.1, "rmax": 300.0, "lmax": 8}
CHECK_EDGES_KPC = tuple(0.1 * k for k in range(13))
CHECK_MIN_PER_SLICE = 30
RESIDUAL_EDGES_KPC = np.concatenate([np.arange(0.0, 2.0, 0.1), np.arange(2.0, 6.01, 0.5)])
RESIDUAL_SECTORS = 8  # equal sectors of azimuth, averaged with equal weights
RESIDUAL_MIN_PER_SECTOR = 20


@dataclass(frozen=True)
class SliceCheck:
    """The median fitted and snapshot Delta Phi ((km/s)^2) of the stars in one |z| slice."""

    z: float
    snapshot: float
    fitted: float
    n: int


@dataclass(frozen=True)
class VerticalResidual:
    """The snapshot minus the fitted potential ((km/s)^2) at the median radius, per |z| slice."""

    z: tuple[float, ...]
    residual: tuple[float, ...]


def fit_potential(galaxy: Galaxy) -> agama.Potential:
    """Return the axisymmetric AGAMA potential of every particle of *galaxy*, in Msun, kpc and km/s."""
    agama.setUnits(mass=1, length=1, velocity=1)
    disk_pos, disk_mass, halo_pos, halo_mass = [], [], [], []
    for ptype, parts in galaxy.by_type().items():
        in_disk = (parts.cylindrical_radius < CYL_R_MAX_KPC) & (np.abs(parts.pos[:, 2]) < CYL_Z_MAX_KPC)
        if ptype not in DISK_TYPES:
            in_disk[:] = False
        off_origin = np.any(parts.pos != 0.0, axis=1)
        disk_pos.append(parts.pos[in_disk & off_origin])
        disk_mass.append(parts.mass[in_disk & off_origin])
        halo_pos.append(parts.pos[~in_disk & off_origin])
        halo_mass.append(parts.mass[~in_disk & off_origin])
    disk = agama.Potential(
        type="CylSpline",
        symmetry="a",
        mmax=0,
        particles=(np.concatenate(disk_pos), np.concatenate(disk_mass)),
        **CYL_GRID,
    )
    halo = agama.Potential(
        type="Multipole",
        symmetry="a",
        particles=(np.concatenate(halo_pos), np.concatenate(halo_mass)),
        **MULTIPOLE_GRID,
    )
    return agama.Potential(disk, halo)


def vertical_potential(pot: agama.Potential, radius: float, z_grid: FloatArray = Z_GRID) -> FloatArray:
    """Return Phi(R, z) - Phi(R, 0) ((km/s)^2) at R = *radius* on *z_grid*."""
    points = np.column_stack([np.full_like(z_grid, radius), np.zeros_like(z_grid), z_grid])
    phi = np.asarray(pot.potential(points), dtype=np.float64)
    return np.asarray(phi - phi[0], dtype=np.float64)


def circular_velocity(pot: agama.Potential, radius: float) -> float:
    """Return the circular velocity (km/s) in the plane at R = *radius*."""
    return float(np.sqrt(-radius * pot.force(np.array([[radius, 0.0, 0.0]]))[0, 0]))


def circularity(pot: agama.Potential, pos: FloatArray, vel: FloatArray) -> FloatArray:
    """Return L_z / L_c(E) of particles in the disk frame; NaN for an unbound one.

    L_c(E) is the angular momentum of the circular orbit in the plane with the particle's
    energy, so a disk star on a circular orbit has 1 and a star without net rotation 0.
    """
    energy = pot.potential(pos) + 0.5 * np.sum(vel**2, axis=1)
    lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    bound = energy < 0.0
    r_c = np.asarray(pot.Rcirc(E=energy[bound]), dtype=np.float64)
    in_plane = np.column_stack([r_c, np.zeros_like(r_c), np.zeros_like(r_c)])
    l_c = r_c * np.sqrt(-r_c * pot.force(in_plane)[:, 0])
    out = np.full(energy.size, np.nan)
    out[bound] = lz[bound] / l_c
    return out


def radial_actions(pot: agama.Potential, pos: FloatArray, vel: FloatArray) -> FloatArray:
    """Return J_R (kpc km/s) with AGAMA's Staeckel fudge; NaN for unbound orbits."""
    finder = agama.ActionFinder(pot, interp=False)
    return np.asarray(finder(np.column_stack([pos, vel])), dtype=np.float64)[:, 0]


def potential_check(pot: agama.Potential, stars: Stars, annulus: BoolArray) -> list[SliceCheck]:
    """Return the median fitted and snapshot Delta Phi of the *annulus* stars in |z| slices.

    Both are the median over a slice minus the median of the first slice, so the
    additive constant of the snapshot potential (and of the mass beyond the selection
    radius) cancels. The snapshot potential carries the non-axisymmetric structure as
    scatter. Slices with fewer than ``CHECK_MIN_PER_SLICE`` stars are left out.
    """
    absz = np.abs(stars.pos[annulus, 2])
    snap = stars.potential[annulus]
    fitted = np.asarray(pot.potential(stars.pos[annulus]), dtype=np.float64)
    rows = []
    for lo, hi in zip(CHECK_EDGES_KPC[:-1], CHECK_EDGES_KPC[1:], strict=True):
        inside = (absz >= lo) & (absz < hi)
        if np.count_nonzero(inside) >= CHECK_MIN_PER_SLICE:
            rows.append((float(np.median(absz[inside])), np.median(snap[inside]), np.median(fitted[inside]), inside))
    if not rows:
        return []
    snap_zero, fit_zero = rows[0][1], rows[0][2]
    return [SliceCheck(z, float(s - snap_zero), float(f - fit_zero), int(np.count_nonzero(m))) for z, s, f, m in rows]


def _sector_average(x: FloatArray, delta: FloatArray, sector: NDArray[np.int64]) -> float:
    """Return the mean over the sectors of the median of *delta* less its gradient in *x* = R - r_mid.

    The slope of the gradient joins the medians of *delta* in the inner and outer halves of
    the stars.
    """
    inner, outer = x < np.median(x), x >= np.median(x)
    slope = (np.median(delta[outer]) - np.median(delta[inner])) / (np.median(x[outer]) - np.median(x[inner]))
    corrected = delta - slope * x
    return float(np.mean([np.median(corrected[sector == k]) for k in range(RESIDUAL_SECTORS)]))


def vertical_residual(
    cyl_r: FloatArray, azimuth: FloatArray, absz: FloatArray, delta: FloatArray, r_mid: float
) -> VerticalResidual:
    """Return the azimuthal average of *delta* at R = *r_mid* in each |z| slice of ``RESIDUAL_EDGES_KPC``.

    Within a slice, a straight line in R - r_mid removes the radial gradient across the
    annulus (:func:`_sector_average`). The corrected delta is reduced to its median in each of ``RESIDUAL_SECTORS``
    equal sectors of *azimuth* (rad), and the sectors are averaged with equal weights. A
    median over all the stars of the slice would weight each azimuth by its number of
    stars, which a bar makes depend on height, and turn the azimuthal structure of the
    potential into a spurious vertical trend. Slices in which a sector holds fewer than
    ``RESIDUAL_MIN_PER_SECTOR`` stars are left out.
    """
    turns = np.mod(azimuth, 2.0 * np.pi) / (2.0 * np.pi)
    sector = np.minimum((turns * RESIDUAL_SECTORS).astype(np.int64), RESIDUAL_SECTORS - 1)
    z_mid, d_mid = [], []
    for lo, hi in zip(RESIDUAL_EDGES_KPC[:-1], RESIDUAL_EDGES_KPC[1:], strict=True):
        inside = (absz >= lo) & (absz < hi)
        if np.bincount(sector[inside], minlength=RESIDUAL_SECTORS).min() < RESIDUAL_MIN_PER_SECTOR:
            continue
        z_mid.append(float(np.median(absz[inside])))
        d_mid.append(_sector_average(cyl_r[inside] - r_mid, delta[inside], sector[inside]))
    return VerticalResidual(tuple(z_mid), tuple(d_mid))


def snapshot_vertical_potential(
    pot: agama.Potential, stars: Stars, annulus: BoolArray, r_mid: float
) -> tuple[FloatArray, VerticalResidual]:
    """Return the snapshot's Delta Phi(R = *r_mid*, z) on ``Z_GRID`` and the residual it adds to the fit.

    The residual delta = Phi_snapshot - Phi_fit of the *annulus* stars, taken at r_mid and
    averaged over azimuth slice by slice (:func:`vertical_residual`), is interpolated on
    ``Z_GRID`` (constant beyond the last slice) and added, less its value in the first
    slice, to the fitted Delta Phi(r_mid, z).
    """
    pos = stars.pos[annulus]
    delta = stars.potential[annulus] - np.asarray(pot.potential(pos), dtype=np.float64)
    azimuth = np.arctan2(pos[:, 1], pos[:, 0])
    residual = vertical_residual(stars.cylindrical_radius[annulus], azimuth, np.abs(pos[:, 2]), delta, r_mid)
    if not residual.z:
        raise ValueError(f"no |z| slice holds {RESIDUAL_MIN_PER_SECTOR} star particles in every sector of the annulus")
    correction = np.interp(Z_GRID, residual.z, residual.residual) - residual.residual[0]
    return np.asarray(vertical_potential(pot, r_mid) + correction, dtype=np.float64), residual
