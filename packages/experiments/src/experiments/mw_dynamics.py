"""The frame, potential and radial actions in which the Milky Way stars are measured.

Every star is placed in the Galactocentric frame of the astroNN catalog of APOGEE DR17,
whose coordinates it reproduces: the Sun at ``R0_KPC`` = 8.125 kpc and ``Z_SUN_KPC`` =
20.8 pc above the plane (Bennett & Bovy 2019, MNRAS 482, 1417), moving at
``V_SUN_KMS`` = (11.1, 245.7, 7.25) km/s. The LAMOST stars are transformed into the same
frame (:func:`galactocentric`).

The potential is the best fit of McMillan (2017, MNRAS 465, 76) as AGAMA distributes it
(``data/McMillan17.ini``, in Msun, kpc and km/s). Its Delta Phi(z) at R_0 makes the
isothermal tracer, its mass within |z| < ``Z_SIGMA_KPC`` at R_0 the Sigma of the sheet,
as for the Auriga disks, and J_R comes from AGAMA's Staeckel fudge in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import agama
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from experiments.populations import Z_SIGMA_KPC
from experiments.potential import radial_actions, vertical_potential

FloatArray = NDArray[np.float64]

KPC = u.Unit("kpc")
DEG = u.Unit("deg")
KM_S = u.Unit("km / s")
MAS_YR = u.Unit("mas / yr")

R0_KPC = 8.125
Z_SUN_KPC = 0.0208
V_SUN_KMS = (11.1, 245.7, 7.25)
FRAME = coord.Galactocentric(
    galcen_distance=R0_KPC * KPC,
    z_sun=Z_SUN_KPC * KPC,
    galcen_v_sun=coord.CartesianDifferential(list(V_SUN_KMS) * KM_S),
)
SIGMA_GRID_POINTS = 1101


@dataclass(frozen=True)
class Phase:
    """Galactocentric cylindrical coordinates: R and z (kpc), v_R, v_T (along the rotation) and v_z (km/s)."""

    r: FloatArray
    z: FloatArray
    v_r: FloatArray
    v_t: FloatArray
    v_z: FloatArray


@dataclass(frozen=True)
class Sky:
    """Heliocentric coordinates: RA and Dec (deg), distance (kpc), proper motions (mas/yr), radial velocity (km/s)."""

    ra: FloatArray
    dec: FloatArray
    distance: FloatArray
    pmra: FloatArray
    pmdec: FloatArray
    rv: FloatArray


def galactocentric(sky: Sky) -> Phase:
    """Return the Galactocentric coordinates of *sky* in ``FRAME``."""
    stars = coord.SkyCoord(
        ra=sky.ra * DEG,
        dec=sky.dec * DEG,
        distance=sky.distance * KPC,
        pm_ra_cosdec=sky.pmra * MAS_YR,
        pm_dec=sky.pmdec * MAS_YR,
        radial_velocity=sky.rv * KM_S,
    ).transform_to(FRAME)
    cyl = stars.represent_as(coord.CylindricalRepresentation, coord.CylindricalDifferential)
    vel = cyl.differentials["s"]
    v_phi = (cyl.rho * vel.d_phi).to_value(KM_S, equivalencies=u.dimensionless_angles())
    return Phase(
        r=np.asarray(cyl.rho.to_value(KPC), dtype=np.float64),
        z=np.asarray(stars.z.to_value(KPC), dtype=np.float64),
        v_r=np.asarray(vel.d_rho.to_value(KM_S), dtype=np.float64),
        v_t=-np.asarray(v_phi, dtype=np.float64),  # the Galaxy rotates clockwise seen from the north pole
        v_z=np.asarray(vel.d_z.to_value(KM_S), dtype=np.float64),
    )


def mcmillan17() -> agama.Potential:
    """Return the potential of McMillan (2017) in Msun, kpc and km/s."""
    agama.setUnits(mass=1, length=1, velocity=1)
    return agama.Potential(file=str(Path(agama.__file__).parent / "data" / "McMillan17.ini"))


def surface_density(pot: agama.Potential, r: float) -> float:
    """Return the mass per area (Msun/pc^2) of *pot* within |z| < ``Z_SIGMA_KPC`` at radius *r* (kpc)."""
    z = np.linspace(0.0, Z_SIGMA_KPC, SIGMA_GRID_POINTS)
    rho = np.asarray(pot.density(np.column_stack([np.full_like(z, r), np.zeros_like(z), z])), dtype=np.float64)
    return float(2.0 * np.trapezoid(rho, z) / 1.0e6)


def solar_vertical_potential(pot: agama.Potential) -> FloatArray:
    """Return Delta Phi(R_0, z) ((km/s)^2) of *pot* on the height grid of :mod:`experiments.heights`."""
    return np.asarray(vertical_potential(pot, R0_KPC), dtype=np.float64)


def cylindrical_actions(pot: agama.Potential, phase: Phase) -> FloatArray:
    """Return J_R (kpc km/s) in *pot* of the stars at *phase*; NaN for unbound orbits."""
    pos = np.column_stack([phase.r, np.zeros_like(phase.r), phase.z])
    vel = np.column_stack([phase.v_r, phase.v_t, phase.v_z])
    return radial_actions(pot, pos, vel)
