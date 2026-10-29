"""Milky Way potentials around McMillan (2017) that match the measured densities near the Sun.

The potential of McMillan (2017, MNRAS 465, 76), as AGAMA distributes it, has at R_0 =
8.125 kpc a midplane density of 0.117 Msun/pc^3 and a surface density of 77.4 Msun/pc^2
within |z| < 1.1 kpc, above the rho_0 = 0.097 +/- 0.013 Msun/pc^3 of McKee et al. (2015, ApJ
814, 13, Eq. 57) and the Sigma_1.1 = 68 +/- 4 Msun/pc^2 of Bovy & Rix (2013, ApJ 779, 115).
The tracer of a population follows Delta Phi(R_0, z), which those two numbers bound. This
module varies the parts of the potential that set them:

- the surface density of the thin disk, by ``THIN_FACTORS``;
- its scale height, over ``THIN_HEIGHTS_KPC`` (McMillan holds it at 0.3 kpc);
- the normalization of the dark halo, by ``HALO_FACTORS``.

A member is admissible when its rho_0 and its Sigma_1.1 both lie within one error of those
measurements. The radial force, and with it the rotation curve, changes too and is not
constrained, because the tracer needs only the vertical potential at R_0.

For the comparison with Imig et al. (2025), every member also gives how the exponential fit
to its tracer grows with sigma_z between the ends of ``GROWTH_SIGMAS_KMS``: below |z| =
``Z_MAX_KPC``, as our counts are fitted, and over the whole profile, as their model is.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import agama
import numpy as np

from experiments.heights import Z_GRID, exponential_profile_hz, tracer_profile
from experiments.mw_analysis import SolarModel, solar_model
from experiments.mw_counts import Z_MAX_KPC
from experiments.mw_dynamics import R0_KPC, surface_density

# McMillan (2017) as in AGAMA's data/McMillan17.ini, in Msun, kpc and km/s.
MCMILLAN17: tuple[dict[str, float | str], ...] = (
    {"type": "Disk", "surfaceDensity": 8.95679e8, "scaleRadius": 2.49955, "scaleHeight": 0.3},
    {"type": "Disk", "surfaceDensity": 1.83444e8, "scaleRadius": 3.02134, "scaleHeight": 0.9},
    {"type": "Disk", "surfaceDensity": 5.31319e7, "scaleRadius": 7.0, "scaleHeight": -0.085, "innerCutoffRadius": 4.0},
    {
        "type": "Disk",
        "surfaceDensity": 2.17995e9,
        "scaleRadius": 1.5,
        "scaleHeight": -0.045,
        "innerCutoffRadius": 12.0,
    },
    {
        "type": "Spheroid",
        "densityNorm": 9.8351e10,
        "axisRatioZ": 0.5,
        "gamma": 0.0,
        "beta": 1.8,
        "scaleRadius": 0.075,
        "outerCutoffRadius": 2.1,
    },
    {
        "type": "Spheroid",
        "densityNorm": 8.53702e6,
        "axisRatioZ": 1.0,
        "gamma": 1.0,
        "beta": 3.0,
        "scaleRadius": 19.5725,
    },
)
THIN, HALO = 0, 5  # the indices of the thin disk and the dark halo in MCMILLAN17
THIN_FACTORS = (0.7, 0.8, 0.9, 1.0)
THIN_HEIGHTS_KPC = (0.25, 0.30, 0.35)
HALO_FACTORS = (0.7, 1.0, 1.3)
RHO0 = (0.097, 0.013)  # Msun/pc^3, McKee et al. (2015), Eq. 57
SIGMA_11 = (68.0, 4.0)  # Msun/pc^2, Bovy & Rix (2013)
GROWTH_SIGMAS_KMS = (10.0, 25.0)


@dataclass(frozen=True)
class FamilyParams:
    """The changes from McMillan (2017): thin-disk surface density factor, its scale height (kpc), halo factor."""

    thin_factor: float
    thin_height_kpc: float
    halo_factor: float


@dataclass(frozen=True)
class Growth:
    """d ln h / d ln sigma_z of the exponential fit to the tracer, below Z_MAX_KPC and over the whole profile."""

    window: float
    whole: float


@dataclass(frozen=True)
class FamilyMember:
    """One potential of the family: its parameters, rho_0 (Msun/pc^3), Sigma_1.1 (Msun/pc^2) and v_c (km/s) at R_0."""

    params: FamilyParams
    rho0: float
    sigma_11: float
    v_circ: float
    admissible: bool
    growth: Growth


def member_potential(params: FamilyParams) -> agama.Potential:
    """Return McMillan (2017) with the thin disk and the halo changed by *params*."""
    agama.setUnits(mass=1, length=1, velocity=1)
    components = [dict(c) for c in MCMILLAN17]
    components[THIN]["surfaceDensity"] = float(components[THIN]["surfaceDensity"]) * params.thin_factor
    components[THIN]["scaleHeight"] = params.thin_height_kpc
    components[HALO]["densityNorm"] = float(components[HALO]["densityNorm"]) * params.halo_factor
    return agama.Potential(*components)


def _growth(model: SolarModel) -> Growth:
    """Return the two-point growth of the exponential fits to the tracer between the ends of GROWTH_SIGMAS_KMS."""
    lo, hi = GROWTH_SIGMAS_KMS
    profiles = [tracer_profile(model.phi, sigma) for sigma in (lo, hi)]

    def slope(limit: float) -> float:
        h_lo, h_hi = (exponential_profile_hz(rho, limit) for rho in profiles)
        return float(np.log(h_hi / h_lo) / np.log(hi / lo))

    return Growth(slope(Z_MAX_KPC), slope(float(Z_GRID[-1])))


def family_member(params: FamilyParams) -> tuple[FamilyMember, SolarModel]:
    """Return the member of the family with *params*, and the vertical model of its potential at R_0."""
    pot = member_potential(params)
    model = solar_model(pot)
    rho0 = float(pot.density(np.array([[R0_KPC, 0.0, 0.0]]))[0]) / 1.0e9
    sigma_11 = surface_density(pot, R0_KPC)
    v_circ = float(np.sqrt(-R0_KPC * pot.force(np.array([[R0_KPC, 0.0, 0.0]]))[0, 0]))
    admissible = abs(rho0 - RHO0[0]) <= RHO0[1] and abs(sigma_11 - SIGMA_11[0]) <= SIGMA_11[1]
    return FamilyMember(params, rho0, sigma_11, v_circ, admissible, _growth(model)), model


def potential_family() -> list[tuple[FamilyMember, SolarModel]]:
    """Return every member of the grid of ``THIN_FACTORS``, ``THIN_HEIGHTS_KPC`` and ``HALO_FACTORS``."""
    return [
        family_member(FamilyParams(thin, height, halo))
        for thin, height, halo in itertools.product(THIN_FACTORS, THIN_HEIGHTS_KPC, HALO_FACTORS)
    ]
