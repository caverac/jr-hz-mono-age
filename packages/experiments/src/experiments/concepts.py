"""What the Concepts pages of the documentation draw for one halo: its potential and the fit of every population.

The potential of the analysis (:func:`experiments.potential.fit_potential`) is the sum of a
CylSpline, for the gas and the stars of the disk region, and a Multipole, for everything
else. This module evaluates the two parts separately: their Delta Phi and vertical force at
the median radius of the annulus, their circular velocities, their densities on a
meridional grid, and the nodes of their grids.

For every age bin of :mod:`experiments.populations` it records what the fit of the scale
height sees: the profile of the star particles, weighted by initial mass, the windowed
h_Z at every trial window of the self-consistent search (:mod:`experiments.heights`), and
the profiles that the three predictions give on the same heights, the isothermal tracer,
the vertical Jeans equation without and with the tilt term, and the sheet. Every profile
is normalized to unit mass over |z| < ``Z_SAMPLE_MAX_KPC``. The bins hold no errors: they
illustrate the measurement, whose record with errors is that of ``auriga analyze``.
"""

from __future__ import annotations

from dataclasses import dataclass

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.analysis import PreparedHalo
from experiments.auriga import HaloSnapshot
from experiments.heights import (
    WINDOW_GRID,
    Z_GRID,
    particle_hz,
    scaled_particle_hz,
    sheet_profile,
    tracer_profile,
)
from experiments.jeans import jeans_density, jeans_profile
from experiments.populations import (
    AGE_BINS,
    K_MAIN,
    R_MAX_KPC,
    R_MIN_KPC,
    Z_MID_MAX_KPC,
    Z_SAMPLE_MAX_KPC,
    AnnulusSample,
    BinStars,
    median_radius,
    outer_sample,
    stars_without_actions,
    surface_density,
)
from experiments.potential import CYL_GRID, MULTIPOLE_GRID, vertical_potential
from experiments.snapshot import Galaxy
from experiments.stats import clipped_dispersion

FloatArray = NDArray[np.float64]

N_PROFILE_BINS = 40  # bins of the histogram of the heights of a population
N_PROFILE_POINTS = 241  # heights at which the predictions are recorded
PROFILE_WINDOWS = 3.0  # the profiles reach this many windows of the population, at most Z_SAMPLE_MAX_KPC
VERTICAL_Z_KPC = np.linspace(0.0, 8.0, 161)  # heights of the vertical potential and force
ROTATION_R_KPC = np.linspace(0.25, 30.0, 120)  # radii of the circular velocities
MAP_R_KPC = np.linspace(0.25, 25.0, 100)  # the meridional grid of the densities
MAP_Z_KPC = np.linspace(0.0, 10.0, 81)
MAP_FLOOR = 1.0e-12  # Msun/pc^3, below which a density enters the maps as this floor


@dataclass(frozen=True)
class ExpansionGrids:
    """The nodes (kpc) of the CylSpline in R and z, the log-spaced radii of the Multipole, and its l_max."""

    cylspline_r: list[float]
    cylspline_z: list[float]
    multipole_r: list[float]
    multipole_lmax: int


@dataclass(frozen=True)
class VerticalStructure:
    """Delta Phi ((km/s)^2) and the vertical force K_z = dPhi/dz ((km/s)^2/kpc) at the median radius, per expansion."""

    z: list[float]
    phi_total: list[float]
    phi_cylspline: list[float]
    phi_multipole: list[float]
    force_total: list[float]
    force_cylspline: list[float]
    force_multipole: list[float]


@dataclass(frozen=True)
class RotationCurve:
    """The circular velocity (km/s) in the plane of the whole potential and of each expansion."""

    r: list[float]
    total: list[float]
    cylspline: list[float]
    multipole: list[float]


@dataclass(frozen=True)
class MeridionalDensity:
    """log10 of the density (Msun/pc^3) of each expansion on the grid of R (rows) and z (columns), in kpc."""

    r: list[float]
    z: list[float]
    log_cylspline: list[list[float]]
    log_multipole: list[list[float]]


@dataclass(frozen=True)
class PopulationFit:  # pylint: disable=too-many-instance-attributes
    """What the fit of the scale height of one age bin sees, normalized to unit mass below ``Z_SAMPLE_MAX_KPC``.

    ``edges`` and ``density`` are the histogram of the heights of the star particles,
    weighted by initial mass; ``z`` and the four profiles are the predictions; ``windows``
    and ``heights`` are the windowed h_Z at every trial window of the self-consistent
    search, NaN where the window cannot measure it.
    """

    age_lo: float
    age_hi: float
    n_stars: int
    sigma_z: float
    hz: float
    window: float
    edges: list[float]
    density: list[float]
    z: list[float]
    tracer: list[float]
    jeans_sigma: list[float]
    jeans: list[float]
    sheet: list[float]
    windows: list[float]
    heights: list[float]


@dataclass(frozen=True)
class ConceptsRecord:  # pylint: disable=too-many-instance-attributes
    """The material of the Concepts pages for one halo."""

    snapshot: HaloSnapshot
    r_mid: float
    sigma_surf: float
    grids: ExpansionGrids
    vertical: VerticalStructure
    rotation: RotationCurve
    meridional: MeridionalDensity
    bins: list[PopulationFit]


def _nodes(size: float, lo: float, hi: float) -> list[float]:
    """Return the *size* nodes of a CylSpline grid from 0 to *hi*, nearly uniform below *lo* and then geometric."""
    return [float(v) for v in agama.nonuniformGrid(int(size), lo, hi)]


def expansion_grids() -> ExpansionGrids:
    """Return the nodes of the grids of :func:`experiments.potential.fit_potential`."""
    return ExpansionGrids(
        cylspline_r=_nodes(CYL_GRID["gridSizeR"], CYL_GRID["rmin"], CYL_GRID["rmax"]),
        cylspline_z=_nodes(CYL_GRID["gridSizeZ"], CYL_GRID["zmin"], CYL_GRID["zmax"]),
        multipole_r=[
            float(r)
            for r in np.geomspace(MULTIPOLE_GRID["rmin"], MULTIPOLE_GRID["rmax"], int(MULTIPOLE_GRID["gridSizeR"]))
        ],
        multipole_lmax=int(MULTIPOLE_GRID["lmax"]),
    )


def _column(radius: float, z: FloatArray) -> FloatArray:
    """Return the points (radius, 0, z) of a vertical line."""
    return np.column_stack([np.full_like(z, radius), np.zeros_like(z), z])


def vertical_structure(pot: agama.Potential, r_mid: float) -> VerticalStructure:
    """Return Delta Phi and the vertical force of *pot* and of its two expansions at *r_mid*."""
    points = _column(r_mid, VERTICAL_Z_KPC)

    def phi(part: agama.Potential) -> list[float]:
        return [float(v) for v in vertical_potential(part, r_mid, VERTICAL_Z_KPC)]

    def force(part: agama.Potential) -> list[float]:
        return [float(v) for v in -np.asarray(part.force(points))[:, 2]]

    return VerticalStructure(
        z=[float(z) for z in VERTICAL_Z_KPC],
        phi_total=phi(pot),
        phi_cylspline=phi(pot[0]),
        phi_multipole=phi(pot[1]),
        force_total=force(pot),
        force_cylspline=force(pot[0]),
        force_multipole=force(pot[1]),
    )


def rotation_curve(pot: agama.Potential) -> RotationCurve:
    """Return the circular velocity of *pot* and of its two expansions on ``ROTATION_R_KPC``."""
    points = np.column_stack([ROTATION_R_KPC, np.zeros_like(ROTATION_R_KPC), np.zeros_like(ROTATION_R_KPC)])

    def v_circ(part: agama.Potential) -> list[float]:
        radial = np.asarray(part.force(points))[:, 0]
        return [float(v) for v in np.sqrt(np.maximum(-ROTATION_R_KPC * radial, 0.0))]

    return RotationCurve(
        r=[float(r) for r in ROTATION_R_KPC],
        total=v_circ(pot),
        cylspline=v_circ(pot[0]),
        multipole=v_circ(pot[1]),
    )


def meridional_density(pot: agama.Potential) -> MeridionalDensity:
    """Return log10 of the density of each expansion of *pot* on ``MAP_R_KPC`` x ``MAP_Z_KPC``."""
    r, z = np.meshgrid(MAP_R_KPC, MAP_Z_KPC, indexing="ij")
    points = np.column_stack([r.ravel(), np.zeros(r.size), z.ravel()])

    def log_density(part: agama.Potential) -> list[list[float]]:
        density = np.asarray(part.density(points)).reshape(r.shape) / 1.0e9  # Msun/kpc^3 to Msun/pc^3
        return [[float(v) for v in row] for row in np.log10(np.maximum(density, MAP_FLOOR)).round(4)]

    return MeridionalDensity(
        r=[float(v) for v in MAP_R_KPC],
        z=[float(v) for v in MAP_Z_KPC],
        log_cylspline=log_density(pot[0]),
        log_multipole=log_density(pot[1]),
    )


def in_annulus(galaxy: Galaxy) -> NDArray[np.bool_]:
    """Return the mask of the star particles of the annulus of the analysis."""
    cyl_r = galaxy.stars.cylindrical_radius
    return np.asarray(
        (cyl_r >= R_MIN_KPC) & (cyl_r < R_MAX_KPC) & (np.abs(galaxy.stars.pos[:, 2]) < Z_SAMPLE_MAX_KPC), dtype=np.bool_
    )


def annulus_stars(galaxy: Galaxy) -> AnnulusSample:
    """Return the star particles of the annulus of the analysis, without their actions."""
    return stars_without_actions(galaxy, in_annulus(galaxy))


def profile_extent(window: float) -> float:
    """Return the height (kpc) to which the profiles of a population with *window* are recorded."""
    if not np.isfinite(window):
        return Z_SAMPLE_MAX_KPC
    return float(min(PROFILE_WINDOWS * window, Z_SAMPLE_MAX_KPC))


def _unit_mass(rho: FloatArray) -> FloatArray:
    """Return the profile *rho* on ``Z_GRID`` divided by its mass below ``Z_SAMPLE_MAX_KPC``."""
    return np.asarray(rho / np.trapezoid(rho, Z_GRID), dtype=np.float64)


def _rounded(values: FloatArray) -> list[float]:
    """Return *values* as a list, to the digits the figures need."""
    return [float(v) for v in np.round(values, 6)]


def height_histogram(absz: FloatArray, mass: FloatArray, extent: float) -> tuple[list[float], list[float]]:
    """Return the edges and the mass per unit height of the heights *absz* below *extent*, as a fraction of all.

    The density is normalized to the whole *mass*, so that it compares with a profile of
    unit mass below ``Z_SAMPLE_MAX_KPC`` on the same heights.
    """
    edges = np.linspace(0.0, extent, N_PROFILE_BINS + 1)
    counts, _ = np.histogram(absz, bins=edges, weights=mass)
    return _rounded(edges), _rounded(counts / np.sum(mass) / np.diff(edges))


def population_fit(
    stars: BinStars, bounds: tuple[float, float], phi_z: FloatArray, sigma_surf: float, r_mid: float
) -> PopulationFit:
    """Return what the fit of one age bin sees, with the potential *phi_z* on ``Z_GRID`` at *r_mid*."""
    absz, mass = np.abs(stars.annulus.z), stars.annulus.initial_mass
    sigma_z = clipped_dispersion(stars.annulus.v_z[absz < Z_MID_MAX_KPC])
    hz, window = scaled_particle_hz(absz, mass, K_MAIN)
    edges, density = height_histogram(absz, mass, profile_extent(window))
    z = np.linspace(0.0, edges[-1], N_PROFILE_POINTS)
    profile = jeans_profile(stars.jeans_stars(), R_MIN_KPC, R_MAX_KPC, r_mid)

    def on_z(rho: FloatArray) -> list[float]:
        return _rounded(np.interp(z, Z_GRID, _unit_mass(rho)))

    return PopulationFit(
        age_lo=bounds[0],
        age_hi=bounds[1],
        n_stars=int(absz.size),
        sigma_z=sigma_z,
        hz=hz,
        window=window,
        edges=edges,
        density=density,
        z=_rounded(z),
        tracer=on_z(tracer_profile(phi_z, sigma_z)),
        jeans_sigma=on_z(jeans_density(phi_z, profile, Z_GRID, with_tilt=False)),
        jeans=on_z(jeans_density(phi_z, profile, Z_GRID, with_tilt=True)),
        sheet=on_z(sheet_profile(sigma_z, sigma_surf)),
        windows=_rounded(WINDOW_GRID),
        heights=[particle_hz(absz, mass, float(w)) for w in WINDOW_GRID],
    )


def measure_concepts(prepared: PreparedHalo) -> ConceptsRecord:
    """Return the material of the Concepts pages for the *prepared* halo."""
    galaxy, pot = prepared.galaxy, prepared.potential
    sample = annulus_stars(galaxy)
    r_mid = median_radius(galaxy, in_annulus(galaxy))
    sigma_surf = surface_density(galaxy, R_MIN_KPC, R_MAX_KPC)
    phi_z = vertical_potential(pot, r_mid)
    outer = outer_sample(galaxy, R_MIN_KPC, R_MAX_KPC)
    bins = [
        population_fit(
            BinStars(
                sample.select((sample.age >= lo) & (sample.age < hi)),
                outer.select((outer.age >= lo) & (outer.age < hi)),
            ),
            (lo, hi),
            phi_z,
            sigma_surf,
            r_mid,
        )
        for lo, hi in AGE_BINS
    ]
    return ConceptsRecord(
        snapshot=prepared.snapshot,
        r_mid=r_mid,
        sigma_surf=sigma_surf,
        grids=expansion_grids(),
        vertical=vertical_structure(pot, r_mid),
        rotation=rotation_curve(pot),
        meridional=meridional_density(pot),
        bins=bins,
    )
