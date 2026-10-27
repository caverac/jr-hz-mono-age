"""Scale heights of a whole disk: all star particles in radial rings and over Palicio's range.

Palicio et al. (2025, Sect. 3.4) give one h_Z per galaxy: the average of the exponential
fits within |Z| < Z_lim and |Z| < Z_lim / 2 over 0.2 L_kpc <= R < L_kpc, weighted by
stellar mass. For every ring of ``RING_EDGES_KPC`` inside that range, and for the whole
range, this module measures:

- h_Z with Palicio's fixed window, and in the self-consistent window ``K_WINDOW`` h_Z;
- the sheet, sigma_z^2 / (pi G Sigma), with the midplane sigma_z of all stars and Sigma
  as in :func:`experiments.populations.surface_density`;
- the isothermal tracer in the fitted potential at the median radius of the ring,
  through the self-consistent window.

All fits weight the star particles by their current mass, as Palicio's histograms do.
"""

from __future__ import annotations

from dataclasses import dataclass

import agama
import numpy as np

from experiments.heights import particle_hz, scaled_particle_hz, scaled_profile_hz, sheet_hz, tracer_profile
from experiments.palicio import PalicioWindow
from experiments.populations import Z_MID_MAX_KPC, Z_SAMPLE_MAX_KPC, clipped_dispersion, surface_density
from experiments.potential import vertical_potential
from experiments.snapshot import Galaxy, Particles, cylindrical_velocities

K_WINDOW = 2.0
RING_EDGES_KPC = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)


@dataclass(frozen=True)
class RingHeights:
    """h_Z (kpc) of one ring: Palicio's, in the self-consistent window (and that window), the sheet, the tracer."""

    palicio: float
    scaled: float
    window: float
    sheet: float
    tracer: float


@dataclass(frozen=True)
class Ring:
    """One radial range of the disk (kpc), its median radius, midplane sigma_z, Sigma and heights."""

    r_lo: float
    r_hi: float
    r_mid: float
    sigma_z_mid: float
    sigma_surf: float
    heights: RingHeights


@dataclass(frozen=True)
class WholeDisk:
    """The rings and the whole of Palicio's radial range for one galaxy."""

    window: PalicioWindow
    rings: list[Ring]
    whole: Ring


def _ring_stars(galaxy: Galaxy, bounds: tuple[float, float]) -> tuple[Particles, float]:
    """Return the star particles with bounds[0] <= R < bounds[1] and |z| < ``Z_SAMPLE_MAX_KPC``, and their median R."""
    stars = galaxy.stars
    cyl_r = stars.cylindrical_radius
    in_ring = (cyl_r >= bounds[0]) & (cyl_r < bounds[1]) & (np.abs(stars.pos[:, 2]) < Z_SAMPLE_MAX_KPC)
    return Particles(stars.pos[in_ring], stars.vel[in_ring], stars.mass[in_ring]), float(np.median(cyl_r[in_ring]))


def measure_ring(galaxy: Galaxy, pot: agama.Potential, bounds: tuple[float, float], z_lim: float) -> Ring:
    """Return the heights of the star particles with bounds[0] <= R < bounds[1] and |z| < ``Z_SAMPLE_MAX_KPC``."""
    stars, r_mid = _ring_stars(galaxy, bounds)
    absz, mass = np.abs(stars.pos[:, 2]), stars.mass
    _, v_z = cylindrical_velocities(stars.pos, stars.vel)
    sigma_z = clipped_dispersion(v_z[absz < Z_MID_MAX_KPC])
    sigma_surf = surface_density(galaxy, *bounds)
    scaled, window = scaled_particle_hz(absz, mass, K_WINDOW)
    heights = RingHeights(
        palicio=particle_hz(absz, mass, z_lim),
        scaled=scaled,
        window=window,
        sheet=sheet_hz(sigma_z, sigma_surf),
        tracer=scaled_profile_hz(tracer_profile(vertical_potential(pot, r_mid), sigma_z), K_WINDOW)[0],
    )
    return Ring(bounds[0], bounds[1], r_mid, sigma_z, sigma_surf, heights)


def measure_whole_disk(galaxy: Galaxy, pot: agama.Potential, window: PalicioWindow) -> WholeDisk:
    """Return the heights of the rings inside Palicio's radial range of *galaxy*, and of the whole range."""
    lo, hi = 0.2 * window.l_kpc, window.l_kpc
    edges = [e for e in RING_EDGES_KPC if lo <= e <= hi]
    return WholeDisk(
        window=window,
        rings=[measure_ring(galaxy, pot, ring, window.z_lim) for ring in zip(edges[:-1], edges[1:], strict=True)],
        whole=measure_ring(galaxy, pot, (lo, hi), window.z_lim),
    )
