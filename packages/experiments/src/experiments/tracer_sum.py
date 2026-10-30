"""The whole disk as a sum of isothermal tracers, against the h_Z of all its star particles.

Palicio et al. (2025) relate J_R,w to one h_Z per galaxy, fitted to all the star particles
over 0.2 L_kpc <= R < L_kpc, whereas the tracer predicts the profile of one mono-age
population from its midplane sigma_z in the potential. If every population of a disk is
such a tracer, the profile of a radial range is the sum over its rings q and age bins a,

    rho(z) = sum_q sum_a M_qa exp(-Delta Phi_q(z) / sigma_qa^2) / (2 int_0^Z exp(-Delta Phi_q / sigma_qa^2) dz),

where M_qa is the current mass of the star particles of ring q and age bin a with
|z| < Z = ``Z_SAMPLE_MAX_KPC``, sigma_qa their clipped sigma_z at |z| < ``Z_MID_MAX_KPC``,
and Delta Phi_q the fitted potential at the center of the ring. This module predicts
h_Z of the rings of :mod:`experiments.whole_disk` inside Palicio's range, and of the whole
range, in two ways:

- ``single``: one tracer per ring of ``RING_KPC``, with the sigma_z of all its stars; the
  sum over rings takes out the change of the potential and of sigma_z with radius, so
  that what remains of the misfit of a single tracer comes from the mixture of ages;
- ``mixture``: one tracer per ring and age bin of ``AGE_EDGES_GYR``.

Both are fitted like the star particles, weighted by current mass as Palicio's histograms
are: within Palicio's Z_lim and in the self-consistent window ``K_WINDOW`` h_Z. A cell
with fewer than ``MIN_MID_STARS`` star particles at |z| < ``Z_MID_MAX_KPC`` has no sigma_z;
its mass is left out of the sum and reported as a fraction of the mass of the range, for
each prediction (``unmodeled``).
Errors are half the 16th-84th percentile interval of bootstrap resamplings of the star
particles of the range, which give every height of one draw from the same stars.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.auriga import HaloSnapshot
from experiments.heights import Z_GRID, particle_hz, profile_hz, scaled_particle_hz, scaled_profile_hz
from experiments.palicio import PalicioWindow
from experiments.populations import SEED, Z_MID_MAX_KPC, Z_SAMPLE_MAX_KPC, Estimate
from experiments.potential import vertical_potential
from experiments.snapshot import Galaxy, cylindrical_velocities
from experiments.stats import clipped_dispersion
from experiments.whole_disk import K_WINDOW, RING_EDGES_KPC

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]

RING_KPC = 1.0
AGE_EDGES_GYR = tuple(float(t) for t in range(15))
MIN_MID_STARS = 30
N_DRAWS = 50
HEIGHTS = ("measured", "single", "mixture")


@dataclass(frozen=True)
class SumStars:
    """Star particles of the range: R and |z| (kpc), v_z (km/s), current mass (Msun) and age (Gyr)."""

    r: FloatArray
    absz: FloatArray
    v_z: FloatArray
    mass: FloatArray
    age: FloatArray

    def select(self, keep: BoolArray | IndexArray) -> SumStars:
        """Return the stars where the mask *keep* is true, or at the indices *keep*, which may repeat."""
        return SumStars(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})


@dataclass(frozen=True)
class SumHeights:
    """h_Z (kpc), or a ratio of two, in Palicio's window Z_lim and in the self-consistent window."""

    palicio: Estimate
    scaled: Estimate


@dataclass(frozen=True)
class Unmodeled:
    """The fraction of the mass of a range that each prediction leaves out."""

    single: float
    mixture: float


@dataclass(frozen=True)
class SumRange:  # pylint: disable=too-many-instance-attributes
    """One radial range (kpc): the mass it leaves out, the measured and predicted h_Z, and their ratios."""

    r_lo: float
    r_hi: float
    unmodeled: Unmodeled
    measured: SumHeights
    single: SumHeights
    mixture: SumHeights
    ratio_single: SumHeights
    ratio_mixture: SumHeights


@dataclass(frozen=True)
class TracerSum:
    """The rings inside Palicio's radial range of one galaxy, and the whole range."""

    window: PalicioWindow
    rings: list[SumRange]
    whole: SumRange


@dataclass(frozen=True)
class TracerSumRecord:
    """The predicted whole-disk h_Z of one snapshot, as ``experiments auriga tracer-sum`` writes it."""

    snapshot: HaloSnapshot
    tracer_sum: TracerSum


def tracer_shape(phi_z: FloatArray, sigma_z: float) -> FloatArray:
    """Return exp(-Delta Phi / sigma_z^2) on ``Z_GRID``, normalized to unit mass over -Z < z < Z."""
    rho = np.exp(-phi_z / sigma_z**2)
    steps = 0.5 * (rho[1:] + rho[:-1]) * np.diff(Z_GRID)
    return np.asarray(rho / (2.0 * np.sum(steps)), dtype=np.float64)


def _tracer(phi_z: FloatArray, v_z: FloatArray, absz: FloatArray, mass: float) -> FloatArray | None:
    """Return the profile of *mass* in the tracer of the midplane stars of (*v_z*, *absz*); None if they are too few."""
    mid = absz < Z_MID_MAX_KPC
    if np.count_nonzero(mid) < MIN_MID_STARS:
        return None
    return mass * tracer_shape(phi_z, clipped_dispersion(v_z[mid]))


@dataclass(frozen=True)
class _Ring:
    """The predicted profiles of one ring and the mass that each leaves out."""

    single: FloatArray
    mixture: FloatArray
    unmodeled_single: float
    unmodeled_mixture: float


def ring_profiles(stars: SumStars, phi_z: FloatArray) -> _Ring:
    """Return the single-tracer and the mixture profiles of the stars of one ring, in its potential *phi_z*."""
    total = float(np.sum(stars.mass))
    single = _tracer(phi_z, stars.v_z, stars.absz, total)
    mixture, unmodeled = np.zeros(Z_GRID.size), 0.0
    for lo, hi in zip(AGE_EDGES_GYR[:-1], AGE_EDGES_GYR[1:], strict=True):
        cell = (stars.age >= lo) & (stars.age < hi)
        mass = float(np.sum(stars.mass[cell]))
        profile = _tracer(phi_z, stars.v_z[cell], stars.absz[cell], mass)
        if profile is None:
            unmodeled += mass
        else:
            mixture += profile
    outside = float(np.sum(stars.mass[(stars.age < AGE_EDGES_GYR[0]) | (stars.age >= AGE_EDGES_GYR[-1])]))
    if single is None:
        return _Ring(np.zeros(Z_GRID.size), mixture, total, unmodeled + outside)
    return _Ring(single, mixture, 0.0, unmodeled + outside)


def _heights(rho: FloatArray, z_lim: float) -> tuple[float, float]:
    """Return h_Z of the profile *rho* in Z_lim and in the self-consistent window."""
    return profile_hz(rho, z_lim), scaled_profile_hz(rho, K_WINDOW)[0]


@dataclass(frozen=True)
class _Layout:
    """What every draw shares: the edges of the rings of ``RING_KPC``, their Delta Phi, the ranges and Z_lim."""

    ring_edges: FloatArray
    phis: list[FloatArray]
    ranges: list[tuple[float, float]]
    z_lim: float


def _layout(pot: agama.Potential, window: PalicioWindow) -> _Layout:
    """Return the rings of Palicio's radial range of *window*, their potentials and the ranges to predict."""
    lo, hi = 0.2 * window.l_kpc, window.l_kpc
    ring_edges = np.arange(lo, hi + 0.5 * RING_KPC, RING_KPC)
    phis = [vertical_potential(pot, float(center)) for center in 0.5 * (ring_edges[1:] + ring_edges[:-1])]
    edges = [e for e in RING_EDGES_KPC if lo <= e <= hi]
    return _Layout(ring_edges, phis, list(zip(edges[:-1], edges[1:], strict=True)) + [(lo, hi)], window.z_lim)


def _range_values(
    stars: SumStars, members: list[_Ring], bounds: tuple[float, float], z_lim: float
) -> tuple[FloatArray, FloatArray]:
    """Return the heights (kind, window) of one range from its stars and its rings, and the unmodeled fractions."""
    rows = stars.select((stars.r >= bounds[0]) & (stars.r < bounds[1]))
    total = float(np.sum(rows.mass))
    heights = np.array(
        [
            (particle_hz(rows.absz, rows.mass, z_lim), scaled_particle_hz(rows.absz, rows.mass, K_WINDOW)[0]),
            _heights(np.sum([ring.single for ring in members], axis=0), z_lim),
            _heights(np.sum([ring.mixture for ring in members], axis=0), z_lim),
        ]
    )
    unmodeled = np.array(
        [
            sum(ring.unmodeled_single for ring in members) / total,
            sum(ring.unmodeled_mixture for ring in members) / total,
        ]
    )
    return heights, unmodeled


def _draw(stars: SumStars, layout: _Layout) -> tuple[FloatArray, FloatArray]:
    """Return the heights (range, kind, window) of one sample of the stars, and the unmodeled mass (range, kind)."""
    edges = layout.ring_edges
    rings = [
        ring_profiles(stars.select((stars.r >= lo) & (stars.r < hi)), phi_z)
        for lo, hi, phi_z in zip(edges[:-1], edges[1:], layout.phis, strict=True)
    ]
    centers = 0.5 * (edges[1:] + edges[:-1])
    values = [
        _range_values(
            stars, [ring for ring, c in zip(rings, centers, strict=True) if lo <= c < hi], (lo, hi), layout.z_lim
        )
        for lo, hi in layout.ranges
    ]
    return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])


def _sum_range(bounds: tuple[float, float], point: FloatArray, draws: FloatArray, unmodeled: Unmodeled) -> SumRange:
    """Return one range from its heights (kind, window) at the point and in each draw (draw, kind, window)."""

    def pair(value: FloatArray, samples: FloatArray) -> SumHeights:
        return SumHeights(*(Estimate.of(float(value[w]), samples[:, w]) for w in range(2)))

    by_kind = {name: pair(point[i], draws[:, i]) for i, name in enumerate(HEIGHTS)}
    ratio = {name: pair(point[i] / point[0], draws[:, i] / draws[:, 0]) for i, name in enumerate(HEIGHTS) if i > 0}
    return SumRange(
        *bounds,
        unmodeled,
        by_kind["measured"],
        by_kind["single"],
        by_kind["mixture"],
        ratio["single"],
        ratio["mixture"],
    )


def sum_stars(galaxy: Galaxy, r_lo: float, r_hi: float) -> SumStars:
    """Return the star particles with r_lo <= R < r_hi and |z| < ``Z_SAMPLE_MAX_KPC``."""
    stars = galaxy.stars
    cyl_r = stars.cylindrical_radius
    keep = (cyl_r >= r_lo) & (cyl_r < r_hi) & (np.abs(stars.pos[:, 2]) < Z_SAMPLE_MAX_KPC)
    _, v_z = cylindrical_velocities(stars.pos[keep], stars.vel[keep])
    age = galaxy.header.cosmology.lookback_gyr(stars.formation_a[keep], galaxy.header.scale_factor)
    return SumStars(cyl_r[keep], np.abs(stars.pos[keep, 2]), v_z, stars.mass[keep], age)


def measure_tracer_sum(
    galaxy: Galaxy, pot: agama.Potential, window: PalicioWindow, n_draws: int = N_DRAWS
) -> TracerSum:
    """Return the measured and predicted h_Z of the rings in Palicio's radial range of *galaxy*, and of all of it."""
    stars = sum_stars(galaxy, 0.2 * window.l_kpc, window.l_kpc)
    layout = _layout(pot, window)
    point, unmodeled = _draw(stars, layout)
    rng = np.random.default_rng(SEED)
    n = stars.r.size
    draws = np.stack([_draw(stars.select(rng.integers(0, n, n)), layout)[0] for _ in range(n_draws)])
    ranges = [
        _sum_range(bounds, point[j], draws[:, j], Unmodeled(*unmodeled[j].tolist()))
        for j, bounds in enumerate(layout.ranges)
    ]
    return TracerSum(window, ranges[:-1], ranges[-1])
