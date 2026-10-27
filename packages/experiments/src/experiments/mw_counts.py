"""Star counts of Milky Way mono-age populations: Gaia DR3 giants split into ages by a spectroscopic survey.

The scale height of a population needs its density against |z|, which a spectroscopic
survey cannot give without its selection function. A complete photometric sample can:
the Gaia DR3 giants of a cylinder of radius ``R_CYL_KPC`` around the Sun, within
|z| < ``Z_MAX_KPC``, whose reddening-free Wesenheit absolute magnitude
M_W = G - 1.9 (BP - RP) + 5 log10(parallax / mas) - 10 lies in a window. Gaia DR3 is
complete for them, so their counts per unit |z| follow the density. Each slice of |z|
(``SLICE_EDGES_KPC``) is split into ages with the fraction of the matched spectroscopic
stars of the slice that belong to the population:

    rho_pop(z) = n_Gaia(z) P(pop | z),

with P interpolated between the centers of the slices. Two surveys give the ages
(:mod:`experiments.mw_catalogs`):

- LAMOST red clump: the Gaia stars inside +/- ``RC_HALF_WIDTH`` mag of the red-clump peak
  of M_W, which a fit of the luminosity function of each slice locates
  (:func:`fit_luminosity_function`), matched to LAMOST within ``MATCH_RADIUS_ARCSEC`` and
  |G_LAMOST - G_DR3| < ``MATCH_MAX_DG``;
- APOGEE: the Gaia giants of the box ``APOGEE_BOX`` of M_W, matched by source_id.

The targeting of both surveys depends on magnitude and color, not on age at a fixed
position, so the fraction of a slice is unbiased although the completeness of the survey
changes with z. h_Z is fitted to the weighted counts as to the star particles of the
Auriga disks (:func:`experiments.heights.particle_hz`), in the self-consistent window
L = k h_Z. The counts stop at ``Z_MAX_KPC``, so a window beyond it has no h_Z.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.coordinates as coord
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from experiments.heights import particle_hz, self_consistent_window
from experiments.mw_catalogs import AgeTracers, GaiaStars
from experiments.mw_dynamics import DEG, Z_SUN_KPC

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

R_CYL_KPC = 1.5
Z_MAX_KPC = 1.2
SLICE_EDGES_KPC = tuple(0.1 * k for k in range(13))
WESENHEIT_COEFF = 1.9  # A_G / E(BP - RP) of red giants
G_MAX = 15.5
BP_RP_MIN = 1.0
PARALLAX_OVER_ERROR_MIN = 10.0
APOGEE_BOX = (-2.6, -0.9)
MW_FIT_RANGE = (-2.7, -0.8)
MW_BIN = 0.025
RC_HALF_WIDTH = 0.3
MATCH_RADIUS_ARCSEC = 3.0
MATCH_MAX_DG = 0.05
# Bounds of (mu_rc, s_rc, mu_2, s_2, gamma). Without them the second Gaussian can widen into
# the wings of the red clump in some slices; the bounds hold every component to the shape
# it has where the fit is unambiguous.
SHAPE_BOUNDS = ((-1.95, -1.55), (0.05, 0.20), (-1.5, -1.0), (0.08, 0.25), (-1.0, 3.0))


@dataclass(frozen=True)
class CountSample:
    """|z| (kpc) of the Gaia stars of a window, and the |z|, age and population of their spectroscopic matches."""

    gaia_absz: FloatArray
    tracer_absz: FloatArray
    tracer_age: FloatArray
    tracer_in_disk: BoolArray

    def resampled(self, rng: np.random.Generator) -> CountSample:
        """Return a bootstrap copy, with the Gaia stars and the matches resampled independently."""
        g = rng.integers(0, self.gaia_absz.size, self.gaia_absz.size)
        t = rng.integers(0, self.tracer_age.size, self.tracer_age.size)
        return CountSample(self.gaia_absz[g], self.tracer_absz[t], self.tracer_age[t], self.tracer_in_disk[t])


@dataclass(frozen=True)
class GaiaCylinder:
    """The Gaia stars of the cylinder that pass the quality cuts: their row in the catalog, |z| (kpc) and M_W."""

    rows: NDArray[np.int64]
    absz: FloatArray
    m_w: FloatArray


def gaia_cylinder(gaia: GaiaStars, m_w_range: tuple[float, float]) -> GaiaCylinder:
    """Return the Gaia stars of the cylinder with precise parallaxes, G < ``G_MAX``, red colors and M_W in range."""
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = 1.0 / gaia.parallax
        lat = np.radians(gaia.b)
        absz = np.abs(distance * np.sin(lat) + Z_SUN_KPC)
        m_w = gaia.g - WESENHEIT_COEFF * gaia.bp_rp + 5.0 * np.log10(gaia.parallax) - 10.0
        keep = (
            (gaia.parallax > PARALLAX_OVER_ERROR_MIN * gaia.parallax_error)
            & (distance * np.cos(lat) < R_CYL_KPC) & (absz < Z_MAX_KPC)
            & (m_w > m_w_range[0]) & (m_w < m_w_range[1])
            & (gaia.g < G_MAX) & (gaia.bp_rp > BP_RP_MIN)
        )  # fmt: skip
    rows = np.flatnonzero(keep)
    return GaiaCylinder(rows, absz[rows], m_w[rows])


def slice_of(absz: FloatArray) -> NDArray[np.int64]:
    """Return the slice of ``SLICE_EDGES_KPC`` that holds each |z|."""
    edges = np.asarray(SLICE_EDGES_KPC)
    return np.asarray(np.clip(np.digitize(absz, edges) - 1, 0, edges.size - 2), dtype=np.int64)


@dataclass(frozen=True)
class LuminosityFit:  # pylint: disable=too-many-instance-attributes
    """The luminosity function of M_W in one slice: red clump, secondary peak and exponential background.

    n(M) = n_rc N(M; mu_rc, s_rc) + n_2 N(M; mu_2, s_2) + n_bg gamma exp(gamma M) / Z, with Z
    normalizing the exponential on ``MW_FIT_RANGE``.
    """

    n_rc: float
    mu_rc: float
    s_rc: float
    n_2: float
    mu_2: float
    s_2: float
    n_bg: float
    gamma: float

    def density(self, m_w: FloatArray) -> FloatArray:
        """Return the stars per magnitude at *m_w*."""
        lo, hi = MW_FIT_RANGE
        width = hi - lo
        norm = width if abs(self.gamma) < 1e-9 else np.expm1(self.gamma * width) / self.gamma
        background = self.n_bg * np.exp(self.gamma * (m_w - lo)) / norm
        return self.n_rc * _gauss(m_w, self.mu_rc, self.s_rc) + self.n_2 * _gauss(m_w, self.mu_2, self.s_2) + background


def _gauss(x: FloatArray, mu: float, s: float) -> FloatArray:
    """Return the normalized Gaussian density at *x*."""
    return np.asarray(np.exp(-0.5 * ((x - mu) / s) ** 2) / (np.sqrt(2.0 * np.pi) * s), dtype=np.float64)


def _unpack(theta: FloatArray) -> LuminosityFit:
    """Return the fit of the parameters *theta*: log amplitudes and shapes."""
    return LuminosityFit(
        float(np.exp(theta[0])), float(theta[1]), float(theta[2]), float(np.exp(theta[3])),
        float(theta[4]), float(theta[5]), float(np.exp(theta[6])), float(theta[7]),
    )  # fmt: skip


def _start(start: LuminosityFit | None, total: float) -> tuple[FloatArray, list[tuple[float, float]]]:
    """Return the first parameters of the fit and their bounds, for *total* stars.

    The fit starts from *start*, with its amplitudes rescaled to *total*, or from a generic
    red clump.
    """
    if start is None:
        start = LuminosityFit(0.4 * total, -1.75, 0.12, 0.1 * total, -1.25, 0.2, 0.5 * total, 1.0)
    scale = total / (start.n_rc + start.n_2 + start.n_bg)
    log_n = (float(np.log(1e-3 * total + 1.0)), float(np.log(2.0 * total + 1.0)))
    mu1, s1, mu2, s2, gam = SHAPE_BOUNDS
    bounds = [log_n, mu1, s1, log_n, mu2, s2, log_n, gam]
    theta = np.array([
        np.log(start.n_rc * scale), start.mu_rc, start.s_rc, np.log(start.n_2 * scale),
        start.mu_2, start.s_2, np.log(start.n_bg * scale), start.gamma,
    ])  # fmt: skip
    return np.clip(theta, [b[0] for b in bounds], [b[1] for b in bounds]), bounds


def fit_luminosity_function(m_w: FloatArray, start: LuminosityFit | None = None) -> LuminosityFit:
    """Return the binned Poisson maximum-likelihood fit of the luminosity function of *m_w* on ``MW_FIT_RANGE``.

    The negative log-likelihood leaves out the terms that do not depend on the parameters.
    The fit starts from *start* (:func:`_start`). Nelder-Mead runs twice, because a collapsed
    simplex can stop short of the minimum.
    """
    lo, hi = MW_FIT_RANGE
    edges = np.arange(lo, hi + 0.5 * MW_BIN, MW_BIN)
    counts, _ = np.histogram(m_w, edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    if counts.sum() == 0:
        raise RuntimeError("no star to fit the luminosity function on")
    theta0, bounds = _start(start, float(counts.sum()))

    def neg_log_like(theta: FloatArray) -> float:
        expected = np.maximum(_unpack(theta).density(centers) * MW_BIN, 1e-12)
        return float(np.sum(expected - counts * np.log(expected)))

    options = {"maxiter": 40_000, "maxfev": 40_000, "xatol": 1e-7, "fatol": 1e-7}
    best = minimize(neg_log_like, theta0, method="Nelder-Mead", bounds=bounds, options=options)
    best = minimize(neg_log_like, best.x, method="Nelder-Mead", bounds=bounds, options=options)
    if not best.success:
        raise RuntimeError(f"the fit of the luminosity function failed: {best.message}")
    return _unpack(np.asarray(best.x, dtype=np.float64))


@dataclass(frozen=True)
class RedClump:
    """The red-clump window of the Gaia cylinder: the fits of the slices, the peak of each, and the stars inside."""

    fits: tuple[LuminosityFit, ...]
    inside: BoolArray  # of the stars of the cylinder


def red_clump_window(cylinder: GaiaCylinder) -> RedClump:
    """Return the stars of *cylinder* within ``RC_HALF_WIDTH`` of the red-clump peak, interpolated in |z|."""
    slices = slice_of(cylinder.absz)
    first = fit_luminosity_function(cylinder.m_w)
    fits = tuple(fit_luminosity_function(cylinder.m_w[slices == k], first) for k in range(len(SLICE_EDGES_KPC) - 1))
    edges = np.asarray(SLICE_EDGES_KPC)
    centers = 0.5 * (edges[1:] + edges[:-1])
    peak = np.interp(cylinder.absz, centers, [f.mu_rc for f in fits])
    return RedClump(fits, np.abs(cylinder.m_w - peak) < RC_HALF_WIDTH)


def lamost_count_sample(gaia: GaiaStars, tracers: AgeTracers) -> tuple[CountSample, RedClump]:
    """Return the red-clump counts of the Gaia cylinder with the ages of their LAMOST matches.

    A LAMOST star matches the nearest Gaia star of the cylinder within
    ``MATCH_RADIUS_ARCSEC`` whose G agrees within ``MATCH_MAX_DG`` and which lies in the
    red-clump window; a Gaia star claimed twice is dropped.
    """
    cylinder = gaia_cylinder(gaia, MW_FIT_RANGE)
    clump = red_clump_window(cylinder)
    gaia_sky = coord.SkyCoord(l=gaia.l[cylinder.rows] * DEG, b=gaia.b[cylinder.rows] * DEG, frame="galactic")
    lamost_sky = coord.SkyCoord(l=tracers.l * DEG, b=tracers.b * DEG, frame="galactic")
    near, separation, _ = lamost_sky.match_to_catalog_sky(gaia_sky)
    matched = (
        (separation.arcsec < MATCH_RADIUS_ARCSEC)
        & (np.abs(tracers.g - gaia.g[cylinder.rows][near]) < MATCH_MAX_DG)
        & clump.inside[near]
    )
    pairs, claims = np.unique(near[matched], return_counts=True)
    matched &= np.isin(near, pairs[claims == 1])
    sample = CountSample(
        cylinder.absz[clump.inside],
        cylinder.absz[near[matched]],
        tracers.age[matched],
        tracers.in_disk[matched],
    )
    return sample, clump


def apogee_count_sample(gaia: GaiaStars, tracers: AgeTracers) -> CountSample:
    """Return the giant-box counts of the Gaia cylinder with the ages of their APOGEE matches by source_id."""
    cylinder = gaia_cylinder(gaia, APOGEE_BOX)
    ids = gaia.source_id[cylinder.rows]
    order = np.argsort(ids)
    position = order[np.clip(np.searchsorted(ids, tracers.source_id, sorter=order), 0, ids.size - 1)]
    matched = (tracers.source_id > 0) & (ids[position] == tracers.source_id)
    return CountSample(
        cylinder.absz,
        cylinder.absz[position[matched]],
        tracers.age[matched],
        tracers.in_disk[matched],
    )


def slice_fraction(sample: CountSample, lo: float, hi: float) -> FloatArray:
    """Return the fraction of the matches of each slice that belong to the population of ages [*lo*, *hi*) Gyr."""
    slices = slice_of(sample.tracer_absz)
    in_pop = sample.tracer_in_disk & (sample.tracer_age >= lo) & (sample.tracer_age < hi)
    n_slices = len(SLICE_EDGES_KPC) - 1
    total = np.bincount(slices, minlength=n_slices).astype(np.float64)
    hits = np.bincount(slices, weights=in_pop.astype(np.float64), minlength=n_slices)
    return np.divide(hits, total, out=np.zeros_like(hits), where=total > 0)


def population_weight(sample: CountSample, lo: float, hi: float) -> FloatArray:
    """Return P(pop | z) at the |z| of every Gaia star, interpolated between the centers of the slices."""
    edges = np.asarray(SLICE_EDGES_KPC)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return np.asarray(np.interp(sample.gaia_absz, centers, slice_fraction(sample, lo, hi)), dtype=np.float64)


def count_hz(sample: CountSample, lo: float, hi: float, k: float) -> tuple[float, float]:
    """Return h_Z (kpc) of the population of ages [*lo*, *hi*) and its window k h_Z; NaN for both beyond the counts."""
    weight = population_weight(sample, lo, hi)
    hz, window = self_consistent_window(lambda w: particle_hz(sample.gaia_absz, weight, w), k)
    if not np.isfinite(window) or window > Z_MAX_KPC:
        return float("nan"), float("nan")
    return hz, window
