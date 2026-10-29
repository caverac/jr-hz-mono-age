"""A reproducible proxy for the J_R,w of Palicio et al. (2025), from a face-on map of the median J_R.

Palicio et al. (2025, A&A 695, A193, Sect. 3.4) relate the vertical scale height h_Z of
each galaxy to J_R,w, "an individual reference value [...] that separates the high and
low radial action regions" of its map of the median J_R, which they read from the map of
each galaxy. That choice has no stated rule, so this module defines a proxy that follows
the construction of their maps (Sect. 2.4 and 2.5) as far as the paper describes it:

- the star particles with |Z| < Z_lim inside the square |X|, |Y| < L_kpc of their maps,
  in square cells of ``CELL_KPC`` = 0.375 kpc, their cell width (footnote 2);
- in each cell with at least ``MIN_PER_CELL`` star particles, the median J_R weighted by
  stellar mass, as they weight it;
- J_R,w is the median of the cell values whose centers lie at 0.2 L_kpc <= R < L_kpc, the
  radial range of their h_Z, so that half of those cells lie above it and half below.

Unlike their maps, the cells are not smoothed with an Epanechnikov kernel of 0.5 kpc,
and J_R comes from the Staeckel fudge of AGAMA in the potential of
:mod:`experiments.potential` rather than from their own fit and code. The proxy is a
statistic of the map, not their value.
"""

from __future__ import annotations

from dataclasses import dataclass

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.auriga import HaloSnapshot
from experiments.palicio import PalicioWindow
from experiments.potential import radial_actions
from experiments.snapshot import Galaxy

FloatArray = NDArray[np.float64]

CELL_KPC = 0.375
MIN_PER_CELL = 10
# Palicio et al. (2025, Sect. 3.4) give the slope of J_R,w against h_Z across galaxies in
# units of the solar angular momentum L_sun = R_0 V_0, which this module takes as the
# companion project does, R_0 = 8.2 kpc and V_0 = 233 km/s; their paper does not state it.
PALICIO_SLOPE_LSUN_PER_KPC = 3.69e-2
L_SUN_KPC_KMS = 8.2 * 233.0


@dataclass(frozen=True)
class JrMap:
    """The proxy J_R,w (kpc km/s) of one galaxy, the spread of its cells and the counts it rests on."""

    window: PalicioWindow
    n_stars: int  # star particles with a finite J_R in the map
    n_cells: int  # cells of the radial range with at least MIN_PER_CELL star particles
    jr_w: float  # the median of the cell values in 0.2 L_kpc <= R < L_kpc
    jr_cells_lo: float  # their 16th percentile
    jr_cells_hi: float  # their 84th percentile


def weighted_median(values: FloatArray, weights: FloatArray) -> float:
    """Return the value below which half of the total *weights* lies."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _cells(x: FloatArray, y: FloatArray, l_kpc: float) -> tuple[NDArray[np.int64], int]:
    """Return the cell of every point of the square |X|, |Y| < *l_kpc*, and the number of cells along a side."""
    n_side = int(np.ceil(2.0 * l_kpc / CELL_KPC))
    ix = np.floor((x + l_kpc) / CELL_KPC).astype(np.int64)
    iy = np.floor((y + l_kpc) / CELL_KPC).astype(np.int64)
    return ix * n_side + iy, n_side


def _center_radius(cell: int, n_side: int, l_kpc: float) -> float:
    """Return the distance (kpc) of the center of *cell* from the center of the map."""
    center_x = (cell // n_side + 0.5) * CELL_KPC - l_kpc
    center_y = (cell % n_side + 0.5) * CELL_KPC - l_kpc
    return float(np.hypot(center_x, center_y))


def cell_medians(
    x: FloatArray, y: FloatArray, jr: FloatArray, mass: FloatArray, l_kpc: float
) -> tuple[FloatArray, FloatArray]:
    """Return the radius of the center and the mass-weighted median J_R of every well-filled cell.

    The cells tile |X|, |Y| < *l_kpc* with side ``CELL_KPC``; a cell enters with at least
    ``MIN_PER_CELL`` star particles.
    """
    if x.size == 0:
        return np.empty(0), np.empty(0)
    cell, n_side = _cells(x, y, l_kpc)
    order = np.argsort(cell, kind="stable")
    cells, starts = np.unique(cell[order], return_index=True)
    filled = [(int(c), members) for c, members in zip(cells, np.split(order, starts[1:]), strict=True)]
    filled = [(c, members) for c, members in filled if members.size >= MIN_PER_CELL]
    radius = np.array([_center_radius(c, n_side, l_kpc) for c, _ in filled], dtype=np.float64)
    median = np.array([weighted_median(jr[members], mass[members]) for _, members in filled], dtype=np.float64)
    return radius, median


def map_proxy(x: FloatArray, y: FloatArray, jr: FloatArray, mass: FloatArray, window: PalicioWindow) -> JrMap:
    """Return the proxy J_R,w of the star particles at (*x*, *y*) (kpc) with *jr* and *mass*, in *window*."""
    finite = np.isfinite(jr)
    radius, median = cell_medians(x[finite], y[finite], jr[finite], mass[finite], window.l_kpc)
    inside = (radius >= 0.2 * window.l_kpc) & (radius < window.l_kpc)
    values = median[inside]
    if values.size == 0:
        nan = float("nan")
        return JrMap(window, int(finite.sum()), 0, nan, nan, nan)
    lo, mid, hi = np.percentile(values, [16.0, 50.0, 84.0])
    return JrMap(window, int(finite.sum()), int(values.size), float(mid), float(lo), float(hi))


def measure_jr_map(galaxy: Galaxy, pot: agama.Potential, window: PalicioWindow) -> JrMap:
    """Return the proxy J_R,w of *galaxy*, in its disk frame, with J_R in the fitted potential *pot*."""
    stars = galaxy.stars
    pos = stars.pos
    in_map = (
        (np.abs(pos[:, 2]) < window.z_lim) & (np.abs(pos[:, 0]) < window.l_kpc) & (np.abs(pos[:, 1]) < window.l_kpc)
    )
    jr = radial_actions(pot, pos[in_map], stars.vel[in_map])
    return map_proxy(pos[in_map, 0], pos[in_map, 1], jr, stars.mass[in_map], window)


@dataclass(frozen=True)
class JrMapRecord:
    """The proxy J_R,w of one snapshot, as ``experiments auriga jr-map`` writes it."""

    snapshot: HaloSnapshot
    jr_map: JrMap


@dataclass(frozen=True)
class Line:
    """The least-squares line y = intercept + slope x, with the standard errors of both."""

    slope: float
    slope_err: float
    intercept: float
    intercept_err: float


def fit_line(x: FloatArray, y: FloatArray) -> Line:
    """Return the ordinary least-squares line through (*x*, *y*), with errors from the scatter about it."""
    design = np.vstack([np.ones_like(x), x]).T
    coef, residual, _, _ = np.linalg.lstsq(design, y)
    scatter = float(residual[0]) / (x.size - 2) if residual.size else 0.0
    cov = scatter * np.linalg.inv(design.T @ design)
    return Line(float(coef[1]), float(np.sqrt(cov[1, 1])), float(coef[0]), float(np.sqrt(cov[0, 0])))


@dataclass(frozen=True)
class JrPoint:
    """One galaxy of the relation: its proxy J_R,w (kpc km/s), and its whole-disk h_Z (kpc) in both windows.

    ``hz_palicio`` and ``hz_scaled`` are measured from the star particles; the ``_mixture``
    heights are predicted by the sum of isothermal tracers of :mod:`experiments.tracer_sum`.
    """

    halo: int
    jr_w: float
    hz_palicio: float
    hz_scaled: float
    hz_palicio_mixture: float
    hz_scaled_mixture: float


@dataclass(frozen=True)
class JrRelation:
    """J_R,w against h_Z across the galaxies, with h_Z in Palicio's windows and in the self-consistent one.

    The lines use the measured h_Z, and the ``_mixture`` lines the predicted one.
    ``palicio_slope`` is the slope of Palicio et al. (2025) in km/s.
    """

    points: list[JrPoint]
    line_palicio: Line
    line_scaled: Line
    line_palicio_mixture: Line
    line_scaled_mixture: Line
    palicio_slope: float


def jr_relation(points: list[JrPoint]) -> JrRelation:
    """Return the lines of J_R,w against the measured and the predicted h_Z through *points*, in both windows."""
    jr = np.array([p.jr_w for p in points])

    def line(height: str) -> Line:
        return fit_line(np.array([getattr(p, height) for p in points]), jr)

    return JrRelation(
        points=points,
        line_palicio=line("hz_palicio"),
        line_scaled=line("hz_scaled"),
        line_palicio_mixture=line("hz_palicio_mixture"),
        line_scaled_mixture=line("hz_scaled_mixture"),
        palicio_slope=PALICIO_SLOPE_LSUN_PER_KPC * L_SUN_KPC_KMS,
    )
