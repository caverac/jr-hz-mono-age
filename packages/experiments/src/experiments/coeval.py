"""Vertical profiles of coeval populations whose vertical actions follow Ting & Rix (2026).

Ting & Rix (2026, arXiv 2608.12156) let a population diffuse in vertical action with a
diffusivity D ~ J_z^b, and find the distribution p(J_z) ~ exp[-(J_z / J_0)^(2 - b)]:
b = 1 for scatterers that fill the volume, which in a harmonic potential is the isothermal
population with J_0 = sigma_z^2 / nu, and b = 1/2 for scatterers in a thin layer at the
midplane, close to the b = 0.51 they measure for red-clump stars of 2-8 Gyr.

In one dimension the phase-space density of a population is proportional to p(J_z), so its
density in the vertical potential Phi(z) at R_mid is

    rho(z) = int f(J_z(Phi(z) + v^2 / 2)) dv,

with the action of the orbit of energy E,

    J_z(E) = (2 / pi) int_0^{z_max(E)} sqrt(2 (E - Phi(z))) dz.

:class:`CoevalModel` computes J_z(E) once per potential, at the energies Phi(z_k) of the
heights of the grid, and :meth:`CoevalModel.profile` returns the profile of the population
whose stars within ``z_mid`` of the plane have the dispersion sigma_z, the stars whose
sigma_z the tracer takes. Orbits that leave the grid are left out, as stars beyond it are
left out of the measured profiles. :func:`coeval_curve` tabulates the h_Z and the half-mass
height of such populations over a grid of dispersions, which :meth:`CoevalCurve.at`
interpolates, and :func:`fit_action_shape` fits b and J_0 to the vertical actions of a
population by maximum likelihood. Because b = 1 is isothermal only in a harmonic potential,
:func:`isothermal_action_shape` gives the b of the isothermal tracer in the same potential,
against which a fitted b is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize

from experiments.heights import Z_GRID, profile_half_mass, scaled_profile_hz

FloatArray = NDArray[np.float64]

THIN_LAYER_B = 0.5  # the b of scatterers in a thin layer at the midplane (Ting & Rix 2026)
N_VELOCITY = 800  # points of the velocity integral at each height, denser near v = 0
J0_BRACKET = (1.0e-3, 1.0e5)  # kpc km/s, the range of J_0 searched for a dispersion
SHAPE_BOUNDS = (0.2, 3.0)  # the range of 2 - b over which the likelihood is maximized
# The isothermal density at the edge of the grid, relative to the midplane, above which a
# population of that sigma_z has orbits beyond the grid that the actions leave out. In the
# potential of Au19 the isothermal b turns over, a sign of the cut, from about 5e-4.
EDGE_DENSITY = 1.0e-4


def orbit_actions(phi_z: FloatArray, z_grid: FloatArray = Z_GRID) -> FloatArray:
    """Return J_z (kpc km/s) of the orbit whose turning point is each height of *z_grid*, in *phi_z* ((km/s)^2).

    *phi_z* must grow with height. The speed of the orbit turning at z_k vanishes above z_k,
    so the integral over the whole grid is the integral up to its turning point.
    """
    speed = np.sqrt(2.0 * np.clip(phi_z[:, None] - phi_z[None, :], 0.0, None))
    return np.asarray((2.0 / np.pi) * np.trapezoid(speed, z_grid, axis=1), dtype=np.float64)


@dataclass(frozen=True)
class CoevalModel:
    """The vertical potential *phi_z* ((km/s)^2) on *z_grid* (kpc), with the actions of its orbits."""

    phi_z: FloatArray
    z_grid: FloatArray = field(default_factory=lambda: Z_GRID)
    z_mid: float = 0.3  # kpc, the stars whose dispersion is sigma_z
    actions: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Compute the actions of the orbits."""
        object.__setattr__(self, "actions", orbit_actions(self.phi_z, self.z_grid))

    def moments(self, j0: float, b: float, n_heights: int | None = None) -> tuple[FloatArray, FloatArray]:
        """Return rho(z) and rho(z) <v_z^2>(z) of the distribution exp[-(J_z / *j0*)^(2 - *b*)].

        Only the first *n_heights* heights of the grid are computed when it is given. At every
        height the velocities run from 0 to the escape speed of the grid, spaced
        quadratically to resolve the core of the coldest populations. They are built on each
        call rather than stored, because a family of models would hold them all at once.
        """
        phi = self.phi_z[:n_heights]
        v_max = np.sqrt(2.0 * (self.phi_z[-1] - phi))
        velocity = v_max[:, None] * (np.linspace(0.0, 1.0, N_VELOCITY) ** 2)[None, :]
        energy = phi[:, None] + 0.5 * velocity**2
        density = np.exp(-((np.interp(energy, self.phi_z, self.actions) / j0) ** (2.0 - b)))
        rho = 2.0 * np.trapezoid(density, velocity, axis=1)
        rho_v2 = 2.0 * np.trapezoid(density * velocity**2, velocity, axis=1)
        return np.asarray(rho, dtype=np.float64), np.asarray(rho_v2, dtype=np.float64)

    def midplane_dispersion(self, j0: float, b: float) -> float:
        """Return the dispersion of v_z (km/s) of the stars within ``z_mid`` of the plane."""
        n_mid = int(np.count_nonzero(self.z_grid <= self.z_mid))
        rho, rho_v2 = self.moments(j0, b, n_mid)
        z = self.z_grid[:n_mid]
        return float(np.sqrt(np.trapezoid(rho_v2, z) / np.trapezoid(rho, z)))

    def reaches_edge(self, sigma_z: float) -> bool:
        """Return whether the isothermal population of *sigma_z* keeps more than ``EDGE_DENSITY`` at the grid edge."""
        return bool(np.exp(-self.phi_z[-1] / sigma_z**2) > EDGE_DENSITY)

    def profile(self, sigma_z: float, b: float) -> FloatArray:
        """Return rho(z) of the population with exponent *b* whose midplane dispersion is *sigma_z*; NaN if none.

        There is none when the population reaches the edge of the grid (:meth:`reaches_edge`),
        or when no J_0 in ``J0_BRACKET`` gives *sigma_z*.
        """
        if self.reaches_edge(sigma_z):
            return np.full(self.z_grid.size, np.nan)

        def excess(log_j0: float) -> float:
            return self.midplane_dispersion(float(np.exp(log_j0)), b) - sigma_z

        lo, hi = np.log(J0_BRACKET)
        if not excess(lo) < 0.0 < excess(hi):
            return np.full(self.z_grid.size, np.nan)
        j0 = float(np.exp(brentq(excess, lo, hi, xtol=1.0e-6)))
        return self.moments(j0, b)[0]


@dataclass(frozen=True)
class CoevalCurve:
    """h_Z and the half-mass height (kpc) of the population with exponent *b* over a grid of dispersions (km/s).

    ``b_isothermal`` is the b that the isothermal tracer of each dispersion has in the same
    potential (:func:`isothermal_action_shape`), the reference for the b fitted to stars.
    """

    b: float
    sigma_z: list[float]
    hz: list[float]
    half_mass: list[float]
    b_isothermal: list[float]

    def at(self, sigma_z: float) -> tuple[float, float]:
        """Return h_Z and the half-mass height at *sigma_z*, interpolated in log-log; NaN off the curve.

        Off the curve means outside the grid of dispersions, or between two points of which
        one has no profile.
        """
        grid = np.asarray(self.sigma_z)
        if not grid[0] <= sigma_z <= grid[-1]:
            return float("nan"), float("nan")
        i = min(max(int(np.searchsorted(grid, sigma_z)), 1), grid.size - 1)
        t = np.log(sigma_z / grid[i - 1]) / np.log(grid[i] / grid[i - 1])

        def interpolate(values: list[float]) -> float:
            return float(np.exp((1.0 - t) * np.log(values[i - 1]) + t * np.log(values[i])))

        return interpolate(self.hz), interpolate(self.half_mass)

    def isothermal_b(self, sigma_z: float) -> float:
        """Return the b of the isothermal tracer at *sigma_z*, interpolated in ln sigma_z; NaN off the grid."""
        grid = np.log(self.sigma_z)
        if not grid[0] <= np.log(sigma_z) <= grid[-1]:
            return float("nan")
        return float(np.interp(np.log(sigma_z), grid, self.b_isothermal))


def coeval_curve(model: CoevalModel, sigma_z: FloatArray, b: float, k: float) -> CoevalCurve:
    """Return the coeval population of *model* with exponent *b* at every dispersion of *sigma_z*.

    h_Z is measured in the self-consistent window L = *k* h_Z, as for the stars.
    """
    profiles = [model.profile(float(s), b) for s in sigma_z]
    return CoevalCurve(
        b=b,
        sigma_z=[float(s) for s in sigma_z],
        hz=[scaled_profile_hz(rho, k, model.z_grid)[0] for rho in profiles],
        half_mass=[profile_half_mass(rho, model.z_grid) for rho in profiles],
        b_isothermal=[isothermal_action_shape(model, float(s)).b for s in sigma_z],
    )


@dataclass(frozen=True)
class ActionShape:
    """b and J_0 (kpc km/s) of exp[-(J_z / J_0)^(2 - b)] fitted to the vertical actions of a population."""

    b: float
    j0: float


def fit_action_shape(jz: FloatArray, weight: FloatArray) -> ActionShape:
    """Return the b and J_0 that maximize the likelihood of the actions *jz*, each weighted by *weight*.

    With q = 2 - b, the normalized distribution is q exp[-(J / J_0)^q] / (J_0 Gamma(1 / q)).
    q is kept within ``SHAPE_BOUNDS``. Actions that are not finite and positive are left out.
    """
    keep = np.isfinite(jz) & (jz > 0.0)
    j, w = jz[keep], weight[keep] / np.sum(weight[keep])
    scale = float(np.sum(w * j))

    def negative_log_likelihood(params: FloatArray) -> float:
        log_j0, q = float(params[0]), float(params[1])
        return float(-np.sum(w * (np.log(q) - log_j0 - math.lgamma(1.0 / q) - (j / np.exp(log_j0)) ** q)))

    result = minimize(
        negative_log_likelihood,
        np.array([np.log(scale), 1.0]),
        method="L-BFGS-B",
        bounds=[(np.log(scale) - 5.0, np.log(scale) + 5.0), SHAPE_BOUNDS],
    )
    return ActionShape(b=2.0 - float(result.x[1]), j0=float(np.exp(result.x[0])))


def isothermal_action_shape(model: CoevalModel, sigma_z: float) -> ActionShape:
    """Return b and J_0 of exp[-(J_z / J_0)^(2 - b)] fitted to the actions of the isothermal tracer of *sigma_z*.

    The isothermal population has p(J_z) ~ exp[-E(J_z) / sigma_z^2], which is exponential in
    J_z, b = 1, only where the potential is harmonic: in a potential ~ |z|^p, E ~ J_z^(2p/(p+2)),
    so that b = 2 - 2p/(p+2), which is 4/3 for a linear potential. Each orbit of the grid of
    *model* enters with the weight p(J_z) dJ_z, so the fit is the maximum-likelihood fit to the
    whole distribution. A population that reaches the edge of the grid (:meth:`CoevalModel.reaches_edge`)
    has no fit, because the orbits beyond the grid would be left out.
    """
    if model.reaches_edge(sigma_z):
        return ActionShape(float("nan"), float("nan"))
    weight = np.exp(-model.phi_z / sigma_z**2) * np.gradient(model.actions)
    return fit_action_shape(model.actions, weight)
