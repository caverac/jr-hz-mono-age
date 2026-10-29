"""Vertical scale heights: the windowed exponential fit, the self-consistent window, and their predictions.

Palicio et al. (2025, A&A 695, A193, Sect. 3.4) fit h_Z as the average of exponential
fits within |Z| < Z_lim and |Z| < Z_lim / 2. Each fit is the maximum-likelihood scale of
an exponential truncated at the window L, whose mean height is

    <z> = h - L / (exp(L / h) - 1),

solved for h. A fixed Z_lim measures the window rather than the population when the
population is thicker than the window. The self-consistent window instead fits every
population inside L = k h(L), the smallest root of that equation, so that populations of
one profile shape are fitted over the same part of their profiles: an exponential
returns its scale height at any k.

Two predictions of h_Z come from the midplane dispersion sigma_z:

- the self-gravitating isothermal sheet, h_Z = sigma_z^2 / (pi G Sigma), with Sigma the
  surface density of all the mass near the plane: the z_0 of its profile sech^2(|z| / z_0),
  which the window L = 2 h_Z returns, so that the scale is compared with the windowed fit;
- the isothermal tracer of the total potential, rho(z) ~ exp(-Delta Phi(z) / sigma_z^2),
  whose profile goes through the same window as the particles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]

G_PC = 4.30091e-3  # gravitational constant, pc Msun^-1 (km/s)^2
Z_GRID = np.linspace(0.0, 12.0, 2401)  # kpc, the heights of a profile
WINDOW_GRID = np.geomspace(0.05, 12.0, 40)  # kpc, the trial windows of the self-consistent search
BISECTION_STEPS = 30
N_MIN_HALF_WINDOW = 20  # particles inside L / 2 for a windowed fit
SCALE_BRACKET_KPC = (1.0e-4, 1.0e3)
MAX_EXPONENT = 700.0  # keeps exp(L / h) finite for the smallest trial scales


def truncated_exp_scale(mean_z: float, window: float) -> float:
    """Return the scale h of an exponential on [0, *window*] with mean *mean_z*; NaN without a root.

    There is no root in ``SCALE_BRACKET_KPC`` when the profile is nearly flat inside the
    window (mean_z close to window / 2) or concentrated at z = 0.
    """

    def residual(scale: float) -> float:
        return float(scale - window / np.expm1(min(window / scale, MAX_EXPONENT)) - mean_z)

    lo, hi = SCALE_BRACKET_KPC
    if not residual(lo) < 0.0 < residual(hi):
        return float("nan")
    return float(brentq(residual, lo, hi))


def particle_hz(absz: FloatArray, weight: FloatArray, window: float) -> float:
    """Return the windowed h_Z of weighted particle heights *absz*; NaN if the window cannot measure it.

    The window cannot measure h_Z when fewer than ``N_MIN_HALF_WINDOW`` particles lie
    inside window / 2, or when a truncated fit has no root.
    """
    if np.count_nonzero(absz <= 0.5 * window) < N_MIN_HALF_WINDOW:
        return float("nan")
    scales = []
    for limit in (window, 0.5 * window):
        inside = absz <= limit
        mean_z = float(np.sum(weight[inside] * absz[inside]) / np.sum(weight[inside]))
        scales.append(truncated_exp_scale(mean_z, limit))
    return float(np.mean(scales))


def _profile_mean(rho: FloatArray, limit: float, z_grid: FloatArray) -> float:
    """Return the mean height of the profile *rho* on *z_grid* over [0, *limit*].

    The integrals end exactly at *limit*, where the profile is interpolated, because the
    truncated exponential that the mean is matched to ends there: stopping at the last
    grid point below *limit* biases the fitted scale low by up to a few percent.
    """
    if limit > z_grid[-1]:
        raise ValueError(f"the window {limit} kpc exceeds the profile grid, {z_grid[-1]} kpc")
    inside = z_grid < limit
    z = np.append(z_grid[inside], limit)
    density = np.append(rho[inside], np.interp(limit, z_grid, rho))
    return float(np.trapezoid(z * density, z) / np.trapezoid(density, z))


def profile_hz(rho: FloatArray, window: float, z_grid: FloatArray = Z_GRID) -> float:
    """Return the windowed h_Z of the density profile *rho* on *z_grid*; NaN if a truncated fit has no root."""
    return float(
        np.mean([truncated_exp_scale(_profile_mean(rho, limit, z_grid), limit) for limit in (window, 0.5 * window)])
    )


def exponential_profile_hz(rho: FloatArray, limit: float, z_grid: FloatArray = Z_GRID) -> float:
    """Return the scale of the exponential on [0, *limit*] that maximum likelihood fits to the profile *rho*.

    This is the fit of a single exponential without a window, which matches the mean height
    of the profile below *limit*; for a profile that is not exponential it differs from the
    windowed h_Z.
    """
    return truncated_exp_scale(_profile_mean(rho, limit, z_grid), limit)


def self_consistent_window(hz_of_window: Callable[[float], float], k: float) -> tuple[float, float]:
    """Return h_Z and the window L at the smallest root of L = k h(L); NaN for both if none on ``WINDOW_GRID``.

    g(L) = L - k h(L) is negative in windows too small to see the profile fall (h is NaN
    or large) and positive in wide ones. The first sign change on ``WINDOW_GRID``
    brackets the root, which ``BISECTION_STEPS`` geometric bisections refine, with NaN
    counted as negative.
    """

    def g(window: float) -> float:
        scale = hz_of_window(window)
        return window - k * scale if np.isfinite(scale) else -np.inf

    first = next((i for i, w in enumerate(WINDOW_GRID) if g(float(w)) >= 0.0), None)
    if first is None or first == 0:
        return float("nan"), float("nan")
    lo, hi = float(WINDOW_GRID[first - 1]), float(WINDOW_GRID[first])
    for _ in range(BISECTION_STEPS):
        mid = float(np.sqrt(lo * hi))
        if g(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    return hz_of_window(hi), hi


def scaled_particle_hz(absz: FloatArray, weight: FloatArray, k: float) -> tuple[float, float]:
    """Return h_Z and the window of weighted particle heights, with the window at k h_Z."""
    return self_consistent_window(lambda window: particle_hz(absz, weight, window), k)


def tracer_profile(phi_z: FloatArray, sigma_z: float) -> FloatArray:
    """Return the isothermal tracer exp(-Delta Phi / sigma_z^2) of the vertical potential *phi_z* ((km/s)^2)."""
    return np.asarray(np.exp(-phi_z / sigma_z**2), dtype=np.float64)


def scaled_profile_hz(rho: FloatArray, k: float, z_grid: FloatArray = Z_GRID) -> tuple[float, float]:
    """Return h_Z and the window of the density profile *rho* on *z_grid*, with the window at k h_Z."""
    return self_consistent_window(lambda window: profile_hz(rho, window, z_grid), k)


def sheet_hz(sigma_z: float, sigma_surf: float) -> float:
    """Return sigma_z^2 / (pi G Sigma) in kpc, for sigma_z in km/s and Sigma in Msun/pc^2."""
    return float(sigma_z**2 / (np.pi * G_PC * sigma_surf) / 1.0e3)


def sheet_profile(sigma_z: float, sigma_surf: float, z_grid: FloatArray = Z_GRID) -> FloatArray:
    """Return the self-gravitating isothermal sheet sech^2(|z| / z_0) on *z_grid*, with z_0 = :func:`sheet_hz`.

    Its h_Z in the self-consistent window L = 2 h_Z is z_0 to 0.2 percent, so at k = 2 the
    profile and the scale are the same prediction. sech^2 x = 4 e^{-2x} / (1 + e^{-2x})^2,
    which does not overflow for thin sheets.
    """
    decay = np.exp(-2.0 * np.abs(z_grid) / sheet_hz(sigma_z, sigma_surf))
    return np.asarray(4.0 * decay / (1.0 + decay) ** 2, dtype=np.float64)
