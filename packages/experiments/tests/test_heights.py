"""Tests of the windowed scale heights, the self-consistent window and the predictions.

The profile checks are the synthetic checks the method was designed against: an
exponential returns its scale height in any self-consistent window, a sech^2 layer returns
a fixed multiple of its z_0 at each k whatever z_0, and the tracer of a linear potential
is an exponential of scale sigma_z^2 / g.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.heights import (
    G_PC,
    N_MIN_HALF_WINDOW,
    Z_GRID,
    exponential_profile_hz,
    half_mass_height,
    particle_hz,
    profile_half_mass,
    profile_hz,
    scaled_particle_hz,
    scaled_profile_hz,
    self_consistent_window,
    sheet_hz,
    sheet_profile,
    tracer_curve,
    tracer_profile,
    truncated_exp_scale,
)


def truncated_mean(scale: float, window: float) -> float:
    """Return the mean of an exponential of *scale* truncated at *window*."""
    return float(scale - window / np.expm1(window / scale))


@pytest.mark.parametrize(("scale", "window"), [(0.3, 1.0), (1.0, 1.0), (2.0, 0.5), (0.05, 2.0)])
def test_truncated_exp_scale_inverts_the_mean(scale: float, window: float) -> None:
    """The scale with the mean of a truncated exponential is that exponential's scale."""
    assert truncated_exp_scale(truncated_mean(scale, window), window) == pytest.approx(scale, rel=1e-9)


@pytest.mark.parametrize("mean_z", [0.5, 0.6, 5.0e-5])
def test_truncated_exp_scale_without_a_root_is_nan(mean_z: float) -> None:
    """A flat or rising profile (mean >= window / 2) and one concentrated at z = 0 have no decaying scale."""
    assert np.isnan(truncated_exp_scale(mean_z, 1.0))


def test_particle_hz_recovers_an_exponential_sample() -> None:
    """The windowed fit of an exponential sample returns its scale height within its statistical error."""
    absz = np.random.default_rng(1).exponential(0.4, 200_000)
    assert particle_hz(absz, np.ones_like(absz), 1.0) == pytest.approx(0.4, rel=0.01)


def test_particle_hz_weights_the_heights() -> None:
    """Doubling the weight of every particle above 0.2 kpc thickens the fitted layer."""
    absz = np.random.default_rng(2).exponential(0.4, 50_000)
    weight = np.where(absz > 0.2, 2.0, 1.0)
    assert particle_hz(absz, weight, 1.0) > particle_hz(absz, np.ones_like(absz), 1.0)


def test_particle_hz_needs_particles_in_the_inner_half() -> None:
    """With fewer than ``N_MIN_HALF_WINDOW`` particles inside window / 2 there is no fit."""
    absz = np.concatenate([np.full(N_MIN_HALF_WINDOW - 1, 0.1), np.full(100, 0.8)])
    assert np.isnan(particle_hz(absz, np.ones_like(absz), 1.0))


@pytest.mark.parametrize("scale", [0.3, 1.0])
def test_profile_hz_of_an_exponential_profile(scale: float) -> None:
    """The windowed fit of an exponential profile on the grid returns its scale to 0.3 percent."""
    assert profile_hz(np.exp(-Z_GRID / scale), 1.5) == pytest.approx(scale, rel=0.003)


@pytest.mark.parametrize("scale", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("k", [1.5, 2.0, 3.0])
def test_self_consistent_window_returns_an_exponential_scale(scale: float, k: float) -> None:
    """In the window k h_Z an exponential returns its scale height to 0.1 percent, whatever k."""
    hz, window = scaled_profile_hz(np.exp(-Z_GRID / scale), k)
    assert hz == pytest.approx(scale, rel=0.001)
    assert window == pytest.approx(k * hz, rel=1e-6)


@pytest.mark.parametrize(("k", "ratio"), [(1.5, 1.114), (2.0, 0.998), (3.0, 0.872)])
def test_self_consistent_window_scales_a_sech2_layer(k: float, ratio: float) -> None:
    """A sech^2(z / z_0) layer returns h_Z / z_0 fixed by k alone, whatever z_0, and h_Z = z_0 at k = 2."""
    ratios = [scaled_profile_hz(np.cosh(Z_GRID / z0) ** -2, k)[0] / z0 for z0 in (0.3, 0.6, 1.2)]
    assert ratios == pytest.approx([ratio] * 3, abs=0.001)
    assert max(ratios) - min(ratios) < 5.0e-4


def test_profile_hz_ends_the_integral_at_the_window() -> None:
    """A window off the grid steps measures an exponential as well as one on them."""
    rho = np.exp(-Z_GRID / 0.4)
    assert profile_hz(rho, 0.7832) == pytest.approx(profile_hz(rho, 0.78), rel=1e-3)
    assert profile_hz(rho, 0.7832) == pytest.approx(0.4, rel=1e-3)


def test_profile_hz_needs_the_window_on_the_grid() -> None:
    """A window beyond the last height of the grid is an error."""
    with pytest.raises(ValueError, match="exceeds the profile grid"):
        profile_hz(np.exp(-Z_GRID), Z_GRID[-1] + 1.0)


def test_self_consistent_window_on_particles() -> None:
    """The self-consistent window of an exponential sample returns its scale height."""
    absz = np.random.default_rng(3).exponential(0.7, 100_000)
    hz, window = scaled_particle_hz(absz, np.ones_like(absz), 2.0)
    assert hz == pytest.approx(0.7, rel=0.02)
    assert window == pytest.approx(2.0 * hz, rel=1e-6)


def test_self_consistent_window_without_a_root() -> None:
    """No window when h is never finite, nor when the first trial window already exceeds k h."""
    assert all(np.isnan(self_consistent_window(lambda _window: float("nan"), 2.0)))
    assert all(np.isnan(self_consistent_window(lambda _window: 0.0, 2.0)))


def test_tracer_of_a_linear_potential_is_an_exponential() -> None:
    """For Delta Phi = g |z| the tracer is exp(-|z| sigma_z^-2 g), of scale sigma_z^2 / g."""
    sigma_z, g = 20.0, 1000.0  # km/s and (km/s)^2 / kpc
    rho = tracer_profile(g * Z_GRID, sigma_z)
    np.testing.assert_allclose(rho, np.exp(-Z_GRID * g / sigma_z**2))
    assert scaled_profile_hz(rho, 2.0)[0] == pytest.approx(sigma_z**2 / g, rel=0.001)


def test_sheet_hz() -> None:
    """The sheet height is sigma_z^2 / (pi G Sigma), converted from pc to kpc."""
    assert sheet_hz(20.0, 50.0) == pytest.approx(400.0 / (np.pi * G_PC * 50.0) / 1.0e3)
    assert sheet_hz(20.0, 50.0) == pytest.approx(0.5921, abs=1e-4)


@pytest.mark.parametrize("sigma_z", [10.0, 15.0, 25.0])
def test_the_sheet_profile_returns_its_scale_in_the_window(sigma_z: float) -> None:
    """At k = 2 the windowed sech^2 layer of the sheet returns sigma_z^2 / (pi G Sigma); a thin sheet stays finite."""
    assert scaled_profile_hz(sheet_profile(sigma_z, 50.0), 2.0)[0] == pytest.approx(sheet_hz(sigma_z, 50.0), rel=0.003)
    thin = sheet_profile(1.0, 500.0)
    assert np.all(np.isfinite(thin)) and thin[0] == pytest.approx(1.0) and thin[-1] == 0.0


@pytest.mark.parametrize(("limit", "ratio"), [(0.6, 0.923), (1.2, 0.730), (3.0, 0.694)])
def test_the_exponential_fit_of_a_profile(limit: float, ratio: float) -> None:
    """An exponential returns its scale at any limit, whereas a sech^2 layer, flat at the plane, comes out thinner.

    Its windowed h_Z at k = 2 is z_0 (test above). The exponential matched to its mean height
    below *limit* thins as the limit grows past the flat core, by *ratio* of the windowed h_Z.
    """
    assert exponential_profile_hz(np.exp(-Z_GRID / 0.3), limit) == pytest.approx(0.3, rel=1e-3)
    sech2 = np.cosh(Z_GRID / 0.4) ** -2
    assert exponential_profile_hz(sech2, limit) / scaled_profile_hz(sech2, 2.0)[0] == pytest.approx(ratio, abs=0.002)


def test_half_mass_height_weights_the_heights() -> None:
    """Half of the weight lies below the weighted median of the heights."""
    absz = np.array([1.0, 2.0, 3.0, 4.0])
    assert half_mass_height(absz, np.ones(4)) == 2.0
    assert half_mass_height(absz, np.array([1.0, 1.0, 1.0, 5.0])) == 4.0


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0])
def test_profile_half_mass_of_an_exponential(scale: float) -> None:
    """An exponential well inside the grid holds half of its mass below scale ln 2."""
    assert profile_half_mass(np.exp(-Z_GRID / scale)) == pytest.approx(scale * np.log(2.0), rel=1e-3)


def test_tracer_curve_slopes_of_a_harmonic_and_a_linear_potential() -> None:
    """The slope eta is 1 in a harmonic potential and 2 in a linear one; a profile without a scale has none."""
    sigma_z = np.geomspace(10.0, 40.0, 9)
    harmonic = tracer_curve(0.5 * 2500.0 * Z_GRID**2, sigma_z, 2.0)
    linear = tracer_curve(1000.0 * Z_GRID, sigma_z, 2.0)
    np.testing.assert_allclose(harmonic.eta, 1.0, atol=0.01)
    np.testing.assert_allclose(linear.eta, 2.0, atol=0.01)
    # The thinnest profile, 0.1 kpc, spans 20 steps of the 5 pc grid, which resolves it to 0.4 percent.
    np.testing.assert_allclose(linear.hz, sigma_z**2 / 1000.0, rtol=0.004)
    assert linear.half_mass == pytest.approx(list(np.log(2.0) * sigma_z**2 / 1000.0), rel=0.004)
    flat = tracer_curve(10.0 * np.log1p(Z_GRID**2), np.array([5.0, 50.0, 500.0]), 2.0)
    assert np.isfinite(flat.hz[0]) and np.isnan(flat.hz[-1]) and np.isnan(flat.eta[-1])
