"""Tests of the vertical Jeans equation with a measured sigma_z(z) and tilt term."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.heights import Z_GRID, scaled_profile_hz, tracer_profile
from experiments.jeans import (
    JEANS_EDGES_KPC,
    TILT_R_MAX_KPC,
    TILT_R_MIN_KPC,
    JeansProfile,
    JeansStars,
    jeans_density,
    jeans_profile,
    slice_tilt,
    tilt_profile,
)

PHI = 900.0 * Z_GRID + 2000.0 * Z_GRID**2  # (km/s)^2, harmonic near the plane and steeper above


def flat(sigma2: float, tilt: float = 0.0) -> JeansProfile:
    """Return a profile with a constant sigma_z^2 and a tilt term t = *tilt* z in every slice."""
    z = tuple(0.5 * (lo + hi) for lo, hi in zip(JEANS_EDGES_KPC[:-1], JEANS_EDGES_KPC[1:]))
    return JeansProfile(z, (sigma2,) * len(z), tuple(tilt * v for v in z))


def test_isothermal_untilted_population_is_the_tracer() -> None:
    """A constant sigma_z and no tilt term give exp(-Delta Phi / sigma_z^2)."""
    nu = jeans_density(PHI, flat(400.0), Z_GRID, with_tilt=True)
    np.testing.assert_allclose(nu, tracer_profile(PHI, 20.0), rtol=1e-3, atol=1e-12)


def test_a_mixture_of_two_tracers_is_recovered_from_its_sigma_z_profile() -> None:
    """Two isothermal tracers satisfy the Jeans equation together, so their sigma_z(z) gives back their sum."""
    cold, hot = tracer_profile(PHI, 15.0), tracer_profile(PHI, 35.0)
    mixture = 0.7 * cold + 0.3 * hot
    sigma2 = (0.7 * 15.0**2 * cold + 0.3 * 35.0**2 * hot) / mixture
    fine = Z_GRID <= 3.0
    profile = JeansProfile(tuple(Z_GRID[fine]), tuple(sigma2[fine]), (0.0,) * int(fine.sum()))
    nu = jeans_density(PHI, profile, Z_GRID, with_tilt=False)
    np.testing.assert_allclose(nu[fine], mixture[fine], rtol=2e-3)


def test_a_positive_tilt_thins_and_counts_only_with_the_tilt() -> None:
    """A positive tilt term adds to the pull of the potential; without the tilt, the profile is the tracer."""
    tracer = scaled_profile_hz(tracer_profile(PHI, 20.0), 2.0)[0]
    tilted = jeans_density(PHI, flat(400.0, tilt=2000.0), Z_GRID, with_tilt=True)
    assert scaled_profile_hz(tilted, 2.0)[0] < 0.95 * tracer
    untilted = jeans_density(PHI, flat(400.0, tilt=2000.0), Z_GRID, with_tilt=False)
    assert scaled_profile_hz(untilted, 2.0)[0] == pytest.approx(tracer, rel=1e-3)


def test_no_slices_give_no_density() -> None:
    """Without a measured slice there is nothing to integrate."""
    assert np.all(np.isnan(jeans_density(PHI, JeansProfile((), (), ()), Z_GRID, with_tilt=True)))


def sample(n: int, alpha: float, rng: np.random.Generator) -> JeansStars:
    """Return stars of the range of the tilt term with nu ~ exp(-R / 3) exp(-|z| / 0.5), sigma_z = 20 + 10 |z|.

    Their sigma_Rz = alpha z does not depend on R, so that t(z) = alpha z (1 / R - 1 / 3).
    """
    r = np.sqrt(rng.uniform(TILT_R_MIN_KPC**2, TILT_R_MAX_KPC**2, 4 * n))
    r = r[rng.uniform(size=r.size) < np.exp(-(r - TILT_R_MIN_KPC) / 3.0)][:n]
    z = rng.exponential(0.5, r.size) * rng.choice([-1.0, 1.0], r.size)
    sigma_z = 20.0 + 10.0 * np.abs(z)
    v_z = rng.normal(0.0, 1.0, r.size) * sigma_z
    v_r = alpha * z * v_z / sigma_z**2 + rng.normal(0.0, 30.0, r.size)
    return JeansStars(r, z, v_r, v_z, np.ones(r.size))


def test_slice_tilt_is_exact_for_an_exponential_density_and_a_linear_sigma_rz() -> None:
    """With nu ~ exp(-R / 3) and sigma_Rz = 40 + 10 R, t = sigma_Rz (1 / R - 1 / 3) + 10 at any R_mid."""
    radius = np.linspace(6.25, 10.25, 9)
    r_mid = 8.2
    tilt = slice_tilt(radius, 5.0 * np.exp(-radius / 3.0), 40.0 + 10.0 * radius, r_mid)
    assert tilt == pytest.approx((40.0 + 10.0 * r_mid) * (1.0 / r_mid - 1.0 / 3.0) + 10.0)


def test_profile_recovers_sigma_z_and_the_tilt_term() -> None:
    """sigma_z^2 follows (20 + 10 |z|)^2 in the annulus, and t(z) = alpha z (1 / R_mid - 1 / 3) near the plane."""
    alpha, r_mid = 300.0, 8.25
    profile = jeans_profile(sample(2_000_000, alpha, np.random.default_rng(3)), 7.5, 9.0, r_mid)
    z, tilt = np.array(profile.z), np.array(profile.tilt)
    assert z.size >= 10
    np.testing.assert_allclose(np.sqrt(profile.sigma2), 20.0 + 10.0 * z, rtol=0.05)
    near = z < 1.0
    np.testing.assert_allclose(tilt[near], alpha * z[near] * (1.0 / r_mid - 1.0 / 3.0), rtol=0.1, atol=15.0)


def test_stars_of_the_annulus_alone_leave_no_tilt() -> None:
    """Without the stars of the rings outside the annulus, no slice has a tilt term."""
    stars = sample(400_000, 300.0, np.random.default_rng(4))
    annulus = stars.select((stars.r >= 7.5) & (stars.r < 9.0))
    profile = jeans_profile(annulus, 7.5, 9.0, 8.25)
    assert profile.z and np.all(np.isnan(profile.tilt))


def test_profile_uses_the_full_second_moment() -> None:
    """A heavy-tailed population keeps its tails: sigma_z^2 is the plain mass-weighted variance, not a clipped one."""
    rng = np.random.default_rng(9)
    n = 400_000
    r = rng.uniform(7.5, 9.0, n)
    z = rng.uniform(-0.19, 0.19, n)
    hot = rng.uniform(size=n) < 0.2
    v_z = np.where(hot, rng.normal(0.0, 80.0, n), rng.normal(0.0, 15.0, n))
    profile = jeans_profile(JeansStars(r, z, np.zeros(n), v_z, np.ones(n)), 7.5, 9.0, 8.25)
    assert profile.sigma2[0] == pytest.approx(0.2 * 80.0**2 + 0.8 * 15.0**2, rel=0.02)


def test_sparse_slices_end_the_profile_and_sparse_rings_leave_no_tilt() -> None:
    """The profile stops at the first slice of too few stars; a slice with a sparse ring has a NaN tilt."""
    stars = sample(20_000, 0.0, np.random.default_rng(5))
    profile = jeans_profile(stars, 7.5, 9.0, 8.25)
    n_slices = len(profile.z)
    assert 0 < n_slices < len(JEANS_EDGES_KPC) - 1
    assert np.isfinite(profile.tilt[0]) and np.isnan(profile.tilt[-1])
    assert np.all(np.isnan(tilt_profile(stars, 8.25)[n_slices:]))
