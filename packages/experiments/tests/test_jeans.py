"""Tests of the vertical Jeans equation with a measured sigma_z(z) and tilt term."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.heights import Z_GRID, scaled_profile_hz, tracer_profile
from experiments.jeans import (
    JEANS_EDGES_KPC,
    JeansProfile,
    JeansStars,
    jeans_density,
    jeans_profile,
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
    """Return stars with nu ~ exp(-R / 3) exp(-|z| / 0.5), sigma_z = 20 + 10 |z| and sigma_Rz = alpha z."""
    r = np.sqrt(rng.uniform(7.5**2, 9.0**2, 4 * n))
    r = r[rng.uniform(size=r.size) < np.exp(-(r - 7.5) / 3.0)][:n]
    z = rng.exponential(0.5, r.size) * rng.choice([-1.0, 1.0], r.size)
    sigma_z = 20.0 + 10.0 * np.abs(z)
    v_z = rng.normal(0.0, 1.0, r.size) * sigma_z
    v_r = alpha * z * v_z / sigma_z**2 + rng.normal(0.0, 30.0, r.size)
    return JeansStars(r, z, v_r, v_z, np.ones(r.size))


def test_profile_recovers_sigma_z_and_the_tilt_term() -> None:
    """sigma_z^2 follows (20 + 10 |z|)^2, and t(z) rises as alpha z (1 / R - 1 / R_d) at R_mid within 20 percent."""
    alpha, r_mid = 300.0, 8.25
    profile = jeans_profile(sample(2_000_000, alpha, np.random.default_rng(3)), 7.5, 9.0, r_mid)
    z, tilt = np.array(profile.z), np.array(profile.tilt)
    assert z.size >= 6
    np.testing.assert_allclose(np.sqrt(profile.sigma2), 20.0 + 10.0 * z, rtol=0.05)
    inner = z < 1.5
    slope = float(np.sum(tilt[inner] * z[inner]) / np.sum(z[inner] ** 2))
    assert slope == pytest.approx(alpha * (1.0 / r_mid - 1.0 / 3.0), rel=0.2)


def test_sparse_slices_end_the_profile_and_sparse_rings_leave_no_tilt() -> None:
    """The profile stops at the first slice of too few stars; a slice whose radial bin is sparse has a NaN tilt."""
    stars = sample(3_000, 0.0, np.random.default_rng(5))
    profile = jeans_profile(stars, 7.5, 9.0, 8.25)
    assert 0 < len(profile.z) < len(JEANS_EDGES_KPC) - 1
    assert np.isnan(profile.tilt).any()
