"""Tests of the ages of star particles from their formation scale factors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from experiments.cosmology import GYR_PER_MPC_KMS, Cosmology

PLANCK = Cosmology(hubble=0.6777, omega_m=0.307, omega_lambda=0.693)


@pytest.mark.parametrize("scale_factor", [1.0e-8, 0.3, 0.9])
def test_lookback_matches_the_integral_of_the_hubble_time(scale_factor: float) -> None:
    """The closed form equals the integral of da / (a H(a)) with H = H_0 sqrt(Omega_m a^-3 + Omega_L)."""
    hubble_time = GYR_PER_MPC_KMS / (100.0 * PLANCK.hubble)

    def dt_da(a: float) -> float:
        return float(hubble_time / (a * np.sqrt(PLANCK.omega_m / a**3 + PLANCK.omega_lambda)))

    expected, _ = quad(dt_da, scale_factor, 1.0, epsabs=1e-12, epsrel=1e-12)
    assert PLANCK.lookback_gyr(np.array([scale_factor]), 1.0)[0] == pytest.approx(expected, rel=1e-9)


def test_age_of_the_universe() -> None:
    """These parameters, without radiation, give a universe 13.82 Gyr old."""
    assert PLANCK.lookback_gyr(np.array([1.0e-12]), 1.0)[0] == pytest.approx(13.8205, abs=1e-4)


def test_lookback_is_zero_now_and_grows_into_the_past() -> None:
    """The lookback time is zero at a = now and grows as a decreases."""
    lookback = PLANCK.lookback_gyr(np.array([1.0, 0.8, 0.5, 0.2]), 1.0)
    assert lookback[0] == 0.0
    assert np.all(np.diff(lookback) > 0.0)


def test_scale_factor_at_lookback_inverts_lookback() -> None:
    """``scale_factor_at_lookback`` returns the scale factor whose lookback time was given."""
    ages = np.linspace(0.0, 13.0, 27)
    for now in (1.0, 0.7):
        back = PLANCK.lookback_gyr(PLANCK.scale_factor_at_lookback(ages * now, now), now)
        np.testing.assert_allclose(back, ages * now, atol=1e-10)
