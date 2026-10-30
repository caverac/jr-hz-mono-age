"""Tests of the vertical profiles of coeval populations (experiments.coeval)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.coeval import (
    THIN_LAYER_B,
    CoevalModel,
    coeval_curve,
    fit_action_shape,
    isothermal_action_shape,
    orbit_actions,
)
from experiments.heights import Z_GRID, profile_half_mass, scaled_profile_hz, tracer_profile

NU2 = 2500.0  # (km/s / kpc)^2, the square of the vertical frequency of the harmonic potential
HARMONIC = 0.5 * NU2 * Z_GRID**2


@pytest.fixture(name="harmonic", scope="module")
def harmonic_model() -> CoevalModel:
    """Return the coeval model of the harmonic potential."""
    return CoevalModel(HARMONIC)


def test_orbit_actions_of_a_harmonic_potential() -> None:
    """In a harmonic potential J_z = E / nu, within the error of the grid at the thinnest orbits."""
    actions = orbit_actions(HARMONIC)
    expected = HARMONIC / np.sqrt(NU2)
    # An orbit that turns 0.1 kpc above the plane spans 20 steps of the 5 pc grid.
    np.testing.assert_allclose(actions[20:], expected[20:], rtol=0.005)
    assert actions[0] == 0.0


def test_exponential_actions_in_a_harmonic_potential_are_isothermal(harmonic: CoevalModel) -> None:
    """With b = 1 the profile is the isothermal tracer, a Gaussian of dispersion sigma_z / nu."""
    rho = harmonic.profile(20.0, 1.0)
    iso = tracer_profile(HARMONIC, 20.0)
    np.testing.assert_allclose((rho / rho[0])[:200], iso[:200], atol=1.0e-4)
    assert harmonic.midplane_dispersion(400.0 / np.sqrt(NU2), 1.0) == pytest.approx(20.0, rel=1.0e-4)


def test_a_thin_layer_population_is_thicker_in_the_window(harmonic: CoevalModel) -> None:
    """At the same midplane dispersion, b = 1/2 has a flat-topped v_z and a wider core than the isothermal tracer."""
    rho = harmonic.profile(20.0, THIN_LAYER_B)
    iso = tracer_profile(HARMONIC, 20.0)
    assert scaled_profile_hz(rho, 2.0)[0] > 1.05 * scaled_profile_hz(iso, 2.0)[0]
    assert profile_half_mass(rho) > profile_half_mass(iso)


def test_coeval_curve_interpolates_in_log_log(harmonic: CoevalModel) -> None:
    """With b = 1 the curve is the harmonic tracer, h_Z proportional to sigma_z, and NaN off its grid."""
    curve = coeval_curve(harmonic, np.array([10.0, 20.0, 40.0]), 1.0, 2.0)
    hz, half_mass = curve.at(28.0)
    assert hz == pytest.approx(curve.hz[1] * 28.0 / 20.0, rel=1.0e-3)
    assert half_mass == pytest.approx(curve.half_mass[1] * 28.0 / 20.0, rel=1.0e-3)
    assert curve.at(10.0)[0] == pytest.approx(curve.hz[0])
    assert curve.at(40.0)[0] == pytest.approx(curve.hz[2])
    assert all(np.isnan(v) for v in (*curve.at(9.0), *curve.at(41.0), *curve.at(float("nan"))))


def test_isothermal_actions_are_exponential_only_in_a_harmonic_potential(harmonic: CoevalModel) -> None:
    """The isothermal tracer has b = 1 and J_0 = sigma^2 / nu in a harmonic potential, and b = 4/3 in a linear one."""
    shape = isothermal_action_shape(harmonic, 20.0)
    assert shape.b == pytest.approx(1.0, abs=1e-3) and shape.j0 == pytest.approx(400.0 / np.sqrt(NU2), rel=1e-3)
    assert isothermal_action_shape(CoevalModel(1000.0 * Z_GRID), 20.0).b == pytest.approx(4.0 / 3.0, abs=0.01)
    curve = coeval_curve(harmonic, np.array([10.0, 20.0]), THIN_LAYER_B, 2.0)
    assert curve.isothermal_b(15.0) == pytest.approx(1.0, abs=1e-3) and np.isnan(curve.isothermal_b(30.0))


def test_a_population_that_reaches_the_edge_of_the_grid_has_no_profile_and_no_shape() -> None:
    """A population whose isothermal density at the edge exceeds EDGE_DENSITY has neither a profile nor a b."""
    shallow = CoevalModel(10.0 * Z_GRID)
    assert shallow.reaches_edge(5.0)
    assert np.all(np.isnan(shallow.profile(5.0, THIN_LAYER_B)))
    assert np.isnan(isothermal_action_shape(shallow, 5.0).b)


def test_a_dispersion_that_no_j0_reaches_has_no_profile(harmonic: CoevalModel) -> None:
    """Below the dispersion of the smallest J_0 searched, the profile is NaN."""
    assert np.all(np.isnan(harmonic.profile(1.0e-3, THIN_LAYER_B)))


@pytest.mark.parametrize("b", [1.0, THIN_LAYER_B])
def test_fit_action_shape_recovers_b(b: float) -> None:
    """Actions drawn from exp[-(J / J_0)^(2 - b)] give back b and J_0; others are ignored."""
    rng = np.random.default_rng(3)
    q = 2.0 - b
    jz = 5.0 * rng.gamma(1.0 / q, 1.0, 20000) ** (1.0 / q)
    shape = fit_action_shape(np.append(jz, [np.nan, -1.0]), np.ones(jz.size + 2))
    assert shape.b == pytest.approx(b, abs=0.05)
    assert shape.j0 == pytest.approx(5.0, rel=0.05)


def test_fit_action_shape_weights_the_stars() -> None:
    """Weights act as multiplicities: doubling the weight of a subset equals repeating it."""
    rng = np.random.default_rng(4)
    jz = rng.exponential(3.0, 2000)
    weight = np.where(jz > 3.0, 2.0, 1.0)
    repeated = np.concatenate([jz, jz[jz > 3.0]])
    weighted, plain = fit_action_shape(jz, weight), fit_action_shape(repeated, np.ones(repeated.size))
    assert weighted.b == pytest.approx(plain.b, abs=1.0e-4)
    assert weighted.j0 == pytest.approx(plain.j0, rel=1.0e-4)
