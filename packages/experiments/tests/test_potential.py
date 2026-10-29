"""Tests of the AGAMA potential fit, the radial actions and the checks against the snapshot potential.

The galaxy is sampled in memory from the mock's known disk and halo, so the fit can be
compared with the potential it came from.
"""

from __future__ import annotations

import re

import agama
import numpy as np
import pytest

from experiments.cosmology import Cosmology
from experiments.heights import Z_GRID
from experiments.mock import true_components
from experiments.potential import (
    RESIDUAL_MIN_PER_SECTOR,
    RESIDUAL_SECTORS,
    circular_velocity,
    fit_potential,
    potential_check,
    radial_actions,
    snapshot_vertical_potential,
    vertical_potential,
    vertical_residual,
)
from experiments.snapshot import Galaxy, Header, Particles, Stars

N_SAMPLE = 40_000
OFFSET = -5.0e4


@pytest.fixture(name="truth", scope="module")
def fixture_truth() -> agama.Potential:
    """Return the mock's disk plus halo potential."""
    return agama.Potential(*true_components())


@pytest.fixture(name="galaxy", scope="module")
def fixture_galaxy(truth: agama.Potential) -> Galaxy:
    """Return the disk as gas and stars and the halo as dark matter, with one star at the origin."""
    agama.setRandomSeed(11)
    disk, halo = true_components()
    disk_pos, disk_mass = (np.asarray(a, dtype=np.float64) for a in disk.sample(N_SAMPLE))
    halo_pos, halo_mass = (np.asarray(a, dtype=np.float64) for a in halo.sample(N_SAMPLE))
    stars_part, gas_part = slice(0, N_SAMPLE // 2), slice(N_SAMPLE // 2, N_SAMPLE)
    star_pos = np.vstack([np.zeros(3), disk_pos[stars_part]])
    n_stars = star_pos.shape[0]
    stars = Stars(
        star_pos,
        np.zeros((n_stars, 3)),
        np.full(n_stars, disk_mass[0]),
        formation_a=np.ones(n_stars),
        initial_mass=np.ones(n_stars),
        potential=np.asarray(truth.potential(star_pos), dtype=np.float64) + OFFSET,
    )
    gas = Particles(disk_pos[gas_part], np.zeros((N_SAMPLE // 2, 3)), disk_mass[gas_part])
    header = Header(1.0, 0.0, 67.77, (0.0,) * 6, Cosmology(0.6777, 0.307, 0.693))
    return Galaxy(header, stars, {0: gas, 1: Particles(halo_pos, np.zeros((N_SAMPLE, 3)), halo_mass)})


@pytest.fixture(name="fitted", scope="module")
def fixture_fitted(galaxy: Galaxy) -> agama.Potential:
    """Return the potential fitted to the sampled galaxy."""
    return fit_potential(galaxy)


def test_fit_recovers_the_vertical_potential_and_the_circular_velocity(
    fitted: agama.Potential, truth: agama.Potential
) -> None:
    """The fit, with a massive star at the origin left out, follows Delta Phi(8 kpc, z) and v_c of the truth."""
    inside = (Z_GRID > 0.05) & (Z_GRID <= 1.5)
    ratio = vertical_potential(fitted, 8.0)[inside] / vertical_potential(truth, 8.0)[inside]
    np.testing.assert_allclose(ratio, 1.0, atol=0.1)
    assert circular_velocity(fitted, 8.0) == pytest.approx(circular_velocity(truth, 8.0), rel=0.02)


def test_vertical_potential_is_zero_in_the_plane(truth: agama.Potential) -> None:
    """Delta Phi starts at zero and grows with |z|."""
    phi = vertical_potential(truth, 8.0)
    assert phi[0] == 0.0
    assert np.all(np.diff(phi) > 0.0)


def test_radial_actions(truth: agama.Potential) -> None:
    """A circular orbit has J_R = 0, an eccentric one J_R > 0, and an unbound one NaN."""
    v_c = circular_velocity(truth, 8.0)
    pos = np.array([[8.0, 0.0, 0.0]] * 3)
    vel = np.array([[0.0, v_c, 0.0], [60.0, v_c, 0.0], [0.0, 2000.0, 0.0]])
    jr = radial_actions(truth, pos, vel)
    assert jr[0] == pytest.approx(0.0, abs=1e-6)
    assert jr[1] > 10.0
    assert np.isnan(jr[2])


def test_potential_check_compares_slices(galaxy: Galaxy, truth: agama.Potential) -> None:
    """Stars that carry the true potential agree with it slice by slice, whatever the constant."""
    annulus = (galaxy.stars.cylindrical_radius > 6.0) & (galaxy.stars.cylindrical_radius < 10.0)
    rows = potential_check(truth, galaxy.stars, annulus)
    assert rows[0].snapshot == rows[0].fitted == 0.0
    for row in rows:
        assert row.snapshot == pytest.approx(row.fitted, abs=1e-6)
        assert row.n >= 30
    assert not potential_check(truth, galaxy.stars, np.zeros_like(annulus))


def test_vertical_residual_removes_the_radial_gradient() -> None:
    """A residual a + b (R - R_mid) + c |z| comes back as a + c |z| at R_mid; sparse slices are left out."""
    rng = np.random.default_rng(4)
    cyl_r, absz = rng.uniform(7.5, 9.0, 20_000), rng.uniform(0.0, 1.0, 20_000)
    azimuth = rng.uniform(-np.pi, np.pi, 20_000)
    delta = 30.0 + 80.0 * (cyl_r - 8.2) + 50.0 * absz
    residual = vertical_residual(cyl_r, azimuth, absz, delta, 8.2)
    assert len(residual.z) == 10
    np.testing.assert_allclose(residual.residual, 30.0 + 50.0 * np.array(residual.z), atol=0.5)
    n_sparse = RESIDUAL_MIN_PER_SECTOR * RESIDUAL_SECTORS * 5
    sparse = vertical_residual(cyl_r[:n_sparse], azimuth[:n_sparse], absz[:n_sparse], delta[:n_sparse], 8.2)
    assert len(sparse.z) < 10


def test_vertical_residual_weights_every_azimuth_equally() -> None:
    """Stars crowding toward a bar's axis with height barely move the azimuthal average of an m = 2 residual.

    The residual is 1000 cos(2 phi), with no vertical trend, and the fraction of stars near
    phi = 0 grows with |z|, so the median over all the stars of the top slice exceeds 500.
    The sector average stays within 10 percent of the amplitude; the remainder comes from
    the crowding inside each sector.
    """
    rng = np.random.default_rng(8)
    n = 200_000
    absz = rng.uniform(0.0, 1.0, n)
    crowded = rng.uniform(size=n) < 0.6 * absz
    azimuth = np.where(crowded, rng.normal(0.0, 0.3, n), rng.uniform(-np.pi, np.pi, n))
    cyl_r = rng.uniform(7.5, 9.0, n)
    delta = 1000.0 * np.cos(2.0 * azimuth)
    residual = vertical_residual(cyl_r, azimuth, absz, delta, 8.2)
    np.testing.assert_allclose(residual.residual, 0.0, atol=100.0)
    top = absz > 0.9
    assert np.median(delta[top]) > 500.0


def test_snapshot_vertical_potential_adds_the_vertical_residual(galaxy: Galaxy, truth: agama.Potential) -> None:
    """The snapshot's Delta Phi is the fit's plus the residual's vertical change; a constant changes nothing."""
    annulus = (galaxy.stars.cylindrical_radius > 6.0) & (galaxy.stars.cylindrical_radius < 10.0)
    phi_snap, residual = snapshot_vertical_potential(truth, galaxy.stars, annulus, 8.0)
    np.testing.assert_allclose(phi_snap, vertical_potential(truth, 8.0), atol=1e-6)
    np.testing.assert_allclose(residual.residual, OFFSET, atol=1e-6)


def test_snapshot_vertical_potential_needs_stars(galaxy: Galaxy, truth: agama.Potential) -> None:
    """Without a |z| slice of enough stars there is no residual to add."""
    with pytest.raises(ValueError, match=re.escape("no |z| slice holds")):
        snapshot_vertical_potential(truth, galaxy.stars, np.zeros(galaxy.stars.mass.size, dtype=bool), 8.0)
