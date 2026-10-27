"""Tests of the snapshot reader: units, the periodic wrap, wind particles and the disk frame.

The snapshots are written with :func:`experiments.mock.write_snapshot` from particles
given in physical units around the center, which sits within 2.2 kpc of two faces of
the periodic box, so that the galaxy straddles them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

from experiments.auriga import HaloSnapshot
from experiments.mock import CENTER, COSMOLOGY, Blocks, rotation_matrix, to_box, write_snapshot
from experiments.snapshot import (
    STAR_TYPE,
    Galaxy,
    Particles,
    cylindrical_velocities,
    galaxy_frame,
    physical_offset,
    read_center,
    read_galaxy,
    read_header,
)

FloatArray = NDArray[np.float64]

ROTATION = rotation_matrix(30.0)
TO_CODE = COSMOLOGY.hubble / 1.0e10
N_STARS, N_WIND, N_FAR = 18, 2, 20
DM_MASS = 3.0e5


@dataclass(frozen=True)
class Truth:
    """The particles written to a synthetic snapshot, in physical units and the galaxy frame."""

    gas_pos: FloatArray
    gas_mass: FloatArray
    dm_pos: FloatArray
    star_pos: FloatArray
    star_vel: FloatArray
    star_mass: FloatArray
    formation_a: FloatArray
    potential: FloatArray


def write_galaxy(root: Path, snapshot: HaloSnapshot) -> Truth:
    """Write gas, dark matter, far-away PartType2 and stars (18 near, 2 wind, 20 far) as *snapshot*."""
    rng = np.random.default_rng(5)
    gas_pos, gas_mass = rng.normal(0.0, 5.0, (40, 3)), rng.uniform(1e4, 1e5, 40)
    dm_pos = rng.normal(0.0, 50.0, (30, 3))
    far = rng.normal(0.0, 1.0, (10, 3)) + 400.0
    near = rng.normal(0.0, 4.0, (N_STARS + N_WIND, 3))
    star_pos = np.concatenate([near, rng.normal(0.0, 1.0, (N_FAR, 3)) + 500.0])
    n = int(star_pos.shape[0])
    star_vel, star_mass = rng.normal(0.0, 50.0, (n, 3)), rng.uniform(1e3, 1e4, n)
    formation_a = rng.uniform(0.2, 1.0, n)
    wind = slice(N_STARS, N_STARS + N_WIND)
    formation_a[wind] = -0.5
    potential = rng.uniform(-2e5, -1e5, n)
    mass_table = np.zeros(6)
    mass_table[1] = DM_MASS * TO_CODE
    blocks: Blocks = {
        0: {"Coordinates": to_box(gas_pos, ROTATION), "Velocities": np.zeros((40, 3)), "Masses": gas_mass * TO_CODE},
        1: {"Coordinates": to_box(dm_pos, ROTATION), "Velocities": np.zeros((30, 3))},
        2: {"Coordinates": to_box(far, ROTATION), "Velocities": np.zeros((10, 3)), "Masses": np.ones(10)},
        STAR_TYPE: {
            "Coordinates": to_box(star_pos, ROTATION),
            "Velocities": star_vel,
            "Masses": star_mass * TO_CODE,
            "GFM_InitialMass": 2.0 * star_mass * TO_CODE,
            "GFM_StellarFormationTime": formation_a,
            "Potential": potential,
        },
    }
    write_snapshot(snapshot, root, blocks, mass_table)
    return Truth(gas_pos, gas_mass, dm_pos, star_pos, star_vel, star_mass, formation_a, potential)


def set_scale_factor(root: Path, snapshot: HaloSnapshot, scale_factor: float) -> None:
    """Set ``Time`` in every snapshot chunk of *snapshot* to *scale_factor*."""
    for path in sorted((snapshot.local_dir(root) / snapshot.subdirs[0]).glob("*.hdf5")):
        with h5py.File(path, "r+") as snap:
            snap["Header"].attrs["Time"] = scale_factor


@pytest.fixture(name="mock24")
def fixture_mock24(tmp_path: Path, au24: HaloSnapshot) -> tuple[Path, Truth]:
    """Return the root of a synthetic Au24 and the particles written to it."""
    return tmp_path, write_galaxy(tmp_path, au24)


def test_read_galaxy_converts_to_physical_units(mock24: tuple[Path, Truth], au24: HaloSnapshot) -> None:
    """Positions come back in kpc around the center across the box faces, with masses, potentials and ages."""
    root, truth = mock24
    galaxy = read_galaxy(au24, root)
    keep = slice(0, N_STARS)  # the near star particles, without the wind and the far ones
    stars = galaxy.stars
    np.testing.assert_allclose(stars.pos, truth.star_pos[keep] @ ROTATION.T, atol=1e-6)
    np.testing.assert_allclose(stars.vel, truth.star_vel[keep])
    np.testing.assert_allclose(stars.mass, truth.star_mass[keep])
    np.testing.assert_allclose(stars.initial_mass, 2.0 * truth.star_mass[keep])
    np.testing.assert_allclose(stars.formation_a, truth.formation_a[keep])
    np.testing.assert_allclose(stars.potential, truth.potential[keep])
    np.testing.assert_allclose(galaxy.others[0].pos, truth.gas_pos @ ROTATION.T, atol=1e-6)
    np.testing.assert_allclose(galaxy.others[0].mass, truth.gas_mass)
    np.testing.assert_allclose(galaxy.others[1].mass, DM_MASS)
    assert list(galaxy.by_type()) == [0, 1, STAR_TYPE]


def test_read_galaxy_at_an_earlier_time(mock24: tuple[Path, Truth], au24: HaloSnapshot) -> None:
    """At a = 0.5, positions shrink by a, velocities by sqrt(a), and the potential grows by 1 / a."""
    root, truth = mock24
    set_scale_factor(root, au24, 0.5)
    stars = read_galaxy(au24, root).stars
    np.testing.assert_allclose(stars.pos, 0.5 * truth.star_pos[:N_STARS] @ ROTATION.T, atol=1e-6)
    np.testing.assert_allclose(stars.vel, np.sqrt(0.5) * truth.star_vel[:N_STARS])
    np.testing.assert_allclose(stars.potential, truth.potential[:N_STARS] / 0.5)


def test_read_galaxy_without_stars_nearby(mock24: tuple[Path, Truth], au24: HaloSnapshot) -> None:
    """A selection radius that holds no star particle is an error."""
    with pytest.raises(ValueError, match="no star particles"):
        read_galaxy(au24, mock24[0], r_select=1.0e-3)


def test_read_header(mock24: tuple[Path, Truth], au24: HaloSnapshot) -> None:
    """The header carries the time, the box and the cosmology."""
    header = read_header(au24.local_dir(mock24[0]) / "snapdir_127" / "snapshot_127.0.hdf5")
    assert header.cosmology == COSMOLOGY
    assert (header.scale_factor, header.redshift) == (1.0, 0.0)
    assert header.mass_table[1] == pytest.approx(DM_MASS * TO_CODE)


def write_catalog(path: Path, center: tuple[float, float, float] | None) -> Path:
    """Write a catalog chunk with one subhalo at *center*, or none."""
    with h5py.File(path, "w") as cat:
        cat.create_group("Header").attrs["Nsubgroups_ThisFile"] = 0 if center is None else 1
        if center is not None:
            cat.create_group("Subhalo")["SubhaloPos"] = np.array([center])
    return path


def test_read_center_skips_chunks_without_subhalos(tmp_path: Path) -> None:
    """The center is the first subhalo of the catalog, in whichever chunk it is."""
    files = [write_catalog(tmp_path / "a.hdf5", None), write_catalog(tmp_path / "b.hdf5", (1.0, 2.0, 3.0))]
    np.testing.assert_array_equal(read_center(files), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="no subhalo"):
        read_center(files[:1])


def test_physical_offset_takes_the_nearest_image(mock24: tuple[Path, Truth], au24: HaloSnapshot) -> None:
    """A particle across a face of the box is close to a center near that face."""
    header = read_header(au24.local_dir(mock24[0]) / "snapdir_127" / "snapshot_127.0.hdf5")
    pos = np.array([[header.box_size - 0.0005, CENTER[1], CENTER[2]]])
    offset = physical_offset(pos, CENTER, header)
    assert offset[0, 0] == pytest.approx(-(0.0005 + CENTER[0]) / COSMOLOGY.hubble * 1.0e3)


def disk_of_stars(axis: FloatArray, bulk: FloatArray) -> Particles:
    """Return star particles on circular orbits about *axis*, moving with *bulk*.

    The phases are evenly spaced, so that the orbital velocities cancel in the mean.
    """
    rng = np.random.default_rng(8)
    e_z = axis / np.linalg.norm(axis)
    e_x = np.cross(e_z, [0.0, 0.0, 1.0] if abs(e_z[2]) < 0.9 else [1.0, 0.0, 0.0])
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    radius, phase = rng.uniform(1.0, 9.0, 500), np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    pos = radius[:, None] * (np.cos(phase)[:, None] * e_x + np.sin(phase)[:, None] * e_y)
    vel = 200.0 * (-np.sin(phase)[:, None] * e_x + np.cos(phase)[:, None] * e_y) + bulk
    return Particles(pos, vel, np.ones(500))


@pytest.mark.parametrize("axis", [np.array([0.3, -0.4, 1.0]), np.array([1.0, 0.1, 0.05])])
def test_galaxy_frame_finds_the_spin_and_the_bulk_motion(axis: FloatArray) -> None:
    """The frame's z axis is the spin of the disk, its axes are right-handed, and the disk rotates forward."""
    bulk = np.array([100.0, -50.0, 20.0])
    stars = disk_of_stars(axis, bulk)
    found_bulk, rotation = galaxy_frame(stars)
    np.testing.assert_allclose(found_bulk, bulk, atol=1e-9)
    np.testing.assert_allclose(rotation[2], axis / np.linalg.norm(axis), atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    disk = stars.in_frame(found_bulk, rotation)
    np.testing.assert_allclose(disk.pos[:, 2], 0.0, atol=1e-12)
    v_phi = (disk.pos[:, 0] * disk.vel[:, 1] - disk.pos[:, 1] * disk.vel[:, 0]) / disk.cylindrical_radius
    np.testing.assert_allclose(v_phi, 200.0)


def test_galaxy_in_frame_moves_every_type(mock24: tuple[Path, Truth], au24: HaloSnapshot) -> None:
    """Moving a galaxy to a frame moves its stars and its other particles alike."""
    galaxy = read_galaxy(au24, mock24[0])
    bulk, flip = np.array([1.0, 2.0, 3.0]), np.diag([1.0, -1.0, -1.0])
    moved = galaxy.in_frame(bulk, flip)
    assert isinstance(moved, Galaxy)
    np.testing.assert_allclose(moved.stars.pos, galaxy.stars.pos @ flip.T)
    np.testing.assert_allclose(moved.others[0].vel, (galaxy.others[0].vel - bulk) @ flip.T)
    np.testing.assert_allclose(moved.stars.potential, galaxy.stars.potential)


def test_particles_select_and_cylindrical_velocities() -> None:
    """``select`` keeps every field of the chosen particles; v_R and v_z are the cylindrical components."""
    pos = np.array([[3.0, 4.0, 1.0], [0.0, 2.0, -1.0]])
    vel = np.array([[3.0, 4.0, 7.0], [5.0, -2.0, 1.0]])
    parts = Particles(pos, vel, np.array([1.0, 2.0]))
    chosen = parts.select(np.array([False, True]))
    np.testing.assert_array_equal(chosen.mass, [2.0])
    np.testing.assert_array_equal(chosen.cylindrical_radius, [2.0])
    v_r, v_z = cylindrical_velocities(pos, vel)
    np.testing.assert_allclose(v_r, [5.0, -2.0])
    np.testing.assert_allclose(v_z, [7.0, 1.0])
