"""Read one Auriga snapshot into physical units, in the frame of the main galaxy's disk.

Arepo writes positions in comoving Mpc/h inside a periodic box, velocities as sqrt(a)
times the peculiar velocity in km/s, masses in 1e10 Msun/h and the gravitational
potential in (km/s)^2 / a. :func:`read_galaxy` keeps the particles of every type within
``R_SELECT_KPC`` of ``SubhaloPos`` of the first subhalo, the main galaxy, and converts
them to kpc, km/s and Msun around that center. Star particles also carry their formation
scale factor, initial mass and potential. Wind particles, which share the star type but
carry a formation scale factor <= 0, are dropped.

:func:`galaxy_frame` then finds the disk frame: its z axis is the angular momentum of the
stars inside ``ALIGN_RADIUS_KPC``, and its origin moves with their mean velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Self

import h5py
import numpy as np
from numpy.typing import NDArray

from experiments.auriga import HaloSnapshot
from experiments.cosmology import Cosmology
from experiments.snapshot_check import N_TYPES, chunks

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

KPC_PER_MPC = 1.0e3
MSUN_PER_MASS_UNIT = 1.0e10
STAR_TYPE = 4
R_SELECT_KPC = 300.0
ALIGN_RADIUS_KPC = 10.0


@dataclass(frozen=True)
class Header:
    """The time, the box and the cosmology of a snapshot."""

    scale_factor: float
    redshift: float
    box_size: float  # comoving Mpc/h
    mass_table: tuple[float, ...]  # 1e10 Msun/h
    cosmology: Cosmology


@dataclass(frozen=True)
class Particles:
    """Particles of one type: positions (kpc), velocities (km/s) and masses (Msun)."""

    pos: FloatArray
    vel: FloatArray
    mass: FloatArray

    def select(self, keep: BoolArray) -> Self:
        """Return the particles where *keep* is true."""
        return replace(self, **{f.name: getattr(self, f.name)[keep] for f in fields(self)})

    def in_frame(self, bulk: FloatArray, rotation: FloatArray) -> Self:
        """Return the particles moved to the frame with velocity *bulk* and axes the rows of *rotation*."""
        return replace(self, pos=self.pos @ rotation.T, vel=(self.vel - bulk) @ rotation.T)

    @property
    def cylindrical_radius(self) -> FloatArray:
        """Return the distance from the z axis (kpc)."""
        return np.asarray(np.hypot(self.pos[:, 0], self.pos[:, 1]), dtype=np.float64)


@dataclass(frozen=True)
class Stars(Particles):
    """Star particles, with formation scale factors, initial masses (Msun) and potentials ((km/s)^2)."""

    formation_a: FloatArray
    initial_mass: FloatArray
    potential: FloatArray


@dataclass(frozen=True)
class Galaxy:
    """The particles of every type around the main galaxy of a snapshot."""

    header: Header
    stars: Stars
    others: dict[int, Particles]

    def by_type(self) -> dict[int, Particles]:
        """Return the particles of every type, keyed and ordered by type number."""
        found: dict[int, Particles] = {**self.others, STAR_TYPE: self.stars}
        return {ptype: found[ptype] for ptype in sorted(found)}

    def in_frame(self, bulk: FloatArray, rotation: FloatArray) -> Galaxy:
        """Return the galaxy moved to the frame with velocity *bulk* and axes the rows of *rotation*."""
        return replace(
            self,
            stars=self.stars.in_frame(bulk, rotation),
            others={ptype: parts.in_frame(bulk, rotation) for ptype, parts in self.others.items()},
        )


def read_header(path: Path) -> Header:
    """Return the header of the snapshot chunk at *path*."""
    with h5py.File(path, "r") as snap:
        attrs = snap["Header"].attrs
        return Header(
            scale_factor=float(attrs["Time"]),
            redshift=float(attrs["Redshift"]),
            box_size=float(attrs["BoxSize"]),
            mass_table=tuple(np.asarray(attrs["MassTable"], dtype=np.float64).tolist()),
            cosmology=Cosmology(float(attrs["HubbleParam"]), float(attrs["Omega0"]), float(attrs["OmegaLambda"])),
        )


def read_center(group_files: list[Path]) -> FloatArray:
    """Return ``SubhaloPos`` of the first subhalo of the catalog, the main galaxy (comoving Mpc/h)."""
    for path in group_files:
        with h5py.File(path, "r") as cat:
            if int(cat["Header"].attrs["Nsubgroups_ThisFile"]) > 0:
                return np.asarray(cat["Subhalo"]["SubhaloPos"][0], dtype=np.float64)
    raise ValueError(f"no subhalo in the catalog {group_files[0].parent}")


def physical_offset(pos: FloatArray, center: FloatArray, header: Header) -> FloatArray:
    """Return the minimum-image offset of the comoving positions *pos* from *center*, in physical kpc."""
    box = header.box_size
    offset = (pos - center + 0.5 * box) % box - 0.5 * box
    return np.asarray(offset * header.scale_factor / header.cosmology.hubble * KPC_PER_MPC, dtype=np.float64)


@dataclass(frozen=True)
class _Selection:
    """The particles of one type in one chunk that lie within the selection radius."""

    particles: Particles
    keep: BoolArray


def _read_particles(
    block: h5py.Group, ptype: int, header: Header, center: FloatArray, r_select: float
) -> _Selection | None:
    """Return the particles of type *ptype* in *block* within *r_select* kpc of *center*, if any."""
    mass_unit = MSUN_PER_MASS_UNIT / header.cosmology.hubble
    pos = physical_offset(np.asarray(block["Coordinates"], dtype=np.float64), center, header)
    keep = np.einsum("ij,ij->i", pos, pos) < r_select**2
    if not keep.any():
        return None
    if "Masses" in block:
        mass = np.asarray(block["Masses"], dtype=np.float64)[keep] * mass_unit
    else:
        mass = np.full(int(keep.sum()), header.mass_table[ptype] * mass_unit)
    vel = np.asarray(block["Velocities"], dtype=np.float64)[keep] * np.sqrt(header.scale_factor)
    return _Selection(Particles(pos[keep], vel, mass), keep)


def _read_stars(block: h5py.Group, header: Header, center: FloatArray, r_select: float) -> Stars | None:
    """Return the star particles in *block* within *r_select* kpc of *center*, if any."""
    if (selection := _read_particles(block, STAR_TYPE, header, center, r_select)) is None:
        return None
    parts, keep = selection.particles, selection.keep
    return Stars(
        parts.pos,
        parts.vel,
        parts.mass,
        formation_a=np.asarray(block["GFM_StellarFormationTime"], dtype=np.float64)[keep],
        initial_mass=np.asarray(block["GFM_InitialMass"], dtype=np.float64)[keep]
        * (MSUN_PER_MASS_UNIT / header.cosmology.hubble),
        potential=np.asarray(block["Potential"], dtype=np.float64)[keep] / header.scale_factor,
    )


def _concatenate[P: Particles](parts: list[P]) -> P:
    """Return the particles of *parts*, one after the other."""
    first = parts[0]
    return replace(first, **{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(first)})


def _read_chunk(
    path: Path, header: Header, center: FloatArray, r_select: float
) -> tuple[Stars | None, dict[int, Particles]]:
    """Return the star particles and the particles of every other type in the chunk *path* near *center*."""
    stars: Stars | None = None
    others: dict[int, Particles] = {}
    with h5py.File(path, "r") as snap:
        in_file = snap["Header"].attrs["NumPart_ThisFile"]
        for ptype in (t for t in range(N_TYPES) if int(in_file[t])):
            block = snap[f"PartType{ptype}"]
            if ptype == STAR_TYPE:
                stars = _read_stars(block, header, center, r_select)
            elif (selection := _read_particles(block, ptype, header, center, r_select)) is not None:
                others[ptype] = selection.particles
    return stars, others


def _read_chunks(
    files: list[Path], header: Header, center: FloatArray, r_select: float
) -> tuple[list[Stars], dict[int, list[Particles]]]:
    """Return the star particles and the particles of every other type near *center*, chunk by chunk."""
    stars: list[Stars] = []
    others: dict[int, list[Particles]] = {}
    for path in files:
        found, rest = _read_chunk(path, header, center, r_select)
        if found is not None:
            stars.append(found)
        for ptype, parts in rest.items():
            others.setdefault(ptype, []).append(parts)
    return stars, others


def read_galaxy(snapshot: HaloSnapshot, root: Path, r_select: float = R_SELECT_KPC) -> Galaxy:
    """Return the particles within *r_select* kpc of the main galaxy of *snapshot*, without wind particles."""
    snapdir, groupdir = (snapshot.local_dir(root) / name for name in snapshot.subdirs)
    files = chunks(snapdir, f"snapshot_{snapshot.snapshot:03d}")
    header = read_header(files[0])
    center = read_center(chunks(groupdir, f"fof_subhalo_tab_{snapshot.snapshot:03d}"))
    stars, others = _read_chunks(files, header, center, r_select)
    if not stars:
        raise ValueError(f"no star particles within {r_select} kpc of the main galaxy of Au{snapshot.halo}")
    merged = _concatenate(stars)
    return Galaxy(header, merged.select(merged.formation_a > 0.0), {t: _concatenate(p) for t, p in others.items()})


def galaxy_frame(stars: Particles) -> tuple[FloatArray, FloatArray]:
    """Return the bulk velocity and the rotation matrix (rows: new x, y, z) of the stellar disk.

    The new z axis is the angular momentum of the stars inside ``ALIGN_RADIUS_KPC``, so
    that the disk rotates with positive v_phi; x and y complete a right-handed frame.
    """
    inner = np.einsum("ij,ij->i", stars.pos, stars.pos) < ALIGN_RADIUS_KPC**2
    mass = stars.mass[inner]
    bulk = np.sum(stars.vel[inner] * mass[:, None], axis=0) / np.sum(mass)
    spin = np.sum(np.cross(stars.pos[inner], stars.vel[inner] - bulk) * mass[:, None], axis=0)
    e_z = spin / np.linalg.norm(spin)
    trial = np.array([1.0, 0.0, 0.0]) if abs(e_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e_x = trial - np.dot(trial, e_z) * e_z
    e_x /= np.linalg.norm(e_x)
    return np.asarray(bulk, dtype=np.float64), np.vstack([e_x, np.cross(e_z, e_x), e_z])


def cylindrical_velocities(pos: FloatArray, vel: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return v_R and v_z (km/s) of particles in the disk frame."""
    cyl_r = np.hypot(pos[:, 0], pos[:, 1])
    v_r = (pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]) / cyl_r
    return np.asarray(v_r, dtype=np.float64), np.asarray(vel[:, 2], dtype=np.float64)
