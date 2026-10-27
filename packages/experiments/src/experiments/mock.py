"""A mock Auriga snapshot of a known axisymmetric galaxy, to check the analysis against its answer.

The analysis reads Arepo HDF5 chunks in comoving units, finds the galaxy, rotates it into
the disk frame, fits a potential to the particles and measures the mono-age populations.
Each step can fail silently (an h factor, a periodic wrap, the sign of the spin), so the
mock is built from a known potential (Msun, kpc, km/s):

- disk: AGAMA ``Disk`` with Sigma_0 = 800 Msun/pc^2, R_d = 2.6 kpc and scaleHeight = -0.3
  kpc, which AGAMA takes as the profile sech^2(z / 0.6 kpc). It is sampled as gas cells
  at rest, which carry the mass but are not tracers;
- halo: AGAMA ``Spheroid``, NFW-like (gamma = 1, beta = 3, alpha = 1), 1e12 Msun, r_s = 20
  kpc, with a cutoff at 150 kpc, sampled as PartType1 with its mass in the ``MassTable``;
- stars: one population per 2 Gyr age range, drawn from a quasi-isothermal DF in the total
  potential with the (sigma_R, sigma_z) of ``POPULATIONS`` at R_0 = 8 kpc, ages uniform in
  the range, and masses (1e3 Msun) too small to matter.

The mock sits 2.2 and 1.5 kpc from two faces of the periodic box, tilted by ``TILT_DEG``
about (1, 1, 0), with a bulk velocity, in two chunks, at h = 0.6777 and a = 1. Its star particles
carry the true potential plus a constant, and two wind particles, which the reader must
drop.

:func:`check_mock` compares the analysis of the mock with the truth computed in the
original frame and the true potential.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import agama
import h5py
import numpy as np
from numpy.typing import NDArray

from experiments.analysis import HaloAnalysis, PreparedHalo
from experiments.auriga import HaloSnapshot
from experiments.cosmology import Cosmology
from experiments.heights import Z_GRID, scaled_particle_hz
from experiments.populations import AGE_BINS, R_MAX_KPC, R_MIN_KPC, Z_JR_MAX_KPC, Z_SAMPLE_MAX_KPC
from experiments.potential import circular_velocity, radial_actions, vertical_potential
from experiments.snapshot import STAR_TYPE
from experiments.snapshot_check import N_TYPES

FloatArray = NDArray[np.float64]
Blocks = dict[int, dict[str, FloatArray]]

COSMOLOGY = Cosmology(hubble=0.6777, omega_m=0.307, omega_lambda=0.693)
BOX = 67.77  # comoving Mpc/h, the 100 Mpc parent box of Auriga
CENTER = np.array([0.0015, 33.0, 67.769])  # 2.2 and 1.5 kpc from two faces of the box
BULK_KMS = np.array([150.0, -80.0, 40.0])
TILT_DEG = 37.0
POTENTIAL_OFFSET = -5.0e4  # the additive constant of the snapshot potential, (km/s)^2
R_DISK_KPC = 2.6
R_SIGMA_KPC = 5.2
R_0_KPC = 8.0
# (sigma_R, sigma_z) in km/s at R_0 and the age range (Gyr) of each population.
POPULATIONS = (
    ((25.0, 10.5), (0.0, 2.0)),
    ((32.0, 14.0), (2.0, 4.0)),
    ((37.0, 17.5), (4.0, 6.0)),
    ((40.0, 21.0), (6.0, 8.0)),
    ((42.0, 23.0), (8.0, 10.0)),
)
STAR_MASS = 1.0e3
N_CHUNKS = 2
SEED = 20261023
DELTA_PHI_RANGE_KPC = (0.05, 1.5)


@dataclass(frozen=True)
class MockSizes:
    """The numbers of disk cells, halo particles and star particles per population."""

    n_disk: int = 500_000
    n_halo: int = 500_000
    n_stars: int = 50_000


@dataclass(frozen=True)
class MockGalaxy:
    """The truth of a mock: galaxy-frame star phase space, ages and masses, the rotation to the box, the potential."""

    xv: FloatArray
    age: FloatArray
    mass: FloatArray
    rotation: FloatArray
    potential: agama.Potential


@dataclass(frozen=True)
class MockBin:
    """h_Z and J_R,med of one age bin from the analysis and from the truth, and the tracer h_Z."""

    age_lo: float
    age_hi: float
    hz: float
    hz_true: float
    jr_median: float
    jr_median_true: float
    hz_tracer: float


@dataclass(frozen=True)
class MockCheck:
    """The analysis of a mock against its truth."""

    spin_error_deg: float
    age_error_gyr: float
    v_circ: float
    v_circ_true: float
    delta_phi_error: float
    bins: list[MockBin]


def true_components() -> tuple[agama.Potential, agama.Potential]:
    """Return the disk and the halo potentials of the mock."""
    agama.setUnits(mass=1, length=1, velocity=1)
    disk = agama.Potential(type="Disk", surfaceDensity=8.0e8, scaleRadius=R_DISK_KPC, scaleHeight=-0.3)
    halo = agama.Potential(
        type="Spheroid", mass=1.0e12, scaleRadius=20.0, gamma=1.0, beta=3.0, alpha=1.0, outerCutoffRadius=150.0
    )
    return disk, halo


def rotation_matrix(tilt_deg: float) -> FloatArray:
    """Return the rotation by *tilt_deg* about (1, 1, 0) / sqrt(2), from the galaxy frame to the box."""
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    angle = np.radians(tilt_deg)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.asarray(np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross, dtype=np.float64)


def to_box(pos: FloatArray, rotation: FloatArray) -> FloatArray:
    """Return galaxy-frame positions (kpc) as wrapped comoving box coordinates (Mpc/h) at a = 1."""
    return np.asarray((pos @ rotation.T / 1.0e3 * COSMOLOGY.hubble + CENTER) % BOX, dtype=np.float64)


def sample_stars(pot: agama.Potential, n_stars: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Return the galaxy-frame phase space and the ages (Gyr) of *n_stars* stars per population."""
    xv_all, age_all = [], []
    scale = np.exp(R_0_KPC / R_SIGMA_KPC)
    for (sigma_r, sigma_z), (lo, hi) in POPULATIONS:
        df = agama.DistributionFunction(
            type="QuasiIsothermal",
            potential=pot,
            Sigma0=1.0,
            Rdisk=R_DISK_KPC,
            sigmar0=sigma_r * scale,
            Rsigmar=R_SIGMA_KPC,
            sigmaz0=sigma_z * scale,
            Rsigmaz=R_SIGMA_KPC,
        )
        xv, _ = agama.GalaxyModel(pot, df).sample(n_stars)
        xv_all.append(np.asarray(xv, dtype=np.float64))
        age_all.append(rng.uniform(lo, hi, n_stars))
    return np.concatenate(xv_all), np.concatenate(age_all)


def _write_chunk(path: Path, blocks: Blocks, chunk: int, mass_table: FloatArray) -> None:
    """Write the part *chunk* of every block of *blocks* to the snapshot chunk *path*."""
    totals = np.zeros(N_TYPES, dtype=np.int64)
    this_file = np.zeros(N_TYPES, dtype=np.int64)
    with h5py.File(path, "w") as snap:
        for ptype, block in blocks.items():
            totals[ptype] = block["Coordinates"].shape[0]
            part = np.array_split(np.arange(totals[ptype]), N_CHUNKS)[chunk]
            group = snap.create_group(f"PartType{ptype}")
            for name, values in block.items():
                group.create_dataset(name, data=values[part])
            this_file[ptype] = part.size
        snap.create_group("Header").attrs.update(
            {
                "Time": 1.0,
                "Redshift": 0.0,
                "HubbleParam": COSMOLOGY.hubble,
                "Omega0": COSMOLOGY.omega_m,
                "OmegaLambda": COSMOLOGY.omega_lambda,
                "BoxSize": BOX,
                "MassTable": mass_table,
                "NumPart_ThisFile": this_file,
                "NumPart_Total": totals,
                "NumFilesPerSnapshot": N_CHUNKS,
            }
        )


def _write_catalog(path: Path, chunk: int) -> None:
    """Write the catalog chunk *path*; the first chunk holds the one subhalo, at ``CENTER``."""
    with h5py.File(path, "w") as cat:
        cat.create_group("Header").attrs["Nsubgroups_ThisFile"] = 1 if chunk == 0 else 0
        if chunk == 0:
            cat.create_group("Subhalo").create_dataset("SubhaloPos", data=CENTER[None, :])


def write_snapshot(snapshot: HaloSnapshot, root: Path, blocks: Blocks, mass_table: FloatArray) -> None:
    """Write *blocks* (datasets per particle type, in code units) as the chunks of *snapshot* under *root*.

    Every block is split in ``N_CHUNKS`` parts. The header holds ``COSMOLOGY`` at a = 1
    in a box of ``BOX``; the group catalog holds one subhalo, at ``CENTER``, in its
    first chunk.
    """
    snapdir, groupdir = (snapshot.local_dir(root) / name for name in snapshot.subdirs)
    snapdir.mkdir(parents=True, exist_ok=True)
    groupdir.mkdir(parents=True, exist_ok=True)
    for chunk in range(N_CHUNKS):
        _write_chunk(snapdir / f"snapshot_{snapshot.snapshot:03d}.{chunk}.hdf5", blocks, chunk, mass_table)
        _write_catalog(groupdir / f"fof_subhalo_tab_{snapshot.snapshot:03d}.{chunk}.hdf5", chunk)


def _matter_blocks(
    disk: agama.Potential, halo: agama.Potential, sizes: MockSizes, rotation: FloatArray
) -> tuple[Blocks, FloatArray]:
    """Return the gas cells of the disk and the dark matter of the halo, in code units, and the mass table."""
    to_code = COSMOLOGY.hubble / 1.0e10
    disk_pos, disk_mass = disk.sample(sizes.n_disk)
    halo_pos, halo_mass = halo.sample(sizes.n_halo)
    mass_table = np.zeros(N_TYPES)
    mass_table[1] = float(halo_mass[0]) * to_code
    blocks: Blocks = {
        0: {
            "Coordinates": to_box(np.asarray(disk_pos, dtype=np.float64), rotation),
            "Velocities": np.tile(BULK_KMS, (sizes.n_disk, 1)),
            "Masses": np.full(sizes.n_disk, float(disk_mass[0]) * to_code),
        },
        1: {
            "Coordinates": to_box(np.asarray(halo_pos, dtype=np.float64), rotation),
            "Velocities": np.tile(BULK_KMS, (sizes.n_halo, 1)),
        },
    }
    return blocks, mass_table


def _star_block(truth: MockGalaxy) -> dict[str, FloatArray]:
    """Return the star datasets of *truth* in code units, followed by two wind particles."""
    to_code = COSMOLOGY.hubble / 1.0e10
    pos = truth.xv[:, :3]
    stars = {
        "Coordinates": to_box(pos, truth.rotation),
        "Velocities": truth.xv[:, 3:] @ truth.rotation.T + BULK_KMS,
        "Masses": truth.mass * to_code,
        "GFM_InitialMass": truth.mass * to_code,
        "GFM_StellarFormationTime": COSMOLOGY.scale_factor_at_lookback(truth.age, 1.0),
        "Potential": np.asarray(truth.potential.potential(pos), dtype=np.float64) + POTENTIAL_OFFSET,
    }
    wind = {name: values[:2].copy() for name, values in stars.items()}
    wind["GFM_StellarFormationTime"][:] = -0.5
    return {name: np.concatenate([values, wind[name]]) for name, values in stars.items()}


def build_mock(snapshot: HaloSnapshot, root: Path, sizes: MockSizes) -> MockGalaxy:
    """Write the mock as *snapshot* under *root*, and return its truth."""
    agama.setRandomSeed(SEED)
    disk, halo = true_components()
    rotation = rotation_matrix(TILT_DEG)
    blocks, mass_table = _matter_blocks(disk, halo, sizes, rotation)
    pot = agama.Potential(disk, halo)
    xv, age = sample_stars(pot, sizes.n_stars, np.random.default_rng(SEED))
    truth = MockGalaxy(xv, age, np.full(age.size, STAR_MASS), rotation, pot)
    write_snapshot(snapshot, root, {**blocks, STAR_TYPE: _star_block(truth)}, mass_table)
    return truth


def _true_bins(mock: MockGalaxy, k: float) -> list[tuple[float, float]]:
    """Return h_Z in the window k h_Z and J_R,med of each age bin, in the original frame and the true potential."""
    pos, vel = mock.xv[:, :3], mock.xv[:, 3:]
    cyl_r, absz = np.hypot(pos[:, 0], pos[:, 1]), np.abs(pos[:, 2])
    annulus = (cyl_r >= R_MIN_KPC) & (cyl_r < R_MAX_KPC) & (absz < Z_SAMPLE_MAX_KPC)
    near = annulus & (absz < Z_JR_MAX_KPC)
    jr = np.full(cyl_r.size, np.nan)
    jr[near] = radial_actions(mock.potential, pos[near], vel[near])
    rows = []
    for lo, hi in AGE_BINS:
        in_bin = annulus & (mock.age >= lo) & (mock.age < hi)
        rows.append((scaled_particle_hz(absz[in_bin], mock.mass[in_bin], k)[0], float(np.median(jr[in_bin & near]))))
    return rows


def check_mock(mock: MockGalaxy, prepared: PreparedHalo, analysis: HaloAnalysis) -> MockCheck:
    """Return the analysis of the mock against its truth."""
    spin_true = mock.rotation @ np.array([0.0, 0.0, 1.0])
    spin_error = float(np.degrees(np.arccos(np.clip(prepared.rotation[2] @ spin_true, -1.0, 1.0))))
    ages = COSMOLOGY.lookback_gyr(COSMOLOGY.scale_factor_at_lookback(mock.age, 1.0), 1.0)
    r_mid = analysis.annulus.r_mid
    phi_fit = vertical_potential(prepared.potential, r_mid)
    phi_true = vertical_potential(mock.potential, r_mid)
    inside = (Z_GRID > DELTA_PHI_RANGE_KPC[0]) & (Z_GRID <= DELTA_PHI_RANGE_KPC[1])
    truth = _true_bins(mock, analysis.mono_age.method.k)
    return MockCheck(
        spin_error_deg=spin_error,
        age_error_gyr=float(np.max(np.abs(ages - mock.age))),
        v_circ=analysis.annulus.v_circ,
        v_circ_true=circular_velocity(mock.potential, r_mid),
        delta_phi_error=float(np.max(np.abs(phi_fit[inside] / phi_true[inside] - 1.0))),
        bins=[
            MockBin(
                age_lo=b.age_lo,
                age_hi=b.age_hi,
                hz=b.heights.measured.value,
                hz_true=hz_true,
                jr_median=b.kinematics.jr_median.value,
                jr_median_true=jr_true,
                hz_tracer=b.predictions.tracer,
            )
            for b, (hz_true, jr_true) in zip(analysis.mono_age.bins, truth, strict=True)
        ],
    )
