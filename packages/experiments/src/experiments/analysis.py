"""The analysis of one Auriga halo: read the snapshot once, fit its potential, and measure everything.

1. Read the particles within 300 kpc of the main galaxy, in physical units, without wind
   particles (:func:`experiments.snapshot.read_galaxy`).
2. Rotate them into the disk frame and fit the axisymmetric AGAMA potential
   (:func:`experiments.potential.fit_potential`).
3. In the annulus, measure the mono-age populations (:mod:`experiments.populations`),
   with the tracer in the fitted potential and in the snapshot's, and the tilt term of
   their Jeans equation over a wider range of radii.
4. Measure the whole disk in rings over Palicio's radial range (:mod:`experiments.whole_disk`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.auriga import HaloSnapshot
from experiments.populations import (
    R_MAX_KPC,
    R_MIN_KPC,
    Method,
    MonoAge,
    VerticalModel,
    annulus_sample,
    measure_mono_age,
    median_radius,
    outer_sample,
    surface_density,
)
from experiments.potential import (
    SliceCheck,
    VerticalResidual,
    circular_velocity,
    fit_potential,
    potential_check,
    snapshot_vertical_potential,
    vertical_potential,
)
from experiments.snapshot import Galaxy, galaxy_frame, read_galaxy
from experiments.whole_disk import WholeDisk, measure_whole_disk

FloatArray = NDArray[np.float64]

MAX_REDSHIFT = 1.0e-3


@dataclass(frozen=True)
class PreparedHalo:
    """A snapshot's galaxy in its disk frame, the rotation from the box to that frame, and the fitted potential."""

    snapshot: HaloSnapshot
    galaxy: Galaxy
    rotation: FloatArray
    potential: agama.Potential


@dataclass(frozen=True)
class Annulus:
    """The annulus of the mono-age populations: bounds and median radius (kpc), v_c (km/s), Sigma (Msun/pc^2)."""

    r_lo: float
    r_hi: float
    r_mid: float
    v_circ: float
    sigma_surf: float
    n_stars: int


@dataclass(frozen=True)
class HaloAnalysis:
    """Everything measured in one halo."""

    snapshot: HaloSnapshot
    annulus: Annulus
    potential_check: list[SliceCheck]
    residual: VerticalResidual
    mono_age: MonoAge
    whole_disk: WholeDisk


def prepare_halo(snapshot: HaloSnapshot, root: Path) -> PreparedHalo:
    """Return *snapshot* under *root* in its disk frame, with its fitted potential; the snapshot must be at z = 0."""
    galaxy = read_galaxy(snapshot, root)
    if abs(galaxy.header.redshift) > MAX_REDSHIFT:
        raise ValueError(f"snapshot {snapshot.snapshot} of Au{snapshot.halo} is at z = {galaxy.header.redshift:.3f}")
    bulk, rotation = galaxy_frame(galaxy.stars)
    galaxy = galaxy.in_frame(bulk, rotation)
    return PreparedHalo(snapshot, galaxy, rotation, fit_potential(galaxy))


def analyze_halo(prepared: PreparedHalo, method: Method) -> HaloAnalysis:
    """Return the measurements of the *prepared* halo with the choices of *method*."""
    galaxy, pot = prepared.galaxy, prepared.potential
    sample, annulus = annulus_sample(galaxy, pot, R_MIN_KPC, R_MAX_KPC)
    r_mid = median_radius(galaxy, annulus)
    sigma_surf = surface_density(galaxy, R_MIN_KPC, R_MAX_KPC)
    phi_snapshot, residual = snapshot_vertical_potential(pot, galaxy.stars, annulus, r_mid)
    model = VerticalModel(sigma_surf, vertical_potential(pot, r_mid), phi_snapshot, r_mid)
    return HaloAnalysis(
        snapshot=prepared.snapshot,
        annulus=Annulus(R_MIN_KPC, R_MAX_KPC, r_mid, circular_velocity(pot, r_mid), sigma_surf, int(annulus.sum())),
        potential_check=potential_check(pot, galaxy.stars, annulus),
        residual=residual,
        mono_age=measure_mono_age(sample, outer_sample(galaxy, R_MIN_KPC, R_MAX_KPC), model, method),
        whole_disk=measure_whole_disk(galaxy, pot, method.window),
    )
