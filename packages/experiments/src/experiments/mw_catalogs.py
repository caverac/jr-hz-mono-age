"""Read the Milky Way catalogs of :mod:`experiments.mw_sources` and select the stars of the analysis.

A mono-age population of the Milky Way is a range of ages of the low-[alpha/Fe] disk with
[Fe/H] > ``FEH_MIN``, the same in the star counts and in the kinematics of both surveys:

- APOGEE DR17 with astroNN: low-[alpha/Fe] below [Mg/Fe] = 0.15 - 0.10 [Fe/H], just below
  the valley of the solar-annulus giants; main-survey targets only (``EXTRATARG`` = 0), one
  row per APOGEE_ID;
- LAMOST primary red clump: low-[alpha/Fe] below [alpha/Fe] = 0.12 - 0.05 [Fe/H], just below
  the valley between the sequences of the solar annulus; ages between 0 and 14 Gyr with
  errors below half the age.

The kinematic samples are the giants of the solar annulus ``R_MIN_KPC`` < R < ``R_MAX_KPC``,
|z| < ``Z_KIN_MAX_KPC``, with distance errors below ``MAX_FRAC_DIST_ERR``; the APOGEE
sample also requires 1.0 < log g < 3.3, 4000 < T_eff < 5200 K and age errors below
``MAX_FRAC_AGE_ERR``. APOGEE gives Galactocentric coordinates and velocity errors in the
astroNN frame; the LAMOST stars are placed in that frame (:mod:`experiments.mw_dynamics`)
with velocity errors from ``N_VELOCITY_DRAWS`` Monte Carlo draws of the distance, proper
motions and line-of-sight velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from experiments.mw_dynamics import Phase, Sky, galactocentric
from experiments.populations import R_MAX_KPC, R_MIN_KPC

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

FEH_MIN = -0.6
Z_KIN_MAX_KPC = 1.5
MAX_FRAC_DIST_ERR = 0.10
MAX_FRAC_AGE_ERR = 0.40
LOGG_RANGE = (1.0, 3.3)
TEFF_RANGE = (4000.0, 5200.0)
LAMOST_AGE_MAX_GYR = 14.0
LAMOST_MAX_FRAC_AGE_ERR = 0.5
N_VELOCITY_DRAWS = 30

ASTRONN_COLUMNS = (
    "APOGEE_ID", "source_id", "TEFF", "LOGG", "FE_H", "MG_H", "weighted_dist", "weighted_dist_error",
    "age_lowess_correct", "age_total_error", "galr", "galz", "galvr", "galvt", "galvz", "galvr_err", "galvz_err",
)  # fmt: skip
LAMOST_COLUMNS = (
    "RA", "DEC", "GLON", "GLAT", "RV", "ERR_RV", "FEH", "AFE", "AGE", "ERR_AGE", "G",
    "DISTANCE", "ERR_DISTANCE", "PMRA", "ERR_PMRA", "PMDEC", "ERR_PMDEC",
)  # fmt: skip
GAIA_COLUMNS = ("source_id", "l", "b", "parallax", "parallax_error", "phot_g_mean_mag", "bp_rp")


def apogee_low_alpha(fe_h: FloatArray, mg_fe: FloatArray) -> BoolArray:
    """Return whether APOGEE stars lie on the low-[alpha/Fe] sequence, below [Mg/Fe] = 0.15 - 0.10 [Fe/H]."""
    return np.asarray(mg_fe < 0.15 - 0.10 * fe_h, dtype=np.bool_)


def lamost_low_alpha(fe_h: FloatArray, alpha_fe: FloatArray) -> BoolArray:
    """Return whether LAMOST stars lie on the low-[alpha/Fe] sequence, below [alpha/Fe] = 0.12 - 0.05 [Fe/H]."""
    return np.asarray(alpha_fe < 0.12 - 0.05 * fe_h, dtype=np.bool_)


@dataclass(frozen=True)
class KinematicStars:
    """Stars of a kinematic sample: age (Gyr), Galactocentric phase space, velocity errors (km/s)."""

    age: FloatArray
    phase: Phase
    v_r_err: FloatArray
    v_z_err: FloatArray

    def select(self, keep: BoolArray | NDArray[np.int64]) -> KinematicStars:
        """Return the stars where the mask *keep* is true, or at the indices *keep*."""
        phase = Phase(**{f.name: getattr(self.phase, f.name)[keep] for f in fields(Phase)})
        return KinematicStars(self.age[keep], phase, self.v_r_err[keep], self.v_z_err[keep])


@dataclass(frozen=True)
class AgeTracers:
    """Stars with ages that split the Gaia counts into populations: Gaia DR3 source_id or sky position, G mag."""

    source_id: NDArray[np.int64]  # 0 where unknown
    l: FloatArray  # deg
    b: FloatArray  # deg
    g: FloatArray
    age: FloatArray
    in_disk: BoolArray  # low-[alpha/Fe] and [Fe/H] > FEH_MIN


@dataclass(frozen=True)
class GaiaStars:
    """Gaia DR3 sources: source_id, l and b (deg), parallax and its error (mas), G and BP - RP (mag)."""

    source_id: NDArray[np.int64]
    l: FloatArray
    b: FloatArray
    parallax: FloatArray
    parallax_error: FloatArray
    g: FloatArray
    bp_rp: FloatArray


def _columns(path: Path, names: tuple[str, ...]) -> dict[str, NDArray[np.generic]]:
    """Return the columns *names* of the first table extension of the FITS file at *path*."""
    table = fits.getdata(path, ext=1, memmap=True)
    return {name: np.asarray(table[name]) for name in names}


def read_gaia(path: Path) -> GaiaStars:
    """Return the Gaia DR3 sources of the file at *path*."""
    cols = _columns(path, GAIA_COLUMNS)
    num = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items() if k != "source_id"}
    return GaiaStars(
        np.asarray(cols["source_id"], dtype=np.int64),
        num["l"], num["b"], num["parallax"], num["parallax_error"], num["phot_g_mean_mag"], num["bp_rp"],
    )  # fmt: skip


@dataclass(frozen=True)
class Apogee:
    """The astroNN columns of the analysis, as floats, with source_id, and the rows to keep."""

    num: dict[str, FloatArray]
    source_id: NDArray[np.int64]
    main: BoolArray  # main survey and the first row of its APOGEE_ID


def read_apogee(astronn: Path, allstar: Path) -> Apogee:
    """Return the astroNN catalog at *astronn*, flagged with the main survey of the row-matched *allstar*."""
    cols = _columns(astronn, ASTRONN_COLUMNS)
    _, first = np.unique(cols["APOGEE_ID"], return_index=True)
    main = np.zeros(cols["APOGEE_ID"].size, dtype=np.bool_)
    main[first] = True
    main &= _columns(allstar, ("EXTRATARG",))["EXTRATARG"] == 0
    num = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items() if k not in ("APOGEE_ID", "source_id")}
    return Apogee(num, np.asarray(cols["source_id"], dtype=np.int64), main)


def apogee_kinematics(apogee: Apogee) -> KinematicStars:
    """Return the low-[alpha/Fe] giants of the solar annulus with good distances and ages."""
    num = apogee.num
    finite = np.all(np.isfinite(np.vstack(list(num.values()))), axis=0)
    age = num["age_lowess_correct"]
    with np.errstate(invalid="ignore"):
        keep = (
            apogee.main & finite
            & (num["LOGG"] > LOGG_RANGE[0]) & (num["LOGG"] < LOGG_RANGE[1])
            & (num["TEFF"] > TEFF_RANGE[0]) & (num["TEFF"] < TEFF_RANGE[1])
            & (num["FE_H"] > FEH_MIN)
            & (num["weighted_dist_error"] < MAX_FRAC_DIST_ERR * num["weighted_dist"])
            & (age > 0.0) & (num["age_total_error"] < MAX_FRAC_AGE_ERR * age)
            & (num["galr"] > R_MIN_KPC) & (num["galr"] < R_MAX_KPC) & (np.abs(num["galz"]) < Z_KIN_MAX_KPC)
            & apogee_low_alpha(num["FE_H"], num["MG_H"] - num["FE_H"])
        )  # fmt: skip
    phase = Phase(num["galr"][keep], num["galz"][keep], num["galvr"][keep], num["galvt"][keep], num["galvz"][keep])
    return KinematicStars(age[keep], phase, num["galvr_err"][keep], num["galvz_err"][keep])


def apogee_age_tracers(apogee: Apogee) -> AgeTracers:
    """Return the main-survey APOGEE stars with a Gaia DR3 source_id, an age and abundances."""
    num = apogee.num
    fe_h, age = num["FE_H"], num["age_lowess_correct"]
    with np.errstate(invalid="ignore"):
        keep = (
            apogee.main & (apogee.source_id > 0) & np.isfinite(age) & (age > 0.0)
            & np.isfinite(num["MG_H"]) & np.isfinite(fe_h)
        )  # fmt: skip
    in_disk = apogee_low_alpha(fe_h[keep], num["MG_H"][keep] - fe_h[keep]) & (fe_h[keep] > FEH_MIN)
    nan = np.full(int(keep.sum()), np.nan)
    return AgeTracers(apogee.source_id[keep], nan, nan, nan, age[keep], in_disk)


def read_lamost(path: Path) -> dict[str, FloatArray]:
    """Return the LAMOST red-clump stars at *path* whose ages are good, as columns of floats."""
    cat = {k: np.asarray(v, dtype=np.float64) for k, v in _columns(path, LAMOST_COLUMNS).items()}
    good = (
        (cat["AGE"] > 0.0) & (cat["AGE"] < LAMOST_AGE_MAX_GYR) & (cat["ERR_AGE"] < LAMOST_MAX_FRAC_AGE_ERR * cat["AGE"])
    )
    return {k: v[good] for k, v in cat.items()}


def lamost_in_disk(cat: dict[str, FloatArray]) -> BoolArray:
    """Return whether the LAMOST stars of *cat* belong to the low-[alpha/Fe] disk with [Fe/H] > ``FEH_MIN``."""
    return lamost_low_alpha(cat["FEH"], cat["AFE"]) & (cat["FEH"] > FEH_MIN)


def lamost_age_tracers(cat: dict[str, FloatArray]) -> AgeTracers:
    """Return the LAMOST red-clump stars as age tracers, to be matched to Gaia by position and G."""
    return AgeTracers(
        np.zeros(cat["AGE"].size, dtype=np.int64), cat["GLON"], cat["GLAT"], cat["G"], cat["AGE"], lamost_in_disk(cat)
    )


def lamost_kinematics(cat: dict[str, FloatArray], rng: np.random.Generator) -> KinematicStars:
    """Return the disk red-clump stars of the solar annulus in the astroNN frame, with Monte Carlo velocity errors."""
    keep = lamost_in_disk(cat) & (cat["ERR_DISTANCE"] < MAX_FRAC_DIST_ERR * cat["DISTANCE"])
    c = {k: v[keep] for k, v in cat.items()}
    phase = galactocentric(Sky(c["RA"], c["DEC"], c["DISTANCE"], c["PMRA"], c["PMDEC"], c["RV"]))
    inside = (phase.r > R_MIN_KPC) & (phase.r < R_MAX_KPC) & (np.abs(phase.z) < Z_KIN_MAX_KPC)
    c = {k: v[inside] for k, v in c.items()}
    shape = (N_VELOCITY_DRAWS, int(inside.sum()))

    def draw(key: str, err: str) -> FloatArray:
        return np.asarray((c[key] + c[err] * rng.standard_normal(shape)).ravel(), dtype=np.float64)

    def tile(key: str) -> FloatArray:
        return np.tile(c[key], N_VELOCITY_DRAWS)

    drawn = galactocentric(
        Sky(
            tile("RA"), tile("DEC"), np.abs(draw("DISTANCE", "ERR_DISTANCE")),
            draw("PMRA", "ERR_PMRA"), draw("PMDEC", "ERR_PMDEC"), draw("RV", "ERR_RV"),
        )
    )  # fmt: skip
    kept = Phase(**{f.name: getattr(phase, f.name)[inside] for f in fields(Phase)})
    return KinematicStars(
        c["AGE"],
        kept,
        np.asarray(np.std(drawn.v_r.reshape(shape), axis=0), dtype=np.float64),
        np.asarray(np.std(drawn.v_z.reshape(shape), axis=0), dtype=np.float64),
    )
