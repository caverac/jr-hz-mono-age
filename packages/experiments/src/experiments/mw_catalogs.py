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

from dataclasses import dataclass, fields, replace
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
ALLSTAR_COLUMNS = (
    "EXTRATARG", "M_H", "ALPHA_M", "J", "K", "H", "AK_TARG", "MIN_H", "MAX_H", "MIN_JK", "MAX_JK",
)  # fmt: skip


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
    in_disk: BoolArray  # in the populations measured: by default low-[alpha/Fe] with [Fe/H] > FEH_MIN


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
class Targeting:
    """The APOGEE targeting of each star: its H, its (J - K_s)_0, and the limits of the cohort it was targeted in.

    (J - K_s)_0 = J - K_s - 1.5 A_Ks, with the A_Ks of the targeting (``AK_TARG``), as in the
    target selection of Zasowski et al. (2013, AJ 146, 81). A cohort without a limit carries
    -9999.99 or 9999.99.
    """

    h: FloatArray
    jk0: FloatArray
    min_h: FloatArray
    max_h: FloatArray
    min_jk: FloatArray
    max_jk: FloatArray


@dataclass(frozen=True)
class TargetingBox:
    """A range of H and of (J - K_s)_0 that holds the same selection in every field.

    A star belongs to the box when it lies in it and its cohort was targeted with limits
    that contain the box, so that the stars of the box are selected alike wherever they are.
    """

    h_lo: float
    h_hi: float
    jk_lo: float
    jk_hi: float

    def holds(self, targeting: Targeting) -> BoolArray:
        """Return which stars of *targeting* belong to the box; False where a value is NaN."""
        t = targeting
        return np.asarray(
            (t.min_h <= self.h_lo) & (t.max_h >= self.h_hi) & (t.min_jk <= self.jk_lo) & (t.max_jk >= self.jk_hi)
            & (t.h >= self.h_lo) & (t.h <= self.h_hi) & (t.jk0 >= self.jk_lo) & (t.jk0 <= self.jk_hi),
            dtype=np.bool_,
        )  # fmt: skip


@dataclass(frozen=True)
class Apogee:  # pylint: disable=too-many-instance-attributes
    """The astroNN columns of the analysis, as floats, with source_id, the rows to keep, abundances and targeting.

    ``m_h`` and ``alpha_m`` are the ASPCAP [M/H] and [alpha/M] of the row-matched allStar,
    in which Imig et al. (2025) bin their populations, and ``targeting`` its targeting. They
    stay out of ``num``, whose columns the kinematic sample requires to be finite.
    """

    num: dict[str, FloatArray]
    source_id: NDArray[np.int64]
    main: BoolArray  # main survey and the first row of its APOGEE_ID
    apogee_id: NDArray[np.str_]
    m_h: FloatArray
    alpha_m: FloatArray
    targeting: Targeting

    def within(self, box: TargetingBox) -> Apogee:
        """Return the catalog with only the stars of *box* kept."""
        return replace(self, main=self.main & box.holds(self.targeting))


def read_apogee(astronn: Path, allstar: Path) -> Apogee:
    """Return the astroNN catalog at *astronn*, flagged with the main survey of the row-matched *allstar*."""
    cols = _columns(astronn, ASTRONN_COLUMNS)
    _, first = np.unique(cols["APOGEE_ID"], return_index=True)
    main = np.zeros(cols["APOGEE_ID"].size, dtype=np.bool_)
    main[first] = True
    star = {k: np.asarray(v, dtype=np.float64) for k, v in _columns(allstar, ALLSTAR_COLUMNS).items()}
    main &= star["EXTRATARG"] == 0
    num = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items() if k not in ("APOGEE_ID", "source_id")}
    ids = np.char.strip(np.asarray(cols["APOGEE_ID"]).astype(np.str_))
    targeting = Targeting(
        star["H"], star["J"] - star["K"] - 1.5 * star["AK_TARG"],
        star["MIN_H"], star["MAX_H"], star["MIN_JK"], star["MAX_JK"],
    )  # fmt: skip
    return Apogee(
        num, np.asarray(cols["source_id"], dtype=np.int64), main, ids, star["M_H"], star["ALPHA_M"], targeting
    )


def with_distmass_ages(apogee: Apogee, distmass: Path) -> Apogee:
    """Return *apogee* with the ages of the distmass catalog at *distmass* in place of the astroNN ages.

    The distmass catalog of Stone-Martinez et al. (2024, AJ 167, 73) is row-matched to
    allStar, as the astroNN catalog is; a row whose APOGEE_ID differs between the two gets
    no age. The age is ``AGE_UNCOR_SS``, the one Imig et al. (2025) adopt, and its error the
    mean of the upper and lower offsets of ``AGE_ERR`` (dex), converted to Gyr, so that the
    cut on the relative age error applies to both age scales alike.
    """
    cols = _columns(distmass, ("APOGEE_ID", "AGE_UNCOR_SS", "AGE_ERR"))
    same = np.char.strip(np.asarray(cols["APOGEE_ID"]).astype(np.str_)) == apogee.apogee_id
    age = np.where(same, np.asarray(cols["AGE_UNCOR_SS"], dtype=np.float64) / 1.0e9, np.nan)
    err = np.asarray(cols["AGE_ERR"], dtype=np.float64)
    fraction = 0.5 * (10.0 ** err[:, 0] - 10.0 ** err[:, 1])
    num = {**apogee.num, "age_lowess_correct": age, "age_total_error": age * fraction}
    return replace(apogee, num=num)


def apogee_kinematics(apogee: Apogee, population: BoolArray | None = None) -> KinematicStars:
    """Return the giants of the solar annulus with good distances and ages in *population* of the rows of *apogee*.

    The population is by default the low-[alpha/Fe] sequence of :func:`apogee_low_alpha`.
    """
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
        )  # fmt: skip
        keep &= apogee_low_alpha(num["FE_H"], num["MG_H"] - num["FE_H"]) if population is None else population
    phase = Phase(num["galr"][keep], num["galz"][keep], num["galvr"][keep], num["galvt"][keep], num["galvz"][keep])
    return KinematicStars(age[keep], phase, num["galvr_err"][keep], num["galvz_err"][keep])


def apogee_age_tracers(apogee: Apogee, population: BoolArray | None = None) -> AgeTracers:
    """Return the main-survey APOGEE stars with a Gaia DR3 source_id, an age and abundances.

    Their populations are the rows where *population* holds, by default the low-[alpha/Fe]
    stars with [Fe/H] > ``FEH_MIN``; the stars kept do not depend on it.
    """
    num = apogee.num
    fe_h, age = num["FE_H"], num["age_lowess_correct"]
    with np.errstate(invalid="ignore"):
        keep = (
            apogee.main & (apogee.source_id > 0) & np.isfinite(age) & (age > 0.0)
            & np.isfinite(num["MG_H"]) & np.isfinite(fe_h)
        )  # fmt: skip
        if population is None:
            population = apogee_low_alpha(fe_h, num["MG_H"] - fe_h) & (fe_h > FEH_MIN)
    nan = np.full(int(keep.sum()), np.nan)
    return AgeTracers(apogee.source_id[keep], nan, nan, nan, age[keep], population[keep])


def scattered_ages(age: FloatArray, err: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Return *age* drawn again from its error *err*, lognormal with err / age as the dispersion of ln age.

    Added to the error the age already carries, the draw doubles its variance. A star without
    a finite, positive age or error keeps its age.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        good = np.isfinite(age) & np.isfinite(err) & (age > 0.0) & (err > 0.0)
        width = np.where(good, err / age, 0.0)
    return np.where(good, age * np.exp(width * rng.standard_normal(age.size)), age)


def with_scattered_ages(apogee: Apogee, rng: np.random.Generator) -> Apogee:
    """Return *apogee* with its ages drawn again from their errors (:func:`scattered_ages`)."""
    num = apogee.num
    age = scattered_ages(num["age_lowess_correct"], num["age_total_error"], rng)
    return replace(apogee, num={**num, "age_lowess_correct": age})


def scattered_lamost(cat: dict[str, FloatArray], rng: np.random.Generator) -> dict[str, FloatArray]:
    """Return the LAMOST catalog *cat* with its ages drawn again from their errors (:func:`scattered_ages`)."""
    return {**cat, "AGE": scattered_ages(cat["AGE"], cat["ERR_AGE"], rng)}


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


def lamost_age_tracers(cat: dict[str, FloatArray], population: BoolArray | None = None) -> AgeTracers:
    """Return the LAMOST red-clump stars as age tracers, to be matched to Gaia by position and G.

    Their populations are the stars where *population* holds, by default those of
    :func:`lamost_in_disk`.
    """
    return AgeTracers(
        np.zeros(cat["AGE"].size, dtype=np.int64), cat["GLON"], cat["GLAT"], cat["G"], cat["AGE"],
        lamost_in_disk(cat) if population is None else population,
    )  # fmt: skip


def lamost_kinematics(
    cat: dict[str, FloatArray], rng: np.random.Generator, population: BoolArray | None = None
) -> KinematicStars:
    """Return the red-clump stars of *population* in the solar annulus, astroNN frame, with Monte Carlo velocity errors.

    The population is by default the disk of :func:`lamost_in_disk`.
    """
    keep = (lamost_in_disk(cat) if population is None else population) & (
        cat["ERR_DISTANCE"] < MAX_FRAC_DIST_ERR * cat["DISTANCE"]
    )
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
