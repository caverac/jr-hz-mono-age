"""The window-free scale heights of Imig et al. (2025) against our star counts and the isothermal tracer.

Imig et al. (2025, ApJ 990, 203) fit every population of APOGEE DR17 in bins of
[alpha/M], [M/H] and age with an exponential vertical profile over the whole survey
footprint, correcting for its selection function, so their heights carry no window of
|z|. Their model (Section III.3, Eqs. 9, 11 and 12) is nu(R, z) = nu_sun Sigma(R) xi(R, z):
Sigma a broken exponential of the midplane density, with scale lengths h_R,in inside
R_break and h_R,out outside, and xi = exp(-|z| / h_Z(R)) with the linear flare
h_Z(R) = h_Z,sun + A_flare (R - R_sun). Their Sun is at R_sun = ``R_SUN_IMIG_KPC``
(Sections II.2 and III.3), nu_sun is a number density of red giants, and their ages come
from the distmass catalog. The grids ``HZ_R_PROFILE`` of their table pivot the flare at
8.122 kpc instead, which would change h_Z at their Sun by at most 0.013 kpc for the
populations kept here.

Our counts are around the Sun, so their model enters them around their Sun, whatever the
radius of the Sun in either frame: :func:`cylinder_slices` integrates it over the disk of
our cylinder around R_sun and over each slice of |z|. Their plane lies 6 pc below ours
(Z_sun = 27 pc against 20.8 pc), which is neglected.

For each of their age bins up to ``AGE_MAX_GYR``, this module measures four h_Z, all
through our counts (:func:`experiments.mw_counts.predicted_hz`), so that each difference
between their heights and our star counts can be taken out in turn:

- ``imig_mass``: their low-[alpha/M] populations with [M/H] > ``M_H_MIN`` pooled by their
  midplane mass densities at R_sun, Sigma_i / (2 h_i);
- ``imig_counts``: the same populations pooled by their number densities of red giants
  nu_i at R_sun, the analog of our counts of giants;
- ``counts_astronn`` and ``counts_distmass``: our counts of the Gaia giants split into ages
  by APOGEE (:mod:`experiments.mw_counts`), in the same range of ages, with the ages of
  astroNN and with those of distmass.

The tracer of each bin uses the sigma_z of the APOGEE kinematic stars of the same ages on
the distmass scale, theirs, so that its comparison with their heights holds the age scale
fixed, and it goes through the same counts. The errors come from ``N_DRAWS`` draws in which
every h_i, Sigma_i and nu_i comes from a split normal with its published lower and upper
errors, and the counts and the kinematic stars are resampled; the flare and the radial
profile keep their best values. Every model height of a draw goes through the counts of
that draw.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from experiments.heights import Z_GRID, tracer_profile
from experiments.mw_analysis import SolarModel
from experiments.mw_catalogs import KinematicStars
from experiments.mw_counts import (
    R_CYL_KPC,
    SLICE_EDGES_KPC,
    CountSample,
    count_hz,
    predicted_hz,
    profile_slices,
)
from experiments.populations import Z_MID_MAX_KPC, Estimate, Trend, fit_trend
from experiments.stats import deconvolved_dispersion

FloatArray = NDArray[np.float64]

M_H_MIN = -0.6
AGE_MAX_GYR = 10.0
R_SUN_IMIG_KPC = 8.3
N_DRAWS = 300
COLUMNS = (
    "MIN_AGE", "MAX_AGE", "H_Z", "H_Z_ERR", "SMD_0", "SMD_0_ERRS", "NU_0", "NU_0_ERRS", "A_FLARE",
    "MIN_M_H", "MAX_M_H", "N_STARS_OBS", "HR_IN", "HR_OUT", "R_BREAK",
)  # fmt: skip
# Nodes of the mean over the disk of our cylinder: Gauss-Legendre in the distance from the Sun,
# midpoints in azimuth. The kink of Sigma at R_break makes the convergence slow; with these, the
# mass of every slice of every population is within 2e-4 of 64 x 256 nodes.
N_RHO_NODES = 32
N_PHI_NODES = 64


@dataclass(frozen=True)
class ImigHeights:
    """The four h_Z (kpc) of one age bin, or a ratio per height: theirs by mass and by counts, ours per age scale."""

    imig_mass: Estimate
    imig_counts: Estimate
    counts_astronn: Estimate
    counts_distmass: Estimate


@dataclass(frozen=True)
class ImigKinematics:
    """The APOGEE kinematic stars of one age bin on the distmass scale: their count, sigma_z, the tracer h_Z.

    The tracer h_Z is the one our distmass counts give its profile.
    """

    n_kin: int
    sigma_z: Estimate
    tracer: float


@dataclass(frozen=True)
class ImigBin:
    """One age bin of Imig et al.: its heights, the kinematics of its ages, and the tracer over each height."""

    age_lo: float
    age_hi: float
    n_populations: int
    heights: ImigHeights
    kinematics: ImigKinematics
    ratios: ImigHeights


@dataclass(frozen=True)
class ImigCheck:
    """The heights of Imig et al. against the counts and the tracer, bin by bin, and the trend of each ratio."""

    bins: list[ImigBin]
    trends: dict[str, Trend]


@dataclass(frozen=True)
class ImigPopulations:  # pylint: disable=too-many-instance-attributes
    """The populations of one age bin: h_Z, Sigma and nu at R_sun with lower and upper errors, the flare, the [M/H] bin.

    ``m_h_lo`` and ``m_h_hi`` are the edges of the [M/H] bin of each population,
    ``n_stars`` the number of their stars in it, and ``hr_in``, ``hr_out`` and ``r_break``
    (kpc) its broken exponential in R.
    """

    age_lo: float
    age_hi: float
    h: FloatArray
    h_err: FloatArray  # (n, 2)
    sigma: FloatArray
    sigma_err: FloatArray  # (n, 2)
    nu: FloatArray
    nu_err: FloatArray  # (n, 2)
    a_flare: FloatArray
    m_h_lo: FloatArray
    m_h_hi: FloatArray
    n_stars: FloatArray
    hr_in: FloatArray
    hr_out: FloatArray
    r_break: FloatArray


@dataclass(frozen=True)
class ImigInputs:
    """What the comparison needs besides their table: our APOGEE counts on both age scales, and the kinematics."""

    counts_astronn: CountSample
    counts_distmass: CountSample
    kinematics: KinematicStars  # on the distmass scale


def read_imig(path: Path) -> list[ImigPopulations]:
    """Return the low-[alpha/M] populations with [M/H] > ``M_H_MIN``, by age bin up to ``AGE_MAX_GYR``."""
    table = fits.getdata(path, ext=1)
    group = np.array([str(value).strip() for value in table["ALPHA_M"]])
    keep = (
        (group == "LOW") & (table["M_H"] > M_H_MIN) & (table["MAX_AGE"] <= AGE_MAX_GYR + 0.01)
        & np.isfinite(table["H_Z"]) & np.isfinite(table["SMD_0"]) & (table["SMD_0"] > 0.0)
        & np.isfinite(table["NU_0"]) & (table["NU_0"] > 0.0)
        & np.isfinite(table["HR_IN"]) & np.isfinite(table["HR_OUT"]) & np.isfinite(table["R_BREAK"])
    )  # fmt: skip
    cols = {k: np.asarray(table[k][keep], dtype=np.float64) for k in COLUMNS}
    bins = []
    for lo in np.unique(cols["MIN_AGE"]):
        rows = cols["MIN_AGE"] == lo
        bins.append(
            ImigPopulations(
                float(lo), float(cols["MAX_AGE"][rows][0]),
                *(cols[k][rows] for k in COLUMNS[2:]),
            )  # fmt: skip
        )
    return bins


def _cylinder_nodes() -> tuple[FloatArray, FloatArray]:
    """Return R (kpc) and the weight of the nodes over the disk of radius ``R_CYL_KPC`` around their Sun.

    The weights sum to one, so a sum over the nodes is a mean over the disk.
    """
    x, w = np.polynomial.legendre.leggauss(N_RHO_NODES)
    rho = 0.5 * R_CYL_KPC * (x + 1.0)
    phi = (np.arange(N_PHI_NODES) + 0.5) * 2.0 * np.pi / N_PHI_NODES
    r = np.sqrt(R_SUN_IMIG_KPC**2 + rho[:, None] ** 2 - 2.0 * R_SUN_IMIG_KPC * rho[:, None] * np.cos(phi[None, :]))
    weight = np.repeat(0.5 * R_CYL_KPC * w * rho * 2.0 * np.pi / N_PHI_NODES, N_PHI_NODES) / (np.pi * R_CYL_KPC**2)
    return np.asarray(r.ravel(), dtype=np.float64), np.asarray(weight, dtype=np.float64)


NODE_R_KPC, NODE_WEIGHT = _cylinder_nodes()


def _model_at_nodes(pops: ImigPopulations, rows: NDArray[np.int64], h: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return Sigma(R) / Sigma(R_sun) and h_Z(R) of the populations *rows* at the nodes, with *h* at R_sun.

    Both have one row per entry of *rows*. Their Eq. (11) is continuous at R_break, so the
    midplane density relative to R_sun needs no constant.
    """

    def log_sigma(r: FloatArray) -> FloatArray:
        r_break = pops.r_break[rows, None]
        inner = -(r - r_break) / pops.hr_in[rows, None]
        return np.asarray(np.where(r <= r_break, inner, -(r - r_break) / pops.hr_out[rows, None]), dtype=np.float64)

    radial = np.exp(log_sigma(NODE_R_KPC[None, :]) - log_sigma(np.full((1, 1), R_SUN_IMIG_KPC)))
    scale = h[:, None] + pops.a_flare[rows, None] * (NODE_R_KPC[None, :] - R_SUN_IMIG_KPC)
    if np.any(scale <= 0.0):
        raise ValueError("the flare makes h_Z negative inside the cylinder")
    return np.asarray(radial, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def cylinder_slices(pops: ImigPopulations, rows: NDArray[np.int64], h: FloatArray) -> FloatArray:
    """Return the mass of each population *rows* of *pops* in every slice of our counts, with *h* at their Sun.

    The mass is nu(R, z) / nu_sun integrated over the slice of |z| and averaged over the disk
    of the cylinder around their Sun: per unit midplane density at R_sun. One row per entry of
    *rows*, so that *rows* may repeat one population with a draw of its h_Z in each.
    """
    radial, scale = _model_at_nodes(pops, rows, h)
    edges = np.asarray(SLICE_EDGES_KPC)
    per_node = -scale[:, :, None] * np.diff(np.exp(-edges[None, None, :] / scale[:, :, None]), axis=2)
    return np.asarray(np.einsum("pns,pn->ps", per_node, radial * NODE_WEIGHT[None, :]), dtype=np.float64)


def cylinder_profile(pops: ImigPopulations, row: int, z_grid: FloatArray = Z_GRID) -> FloatArray:
    """Return the density of population *row* on *z_grid*, averaged over the disk of the cylinder, per nu_sun."""
    radial, scale = _model_at_nodes(pops, np.array([row]), pops.h[[row]])
    return np.asarray(np.exp(-z_grid[:, None] / scale[0][None, :]) @ (radial[0] * NODE_WEIGHT), dtype=np.float64)


def pooled_slices(pops: ImigPopulations, h: FloatArray, midplane: FloatArray) -> FloatArray:
    """Return the mass in every slice of all the populations of *pops*, of scales *h* and midplane densities *midplane*.

    The scales *h* are at their Sun, and *midplane* are the densities there.
    """
    return np.asarray(midplane @ cylinder_slices(pops, np.arange(h.size), h), dtype=np.float64)


def split_normal(rng: np.random.Generator, value: FloatArray, err: FloatArray) -> FloatArray:
    """Return one draw per value from a split normal with lower and upper errors ``err[:, 0]`` and ``err[:, 1]``."""
    z = rng.standard_normal(value.size)
    return np.asarray(np.maximum(value + np.where(z < 0.0, err[:, 0], err[:, 1]) * z, 1e-3 * value), dtype=np.float64)


@dataclass(frozen=True)
class _Parameters:
    """Their h_Z, Sigma and nu at R_sun of the populations of one bin: the best values or one draw."""

    h: FloatArray
    sigma: FloatArray
    nu: FloatArray

    @classmethod
    def of(cls, pops: ImigPopulations, rng: np.random.Generator | None) -> _Parameters:
        """Return the best values (*rng* None), or one draw of every parameter from its split normal."""
        if rng is None:
            return cls(pops.h, pops.sigma, pops.nu)
        pairs = ((pops.h, pops.h_err), (pops.sigma, pops.sigma_err), (pops.nu, pops.nu_err))
        h, sigma, nu = (split_normal(rng, value, err) for value, err in pairs)
        return cls(h, sigma, nu)

    def heights(self, pops: ImigPopulations, counts: CountSample, k: float) -> tuple[float, float]:
        """Return h_Z through *counts* of the populations pooled by mass and by number of giants."""
        by_mass = pooled_slices(pops, self.h, self.sigma / (2.0 * self.h))
        return predicted_hz(counts, by_mass, k)[0], predicted_hz(counts, pooled_slices(pops, self.h, self.nu), k)[0]


@dataclass(frozen=True)
class _Values:
    """What one draw of a bin gives: the four heights, sigma_z and the tracer, all through the counts of the draw."""

    heights: tuple[float, float, float, float]
    sigma_z: float
    tracer: float


def _values(pops: ImigPopulations, inputs: ImigInputs, model: SolarModel, rng: np.random.Generator | None) -> _Values:
    """Return the heights, sigma_z and the tracer of one bin, at the best values (*rng* None) or for one draw."""
    kin = inputs.kinematics
    in_age = (kin.age >= pops.age_lo) & (kin.age < pops.age_hi) & (np.abs(kin.phase.z) < Z_MID_MAX_KPC)
    v_z, v_z_err = kin.phase.v_z[in_age], kin.v_z_err[in_age]
    parameters = _Parameters.of(pops, rng)
    astronn, distmass = inputs.counts_astronn, inputs.counts_distmass
    if rng is not None:
        rows = rng.integers(0, v_z.size, v_z.size)
        v_z, v_z_err = v_z[rows], v_z_err[rows]
        astronn, distmass = astronn.resampled(rng), distmass.resampled(rng)
    heights = (
        *parameters.heights(pops, distmass, model.k),
        count_hz(astronn, pops.age_lo, pops.age_hi, model.k)[0],
        count_hz(distmass, pops.age_lo, pops.age_hi, model.k)[0],
    )
    sigma_z = deconvolved_dispersion(v_z, v_z_err)
    tracer = predicted_hz(distmass, profile_slices(tracer_profile(model.phi, sigma_z)), model.k)[0]
    return _Values(heights, sigma_z, tracer)


def measure_imig_bin(
    pops: ImigPopulations, inputs: ImigInputs, model: SolarModel, rng: np.random.Generator, n_draws: int
) -> ImigBin:
    """Return the four heights of one age bin, the tracer of the kinematic stars of its ages, and their ratios."""
    point = _values(pops, inputs, model, None)
    draws = [_values(pops, inputs, model, rng) for _ in range(n_draws)]
    tracers = np.array([d.tracer for d in draws])
    tracer = point.tracer
    n_heights = len(fields(ImigHeights))

    kin = inputs.kinematics
    in_age = (kin.age >= pops.age_lo) & (kin.age < pops.age_hi) & (np.abs(kin.phase.z) < Z_MID_MAX_KPC)
    return ImigBin(
        age_lo=pops.age_lo,
        age_hi=pops.age_hi,
        n_populations=int(pops.h.size),
        heights=ImigHeights(
            *(Estimate.of(point.heights[j], np.array([d.heights[j] for d in draws])) for j in range(n_heights))
        ),
        kinematics=ImigKinematics(
            int(in_age.sum()), Estimate.of(point.sigma_z, np.array([d.sigma_z for d in draws])), tracer
        ),
        ratios=ImigHeights(
            *(
                Estimate.of(tracer / point.heights[j], tracers / np.array([d.heights[j] for d in draws]))
                for j in range(n_heights)
            )
        ),
    )


def measure_imig(
    populations: list[ImigPopulations],
    inputs: ImigInputs,
    model: SolarModel,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
) -> ImigCheck:
    """Return the comparison in every age bin of *populations*, and the trend of each ratio with the mean age.

    A trend uses the bins whose ratio is finite; the age of a bin is the geometric mean of
    its edges.
    """
    bins = [measure_imig_bin(pops, inputs, model, rng, n_draws) for pops in populations]
    age = np.sqrt(np.array([b.age_lo * b.age_hi for b in bins]))
    trends = {}
    for f in fields(ImigHeights):
        ratio = np.array([getattr(b.ratios, f.name).value for b in bins])
        err = np.array([getattr(b.ratios, f.name).err for b in bins])
        ok = np.isfinite(ratio) & np.isfinite(err) & (err > 0.0)
        trends[f.name] = fit_trend(age[ok], ratio[ok], err[ok])
    return ImigCheck(bins, trends)
