"""The local vertical potential from the tracer: the (rho_0, Sigma_1.1) at which every age bin of a survey agrees.

Every age bin carries the ratio of a predicted to the measured h_Z in each potential of the
family around McMillan (2017) (:mod:`experiments.mw_potentials`). Across the family, the
logarithm of that ratio is close to linear in ln rho_0 and ln Sigma_1.1, two numbers that
bound the vertical potential near the plane, so for each bin b a least-squares fit over the
members gives

    ln r_b = a_b + c_b ln rho_0 + d_b ln Sigma_1.1.

Where the potential is harmonic h ~ rho_0^(-1/2), so a cold bin has c_b near -1/2 and d_b
near 0, whereas a hot bin reaches heights that Sigma_1.1 also sets. If the prediction holds,
one (rho_0, Sigma_1.1) brings every bin to r_b = 1. We find it by weighted least squares,
each bin weighted by the relative bootstrap error of its tracer ratio, and the chi^2 per
degree of freedom says whether the bins agree; the errors are scaled by its root where it
exceeds 1. This is the inverse use of the tracer of Bovy & Rix (2013, ApJ 779, 115), with the
family standing in for a parameterized potential. ``bin_rho0`` gives, for each bin, the
rho_0 that brings it alone to 1 at the fitted Sigma_1.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.populations import Estimate

FloatArray = NDArray[np.float64]

N_MIN_INVERSION_BINS = 3  # bins for two parameters and one degree of freedom


@dataclass(frozen=True)
class Inversion:  # pylint: disable=too-many-instance-attributes
    """The rho_0 (Msun/pc^3) and Sigma_1.1 (Msun/pc^2) that bring every bin of one survey to the measured h_Z.

    ``correlation`` is that of ln rho_0 and ln Sigma_1.1, ``model_rms`` the largest rms of the
    fit of a bin across the family, in ln r, which bounds how well two numbers describe the
    potentials of the family.
    """

    survey: str
    n_bins: int
    rho0: Estimate
    sigma_11: Estimate
    correlation: float
    chi2_dof: float
    model_rms: float
    bin_rho0: list[Estimate]


def _nan_inversion(name: str, n_bins: int) -> Inversion:
    """Return the inversion of a survey with too few bins."""
    none = Estimate(float("nan"), float("nan"))
    return Inversion(name, n_bins, none, none, float("nan"), float("nan"), float("nan"), [])


@dataclass(frozen=True)
class FamilyRatios:
    """One survey's ratios over the family: rho_0 and Sigma_1.1 of each member, and per bin its ratios and error.

    ``ratios`` has one row per bin and one column per member; ``rel_err`` is the relative
    bootstrap error of the tracer ratio of each bin, which weights it.
    """

    rho0: FloatArray
    sigma_11: FloatArray
    ratios: FloatArray
    rel_err: FloatArray


def _bin_fits(family: FamilyRatios) -> tuple[list[FloatArray], list[float], list[float]]:
    """Return, for each bin that enters, the coefficients (a, c, d) of its fit, its relative error and its rms."""
    design = np.column_stack([np.ones(family.rho0.size), np.log(family.rho0), np.log(family.sigma_11)])
    coefs, rels, rms = [], [], []
    for r, rel in zip(family.ratios, family.rel_err, strict=True):
        if not (np.all(np.isfinite(r)) and np.all(r > 0.0) and np.isfinite(rel) and rel > 0.0):
            continue
        coef, *_ = np.linalg.lstsq(design, np.log(r), rcond=None)
        coefs.append(coef)
        rels.append(float(rel))
        rms.append(float(np.sqrt(np.mean((np.log(r) - design @ coef) ** 2))))
    return coefs, rels, rms


def invert(name: str, family: FamilyRatios) -> Inversion:
    """Return the (rho_0, Sigma_1.1) at which the ratios of every bin of the survey *name* equal 1.

    A bin enters when its error and all its ratios are finite and positive.
    """
    coefs, rels, rms = _bin_fits(family)
    if len(coefs) < N_MIN_INVERSION_BINS:
        return _nan_inversion(name, len(coefs))
    slopes = np.array([coef[1:] for coef in coefs])
    target = -np.array([coef[0] for coef in coefs])
    weight = 1.0 / np.array(rels) ** 2
    cov = np.linalg.inv(slopes.T @ (slopes * weight[:, None]))
    solution = cov @ (slopes.T @ (weight * target))
    chi2_dof = float(np.sum(weight * (slopes @ solution - target) ** 2)) / (len(coefs) - 2)
    cov = cov * max(1.0, chi2_dof)
    err = np.sqrt(np.diag(cov))
    log_rho0, log_sigma = (float(v) for v in solution)
    return Inversion(
        survey=name,
        n_bins=len(coefs),
        rho0=Estimate(float(np.exp(log_rho0)), float(np.exp(log_rho0) * err[0])),
        sigma_11=Estimate(float(np.exp(log_sigma)), float(np.exp(log_sigma) * err[1])),
        correlation=float(cov[0, 1] / (err[0] * err[1])),
        chi2_dof=chi2_dof,
        model_rms=max(rms),
        bin_rho0=[_bin_rho0(coef, rel, log_sigma) for coef, rel in zip(coefs, rels, strict=True)],
    )


def _bin_rho0(coef: FloatArray, rel: float, log_sigma: float) -> Estimate:
    """Return the rho_0 that brings one bin, with the fit *coef* and relative error *rel*, to 1 at Sigma_1.1."""
    log_b = (-coef[0] - coef[2] * log_sigma) / coef[1]
    return Estimate(float(np.exp(log_b)), float(np.exp(log_b) * rel / abs(coef[1])))
