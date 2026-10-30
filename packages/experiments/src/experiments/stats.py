"""Robust statistics of the star particles and of bootstrap draws."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

VEL_CLIP_SIGMA = 4.0
MAD_TO_SIGMA = 1.4826


def clipped_dispersion(v: FloatArray) -> float:
    """Return the dispersion of *v* within ``VEL_CLIP_SIGMA`` MAD sigmas of its median."""
    center = np.median(v)
    mad = MAD_TO_SIGMA * np.median(np.abs(v - center))
    return float(np.std(v[np.abs(v - center) < VEL_CLIP_SIGMA * mad]))


def weighted_median(values: FloatArray, weights: FloatArray) -> float:
    """Return the value below which half of the total *weights* lies."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def half_width(draws: FloatArray) -> float:
    """Return half the 16th-84th percentile interval of the finite *draws*; NaN if none is finite."""
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan")
    lo, hi = np.percentile(finite, [16.0, 84.0])
    return float(0.5 * (hi - lo))


def deconvolved_dispersion(v: FloatArray, err: FloatArray) -> float:
    """Return the dispersion of *v* within ``VEL_CLIP_SIGMA`` MAD sigmas of its median, less the median of *err*.

    The measurement errors *err* of the kept values are removed in quadrature; a dispersion
    smaller than them is 0.
    """
    center = np.median(v)
    mad = MAD_TO_SIGMA * np.median(np.abs(v - center))
    kept = np.abs(v - center) < VEL_CLIP_SIGMA * mad
    return float(np.sqrt(max(float(np.var(v[kept])) - float(np.median(err[kept])) ** 2, 0.0)))
