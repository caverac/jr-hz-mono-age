"""Checks of the mono-age result that the main measurement leaves open.

Three checks, each over its own bins of the stars of the annulus, with
``N_CHECK_BOOTSTRAP`` resamplings (fewer if the main measurement takes fewer) drawn from a
generator of their own, so that the main measurement of :mod:`experiments.populations`
does not change:

- ``all_stars``: the bins of 1 Gyr of the main measurement, with the tracer of the clipped
  sigma_z and of the unclipped one, which keeps the tails of a population that mixes
  several dispersions;
- ``disk``: the same bins with only the stars whose circularity L_z / L_c(E) exceeds
  ``DISK_CIRCULARITY``, which leaves out the stars on orbits closer to a spheroid's;
- ``fine``: bins of ``FINE_BIN_GYR``, which mix fewer dispersions than bins of 1 Gyr.

Each bin gives sigma_z, h_Z in the self-consistent window and the tracer over h_Z, with the
clipped and the unclipped dispersion; each set gives the trends of both ratios with age
over ``CHECK_RANGES``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.heights import scaled_particle_hz, scaled_profile_hz, tracer_profile
from experiments.populations import (
    AGE_BINS,
    Z_MID_MAX_KPC,
    AnnulusSample,
    Estimate,
    Method,
    Trend,
    VerticalModel,
    fit_trend,
)
from experiments.stats import clipped_dispersion, half_width

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DISK_CIRCULARITY = 0.7
FINE_BIN_GYR = 0.25
FINE_BINS = tuple((FINE_BIN_GYR * i, FINE_BIN_GYR * (i + 1)) for i in range(int(10.0 / FINE_BIN_GYR)))
N_CHECK_BOOTSTRAP = 100
N_MIN_MID = 30  # stars within Z_MID_MAX_KPC for a dispersion
N_MIN_TREND_BINS = 3  # bins with finite ratios for a trend with one degree of freedom
CHECK_SEED = 20261030
CHECK_RANGES = ((0.0, 10.0), (2.0, 10.0))


@dataclass(frozen=True)
class CheckValues:
    """sigma_z (km/s) and h_Z (kpc) of one bin, and the tracer of its clipped and unclipped sigma_z over h_Z."""

    sigma_z: Estimate
    hz: Estimate
    tracer: Estimate
    tracer_unclipped: Estimate


@dataclass(frozen=True)
class CheckBin:
    """One bin of a check: its ages (Gyr), the fraction of the initial mass of its ages that it keeps, and values."""

    age_lo: float
    age_hi: float
    age_median: float
    mass_fraction: float
    values: CheckValues


@dataclass(frozen=True)
class CheckRange:
    """The trends of the two tracer ratios with age over the bins inside [lo, hi] Gyr."""

    lo: float
    hi: float
    tracer: Trend
    tracer_unclipped: Trend


@dataclass(frozen=True)
class CheckSet:
    """The bins of one check and the trends of its ratios."""

    bins: list[CheckBin]
    ranges: list[CheckRange]


@dataclass(frozen=True)
class Checks:
    """The three checks of one annulus."""

    all_stars: CheckSet
    disk: CheckSet
    fine: CheckSet


@dataclass(frozen=True)
class _Setting:
    """What every bin of a check is measured with: the model, the window factor k, the draws and their generator."""

    model: VerticalModel
    k: float
    n_draws: int
    rng: np.random.Generator


def _values(rows: AnnulusSample, model: VerticalModel, k: float) -> tuple[float, float, float, float]:
    """Return sigma_z, h_Z and the two tracer ratios of the stars *rows*; NaN without enough stars at the plane."""
    absz = np.abs(rows.z)
    mid = rows.v_z[absz < Z_MID_MAX_KPC]
    if mid.size < N_MIN_MID:
        return (float("nan"),) * 4
    sigma_z = clipped_dispersion(mid)
    hz = scaled_particle_hz(absz, rows.initial_mass, k)[0]
    tracer = scaled_profile_hz(tracer_profile(model.phi_fit, sigma_z), k)[0]
    unclipped = scaled_profile_hz(tracer_profile(model.phi_fit, float(np.std(mid))), k)[0]
    return sigma_z, hz, tracer / hz, unclipped / hz


def _check_bin(rows: AnnulusSample, bounds: tuple[float, float], fraction: float, setting: _Setting) -> CheckBin:
    """Return the values of the stars *rows* of one bin, with errors from the resamplings of *setting*."""
    point = _values(rows, setting.model, setting.k)
    n = rows.z.size
    draws = np.array(
        [_values(rows.select(setting.rng.integers(0, n, n)), setting.model, setting.k) for _ in range(setting.n_draws)]
        if n > 0
        else np.full((setting.n_draws, 4), np.nan)
    )
    estimates = [Estimate(point[i], half_width(draws[:, i])) for i in range(4)]
    median_age = float(np.median(rows.age)) if n > 0 else float("nan")
    return CheckBin(*bounds, median_age, fraction, CheckValues(*estimates))


def _ranges(bins: list[CheckBin]) -> list[CheckRange]:
    """Return the trends of the two tracer ratios over ``CHECK_RANGES``, from the bins with finite ratios.

    A range with fewer than ``N_MIN_TREND_BINS`` such bins has NaN trends.
    """
    out = []
    for lo, hi in CHECK_RANGES:
        chosen = [
            b
            for b in bins
            if b.age_lo >= lo
            and b.age_hi <= hi
            and np.isfinite(b.values.tracer.value)
            and np.isfinite(b.values.tracer_unclipped.value)
            and b.values.tracer.err > 0.0
            and b.values.tracer_unclipped.err > 0.0
        ]
        if len(chosen) < N_MIN_TREND_BINS:
            none = Trend(float("nan"), float("nan"), float("nan"))
            out.append(CheckRange(lo, hi, none, none))
            continue
        age = np.array([b.age_median for b in chosen])

        def trend(ratio: str, rows: list[CheckBin] = chosen, ages: FloatArray = age) -> Trend:
            value = np.array([getattr(b.values, ratio).value for b in rows])
            err = np.array([getattr(b.values, ratio).err for b in rows])
            return fit_trend(ages, value, err)

        out.append(CheckRange(lo, hi, trend("tracer"), trend("tracer_unclipped")))
    return out


def check_set(
    sample: AnnulusSample, keep: BoolArray, bins: tuple[tuple[float, float], ...], setting: _Setting
) -> CheckSet:
    """Return one check over *bins* of the stars of *sample* that *keep* selects."""
    out = []
    for lo, hi in bins:
        in_bin = (sample.age >= lo) & (sample.age < hi)
        total = float(np.sum(sample.initial_mass[in_bin]))
        kept = float(np.sum(sample.initial_mass[in_bin & keep]))
        fraction = kept / total if total > 0.0 else float("nan")
        out.append(_check_bin(sample.select(in_bin & keep), (lo, hi), fraction, setting))
    return CheckSet(out, _ranges(out))


def measure_checks(sample: AnnulusSample, circularity: FloatArray, model: VerticalModel, method: Method) -> Checks:
    """Return the three checks of the annulus *sample*, whose stars have the *circularity* L_z / L_c(E)."""
    setting = _Setting(model, method.k, min(N_CHECK_BOOTSTRAP, method.n_bootstrap), np.random.default_rng(CHECK_SEED))
    everyone = np.ones(sample.z.size, dtype=bool)
    return Checks(
        all_stars=check_set(sample, everyone, AGE_BINS, setting),
        disk=check_set(sample, circularity > DISK_CIRCULARITY, AGE_BINS, setting),
        fine=check_set(sample, everyone, FINE_BINS, setting),
    )
