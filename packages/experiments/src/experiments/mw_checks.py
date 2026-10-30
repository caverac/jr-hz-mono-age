"""Checks of the Milky Way test against the errors of the ages and the selection of the surveys.

Both checks remeasure every survey of :mod:`experiments.mw_pipeline` with ``n_bootstrap``
draws, without the family of potentials:

- the errors of the ages: every age, in the counts and in the kinematics, is drawn again
  from its error (:func:`experiments.mw_catalogs.scattered_ages`), which doubles the variance
  of the errors, in ``N_REALIZATIONS`` realizations. The errors mix neighboring bins of 2 Gyr,
  so if they set the ratios or the trends, doubling them moves them;
- the selection: in each slice of |z|, the matches of each survey are split at their median G
  into a bright and a faint half, each of which gives its own population fractions in the
  counts. A targeting that does not depend on the age of a star at a fixed position gives
  both halves the same h_Z, within their errors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.mw_analysis import MwSurvey, SolarModel, Survey, measure_survey, solar_model
from experiments.mw_catalogs import (
    Apogee,
    GaiaStars,
    read_apogee,
    read_gaia,
    read_lamost,
    scattered_lamost,
    with_distmass_ages,
    with_scattered_ages,
)
from experiments.mw_counts import brighter_in_slice
from experiments.mw_dynamics import mcmillan17
from experiments.mw_pipeline import Progress, apogee_survey, lamost_survey, source_path
from experiments.mw_sources import GAIA_PATH

FloatArray = NDArray[np.float64]

N_REALIZATIONS = 4
CHECK_BOOTSTRAP = 100
CHECK_SEED = 20261031


@dataclass(frozen=True)
class MagnitudeSplit:
    """The median G of the matches of one survey, and the survey measured with each half of every slice of |z|."""

    survey: str
    g_split: float
    bright: MwSurvey
    faint: MwSurvey


@dataclass(frozen=True)
class MwChecks:
    """The surveys remeasured with the ages drawn again, once per realization, and with each half of the tracers."""

    n_bootstrap: int
    age_errors: list[list[MwSurvey]]
    magnitude: list[MagnitudeSplit]


@dataclass(frozen=True)
class _Catalogs:
    """The catalogs that the surveys are built from."""

    gaia: GaiaStars
    astronn: Apogee
    distmass: Apogee
    lamost: dict[str, FloatArray]


@dataclass(frozen=True)
class _Setting:
    """What every survey is measured with: the potential, its vertical model, the generator and the draws."""

    pot: agama.Potential
    model: SolarModel
    rng: np.random.Generator
    n_bootstrap: int


def _surveys(catalogs: _Catalogs, setting: _Setting) -> list[Survey]:
    """Return LAMOST, APOGEE with astroNN ages and APOGEE with distmass ages, built from *catalogs*."""
    lamost, _ = lamost_survey(catalogs.lamost, catalogs.gaia, setting.pot, setting.rng)
    return [
        lamost,
        apogee_survey("apogee", catalogs.astronn, catalogs.gaia, setting.pot),
        apogee_survey("apogee_distmass", catalogs.distmass, catalogs.gaia, setting.pot),
    ]


def _split(survey: Survey, setting: _Setting) -> MagnitudeSplit:
    """Return *survey* measured with its matches brighter, and fainter, than the median G of their slice of |z|."""
    counts = survey.counts
    bright = brighter_in_slice(counts)
    halves = (bright, ~bright)
    measured = [
        measure_survey(
            replace(survey, counts=counts.with_matches(half)), setting.model, setting.rng, setting.n_bootstrap
        )
        for half in halves
    ]
    return MagnitudeSplit(survey.name, float(np.nanmedian(counts.tracer_g)), measured[0], measured[1])


def _age_errors(
    catalogs: _Catalogs, setting: _Setting, n_realizations: int, progress: Progress
) -> list[list[MwSurvey]]:
    """Return every survey remeasured with its ages drawn again, once per realization."""
    out = []
    for i in range(n_realizations):
        scattered = _Catalogs(
            catalogs.gaia,
            with_scattered_ages(catalogs.astronn, setting.rng),
            with_scattered_ages(catalogs.distmass, setting.rng),
            scattered_lamost(catalogs.lamost, setting.rng),
        )
        out.append(
            [measure_survey(s, setting.model, setting.rng, setting.n_bootstrap) for s in _surveys(scattered, setting)]
        )
        progress(f"ages drawn again, realization {i + 1} of {n_realizations}")
    return out


def _magnitude_splits(catalogs: _Catalogs, setting: _Setting) -> list[MagnitudeSplit]:
    """Return every survey measured with the bright and with the faint half of its matches."""
    return [_split(survey, setting) for survey in _surveys(catalogs, setting)]


def check_milky_way(
    root: Path,
    n_realizations: int = N_REALIZATIONS,
    n_bootstrap: int = CHECK_BOOTSTRAP,
    progress: Progress = lambda _: None,
) -> MwChecks:
    """Return both checks of the Milky Way test from the inputs under *root*, reporting each stage to *progress*."""
    pot = mcmillan17()
    setting = _Setting(pot, solar_model(pot), np.random.default_rng(CHECK_SEED), n_bootstrap)
    astronn = read_apogee(source_path(root, "astronn"), source_path(root, "allstar"))
    catalogs = _Catalogs(
        read_gaia(root / GAIA_PATH),
        astronn,
        with_distmass_ages(astronn, source_path(root, "distmass")),
        read_lamost(source_path(root, "lamost")),
    )
    age_errors = _age_errors(catalogs, setting, n_realizations, progress)
    magnitude = _magnitude_splits(catalogs, setting)
    progress("bright and faint halves of the tracers measured")
    return MwChecks(n_bootstrap, age_errors, magnitude)
