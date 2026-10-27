"""The Milky Way analysis from the local catalogs to its record.

:func:`analyze_milky_way` reads the five inputs under the root
(:mod:`experiments.mw_sources`), places the kinematic stars of both surveys in McMillan
(2017) with their J_R, builds the star counts of each survey, and measures the test of
:mod:`experiments.mw_analysis` with each, and the check against Imig et al. (2025) with the
APOGEE kinematics (:mod:`experiments.mw_imig`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.mw_analysis import N_BOOTSTRAP, SEED, MwSurvey, Survey, measure_survey, solar_model
from experiments.mw_catalogs import (
    apogee_age_tracers,
    apogee_kinematics,
    lamost_age_tracers,
    lamost_kinematics,
    read_apogee,
    read_gaia,
    read_lamost,
)
from experiments.mw_counts import apogee_count_sample, lamost_count_sample
from experiments.mw_dynamics import R0_KPC, cylindrical_actions, mcmillan17
from experiments.mw_imig import N_DRAWS, ImigCheck, measure_imig, read_imig
from experiments.mw_sources import FILE_SOURCES, GAIA_PATH


@dataclass(frozen=True)
class MilkyWay:
    """The Milky Way test: R_0 (kpc), Sigma (Msun/pc^2), k, the bootstrap size, the red-clump peaks, the surveys."""

    r0_kpc: float
    sigma_surf: float
    k: float
    n_bootstrap: int
    rc_peak: tuple[float, ...]  # M_W of the red-clump peak in each |z| slice of the counts
    surveys: list[MwSurvey]
    imig: ImigCheck


def source_path(root: Path, name: str) -> Path:
    """Return the local path of the input *name* under *root*."""
    return root / next(source.path for source in FILE_SOURCES if source.name == name)


def analyze_milky_way(root: Path, n_bootstrap: int = N_BOOTSTRAP, n_draws: int = N_DRAWS) -> MilkyWay:
    """Return the Milky Way test from the inputs under *root*."""
    rng = np.random.default_rng(SEED)
    pot = mcmillan17()
    model = solar_model(pot)
    gaia = read_gaia(root / GAIA_PATH)
    lamost = read_lamost(source_path(root, "lamost"))
    apogee = read_apogee(source_path(root, "astronn"), source_path(root, "allstar"))
    lamost_kin = lamost_kinematics(lamost, rng)
    apogee_kin = apogee_kinematics(apogee)
    lamost_counts, clump = lamost_count_sample(gaia, lamost_age_tracers(lamost))
    surveys = [
        Survey("lamost", lamost_counts, lamost_kin, cylindrical_actions(pot, lamost_kin.phase)),
        Survey(
            "apogee",
            apogee_count_sample(gaia, apogee_age_tracers(apogee)),
            apogee_kin,
            cylindrical_actions(pot, apogee_kin.phase),
        ),
    ]
    return MilkyWay(
        r0_kpc=R0_KPC,
        sigma_surf=model.sigma_surf,
        k=model.k,
        n_bootstrap=n_bootstrap,
        rc_peak=tuple(fit.mu_rc for fit in clump.fits),
        surveys=[measure_survey(survey, model, rng, n_bootstrap) for survey in surveys],
        imig=measure_imig(read_imig(source_path(root, "imig")), apogee_kin, model, rng, n_draws),
    )
