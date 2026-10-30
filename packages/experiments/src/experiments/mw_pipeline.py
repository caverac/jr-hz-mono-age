"""The Milky Way analysis from the local catalogs to its record.

:func:`analyze_milky_way` reads the six inputs under the root
(:mod:`experiments.mw_sources`), places the kinematic stars of both surveys in McMillan
(2017) with their J_R, builds the star counts of each survey, and measures the test of
:mod:`experiments.mw_analysis` with LAMOST, with APOGEE and its astroNN ages, and with
APOGEE and the ages of distmass. It then compares the heights of Imig et al. (2025) with
the APOGEE counts on both age scales and with the tracer (:mod:`experiments.mw_imig`), and
each of their populations with the counts of the same ages and [M/H]
(:mod:`experiments.mw_imig_cells`), with all the matches, with those of each targeting
box, and with the LAMOST counts split by [Fe/H]. From the family of potentials that every
bin carries, it inverts the tracer, and the thin-layer population, for the rho_0 and
Sigma_1.1 at which every bin of a survey agrees (:mod:`experiments.mw_inversion`). A run
takes over an hour, so it reports each stage to *progress* as it ends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import agama
import numpy as np
from numpy.typing import NDArray

from experiments.mw_analysis import N_BOOTSTRAP, SEED, MwBin, MwSurvey, Survey, measure_survey, solar_model
from experiments.mw_catalogs import (
    Apogee,
    GaiaStars,
    apogee_age_tracers,
    apogee_kinematics,
    lamost_age_tracers,
    lamost_kinematics,
    read_apogee,
    read_gaia,
    read_lamost,
    with_distmass_ages,
)
from experiments.mw_counts import SLICE_EDGES_KPC, RedClump, apogee_count_sample, lamost_count_sample
from experiments.mw_dynamics import R0_KPC, cylindrical_actions, mcmillan17
from experiments.mw_imig import N_DRAWS, ImigCheck, ImigInputs, measure_imig, read_imig
from experiments.mw_imig_cells import (
    TARGETING_BOXES,
    CellSettings,
    ImigCells,
    TargetedCells,
    TracerEstimator,
    lamost_metallicity_counts,
    measure_cells,
    metallicity_counts,
    tracer_estimators,
)
from experiments.mw_inversion import FamilyRatios, Inversion, invert
from experiments.mw_potentials import FamilyMember, potential_family
from experiments.mw_sources import FILE_SOURCES, GAIA_PATH

FloatArray = NDArray[np.float64]
Progress = Callable[[str], None]  # receives one line at the end of each stage of a run


def _silent(_: str) -> None:
    """Report nothing."""


@dataclass(frozen=True)
class MilkyWay:  # pylint: disable=too-many-instance-attributes
    """The Milky Way test: R_0 (kpc), Sigma (Msun/pc^2), k, the bootstrap size, the red-clump peaks, the surveys.

    ``imig`` compares the pooled heights of Imig et al. with the counts and the tracer,
    ``imig_cells`` each of their populations with the counts of its ages and [M/H], and
    ``imig_cells_targeting`` the same with the matches of each targeting box, and
    ``imig_cells_lamost`` with the LAMOST counts, whose ages are not on the scale of Imig et al.
    ``estimators`` holds the windowed h_Z of the tracer against the exponential fit to it.
    ``family`` holds the potentials around McMillan (2017) whose tracers every bin also
    carries (:mod:`experiments.mw_potentials`), and ``inversions`` and ``inversions_coeval``
    the rho_0 and Sigma_1.1 of each survey that bring its bins to the measured h_Z with the
    tracer and with the thin-layer population.
    """

    r0_kpc: float
    sigma_surf: float
    k: float
    n_bootstrap: int
    rc_peak: tuple[float, ...]  # M_W of the red-clump peak in each |z| slice of the counts
    slice_edges_kpc: tuple[float, ...]  # the |z| slices of the counts
    surveys: list[MwSurvey]
    imig: ImigCheck
    imig_cells: ImigCells
    imig_cells_targeting: list[TargetedCells]
    imig_cells_lamost: ImigCells
    estimators: list[TracerEstimator]
    family: list[FamilyMember]
    inversions: list[Inversion]
    inversions_coeval: list[Inversion]


def survey_inversion(
    survey: MwSurvey, members: list[FamilyMember], ratios: Callable[[MwBin], list[float]]
) -> Inversion:
    """Return the inversion of *survey* over the potentials *members*, with the ratios *ratios* of each bin."""
    family = FamilyRatios(
        rho0=np.array([m.rho0 for m in members]),
        sigma_11=np.array([m.sigma_11 for m in members]),
        ratios=np.array([ratios(b) for b in survey.bins], dtype=np.float64).reshape(len(survey.bins), len(members)),
        rel_err=np.array([b.ratios.tracer.err / b.ratios.tracer.value for b in survey.bins]),
    )
    return invert(survey.name, family)


def source_path(root: Path, name: str) -> Path:
    """Return the local path of the input *name* under *root*."""
    return root / next(source.path for source in FILE_SOURCES if source.name == name)


def apogee_survey(name: str, apogee: Apogee, gaia: GaiaStars, pot: agama.Potential) -> Survey:
    """Return the APOGEE survey *name* with the ages of *apogee*: its counts, kinematics and J_R in *pot*."""
    kinematics = apogee_kinematics(apogee)
    counts = apogee_count_sample(gaia, apogee_age_tracers(apogee))
    return Survey(name, counts, kinematics, cylindrical_actions(pot, kinematics.phase))


def lamost_survey(
    lamost: dict[str, FloatArray], gaia: GaiaStars, pot: agama.Potential, rng: np.random.Generator
) -> tuple[Survey, RedClump]:
    """Return the LAMOST survey of the catalog *lamost*, and the red-clump window of its counts."""
    kinematics = lamost_kinematics(lamost, rng)
    counts, clump = lamost_count_sample(gaia, lamost_age_tracers(lamost))
    return Survey("lamost", counts, kinematics, cylindrical_actions(pot, kinematics.phase)), clump


def analyze_milky_way(  # pylint: disable=too-many-locals
    root: Path, n_bootstrap: int = N_BOOTSTRAP, n_draws: int = N_DRAWS, progress: Progress = _silent
) -> MilkyWay:
    """Return the Milky Way test from the inputs under *root*, reporting each stage to *progress*.

    The stages run in a fixed order, the order in which they draw from one generator, so
    that a record reproduces.
    """
    rng = np.random.default_rng(SEED)
    pot = mcmillan17()
    model = solar_model(pot)
    family = potential_family()
    family_models = tuple(member_model for _, member_model in family)
    gaia = read_gaia(root / GAIA_PATH)
    apogee = read_apogee(source_path(root, "astronn"), source_path(root, "allstar"))
    lamost_catalog = read_lamost(source_path(root, "lamost"))
    lamost, clump = lamost_survey(lamost_catalog, gaia, pot, rng)
    astronn = apogee_survey("apogee", apogee, gaia, pot)
    distmass_apogee = with_distmass_ages(apogee, source_path(root, "distmass"))
    distmass = apogee_survey("apogee_distmass", distmass_apogee, gaia, pot)
    populations = read_imig(source_path(root, "imig"))
    progress(f"inputs read: {gaia.source_id.size:,} Gaia stars, {len(populations)} age bins of Imig et al.")

    def reported(label: str, measured: ImigCells) -> ImigCells:
        progress(
            f"{label}: {measured.n_measured} of {len(measured.cells)} measured, median {measured.median_ratio:.2f}"
        )
        return measured

    def apogee_cells(catalog: Apogee) -> ImigCells:
        return measure_cells(
            populations, metallicity_counts(gaia, catalog, populations), model, rng, CellSettings(n_draws)
        )

    surveys = []
    for survey in (lamost, astronn, distmass):
        surveys.append(measure_survey(survey, model, rng, n_bootstrap, family_models))
        tracer = [b.ratios.tracer.value for b in surveys[-1].bins]
        progress(f"{survey.name}: {len(tracer)} age bins, tracer over h_Z {min(tracer):.2f} to {max(tracer):.2f}")
    imig = measure_imig(
        populations, ImigInputs(astronn.counts, distmass.counts, distmass.kinematics), model, rng, n_draws
    )
    progress(f"Imig et al.: {len(imig.bins)} age bins")
    imig_cells = reported("cells", apogee_cells(distmass_apogee))
    targeting = [
        TargetedCells(box, reported(f"cells in box {i + 1}", apogee_cells(distmass_apogee.within(box))))
        for i, box in enumerate(TARGETING_BOXES)
    ]
    lamost_metals = lamost_metallicity_counts(gaia, lamost_catalog, populations, rng)
    lamost_cells = reported(
        "LAMOST cells", measure_cells(populations, lamost_metals, model, rng, CellSettings(n_draws))
    )
    members = [member for member, _ in family]
    return MilkyWay(
        r0_kpc=R0_KPC,
        sigma_surf=model.sigma_surf,
        k=model.k,
        n_bootstrap=n_bootstrap,
        rc_peak=tuple(fit.mu_rc for fit in clump.fits),
        slice_edges_kpc=SLICE_EDGES_KPC,
        surveys=surveys,
        imig=imig,
        imig_cells=imig_cells,
        imig_cells_targeting=targeting,
        imig_cells_lamost=lamost_cells,
        estimators=tracer_estimators(model),
        family=members,
        inversions=[survey_inversion(s, members, lambda b: b.ratios.family) for s in surveys],
        inversions_coeval=[survey_inversion(s, members, lambda b: b.ratios.family_coeval) for s in surveys],
    )
