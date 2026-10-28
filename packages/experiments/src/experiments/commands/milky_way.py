"""``experiments milky-way``: fetch and check the Milky Way catalogs, and measure the test of the tracer.

``fetch`` downloads the six inputs of :mod:`experiments.mw_sources` into the root, which
git ignores, skipping those already there; ``check`` compares the local copies with their
pinned sizes and checksums; ``analyze`` measures the test (:mod:`experiments.mw_pipeline`)
and writes its record to ``<out-dir>/milky-way.json``::

    experiments milky-way fetch
    experiments milky-way check
    OMP_NUM_THREADS=4 experiments milky-way analyze
"""

from __future__ import annotations

from pathlib import Path

import click
import pyvo

from experiments._paths import MW_DATA_DIR, MW_ROOT
from experiments.commands._report import write_json
from experiments.mw_analysis import N_BOOTSTRAP, MwSurvey
from experiments.mw_pipeline import MilkyWay, analyze_milky_way
from experiments.mw_sources import FILE_SOURCES, GAIA_TAP_URL, check_file, check_gaia, fetch_file, fetch_gaia

NAMES = tuple(source.name for source in FILE_SOURCES) + ("gaia",)
RESULT_NAME = "milky-way.json"

root_option = click.option(
    "--root",
    default=MW_ROOT,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Local root of the Milky Way catalogs.",
)


@click.group("milky-way")
def milky_way() -> None:
    """Fetch and check the Milky Way catalogs, and measure the test of the isothermal tracer."""


@milky_way.command("fetch")
@root_option
@click.option("--only", type=click.Choice(NAMES), multiple=True, help="Fetch only these inputs (repeatable).")
def fetch(root: Path, only: tuple[str, ...]) -> None:
    """Download the Milky Way inputs that are not yet under the root."""
    wanted = set(only or NAMES)
    for source in FILE_SOURCES:
        if source.name in wanted:
            click.echo(f"{source.name}: {fetch_file(source, root)}")
    if "gaia" in wanted:
        click.echo(f"gaia: {fetch_gaia(root, pyvo.dal.TAPService(GAIA_TAP_URL))}")


@milky_way.command("check")
@root_option
def check(root: Path) -> None:
    """Compare the local Milky Way inputs with their pinned sizes and checksums."""
    results = [check_file(source, root) for source in FILE_SOURCES] + [check_gaia(root)]
    for result in results:
        click.echo(f"{result.name}: {'ok' if result.ok else 'FAILED'}, {result.detail} ({result.path})")
    if not all(result.ok for result in results):
        raise click.ClickException("some inputs are missing or differ; fetch them with `experiments milky-way fetch`")


def survey_lines(survey: MwSurvey) -> list[str]:
    """Return the printed table of one survey."""
    lines = [
        f"{survey.name}: {survey.n_gaia:,} Gaia stars, {survey.n_matched:,} matches",
        "    ages     med  N_pop  N_kin      sigma_z          h_Z   tracer/h_Z    sheet/h_Z",
    ]
    for b in survey.bins:
        h, kin, r = b.heights, b.kinematics, b.ratios
        lines.append(
            f"  {b.age_lo:3.0f}-{b.age_hi:<3.0f} {b.age_median:5.2f} {h.n_matched:6d} {kin.n_kin:6d} "
            f"{kin.sigma_z.value:6.1f}+/-{kin.sigma_z.err:4.1f} {h.measured.value:6.3f}+/-{h.measured.err:.3f} "
            f"{r.tracer.value:5.2f}+/-{r.tracer.err:.2f} {r.sheet.value:5.2f}+/-{r.sheet.err:.2f}"
        )
    for rng in survey.ranges:
        trends = ", ".join(f"{k} {t.value:+.3f}+/-{t.err:.3f}" for k, t in rng.trends.items())
        lines.append(f"  {rng.lo:g}-{rng.hi:g} Gyr ({rng.n_bins} bins): trends {trends}; eta {rng.eta:.2f}")
    return lines


def report_lines(result: MilkyWay) -> list[str]:
    """Return the printed report of the Milky Way test."""
    lines = [f"Milky Way at R_0 = {result.r0_kpc} kpc, Sigma = {result.sigma_surf:.1f} Msun/pc^2, k = {result.k}"]
    for survey in result.surveys:
        lines.extend(survey_lines(survey))
    lines.append("imig:  ages      h_Z by mass  h_Z by counts  counts astroNN  counts distmass   sigma_z  tracer/h")
    for b in result.imig.bins:
        h, r = b.heights, b.ratios
        lines.append(
            f"  {b.age_lo:5.2f}-{b.age_hi:<5.2f} "
            + " ".join(
                f"{e.value:6.3f}+/-{e.err:.3f}"
                for e in (h.imig_mass, h.imig_counts, h.counts_astronn, h.counts_distmass)
            )
            + f" {b.kinematics.sigma_z.value:5.1f} {r.imig_counts.value:5.2f}+/-{r.imig_counts.err:.2f}"
        )
    trends = ", ".join(f"{k} {t.value:+.3f}+/-{t.err:.3f}" for k, t in result.imig.trends.items())
    lines.append(f"  trends of tracer/h: {trends}")
    return lines


@milky_way.command("analyze")
@root_option
@click.option(
    "--n-bootstrap", default=N_BOOTSTRAP, show_default=True, type=click.IntRange(min=2), help="Bootstrap resamplings."
)
@click.option(
    "--out-dir",
    default=MW_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON record.",
)
def analyze(root: Path, n_bootstrap: int, out_dir: Path) -> None:
    """Measure the test of the isothermal tracer with the Milky Way catalogs under the root."""
    try:
        result = analyze_milky_way(root, n_bootstrap, n_bootstrap)
    except FileNotFoundError as err:
        raise click.ClickException(f"{err}; fetch the inputs with `experiments milky-way fetch`") from err
    for line in report_lines(result):
        click.echo(line)
    click.echo(f"  wrote {write_json(result, out_dir / RESULT_NAME)}")
