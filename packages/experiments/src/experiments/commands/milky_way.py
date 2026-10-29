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

import time
from pathlib import Path

import click
import numpy as np
import pyvo

from experiments._paths import MW_DATA_DIR, MW_ROOT
from experiments.commands._report import write_json
from experiments.mw_analysis import N_BOOTSTRAP, MwSurvey
from experiments.mw_imig_cells import ImigCells
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
    bias = ", ".join(f"{h:.2f} {r:.3f}" for h, r in zip(survey.bias.scales, survey.bias.ratios, strict=True))
    lines.append(f"  counts of an exponential over its scale (kpc): {bias}")
    gradients = ", ".join(f"{b.age_lo:g}-{b.age_hi:g} {b.heights.gradient:+.2f}" for b in survey.bins)
    lines.append(f"  gradient of the fraction exp(b x) toward the center (1/kpc): {gradients}")
    return lines


def cell_lines(cells: ImigCells, title: str = "imig cells") -> list[str]:
    """Return the printed summary of the populations of Imig et al. one at a time, by [M/H] bin.

    The medians are of our windowed h_Z over their model through our counts, of our
    exponential fit over theirs through our counts, and of the tracer over our windowed h_Z.
    """
    lines = [
        f"{title}, {cells.n_matches:,} matches, cells with {cells.min_matches}+:"
        "   [M/H]  cells  matches  ours/theirs  exponential/theirs  tracer/ours"
    ]

    def median(values: list[float]) -> str:
        finite = [v for v in values if np.isfinite(v)]
        return f"{np.median(finite):.2f}" if finite else "none"

    for lo in sorted({c.m_h_lo for c in cells.cells}):
        measured = [
            c
            for c in cells.cells
            if c.m_h_lo == lo and c.n_matched >= cells.min_matches and np.isfinite(c.ratios.counted.value)
        ]
        lines.append(
            f"  {lo:+.1f}  {len(measured):5d} {sum(c.n_matched for c in measured):8d}"
            f"  {median([c.ratios.counted.value for c in measured]):>11s}"
            f"  {median([c.ratios.exponential.value for c in measured]):>18s}"
            f"  {median([c.ratios.tracer.value for c in measured]):>11s}"
        )
    trends = ", ".join(
        f"{name} {t.age.value:+.2f}+/-{t.age.err:.2f} ({t.metallicity.value:+.2f}+/-{t.metallicity.err:.2f})"
        for name, t in (
            ("counted", cells.trends.counted),
            ("exponential", cells.trends.exponential),
            ("tracer", cells.trends.tracer),
        )
    )
    lines.append(
        f"  {cells.n_measured} cells: median {cells.median_ratio:.2f}; "
        f"slope of ln ratio per ln age at fixed [M/H] (per dex at fixed age): {trends}"
    )
    slopes = ", ".join(
        f"{name} {getattr(cells.slopes, name).value:+.2f}+/-{getattr(cells.slopes, name).err:.2f}"
        for name in ("counts", "counts_exponential", "imig", "imig_sun", "tracer", "sigma_z")
    )
    lines.append(f"  slope of ln h_Z (ln sigma_z) per dex at fixed age: {slopes}")
    dispersion = ", ".join(
        f"{name} {getattr(cells.dispersion, name).value:+.2f}+/-{getattr(cells.dispersion, name).err:.2f}"
        for name in ("counts_exponential", "imig_exponential", "imig_sun")
    )
    lines.append(f"  slope of ln h_Z per ln sigma_z at fixed [M/H]: {dispersion}")
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
    lines.extend(cell_lines(result.imig_cells))
    for targeted in result.imig_cells_targeting:
        box = targeted.box
        title = f"imig cells in {box.h_lo:g} < H < {box.h_hi:g}, {box.jk_lo:g} < (J-K)0 < {box.jk_hi:g}"
        lines.extend(cell_lines(targeted.cells, title))
    lines.extend(cell_lines(result.imig_cells_lamost, "imig cells with LAMOST, LAMOST ages"))
    estimators = ", ".join(
        f"{e.sigma_z:g} km/s {e.windowed / e.exponential:.2f} ({e.windowed / e.exponential_whole:.2f})"
        for e in result.estimators
    )
    lines.append(f"tracer, windowed h_Z over the exponential fit to |z| < 1.2 kpc (to the whole profile): {estimators}")
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
    """Measure the test of the isothermal tracer with the Milky Way catalogs under the root.

    Each stage prints one line to stderr as it ends, with the minutes since the start.
    """
    start = time.monotonic()

    def progress(message: str) -> None:
        click.echo(f"[{(time.monotonic() - start) / 60.0:5.1f} min] {message}", err=True)

    try:
        result = analyze_milky_way(root, n_bootstrap, n_bootstrap, progress)
    except FileNotFoundError as err:
        raise click.ClickException(f"{err}; fetch the inputs with `experiments milky-way fetch`") from err
    for line in report_lines(result):
        click.echo(line)
    click.echo(f"  wrote {write_json(result, out_dir / RESULT_NAME)}")
