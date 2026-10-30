"""``experiments paper-figures``: draw the figures of the paper from the records under the data directory.

The records are those of ``auriga analyze`` and ``auriga tracer-sum`` for each of HALOS,
``window-injection`` and ``milky-way analyze``; the figures go to the figures directory as
PNG and PDF (:mod:`experiments.paper_figures`)::

    experiments paper-figures 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import DATA_DIR, FIGURES_DIR
from experiments._plotting import save_figure
from experiments.auriga import HaloSnapshot
from experiments.commands.auriga_analyze import result_name
from experiments.commands.auriga_tracer_sum import tracer_sum_name
from experiments.paper_figures import PaperRecords, paper_figures
from experiments.records import load_analysis, load_injection, load_milky_way, load_tracer_sum


@click.command("paper-figures")
@click.argument("halos", nargs=-1, type=int, required=True)
@click.option(
    "--data-dir",
    default=DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the records, with auriga/ and milky-way/ under it.",
)
@click.option(
    "--figures-dir",
    default=FIGURES_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the figures (PNG and PDF).",
)
def paper_figures_command(halos: tuple[int, ...], data_dir: Path, figures_dir: Path) -> None:
    """Draw the figures of the paper from the records of HALOS and of the Milky Way."""
    snapshots = [HaloSnapshot(halo) for halo in halos]
    records = PaperRecords(
        halos=[load_analysis(data_dir / "auriga" / result_name(s)) for s in snapshots],
        sums=[load_tracer_sum(data_dir / "auriga" / tracer_sum_name(s)) for s in snapshots],
        injection=load_injection(data_dir / "window-injection.json"),
        milky_way=load_milky_way(data_dir / "milky-way" / "milky-way.json"),
    )
    for name, fig in paper_figures(records).items():
        click.echo(save_figure(fig, name, figures_dir).line())
