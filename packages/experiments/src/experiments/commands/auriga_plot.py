"""``experiments auriga plot``: draw the mono-age figure of each halo from the record of ``analyze``.

The figure (:func:`experiments.figures.mono_age_figure`) goes to
``<figures-dir>/<stem>-mono-age`` as PNG and PDF::

    experiments auriga plot 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._plotting import save_figure
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import figure_options, require, snapshots
from experiments.figures import mono_age_figure
from experiments.records import load_analysis


def plot_halos(selected: list[HaloSnapshot], data_dir: Path, figures_dir: Path) -> list[str]:
    """Draw the mono-age figure of every halo of *selected*; return the report."""
    lines = []
    for snap in selected:
        analysis = load_analysis(require(data_dir / f"{snap.stem}.json", "analyze"))
        lines.append(save_figure(mono_age_figure(analysis), f"{snap.stem}-mono-age", figures_dir).line())
    return lines


@click.command("plot")
@figure_options
def auriga_plot(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...], level: int, suite: str, snapshot: int, data_dir: Path, figures_dir: Path
) -> None:
    """Draw the mono-age figure of each of HALOS."""
    for line in plot_halos(snapshots(halos, level, suite, snapshot), data_dir, figures_dir):
        click.echo(line)
