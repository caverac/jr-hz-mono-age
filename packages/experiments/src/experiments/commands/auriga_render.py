"""``experiments auriga render``: draw the images of the disks from the maps of ``render-data``.

For each halo the face-on and edge-on portrait goes to ``<figures-dir>/<stem>-portrait.jpg``.
The portraits are drawn for the documentation site only::

    experiments auriga render 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._plotting import save_render
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import figure_options, require, snapshots
from experiments.render import load_maps, portrait_figure


def render_halos(selected: list[HaloSnapshot], data_dir: Path, figures_dir: Path) -> list[str]:
    """Draw the portrait of every halo of *selected*; return the report."""
    lines = []
    for snap in selected:
        maps = load_maps(require(data_dir / f"{snap.stem}-maps.npz", "render-data"))
        portrait = portrait_figure(maps, f"Au{snap.halo}")
        lines.append(save_render(portrait, f"{snap.stem}-portrait", figures_dir).line())
    return lines


@click.command("render")
@figure_options
def auriga_render(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...], level: int, suite: str, snapshot: int, data_dir: Path, figures_dir: Path
) -> None:
    """Draw the portrait of each of HALOS, colored by stellar age."""
    for line in render_halos(snapshots(halos, level, suite, snapshot), data_dir, figures_dir):
        click.echo(line)
