"""``experiments figures``: draw every figure from the data that the data commands cached.

The halos are those of Palicio et al. (2025, Table 1) at Level 4, z = 0. A halo enters
the mono-age figures when its ``analyze`` record exists, and the images when its
``render-data`` maps exist; the others are reported and skipped::

    experiments figures
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR, FIGURES_DIR
from experiments.auriga import HaloSnapshot
from experiments.commands.auriga_plot import plot_halos
from experiments.commands.auriga_render import render_halos
from experiments.palicio import PALICIO_WINDOWS


@click.command("figures")
@click.option(
    "--data-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the records that the data commands wrote.",
)
@click.option(
    "--figures-dir",
    default=FIGURES_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the figures (PNG and PDF).",
)
def figures(data_dir: Path, figures_dir: Path) -> None:
    """Draw every figure whose data is cached under the data directory."""
    halos = [HaloSnapshot(halo) for halo in sorted(PALICIO_WINDOWS)]
    analyzed = [snap for snap in halos if (data_dir / f"{snap.stem}.json").exists()]
    rendered = [snap for snap in halos if (data_dir / f"{snap.stem}-maps.npz").exists()]
    for kind, have, command in (("analysis", analyzed, "analyze"), ("maps", rendered, "render-data")):
        missing = sorted({snap.halo for snap in halos} - {snap.halo for snap in have})
        if missing:
            click.echo(f"  no {kind} for Au{', Au'.join(map(str, missing))}: run `experiments auriga {command}`")
    for line in (*plot_halos(analyzed, data_dir, figures_dir), *render_halos(rendered, data_dir, figures_dir)):
        click.echo(line)
