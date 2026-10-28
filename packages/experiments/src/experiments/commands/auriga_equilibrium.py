"""``experiments auriga equilibrium``: the mean vertical motions of every population of each halo.

Each halo is read and its potential fitted (:mod:`experiments.analysis`), and the mean
motions of :mod:`experiments.equilibrium` are measured in every age bin. The record goes
to ``<out-dir>/<stem>-equilibrium.json``::

    experiments auriga equilibrium 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import local_halo, snapshot_options, snapshots
from experiments.commands._report import write_json
from experiments.equilibrium import EquilibriumBin, EquilibriumRecord, measure_equilibrium
from experiments.populations import N_BOOTSTRAP

HEADER = "    age  N_stars    h_Z  thickening(1/Gyr)  bending(km/s)      asymmetry      warp(kpc)"


def equilibrium_name(snapshot: HaloSnapshot) -> str:
    """Return the file name of the equilibrium record of *snapshot*."""
    return f"{snapshot.stem}-equilibrium.json"


def equilibrium_lines(bins: list[EquilibriumBin]) -> list[str]:
    """Return the printed table of the mean motions of one halo."""
    lines = [HEADER]
    for b in bins:
        m = b.motions
        lines.append(
            f"  {b.age_lo:2.0f}-{b.age_hi:<2.0f} {b.n_stars:8d} {b.hz.value:6.3f} "
            f"{m.thickening.value:+7.2f}+/-{m.thickening.err:5.2f} {m.bending.value:+6.2f}+/-{m.bending.err:4.2f} "
            f"{m.asymmetry.value:+.3f}+/-{m.asymmetry.err:.3f} {m.warp.value:.3f}+/-{m.warp.err:.3f}"
        )
    return lines


@click.command("equilibrium")
@snapshot_options
@click.option(
    "--n-bootstrap", default=N_BOOTSTRAP, show_default=True, type=click.IntRange(min=2), help="Bootstrap resamplings."
)
@click.option(
    "--out-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON records.",
)
def auriga_equilibrium(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    n_bootstrap: int,
    out_dir: Path,
) -> None:
    """Measure the mean vertical motions of every population of each of HALOS."""
    for snap in snapshots(halos, level, suite, snapshot):
        prepared = local_halo(snap, root)
        result = measure_equilibrium(prepared.galaxy, n_bootstrap)
        click.echo(f"Au{snap.halo}:")
        for line in equilibrium_lines(result):
            click.echo(line)
        click.echo(f"  wrote {write_json(EquilibriumRecord(snap, result), out_dir / equilibrium_name(snap))}")
