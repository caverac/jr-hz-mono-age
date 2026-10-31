"""``experiments auriga concepts``: what the Concepts pages of the documentation draw for each halo.

Each halo is read and its potential fitted (:mod:`experiments.analysis`), and
:mod:`experiments.concepts` evaluates the two expansions of the potential and the fit of
every population. The record goes to ``<out-dir>/<stem>-concepts.json``::

    experiments auriga concepts 2 3 16 19 23 24 27
"""

from __future__ import annotations

import math
from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import local_halo, snapshot_options, snapshots
from experiments.commands._report import write_json
from experiments.concepts import measure_concepts


def concepts_name(snapshot: HaloSnapshot) -> str:
    """Return the file name of the Concepts record of *snapshot*."""
    return f"{snapshot.stem}-concepts.json"


@click.command("concepts")
@snapshot_options
@click.option(
    "--out-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON records.",
)
def auriga_concepts(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    out_dir: Path,
) -> None:
    """Evaluate the expansions of the potential and the fit of every population of each of HALOS."""
    for snap in snapshots(halos, level, suite, snapshot):
        record = measure_concepts(local_halo(snap, root))
        measured = sum(1 for b in record.bins if math.isfinite(b.hz))
        path = write_json(record, out_dir / concepts_name(snap))
        click.echo(
            f"Au{snap.halo}: R_mid = {record.r_mid:.2f} kpc, {measured} of {len(record.bins)} bins; wrote {path}"
        )
