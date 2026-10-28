"""``experiments auriga jr-map``: the proxy J_R,w of each halo, from its face-on map of the median J_R.

Each halo is read and its potential fitted (:mod:`experiments.analysis`), and the proxy
of :mod:`experiments.jr_map` is measured in Palicio's window of the halo. The record goes
to ``<out-dir>/<stem>-jr-map.json``::

    OMP_NUM_THREADS=4 experiments auriga jr-map 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import local_halo, snapshot_options, snapshots
from experiments.commands._report import write_json
from experiments.commands.auriga_analyze import POSITIVE, palicio_window
from experiments.jr_map import JrMapRecord, measure_jr_map


def jr_map_name(snapshot: HaloSnapshot) -> str:
    """Return the file name of the J_R map record of *snapshot*."""
    return f"{snapshot.stem}-jr-map.json"


@click.command("jr-map")
@snapshot_options
@click.option("--z-lim", type=POSITIVE, default=None, help="Palicio's Z_lim (kpc), in place of their Table 1.")
@click.option("--l-kpc", type=POSITIVE, default=None, help="Palicio's L_kpc (kpc), in place of their Table 1.")
@click.option(
    "--out-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON records.",
)
def auriga_jr_map(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    z_lim: float | None,
    l_kpc: float | None,
    out_dir: Path,
) -> None:
    """Measure the proxy J_R,w of each of HALOS from its map of the median J_R."""
    for snap in snapshots(halos, level, suite, snapshot):
        window = palicio_window(snap.halo, z_lim, l_kpc)
        prepared = local_halo(snap, root)
        jr_map = measure_jr_map(prepared.galaxy, prepared.potential, window)
        click.echo(
            f"Au{snap.halo}: J_R,w {jr_map.jr_w:.1f} kpc km/s (cells {jr_map.jr_cells_lo:.1f} to "
            f"{jr_map.jr_cells_hi:.1f}, {jr_map.n_cells} cells, {jr_map.n_stars:,} star particles)"
        )
        click.echo(f"  wrote {write_json(JrMapRecord(snap, jr_map), out_dir / jr_map_name(snap))}")
