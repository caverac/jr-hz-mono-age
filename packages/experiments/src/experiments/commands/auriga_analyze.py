"""``experiments auriga analyze``: measure the mono-age populations and the whole disk of each halo.

Each halo is read and its potential fitted once (:mod:`experiments.analysis`). The report
is printed, and the full record goes to ``<out-dir>/auriga-au<N>-level<L>-<suite>-snap<SSS>.json``.
Palicio's window of each halo comes from Palicio et al. (2025, Table 1) unless
``--z-lim`` and ``--l-kpc`` set it::

    OMP_NUM_THREADS=4 caffeinate -i experiments auriga analyze 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import RESULTS_DIR
from experiments.analysis import analyze_halo, prepare_halo
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import snapshot_options, snapshots
from experiments.commands._report import report_lines, write_json
from experiments.palicio import PALICIO_WINDOWS, PalicioWindow
from experiments.populations import K_MAIN, N_BOOTSTRAP, Method

POSITIVE = click.FloatRange(min=0.0, min_open=True)


def result_name(snapshot: HaloSnapshot) -> str:
    """Return the file name of the record of *snapshot*."""
    return f"auriga-au{snapshot.halo}-level{snapshot.level}-{snapshot.suite}-snap{snapshot.snapshot:03d}.json"


def palicio_window(halo: int, z_lim: float | None, l_kpc: float | None) -> PalicioWindow:
    """Return Palicio's window of *halo*, with *z_lim* and *l_kpc* in place of the table's where given."""
    if z_lim is not None and l_kpc is not None:
        return PalicioWindow(l_kpc, z_lim)
    if (table := PALICIO_WINDOWS.get(halo)) is None:
        raise click.ClickException(f"Palicio et al. (2025) list no window for Au{halo}: pass --z-lim and --l-kpc")
    return PalicioWindow(table.l_kpc if l_kpc is None else l_kpc, table.z_lim if z_lim is None else z_lim)


@click.command("analyze")
@snapshot_options
@click.option("--k", default=K_MAIN, show_default=True, type=POSITIVE, help="The self-consistent window is k h_Z.")
@click.option(
    "--n-bootstrap", default=N_BOOTSTRAP, show_default=True, type=click.IntRange(min=2), help="Bootstrap resamplings."
)
@click.option("--z-lim", type=POSITIVE, default=None, help="Palicio's Z_lim (kpc), in place of their Table 1.")
@click.option("--l-kpc", type=POSITIVE, default=None, help="Palicio's L_kpc (kpc), in place of their Table 1.")
@click.option(
    "--out-dir",
    default=RESULTS_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON records.",
)
def auriga_analyze(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    k: float,
    n_bootstrap: int,
    z_lim: float | None,
    l_kpc: float | None,
    out_dir: Path,
) -> None:
    """Measure the mono-age populations and the whole disk of each of HALOS."""
    for snap in snapshots(halos, level, suite, snapshot):
        method = Method(palicio_window(snap.halo, z_lim, l_kpc), k, n_bootstrap)
        try:
            prepared = prepare_halo(snap, root)
        except FileNotFoundError as err:
            raise click.ClickException(f"{err}; download Au{snap.halo} with `experiments auriga download`") from err
        analysis = analyze_halo(prepared, method)
        for line in report_lines(analysis):
            click.echo(line)
        click.echo(f"  wrote {write_json(analysis, out_dir / result_name(snap))}")
