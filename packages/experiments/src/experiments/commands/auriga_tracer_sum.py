"""``experiments auriga tracer-sum``: the whole-disk h_Z of each halo predicted as a sum of isothermal tracers.

Each halo is read and its potential fitted (:mod:`experiments.analysis`), and the
predictions of :mod:`experiments.tracer_sum` are measured in Palicio's window of the halo.
The record goes to ``<out-dir>/<stem>-tracer-sum.json``::

    OMP_NUM_THREADS=4 experiments auriga tracer-sum 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR
from experiments.auriga import HaloSnapshot
from experiments.commands._auriga import local_halo, snapshot_options, snapshots
from experiments.commands._report import write_json
from experiments.commands.auriga_analyze import POSITIVE, palicio_window
from experiments.tracer_sum import N_DRAWS, SumRange, TracerSum, TracerSumRecord, measure_tracer_sum

HEADER = "    R (kpc)  unmodeled   h_Z(Z_lim)  single  mixture   h_Z(k h_Z)  single  mixture"


def tracer_sum_name(snapshot: HaloSnapshot) -> str:
    """Return the file name of the tracer-sum record of *snapshot*."""
    return f"{snapshot.stem}-tracer-sum.json"


def _range_line(r: SumRange) -> str:
    """Return the row of one radial range: the heights in Z_lim, then in the self-consistent window."""
    m, single, mixture = r.measured, r.ratio_single, r.ratio_mixture
    return (
        f"  {r.r_lo:4.1f}-{r.r_hi:<4.1f} {r.unmodeled.mixture:9.4f} "
        f"{m.palicio.value:6.3f}+/-{m.palicio.err:.3f} {single.palicio.value:6.3f} {mixture.palicio.value:8.3f} "
        f"{m.scaled.value:6.3f}+/-{m.scaled.err:.3f} {single.scaled.value:6.3f} {mixture.scaled.value:8.3f}"
    )


def tracer_sum_lines(result: TracerSum) -> list[str]:
    """Return the printed table of one halo: measured h_Z, and each prediction over it."""
    return [HEADER, *(_range_line(r) for r in result.rings), _range_line(result.whole)]


@click.command("tracer-sum")
@snapshot_options
@click.option("--z-lim", type=POSITIVE, default=None, help="Palicio's Z_lim (kpc), in place of their Table 1.")
@click.option("--l-kpc", type=POSITIVE, default=None, help="Palicio's L_kpc (kpc), in place of their Table 1.")
@click.option("--n-draws", default=N_DRAWS, show_default=True, type=click.IntRange(min=2), help="Bootstrap draws.")
@click.option(
    "--out-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the JSON records.",
)
def auriga_tracer_sum(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    z_lim: float | None,
    l_kpc: float | None,
    n_draws: int,
    out_dir: Path,
) -> None:
    """Predict the whole-disk h_Z of each of HALOS as a sum of isothermal tracers."""
    for snap in snapshots(halos, level, suite, snapshot):
        window = palicio_window(snap.halo, z_lim, l_kpc)
        prepared = local_halo(snap, root)
        result = measure_tracer_sum(prepared.galaxy, prepared.potential, window, n_draws)
        click.echo(f"Au{snap.halo}:")
        for line in tracer_sum_lines(result):
            click.echo(line)
        click.echo(f"  wrote {write_json(TracerSumRecord(snap, result), out_dir / tracer_sum_name(snap))}")
