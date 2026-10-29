"""``experiments auriga jr-relation``: J_R,w against h_Z across the halos, as Palicio et al. (2025) fit it.

The command reads, for each halo, the proxy J_R,w of ``experiments auriga jr-map``, the
whole-disk h_Z of ``experiments auriga analyze`` and the h_Z that ``experiments auriga
tracer-sum`` predicts for it, in Palicio's windows and in the self-consistent one, fits a
line of J_R,w against each, and writes the result to ``<data-dir>/jr-relation.json``::

    experiments auriga jr-relation 2 3 16 19 23 24 27
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import AURIGA_DATA_DIR
from experiments.commands._auriga import require, snapshots
from experiments.commands._report import write_json
from experiments.commands.auriga_analyze import result_name
from experiments.commands.auriga_jr_map import jr_map_name
from experiments.commands.auriga_tracer_sum import tracer_sum_name
from experiments.jr_map import JrPoint, jr_relation
from experiments.records import load_analysis, load_jr_map, load_tracer_sum

RELATION_NAME = "jr-relation.json"


@click.command("jr-relation")
@click.argument("halos", nargs=-1, required=True, type=click.IntRange(min=1))
@click.option("--level", default=4, show_default=True, help="Resolution level of the release.")
@click.option("--suite", default="Original", show_default=True, help="Simulation suite of the release.")
@click.option("--snapshot", default=127, show_default=True, help="Snapshot number; 127 is z = 0 at Level 4.")
@click.option(
    "--data-dir",
    default=AURIGA_DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the records, and of the result.",
)
def auriga_jr_relation(halos: tuple[int, ...], level: int, suite: str, snapshot: int, data_dir: Path) -> None:
    """Fit J_R,w against the measured and the predicted whole-disk h_Z across HALOS."""
    points = []
    for snap in snapshots(halos, level, suite, snapshot):
        proxy = load_jr_map(require(data_dir / jr_map_name(snap), "jr-map")).jr_map
        whole = load_analysis(require(data_dir / result_name(snap), "analyze")).whole_disk.whole.heights
        mixture = load_tracer_sum(require(data_dir / tracer_sum_name(snap), "tracer-sum")).tracer_sum.whole.mixture
        points.append(
            JrPoint(snap.halo, proxy.jr_w, whole.palicio, whole.scaled, mixture.palicio.value, mixture.scaled.value)
        )
    relation = jr_relation(points)
    lines = (
        ("Palicio's windows", relation.line_palicio),
        ("self-consistent", relation.line_scaled),
        ("Palicio's windows, predicted", relation.line_palicio_mixture),
        ("self-consistent, predicted", relation.line_scaled_mixture),
    )
    for label, line in lines:
        click.echo(
            f"J_R,w = ({line.intercept:.1f} +/- {line.intercept_err:.1f}) "
            f"+ ({line.slope:.1f} +/- {line.slope_err:.1f}) h_Z with h_Z in {label}; "
            f"Palicio et al.: slope {relation.palicio_slope:.1f} km/s"
        )
    click.echo(f"  wrote {write_json(relation, data_dir / RELATION_NAME)}")
