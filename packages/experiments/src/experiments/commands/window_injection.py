"""``experiments window-injection``: run the injection test of the scale-height estimators.

The test (:mod:`experiments.injection`) needs no snapshot. Its record goes to
``<out-dir>/window-injection.json``, which the documentation site reads::

    experiments window-injection
"""

from __future__ import annotations

from pathlib import Path

import click

from experiments._paths import DATA_DIR
from experiments.commands._report import write_json
from experiments.injection import DRAWS, N_STARS, run_injection

RECORD = "window-injection.json"


@click.command("window-injection")
@click.option("--n-stars", default=N_STARS, show_default=True, type=click.IntRange(min=100), help="Heights per draw.")
@click.option("--draws", default=DRAWS, show_default=True, type=click.IntRange(min=2), help="Draws per profile.")
@click.option(
    "--out-dir",
    default=DATA_DIR,
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of the record.",
)
def window_injection(n_stars: int, draws: int, out_dir: Path) -> None:
    """Fit profiles of known shape in the fixed and the self-consistent windows, and write the ratios."""
    injection = run_injection(n_stars, draws)
    for point in injection.points:
        click.echo(
            f"{point.family:11s} scale {point.scale:4.1f} kpc, reference {point.reference:.3f} kpc: "
            f"fixed {point.fixed.median:.3f} [{point.fixed.lo:.3f}, {point.fixed.hi:.3f}] "
            f"({point.fixed_failed:.0%} failed), self-consistent {point.scaled.median:.3f} "
            f"[{point.scaled.lo:.3f}, {point.scaled.hi:.3f}]"
        )
    click.echo(f"  wrote {write_json(injection, out_dir / RECORD)}")
