"""Command line of the analysis engine.

Every number and figure of the paper is derived through this command line. The
engine reads the Auriga snapshots from ``staging/auriga/`` and the Milky Way catalogs from
``staging/milky-way/`` (both ignored by git), writes the
results of the data commands to ``assets/data/`` and the figures of the plot commands to
``assets/figures/``. Each subcommand lives in its own module under
:mod:`experiments.commands`::

    experiments paths
    experiments auriga download 2 3 16 19 23 24 27
    experiments auriga analyze 2 3 16 19 23 24 27
    experiments milky-way analyze
    experiments figures
    experiments window-injection
"""

from __future__ import annotations

import click

from experiments import __version__
from experiments._paths import (
    AURIGA_DATA_DIR,
    AURIGA_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    MOCK_ROOT,
    MW_DATA_DIR,
    MW_ROOT,
    REPO_ROOT,
)
from experiments.commands.all_figures import figures
from experiments.commands.auriga import auriga
from experiments.commands.milky_way import milky_way
from experiments.commands.paper_figures import paper_figures_command
from experiments.commands.window_injection import window_injection

__all__ = [
    "AURIGA_DATA_DIR",
    "AURIGA_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "MOCK_ROOT",
    "MW_DATA_DIR",
    "MW_ROOT",
    "REPO_ROOT",
    "main",
]


@click.group()
@click.version_option(__version__, prog_name="experiments")
def main() -> None:
    """Derive the numbers and figures of the mono-age J_R-h_Z paper."""


@main.command()
def paths() -> None:
    """Print where the engine reads snapshots and writes results, tables and figures."""
    for label, path in (
        ("auriga snapshots", AURIGA_ROOT),
        ("mock snapshot", MOCK_ROOT),
        ("milky way catalogs", MW_ROOT),
        ("result tables", DATA_DIR),
        ("auriga results", AURIGA_DATA_DIR),
        ("milky way results", MW_DATA_DIR),
        ("figures", FIGURES_DIR),
    ):
        click.echo(f"{label}: {path}")


main.add_command(auriga)
main.add_command(milky_way)
main.add_command(figures)
main.add_command(window_injection)
main.add_command(paper_figures_command)
