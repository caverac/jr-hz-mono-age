"""Command line of the analysis engine.

Every number and figure of the paper is derived through this command line. The
engine reads the Auriga snapshots from ``staging/auriga/`` and writes the results of the
analysis to ``staging/results/``, both ignored by git until the paper is public. Each
subcommand lives in its own module under :mod:`experiments.commands`::

    experiments paths
    experiments auriga download 2 3 16 19 23 24 27
    experiments auriga analyze 2 3 16 19 23 24 27
"""

from __future__ import annotations

import click

from experiments import __version__
from experiments._paths import AURIGA_ROOT, DATA_DIR, FIGURES_DIR, MOCK_ROOT, REPO_ROOT, RESULTS_DIR
from experiments.commands.auriga import auriga

__all__ = ["AURIGA_ROOT", "DATA_DIR", "FIGURES_DIR", "MOCK_ROOT", "REPO_ROOT", "RESULTS_DIR", "main"]


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
        ("results", RESULTS_DIR),
        ("result tables", DATA_DIR),
        ("figures", FIGURES_DIR),
    ):
        click.echo(f"{label}: {path}")


main.add_command(auriga)
