"""Command line of the analysis engine.

Every number and figure of the paper is derived through this command line. The
engine reads the Auriga snapshots from ``staging/auriga/`` (local-only, ignored by
git) and writes small result tables to ``assets/data/`` and figures to
``assets/figures/``. Each subcommand lives in its own module under
:mod:`experiments.commands`::

    experiments paths
    experiments auriga download 2 3 16 19 23 27
"""

from __future__ import annotations

import click

from experiments import __version__
from experiments._paths import AURIGA_ROOT, DATA_DIR, FIGURES_DIR, REPO_ROOT
from experiments.commands.auriga import auriga

__all__ = ["AURIGA_ROOT", "DATA_DIR", "FIGURES_DIR", "REPO_ROOT", "main"]


@click.group()
@click.version_option(__version__, prog_name="experiments")
def main() -> None:
    """Derive the numbers and figures of the mono-age J_R-h_Z paper."""


@main.command()
def paths() -> None:
    """Print where the engine reads snapshots and writes tables and figures."""
    for label, path in (("auriga snapshots", AURIGA_ROOT), ("result tables", DATA_DIR), ("figures", FIGURES_DIR)):
        click.echo(f"{label}: {path}")


main.add_command(auriga)
