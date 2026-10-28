"""The arguments and options shared by the ``auriga`` subcommands."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

from experiments._paths import AURIGA_DATA_DIR, AURIGA_ROOT, FIGURES_DIR
from experiments.analysis import PreparedHalo, prepare_halo
from experiments.auriga import HaloSnapshot

CLIENT_ID_ENV = "JR_HZ_GLOBUS_CLIENT_ID"

F = TypeVar("F", bound=Callable[..., None])

client_id_option = click.option(
    "--client-id",
    envvar=CLIENT_ID_ENV,
    show_envvar=True,
    required=True,
    help="Client ID of the native app registered with Globus Auth.",
)


def snapshot_options(command: F) -> F:
    """Add the HALOS argument and the options that select the snapshot and the local root."""
    command = click.option(
        "--root",
        default=AURIGA_ROOT,
        show_default=True,
        type=click.Path(file_okay=False, path_type=Path),
        help="Local root of the release; download needs the directory that docker/compose.yaml mounts.",
    )(command)
    command = click.option(
        "--snapshot", default=127, show_default=True, help="Snapshot number; 127 is z = 0 at Level 4."
    )(command)
    command = click.option("--suite", default="Original", show_default=True, help="Simulation suite of the release.")(
        command
    )
    command = click.option("--level", default=4, show_default=True, help="Resolution level of the release.")(command)
    return click.argument("halos", nargs=-1, required=True, type=click.IntRange(min=1))(command)


def snapshots(halos: tuple[int, ...], level: int, suite: str, snapshot: int) -> list[HaloSnapshot]:
    """Return the selected snapshot of each halo in *halos*."""
    return [HaloSnapshot(halo, level, suite, snapshot) for halo in halos]


def local_halo(snapshot: HaloSnapshot, root: Path) -> PreparedHalo:
    """Return *snapshot* under *root* in its disk frame with its fitted potential, or say how to download it."""
    try:
        return prepare_halo(snapshot, root)
    except FileNotFoundError as err:
        raise click.ClickException(f"{err}; download Au{snapshot.halo} with `experiments auriga download`") from err


def figure_options(command: F) -> F:
    """Add the HALOS argument, the options that select the snapshot, and the data and figure directories."""
    command = click.option(
        "--figures-dir",
        default=FIGURES_DIR,
        show_default=True,
        type=click.Path(file_okay=False, path_type=Path),
        help="Directory of the figures (PNG and PDF).",
    )(command)
    command = click.option(
        "--data-dir",
        default=AURIGA_DATA_DIR,
        show_default=True,
        type=click.Path(file_okay=False, path_type=Path),
        help="Directory of the records that the data commands wrote.",
    )(command)
    command = click.option(
        "--snapshot", default=127, show_default=True, help="Snapshot number; 127 is z = 0 at Level 4."
    )(command)
    command = click.option("--suite", default="Original", show_default=True, help="Simulation suite of the release.")(
        command
    )
    command = click.option("--level", default=4, show_default=True, help="Resolution level of the release.")(command)
    return click.argument("halos", nargs=-1, required=True, type=click.IntRange(min=1))(command)


def require(path: Path, command: str) -> Path:
    """Return *path*, or fail with the *command* that writes it."""
    if not path.exists():
        raise click.ClickException(f"{path} does not exist; write it with `experiments auriga {command}`")
    return path
