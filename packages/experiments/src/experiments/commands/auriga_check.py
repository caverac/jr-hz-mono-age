"""``experiments auriga check``: check the local copies of Auriga snapshots."""

from __future__ import annotations

from pathlib import Path

import click

from experiments.commands._auriga import snapshot_options, snapshots
from experiments.snapshot_check import check_download


@click.command("check")
@snapshot_options
def auriga_check(halos: tuple[int, ...], level: int, suite: str, snapshot: int, root: Path) -> None:
    """Check that the local copy of each halo in HALOS is complete and readable.

    The files must match the release listing of transfer.json in number and size, and
    the HDF5 chunks of the snapshot and the group catalog must hold the rows their headers
    count. The command fails if any check fails. The files are downloads, so they are
    only read as data.
    """
    failed = []
    for snap in snapshots(halos, level, suite, snapshot):
        report = check_download(snap, root)
        for line in report.lines:
            click.echo(line)
        if report.problems:
            failed.append(f"Au{snap.halo}")
    if failed:
        raise click.ClickException(f"problems in {', '.join(failed)}")
