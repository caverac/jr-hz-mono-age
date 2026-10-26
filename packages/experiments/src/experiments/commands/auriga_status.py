"""``experiments auriga status``: follow the Globus transfers of Auriga snapshots."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import click
import globus_sdk

from experiments.auriga import TaskStatus, TransferRecord, read_record, write_record
from experiments.commands._auriga import client_id_option, snapshot_options, snapshots
from experiments.globus_transfer import make_client, task_status


def _poll(client: globus_sdk.TransferClient, record: TransferRecord, path: Path) -> tuple[TransferRecord, TaskStatus]:
    """Return *record* with the current state of its task, stored in *path*; an ended task is not polled."""
    if record.task is not None and record.task.finished:
        return record, record.task
    task = task_status(client, record.submission.task_id)
    record = replace(record, task=task)
    write_record(record, path)
    return record, task


def _progress(record: TransferRecord, task: TaskStatus) -> str:
    """Return one line on the state *task* of the transfer of *record*."""
    counts = task.counts
    line = (
        f"Au{record.snapshot.halo}: {task.status}, {counts.files_transferred} copied and {counts.files_skipped} "
        f"skipped of {counts.files} files, {counts.bytes_transferred / 1e9:.2f} of {record.total_bytes / 1e9:.2f} GB, "
        f"{task.bytes_per_second / 1e6:.2f} MB/s, {counts.faults} faults"
    )
    return line if task.nice_status is None else f"{line} ({task.nice_status})"


def _follow(
    client: globus_sdk.TransferClient, paths: list[Path], wait: bool, interval: int
) -> list[tuple[TransferRecord, TaskStatus]]:
    """Poll the tasks recorded in *paths* and print their progress; with *wait*, every *interval* s until all end."""
    polled = [_poll(client, read_record(path), path) for path in paths]
    while True:
        for record, task in polled:
            click.echo(_progress(record, task))
        if not wait or all(task.finished for _, task in polled):
            return polled
        time.sleep(interval)
        polled = [_poll(client, record, path) for (record, _), path in zip(polled, paths, strict=True)]


@click.command("status")
@snapshot_options
@click.option("--wait", is_flag=True, help="Poll until every task has ended.")
@click.option("--interval", default=300, show_default=True, type=click.IntRange(min=10), help="Seconds between polls.")
@client_id_option
def auriga_status(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    wait: bool,
    interval: int,
    client_id: str,
) -> None:
    """Report the transfer task of each halo in HALOS, and store its state in transfer.json.

    A task that has ended is not polled again. With --wait the command returns when every
    task has ended; run it under ``caffeinate -i`` so that the Mac, and with it the GCP
    endpoint, stays awake. The command fails if a task failed.
    """
    paths = [snap.manifest_path(root) for snap in snapshots(halos, level, suite, snapshot)]
    if missing := [str(path) for path in paths if not path.is_file()]:
        raise click.ClickException(f"no transfer submitted: {', '.join(missing)} missing")
    polled = _follow(make_client(client_id), paths, wait, interval)
    if failed := [f"Au{record.snapshot.halo}" for record, task in polled if task.status == "FAILED"]:
        raise click.ClickException(f"failed: {', '.join(failed)}")
