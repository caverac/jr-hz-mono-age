"""``experiments auriga download``: submit the Globus transfers of Auriga snapshots."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import click
import globus_sdk

from experiments.auriga import AURIGA_COLLECTION_ID, HaloSnapshot, read_record, write_record
from experiments.commands._auriga import client_id_option, snapshot_options, snapshots
from experiments.globus_transfer import (
    Collection,
    find_collection,
    get_collection,
    make_client,
    request_data_access,
    submit_snapshot,
)

# The GCP collection of docker/compose.yaml, and the container path of its mount of
# staging/auriga/.
DESTINATION = "jr-hz-mono-age-auriga"
DESTINATION_ROOT = PurePosixPath("/data/auriga")


def _existing_records(selected: list[HaloSnapshot], root: Path, resubmit: bool) -> list[Path]:
    """Return the ``transfer.json`` files of *selected* that exist; without *resubmit*, any is an error."""
    existing = [path for snap in selected if (path := snap.manifest_path(root)).exists()]
    if existing and not resubmit:
        raise click.ClickException(f"{', '.join(map(str, existing))} exist; pass --resubmit to submit again")
    return existing


def _endpoints(client: globus_sdk.TransferClient, destination: str) -> tuple[Collection, Collection]:
    """Return the Auriga collection and the GCP collection *destination*, after printing both."""
    source = get_collection(client, AURIGA_COLLECTION_ID)
    target = find_collection(client, destination)
    click.echo(
        f"source: {source.display_name} ({source.collection_id}, {source.entity_type}, "
        f"HTTPS server: {source.https_server or 'none'})"
    )
    click.echo(f"destination: {target.display_name} ({target.collection_id}, {target.entity_type})")
    if target.gcp_connected is False:
        click.echo(
            "warning: the destination is not connected; the tasks wait until it is (docker compose up -d globus)"
        )
    return source, target


@click.command("download")
@snapshot_options
@click.option("--destination", default=DESTINATION, show_default=True, help="Display name or ID of the GCP collection.")
@click.option("--resubmit", is_flag=True, help="Submit although transfer.json exists; the old one is kept.")
@client_id_option
def auriga_download(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    halos: tuple[int, ...],
    level: int,
    suite: str,
    snapshot: int,
    root: Path,
    destination: str,
    resubmit: bool,
    client_id: str,
) -> None:
    """Submit one Globus transfer per halo in HALOS, of its snapshot and group catalog.

    The files go from the Auriga collection to the GCP collection DESTINATION, at the same
    paths under its /data/auriga/, which is the local root. Before submitting, the files
    of the release are listed into the halo's transfer.json, with the ID of the task.
    A file whose checksum already matches the release is not copied again, so --resubmit
    completes a partial copy; the record it replaces is renamed transfer-<task id>.json.
    The first call opens a Globus login on the command line.
    """
    selected = snapshots(halos, level, suite, snapshot)
    existing = _existing_records(selected, root, resubmit)
    client = make_client(client_id)
    source, target = _endpoints(client, destination)
    request_data_access(client, source)
    for path in existing:
        path.rename(path.with_name(f"transfer-{read_record(path).submission.task_id}.json"))
    for snap in selected:
        record = submit_snapshot(client, snap, source, target, DESTINATION_ROOT)
        write_record(record, snap.manifest_path(root))
        click.echo(
            f"Au{snap.halo}: task {record.submission.task_id}, "
            f"{len(record.files)} files, {record.total_bytes / 1e9:.2f} GB"
        )
