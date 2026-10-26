"""Transfers of the Auriga release through the Globus Transfer API.

A download is a Globus task from the Auriga collection to a Globus Connect Personal (GCP)
collection on this machine. GCP runs as the ``globus`` service of ``docker/compose.yaml``,
which mounts ``staging/auriga/`` at ``/data/auriga``. Submitting the task from here,
rather than from the File Manager of app.globus.org, puts the request and its outcome in
the halo's ``transfer.json``.

Logging in needs the client ID of a native app registered with Globus Auth. The Globus
SDK keeps the tokens under ``~/.globus/app/``, outside the repository.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Final

import globus_sdk

from experiments.auriga import (
    Endpoint,
    HaloSnapshot,
    RemoteFile,
    Submission,
    TaskCounts,
    TaskStatus,
    TransferRecord,
)

APP_NAME = "jr-hz-mono-age"
# Files whose checksum already matches on the destination are skipped, so a task
# submitted again only copies what is missing or different.
SYNC_LEVEL: Final = "checksum"
# Only standard GCS mapped collections take a data_access scope; asking for one on any
# other collection type fails the login with "Unknown Scope".
MAPPED_COLLECTION = "GCSv5_mapped_collection"


@dataclass(frozen=True)
class Collection:
    """The fields of a Globus collection document that a transfer needs."""

    collection_id: str
    display_name: str
    entity_type: str
    gcp_connected: bool | None
    https_server: str | None


def make_client(client_id: str) -> globus_sdk.TransferClient:
    """Return a Transfer client that logs in on the command line when it first needs a token."""
    config = globus_sdk.GlobusAppConfig(request_refresh_tokens=True)
    return globus_sdk.TransferClient(app=globus_sdk.UserApp(APP_NAME, client_id=client_id, config=config))


def _optional_str(value: object) -> str | None:
    """Return *value* as a string, keeping ``None``."""
    return None if value is None else str(value)


def _collection(doc: Mapping[str, object]) -> Collection:
    """Return the :class:`Collection` of a collection document."""
    connected = doc.get("gcp_connected")
    return Collection(
        collection_id=str(doc["id"]),
        display_name=str(doc["display_name"]),
        entity_type=str(doc["entity_type"]),
        gcp_connected=connected if isinstance(connected, bool) else None,
        https_server=_optional_str(doc.get("https_server")),
    )


def get_collection(client: globus_sdk.TransferClient, collection_id: str) -> Collection:
    """Return the collection with ID *collection_id*."""
    return _collection(client.get_endpoint(collection_id).data)


def find_collection(client: globus_sdk.TransferClient, name_or_id: str) -> Collection:
    """Return the collection with ID *name_or_id*, or else the one of the user's own with that display name."""
    try:
        uuid.UUID(name_or_id)
    except ValueError:
        matches = [
            doc
            for doc in client.endpoint_search(name_or_id, filter_scope="my-endpoints")
            if doc["display_name"] == name_or_id
        ]
        if len(matches) != 1:
            raise LookupError(f"{len(matches)} of your collections are named {name_or_id!r}") from None
        return _collection(matches[0])
    return get_collection(client, name_or_id)


def request_data_access(client: globus_sdk.TransferClient, collection: Collection) -> None:
    """Ask for consent to the data of *collection* at the next login, if its type needs it."""
    if collection.entity_type == MAPPED_COLLECTION:
        client.add_app_data_access_scope(collection.collection_id)


def list_files(
    client: globus_sdk.TransferClient, collection_id: str, directory: PurePosixPath, relative: PurePosixPath
) -> Iterator[RemoteFile]:
    """Yield every file under *directory* of a collection, with its path under *relative*."""
    for entry in client.operation_ls(collection_id, path=f"/{directory}/"):
        name = str(entry["name"])
        if entry["type"] == "dir":
            yield from list_files(client, collection_id, directory / name, relative / name)
        else:
            yield RemoteFile(str(relative / name), int(entry["size"]))


def list_snapshot(
    client: globus_sdk.TransferClient, collection_id: str, snapshot: HaloSnapshot
) -> tuple[RemoteFile, ...]:
    """Return the files of the snapshot and group-catalog directories of *snapshot*."""
    files = tuple(
        item
        for subdir in snapshot.subdirs
        for item in list_files(client, collection_id, snapshot.release_dir / subdir, PurePosixPath(subdir))
    )
    if not files:
        raise LookupError(f"the release lists no files under /{snapshot.release_dir}/{{{','.join(snapshot.subdirs)}}}/")
    return files


def submit_snapshot(
    client: globus_sdk.TransferClient,
    snapshot: HaloSnapshot,
    source: Collection,
    destination: Collection,
    destination_root: PurePosixPath,
) -> TransferRecord:
    """List the files of *snapshot* on *source*, submit their transfer to *destination* and return the record.

    Globus compares the checksums of source and destination after each file is copied,
    and keeps the modification times of the release.
    """
    files = list_snapshot(client, source.collection_id, snapshot)
    source_dir = PurePosixPath("/") / snapshot.release_dir
    destination_dir = destination_root / snapshot.release_dir
    data = globus_sdk.TransferData(
        source.collection_id,
        destination.collection_id,
        label=snapshot.label,
        sync_level=SYNC_LEVEL,
        verify_checksum=True,
        preserve_timestamp=True,
    )
    for subdir in snapshot.subdirs:
        data.add_item(f"{source_dir / subdir}/", f"{destination_dir / subdir}/", recursive=True)
    task_id = str(client.submit_transfer(data)["task_id"])
    return TransferRecord(
        snapshot=snapshot,
        source=Endpoint(source.collection_id, source.display_name, str(source_dir)),
        destination=Endpoint(destination.collection_id, destination.display_name, str(destination_dir)),
        files=files,
        submission=Submission(
            task_id=task_id,
            submitted=datetime.now(UTC).isoformat(timespec="seconds"),
            globus_sdk=globus_sdk.__version__,
            sync_level=SYNC_LEVEL,
        ),
    )


def task_status(client: globus_sdk.TransferClient, task_id: str) -> TaskStatus:
    """Return the current state of the transfer task *task_id*."""
    doc = client.get_task(task_id)
    return TaskStatus(
        status=str(doc["status"]),
        nice_status=_optional_str(doc.get("nice_status")),
        request_time=str(doc["request_time"]),
        completion_time=_optional_str(doc.get("completion_time")),
        bytes_per_second=int(doc["effective_bytes_per_second"]),
        counts=TaskCounts(
            files=int(doc["files"]),
            files_transferred=int(doc["files_transferred"]),
            files_skipped=int(doc["files_skipped"]),
            bytes_transferred=int(doc["bytes_transferred"]),
            faults=int(doc["faults"]),
        ),
    )
