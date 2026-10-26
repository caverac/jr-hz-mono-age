"""Tests for the Globus transfers, against canned answers of the Transfer API."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import globus_sdk
import pytest
from conftest import AURIGA_ID, GCP_ID, TransferAPI

from experiments.auriga import HaloSnapshot, RemoteFile
from experiments.globus_transfer import (
    Collection,
    find_collection,
    get_collection,
    list_files,
    list_snapshot,
    make_client,
    request_data_access,
    submit_snapshot,
    task_status,
)

GUEST = Collection(AURIGA_ID, "The Auriga Simulations", "GCSv5_guest_collection", None, None)
GCP = Collection(GCP_ID, "jr-hz-mono-age-auriga", "GCP_mapped_collection", True, None)


def test_make_client_logs_in_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The client is built without a request; its tokens live under the home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    client = make_client("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    assert isinstance(client, globus_sdk.TransferClient)
    assert (tmp_path / ".globus" / "app").is_dir()


def test_get_collection(client: globus_sdk.TransferClient, transfer_api: TransferAPI) -> None:
    """A collection document maps onto a Collection, with the optional fields kept."""
    transfer_api.collection(AURIGA_ID, "The Auriga Simulations", "GCSv5_mapped_collection", https_server="https://x")
    transfer_api.collection(GCP_ID, "jr-hz-mono-age-auriga", "GCP_mapped_collection", gcp_connected=False)
    auriga = get_collection(client, AURIGA_ID)
    assert auriga == Collection(AURIGA_ID, "The Auriga Simulations", "GCSv5_mapped_collection", None, "https://x")
    assert get_collection(client, GCP_ID).gcp_connected is False


def test_find_collection(client: globus_sdk.TransferClient, transfer_api: TransferAPI) -> None:
    """A collection is found by ID, or by the exact display name among the user's own."""
    transfer_api.collection(GCP_ID, "jr-hz-mono-age-auriga", "GCP_mapped_collection", gcp_connected=True)
    transfer_api.search(
        {"id": GCP_ID, "display_name": "jr-hz-mono-age-auriga", "entity_type": "GCP_mapped_collection"},
        {"id": AURIGA_ID, "display_name": "jr-hz-mono-age-auriga old", "entity_type": "GCP_mapped_collection"},
    )
    assert find_collection(client, GCP_ID) == GCP
    assert find_collection(client, "jr-hz-mono-age-auriga").collection_id == GCP_ID
    with pytest.raises(LookupError, match="0 of your collections are named 'missing'"):
        find_collection(client, "missing")


def test_request_data_access(client: globus_sdk.TransferClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a GCS mapped collection asks for a data_access consent."""
    requested: list[str] = []

    def add_scope(self: globus_sdk.TransferClient, collection_ids: str) -> globus_sdk.TransferClient:
        requested.append(collection_ids)
        return self

    monkeypatch.setattr(globus_sdk.TransferClient, "add_app_data_access_scope", add_scope)
    request_data_access(client, GUEST)
    request_data_access(client, GCP)
    assert not requested
    request_data_access(client, Collection(AURIGA_ID, "Auriga", "GCSv5_mapped_collection", None, None))
    assert requested == [AURIGA_ID]


def test_list_files_recurses(client: globus_sdk.TransferClient, transfer_api: TransferAPI) -> None:
    """Files below subdirectories are listed with their path from the listed directory."""
    transfer_api.listing(AURIGA_ID, "/top/", ("a.hdf5", "file", 3), ("sub", "dir", 0))
    transfer_api.listing(AURIGA_ID, "/top/sub/", ("b.hdf5", "file", 4))
    files = list(list_files(client, AURIGA_ID, PurePosixPath("top"), PurePosixPath("top")))
    assert files == [RemoteFile("top/a.hdf5", 3), RemoteFile("top/sub/b.hdf5", 4)]


def test_list_snapshot(client: globus_sdk.TransferClient, transfer_api: TransferAPI, au24: HaloSnapshot) -> None:
    """A snapshot lists the files of both of its directories; an empty one is an error."""
    transfer_api.release(au24, 10, 20)
    assert [item.path for item in list_snapshot(client, AURIGA_ID, au24)] == [
        "snapdir_127/snapshot_127.0.hdf5",
        "snapdir_127/snapshot_127.1.hdf5",
        "groups_127/fof_subhalo_tab_127.0.hdf5",
        "groups_127/fof_subhalo_tab_127.1.hdf5",
    ]
    au2 = HaloSnapshot(2)
    transfer_api.release(au2)
    with pytest.raises(LookupError, match=r"no files under /level4/Original/halo_2/\{snapdir_127,groups_127\}/"):
        list_snapshot(client, AURIGA_ID, au2)


def test_submit_snapshot(client: globus_sdk.TransferClient, transfer_api: TransferAPI, au24: HaloSnapshot) -> None:
    """The submitted task copies both directories with checksums, and the record describes it."""
    transfer_api.release(au24, 10, 20)
    transfer_api.accept_transfers()
    record = submit_snapshot(client, au24, GUEST, GCP, PurePosixPath("/data/auriga"))
    request_body = transfer_api.mock.calls[-1].request.body
    assert isinstance(request_body, bytes | str)
    body = json.loads(request_body)
    assert body["label"] == "jr-hz-mono-age Au24 level4 Original snap127"
    # The SDK sends sync_level as the API's integer code, in which 3 is "checksum".
    assert (body["sync_level"], body["verify_checksum"], body["preserve_timestamp"]) == (3, True, True)
    assert [(item["source_path"], item["destination_path"], item["recursive"]) for item in body["DATA"]] == [
        ("/level4/Original/halo_24/snapdir_127/", "/data/auriga/level4/Original/halo_24/snapdir_127/", True),
        ("/level4/Original/halo_24/groups_127/", "/data/auriga/level4/Original/halo_24/groups_127/", True),
    ]
    assert record.submission.task_id == "task-0"
    assert record.source.directory == "/level4/Original/halo_24"
    assert record.destination.directory == "/data/auriga/level4/Original/halo_24"
    assert record.total_bytes == 60
    assert record.task is None


def test_task_status(client: globus_sdk.TransferClient, transfer_api: TransferAPI) -> None:
    """A task document maps onto a TaskStatus."""
    transfer_api.task("task-0", "INACTIVE", nice_status="ENDPOINT_ERROR", completion_time="t1")
    status = task_status(client, "task-0")
    assert (status.status, status.nice_status, status.completion_time) == ("INACTIVE", "ENDPOINT_ERROR", "t1")
    assert (status.counts.files, status.counts.bytes_transferred, status.bytes_per_second) == (
        16,
        2_000_000_000,
        490_000,
    )
