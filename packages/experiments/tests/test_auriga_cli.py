"""Tests for the ``auriga`` commands, against canned answers of the Transfer API."""

from __future__ import annotations

from pathlib import Path

import globus_sdk
import pytest
from conftest import AURIGA_ID, GCP_ID, TASK_ID, TransferAPI, invoke, run

from experiments.auriga import HaloSnapshot, read_record
from experiments.commands._auriga import CLIENT_ID_ENV


@pytest.fixture(autouse=True)
def fixture_globus(client: globus_sdk.TransferClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand the commands the token-free client, and set the client ID."""
    for module in ("auriga_download", "auriga_status"):
        monkeypatch.setattr(f"experiments.commands.{module}.make_client", lambda _client_id: client)
    monkeypatch.setenv(CLIENT_ID_ENV, "6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _collections(transfer_api: TransferAPI, *, connected: bool) -> None:
    """Answer the lookups of the Auriga collection and of the GCP collection by name."""
    transfer_api.collection(AURIGA_ID, "The Auriga Simulations", "GCSv5_guest_collection", https_server=None)
    gcp = {"id": GCP_ID, "display_name": "jr-hz-mono-age-auriga", "entity_type": "GCP_mapped_collection"}
    transfer_api.search({**gcp, "gcp_connected": connected})


def test_download_writes_records(tmp_path: Path, transfer_api: TransferAPI) -> None:
    """One task per halo is submitted and recorded; a disconnected endpoint is a warning."""
    _collections(transfer_api, connected=False)
    for halo in (2, 3):
        transfer_api.release(HaloSnapshot(halo), 100, 200)
    transfer_api.accept_transfers()
    output = invoke("auriga", "download", "2", "3", "--root", str(tmp_path))
    assert "source: The Auriga Simulations" in output and "HTTPS server: none" in output
    assert "warning: the destination is not connected" in output
    assert "Au2: task task-0, 4 files, 0.00 GB" in output
    assert "Au3: task task-1" in output
    assert read_record(HaloSnapshot(3).manifest_path(tmp_path)).submission.task_id == "task-1"


def test_download_keeps_existing_records(copy_root: Path, au24: HaloSnapshot, transfer_api: TransferAPI) -> None:
    """An existing record stops the download, unless --resubmit, which keeps it under its task ID."""
    result = run("auriga", "download", "24", "--root", str(copy_root))
    assert result.exit_code != 0
    assert "exist; pass --resubmit to submit again" in result.output
    _collections(transfer_api, connected=True)
    transfer_api.release(au24, 100)
    transfer_api.accept_transfers()
    output = invoke("auriga", "download", "24", "--resubmit", "--root", str(copy_root))
    assert "warning" not in output
    assert read_record(au24.manifest_path(copy_root)).submission.task_id == "task-0"
    assert (au24.local_dir(copy_root) / f"transfer-{TASK_ID}.json").is_file()


def test_status_needs_a_record(tmp_path: Path) -> None:
    """Status of a halo that was never submitted is an error."""
    result = run("auriga", "status", "24", "--root", str(tmp_path))
    assert result.exit_code != 0
    assert "no transfer submitted" in result.output


def test_status_polls_and_records(copy_root: Path, au24: HaloSnapshot, transfer_api: TransferAPI) -> None:
    """A poll prints the progress and stores it in the record."""
    transfer_api.task(TASK_ID, "ACTIVE")
    output = invoke("auriga", "status", "24", "--root", str(copy_root))
    assert "Au24: ACTIVE, 4 copied and 0 skipped of 16 files, 2.00 of 0.00 GB, 0.49 MB/s, 0 faults" in output
    task = read_record(au24.manifest_path(copy_root)).task
    assert task is not None and task.status == "ACTIVE"


def test_status_waits_until_the_end(
    copy_root: Path, au24: HaloSnapshot, transfer_api: TransferAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --wait the command polls until the task ends, and an ended task is not polled again."""
    sleeps: list[float] = []
    monkeypatch.setattr("experiments.commands.auriga_status.time.sleep", sleeps.append)
    transfer_api.task(TASK_ID, "ACTIVE")
    transfer_api.task(TASK_ID, "SUCCEEDED", completion_time="2026-10-26T16:00:00+00:00")
    output = invoke("auriga", "status", "24", "--wait", "--interval", "60", "--root", str(copy_root))
    assert "Au24: ACTIVE" in output and "Au24: SUCCEEDED" in output
    assert sleeps == [60]
    assert "Au24: SUCCEEDED" in invoke("auriga", "status", "24", "--root", str(copy_root))
    task = read_record(au24.manifest_path(copy_root)).task
    assert task is not None and task.completion_time == "2026-10-26T16:00:00+00:00"


def test_status_reports_failure(copy_root: Path, transfer_api: TransferAPI) -> None:
    """A failed task makes the command fail, with the reason Globus gives."""
    transfer_api.task(TASK_ID, "FAILED", nice_status="PERMISSION_DENIED")
    result = run("auriga", "status", "24", "--root", str(copy_root))
    assert result.exit_code != 0
    assert "(PERMISSION_DENIED)" in result.output
    assert "failed: Au24" in result.output


def test_check_command(copy_root: Path, au24: HaloSnapshot) -> None:
    """The check passes on a complete copy and fails on a broken one."""
    assert "on SubhaloPos" in invoke("auriga", "check", "24", "--root", str(copy_root))
    (au24.local_dir(copy_root) / au24.subdirs[0] / "snapshot_127.1.hdf5").write_bytes(b"")
    result = run("auriga", "check", "24", "--root", str(copy_root))
    assert result.exit_code != 0
    assert "problems in Au24" in result.output
