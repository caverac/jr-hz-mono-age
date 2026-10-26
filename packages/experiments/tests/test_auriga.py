"""Tests for the release layout and the transfer records."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath

from conftest import make_record

from experiments.auriga import HaloSnapshot, RemoteFile, TaskCounts, TaskStatus, read_record, write_record


def test_layout_of_a_snapshot(tmp_path: Path) -> None:
    """The release paths, the local copy and the label follow the release layout."""
    snap = HaloSnapshot(2, level=3, suite="Original", snapshot=63)
    assert snap.release_dir == PurePosixPath("level3/Original/halo_2")
    assert snap.subdirs == ("snapdir_063", "groups_063")
    assert snap.local_dir(tmp_path) == tmp_path / "level3" / "Original" / "halo_2"
    assert snap.manifest_path(tmp_path) == tmp_path / "level3" / "Original" / "halo_2" / "transfer.json"
    assert snap.label == "jr-hz-mono-age Au2 level3 Original snap063"


def test_record_round_trip(tmp_path: Path, au24: HaloSnapshot) -> None:
    """A record reads back equal to what was written, before and after a poll."""
    files = (RemoteFile("snapdir_127/snapshot_127.0.hdf5", 10), RemoteFile("groups_127/fof_subhalo_tab_127.0.hdf5", 5))
    record = make_record(au24, files)
    path = tmp_path / "halo" / "transfer.json"
    write_record(record, path)
    assert read_record(path) == record
    assert record.total_bytes == 15
    counts = TaskCounts(files=2, files_transferred=2, files_skipped=0, bytes_transferred=15, faults=0)
    polled = replace(record, task=TaskStatus("SUCCEEDED", None, "t0", "t1", 1, counts))
    write_record(polled, path)
    assert read_record(path) == polled
    assert polled.task is not None and polled.task.finished
    assert not replace(polled.task, status="ACTIVE").finished
