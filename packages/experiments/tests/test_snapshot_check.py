"""Tests for the integrity check of a local copy, on synthetic snapshots."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.auriga import HaloSnapshot, RemoteFile, read_record, write_record
from experiments.snapshot_check import CheckReport, check_download, chunks


def _snapshot_chunk(root: Path, snap: HaloSnapshot, i: int) -> Path:
    """Return snapshot chunk *i* of the copy under *root*."""
    return snap.local_dir(root) / snap.subdirs[0] / f"snapshot_{snap.snapshot:03d}.{i}.hdf5"


def _catalog_chunk(root: Path, snap: HaloSnapshot, i: int) -> Path:
    """Return group-catalog chunk *i* of the copy under *root*."""
    return snap.local_dir(root) / snap.subdirs[1] / f"fof_subhalo_tab_{snap.snapshot:03d}.{i}.hdf5"


def _set_attr(path: Path, name: str, value: object) -> None:
    """Overwrite the header attribute *name* of an HDF5 file."""
    with h5py.File(path, "r+") as handle:
        handle["Header"].attrs[name] = value


def _problems(root: Path, snap: HaloSnapshot) -> str:
    """Return the problems of the check, one per line."""
    return "\n".join(check_download(snap, root).problems)


def test_complete_copy_passes(copy_root: Path, au24: HaloSnapshot) -> None:
    """A complete copy reports its counts, its task and the star on the center."""
    report = check_download(au24, copy_root)
    assert not report.problems
    output = "\n".join(report.lines)
    assert "4 files of the release listing" in output
    assert "Globus task aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee: not polled yet" in output
    assert "NumPart_Total [8, 0, 0, 0, 6, 0] at z = 0.000: matches the sum over the chunks" in output
    assert "groups 4, subhalos 2 in 2 chunks: match the header totals" in output
    assert "on SubhaloPos: snapshot_127.0.hdf5 PartType4 x1" in output


def test_copy_without_record(bare_root: Path, au24: HaloSnapshot) -> None:
    """A copy made outside auriga-download is checked without the listing."""
    report = check_download(au24, bare_root)
    assert not report.problems
    assert "no transfer.json" in "\n".join(report.lines)


def test_listing_mismatch(copy_root: Path, au24: HaloSnapshot) -> None:
    """A listed file that is missing or of another size is a problem."""
    path = au24.manifest_path(copy_root)
    record = read_record(path)
    first = record.files[0]
    files = (RemoteFile(first.path, first.size + 1), *record.files[1:], RemoteFile("snapdir_127/extra.hdf5", 1))
    write_record(replace(record, files=files), path)
    problems = _problems(copy_root, au24)
    assert f"{first.path}: {first.size} bytes, the release lists {first.size + 1}" in problems
    assert "snapdir_127/extra.hdf5: missing" in problems


def test_short_dataset(bare_root: Path, au24: HaloSnapshot) -> None:
    """A dataset with fewer rows than its header counts is a problem, and stops the check."""
    with h5py.File(_snapshot_chunk(bare_root, au24, 1), "r+") as snap:
        del snap["PartType0/Masses"]
        snap["PartType0/Masses"] = np.ones(2)
    report = check_download(au24, bare_root)
    assert report.problems == ["snapshot_127.1.hdf5 PartType0/Masses: 2 rows, header 3"]
    assert "on SubhaloPos" not in "\n".join(report.lines)


def test_snapshot_header_mismatches(bare_root: Path, au24: HaloSnapshot) -> None:
    """Chunk counts, redshifts and totals that disagree are problems."""
    _set_attr(_snapshot_chunk(bare_root, au24, 0), "NumFilesPerSnapshot", 3)
    _set_attr(_snapshot_chunk(bare_root, au24, 1), "Redshift", 1.0)
    problems = _problems(bare_root, au24)
    assert "snapshot_127.0.hdf5: NumFilesPerSnapshot 3, 2 chunks on disk" in problems
    assert "the chunks give 1 values of NumPart_Total and 2 redshifts" in problems


def test_high_word_counts(bare_root: Path, au24: HaloSnapshot) -> None:
    """The high word of NumPart_Total counts 2^32 particles."""
    for i in range(2):
        _set_attr(_snapshot_chunk(bare_root, au24, i), "NumPart_Total_HighWord", np.array([1, 0, 0, 0, 0, 0]))
    assert "NumPart_Total [4294967304, 0, 0, 0, 6, 0], summed over the chunks [8, 0, 0, 0, 6, 0]" in _problems(
        bare_root, au24
    )


def test_unreadable_chunks(bare_root: Path, au24: HaloSnapshot) -> None:
    """A chunk that is not HDF5 is reported, and the sums no longer match."""
    _snapshot_chunk(bare_root, au24, 1).write_bytes(b"not hdf5")
    _catalog_chunk(bare_root, au24, 1).write_bytes(b"not hdf5")
    problems = _problems(bare_root, au24)
    assert "snapshot_127.1.hdf5: unreadable" in problems
    assert "NumPart_Total [8, 0, 0, 0, 6, 0], summed over the chunks [5, 0, 0, 0, 4, 0]" in problems
    assert "fof_subhalo_tab_127.1.hdf5: unreadable" in problems
    assert "groups and subhalos: [3, 2] in the chunks, header totals [(4, 2)]" in problems


def test_catalog_header_mismatches(bare_root: Path, au24: HaloSnapshot) -> None:
    """A wrong chunk count and totals that disagree between chunks are problems."""
    _set_attr(_catalog_chunk(bare_root, au24, 0), "NumFiles", 1)
    _set_attr(_catalog_chunk(bare_root, au24, 1), "Ngroups_Total", 5)
    problems = _problems(bare_root, au24)
    assert "fof_subhalo_tab_127.0.hdf5: NumFiles 1, 2 chunks on disk" in problems
    assert "header totals [(4, 2), (5, 2)]" in problems


def test_catalog_without_subhalo(bare_root: Path, au24: HaloSnapshot) -> None:
    """A catalog with no subhalo has no center."""
    with h5py.File(_catalog_chunk(bare_root, au24, 0), "r+") as cat:
        del cat["Subhalo"]
        cat["Header"].attrs["Nsubgroups_ThisFile"] = 0
    for i in range(2):
        _set_attr(_catalog_chunk(bare_root, au24, i), "Nsubgroups_Total", 0)
    assert _problems(bare_root, au24) == "the catalog holds no subhalo"


def test_no_particle_on_center(bare_root: Path, au24: HaloSnapshot) -> None:
    """When no particle sits on the center, the check says so."""
    with h5py.File(_snapshot_chunk(bare_root, au24, 0), "r+") as snap:
        snap["PartType4/Coordinates"][0] = (0.0, 0.0, 0.0)
    assert "on SubhaloPos: none" in "\n".join(check_download(au24, bare_root).lines)


def test_missing_directory(tmp_path: Path, au24: HaloSnapshot) -> None:
    """A copy without chunks is a problem."""
    assert _problems(tmp_path, au24).startswith("no snapshot_127.*.hdf5 in ")


def test_chunks_sort_numerically(tmp_path: Path) -> None:
    """Chunk 10 comes after chunk 2."""
    for i in (10, 2, 0):
        (tmp_path / f"snap.{i}.hdf5").touch()
    assert [path.name for path in chunks(tmp_path, "snap")] == ["snap.0.hdf5", "snap.2.hdf5", "snap.10.hdf5"]
    with pytest.raises(FileNotFoundError):
        chunks(tmp_path, "other")


def test_report_lines() -> None:
    """Notes and problems are indented, and problems are marked."""
    report = CheckReport(["title"])
    report.note("fine")
    report.problem("broken")
    assert report.lines == ["title", "  fine", "  PROBLEM broken"]
    assert report.problems == ["broken"]
