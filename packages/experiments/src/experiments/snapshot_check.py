"""Integrity of the local copy of one Auriga snapshot.

Globus compares the checksum of every file it copies, but it does not know what a
snapshot should contain. Before the analysis reads a halo, this check confirms that:

1. every file of the release listing in ``transfer.json`` is on disk with its listed
   size;
2. every snapshot chunk opens, each particle dataset has ``NumPart_ThisFile`` rows and
   its last row is readable (a truncated HDF5 file fails there), all chunks agree on
   ``NumPart_Total`` and on the redshift, and ``NumPart_ThisFile`` summed over the
   chunks equals ``NumPart_Total``;
3. the same holds for the group catalog, against ``Ngroups_Total`` and
   ``Nsubgroups_Total``.

It also lists the particles that sit exactly on ``SubhaloPos`` of the first subhalo,
the main galaxy (Grand et al. 2024, Sect. 3.2). AGAMA's potential expansions reject a
massive particle at r = 0, so the potential fit has to leave these out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from experiments.auriga import MANIFEST_NAME, HaloSnapshot, TransferRecord, read_record

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

N_TYPES = 6


@dataclass
class CheckReport:
    """The lines of a check, and the problems among them."""

    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        """Add a line that reports a passed check or a fact."""
        self.lines.append(f"  {line}")

    def problem(self, line: str) -> None:
        """Add a line that reports a failed check."""
        self.problems.append(line)
        self.lines.append(f"  PROBLEM {line}")


def chunks(directory: Path, stem: str) -> list[Path]:
    """Return the chunks ``<stem>.0.hdf5``, ``<stem>.1.hdf5``, ... of *directory* in numerical order."""
    files = sorted(directory.glob(f"{stem}.*.hdf5"), key=lambda path: int(path.name.split(".")[1]))
    if not files:
        raise FileNotFoundError(f"no {stem}.*.hdf5 in {directory}")
    return files


def check_listing(record: TransferRecord, halo_dir: Path, report: CheckReport) -> None:
    """Check that every file of the release listing is in *halo_dir* with its listed size."""
    n_problems = len(report.problems)
    for item in record.files:
        path = halo_dir / item.path
        if not path.is_file():
            report.problem(f"{item.path}: missing")
        elif (size := path.stat().st_size) != item.size:
            report.problem(f"{item.path}: {size} bytes, the release lists {item.size}")
    if len(report.problems) == n_problems:
        report.note(
            f"{len(record.files)} files of the release listing ({record.total_bytes:,} bytes) present with their sizes"
        )
    status = "not polled yet" if record.task is None else record.task.status
    report.note(f"Globus task {record.submission.task_id}: {status}")


def _particle_total(attrs: h5py.AttributeManager) -> IntArray:
    """Return ``NumPart_Total`` with its high word, which counts beyond 2^32 particles."""
    total = np.asarray(attrs["NumPart_Total"], dtype=np.int64)
    if "NumPart_Total_HighWord" in attrs:
        total += np.asarray(attrs["NumPart_Total_HighWord"], dtype=np.int64) << 32
    return total


def _check_rows(group: h5py.Group, n_rows: int, where: str, report: CheckReport) -> None:
    """Check that every dataset of *group* has *n_rows* rows, and read the last one."""
    for key, dset in group.items():
        if dset.shape[0] != n_rows:
            report.problem(f"{where}/{key}: {dset.shape[0]} rows, header {n_rows}")
        elif n_rows:
            _ = dset[-1]


def _check_snapshot_chunk(path: Path, n_files: int, report: CheckReport) -> tuple[IntArray, IntArray, float]:
    """Check one snapshot chunk; return its ``NumPart_ThisFile``, ``NumPart_Total`` and redshift."""
    with h5py.File(path, "r") as snap:
        attrs = snap["Header"].attrs
        if int(attrs["NumFilesPerSnapshot"]) != n_files:
            report.problem(f"{path.name}: NumFilesPerSnapshot {attrs['NumFilesPerSnapshot']}, {n_files} chunks on disk")
        this = np.asarray(attrs["NumPart_ThisFile"], dtype=np.int64)
        for ptype in np.flatnonzero(this):
            _check_rows(snap[f"PartType{ptype}"], int(this[ptype]), f"{path.name} PartType{ptype}", report)
        report.note(f"{path.name}: NumPart_ThisFile {this.tolist()}")
        return this, _particle_total(attrs), float(attrs["Redshift"])


def check_snapshot(files: list[Path], report: CheckReport) -> None:
    """Check every chunk of a snapshot, and the chunk counts against the header total."""
    totals: set[tuple[int, ...]] = set()
    redshifts: set[float] = set()
    summed = np.zeros(N_TYPES, dtype=np.int64)
    for path in files:
        try:
            this, total, redshift = _check_snapshot_chunk(path, len(files), report)
        except (OSError, KeyError) as err:
            report.problem(f"{path.name}: unreadable ({err})")
            continue
        summed += this
        totals.add(tuple(total.tolist()))
        redshifts.add(redshift)
    if len(totals) != 1 or len(redshifts) != 1:
        report.problem(f"the chunks give {len(totals)} values of NumPart_Total and {len(redshifts)} redshifts")
    elif (total_list := list(totals.pop())) != summed.tolist():
        report.problem(f"NumPart_Total {total_list}, summed over the chunks {summed.tolist()}")
    else:
        report.note(f"NumPart_Total {total_list} at z = {redshifts.pop():.3f}: matches the sum over the chunks")


def check_groups(files: list[Path], report: CheckReport) -> FloatArray | None:
    """Check every chunk of a group catalog; return ``SubhaloPos`` of the first subhalo, if any."""
    counts = np.zeros(2, dtype=np.int64)
    totals: set[tuple[int, int]] = set()
    center: FloatArray | None = None
    for path in files:
        try:
            with h5py.File(path, "r") as cat:
                attrs = cat["Header"].attrs
                if int(attrs["NumFiles"]) != len(files):
                    report.problem(f"{path.name}: NumFiles {attrs['NumFiles']}, {len(files)} chunks on disk")
                this = (int(attrs["Ngroups_ThisFile"]), int(attrs["Nsubgroups_ThisFile"]))
                for name, n_rows in zip(("Group", "Subhalo"), this, strict=True):
                    if n_rows:
                        _check_rows(cat[name], n_rows, f"{path.name} {name}", report)
                if center is None and this[1]:
                    center = np.asarray(cat["Subhalo"]["SubhaloPos"][0], dtype=np.float64)
                counts += this
                totals.add((int(attrs["Ngroups_Total"]), int(attrs["Nsubgroups_Total"])))
        except (OSError, KeyError) as err:
            report.problem(f"{path.name}: unreadable ({err})")
    if len(totals) == 1 and (total := next(iter(totals))) == tuple(counts.tolist()):
        report.note(f"groups {total[0]:,}, subhalos {total[1]:,} in {len(files)} chunks: match the header totals")
    else:
        report.problem(f"groups and subhalos: {counts.tolist()} in the chunks, header totals {sorted(totals)}")
    return center


def particles_at(files: list[Path], center: FloatArray) -> list[str]:
    """Return the chunk, type and number of every set of particles exactly at *center*."""
    hits = []
    for path in files:
        with h5py.File(path, "r") as snap:
            for ptype in range(N_TYPES):
                name = f"PartType{ptype}"
                if name in snap:
                    positions = np.asarray(snap[name]["Coordinates"], dtype=np.float64)
                    if n_hit := int(np.count_nonzero(np.all(positions == center, axis=1))):
                        hits.append(f"{path.name} {name} x{n_hit}")
    return hits


def check_download(snapshot: HaloSnapshot, root: Path) -> CheckReport:
    """Check the local copy of *snapshot* under *root*."""
    halo_dir = snapshot.local_dir(root)
    report = CheckReport([f"Au{snapshot.halo} level {snapshot.level} {snapshot.suite} snapshot {snapshot.snapshot}"])
    manifest = snapshot.manifest_path(root)
    if manifest.is_file():
        check_listing(read_record(manifest), halo_dir, report)
    else:
        report.note(f"no {MANIFEST_NAME}: this copy was not made by auriga-download, so its file sizes are not checked")
    snapdir, groupdir = snapshot.subdirs
    try:
        snap_files = chunks(halo_dir / snapdir, f"snapshot_{snapshot.snapshot:03d}")
        group_files = chunks(halo_dir / groupdir, f"fof_subhalo_tab_{snapshot.snapshot:03d}")
    except FileNotFoundError as err:
        report.problem(str(err))
        return report
    check_snapshot(snap_files, report)
    center = check_groups(group_files, report)
    if report.problems:
        return report
    if center is None:
        report.problem("the catalog holds no subhalo")
    else:
        report.note("on SubhaloPos: " + (", ".join(particles_at(snap_files, center)) or "none"))
    return report
