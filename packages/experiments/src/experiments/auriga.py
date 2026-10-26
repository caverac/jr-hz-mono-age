"""Layout of the Auriga public release and the record of each download.

The release (Grand et al. 2024, MNRAS 532, 1814) is served only through Globus, from the
collection "The Auriga Simulations" on the MPCDF DataHub. A halo of a suite at Level L
lives under ``/level<L>/<suite>/halo_<N>/``, with one ``snapdir_<SSS>/`` and one
``groups_<SSS>/`` per snapshot; at Level 4, snapshot 127 is z = 0. The local copy keeps
the same layout under ``staging/auriga/``. Each halo directory also holds a
``transfer.json``, which records the files the release listed, the Globus task that
copied them and how that task ended.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

AURIGA_COLLECTION_ID = "02a2dbb8-f64d-4440-bafe-44b60b964501"
MANIFEST_NAME = "transfer.json"


@dataclass(frozen=True)
class HaloSnapshot:
    """One snapshot of one Auriga halo, the unit of a download."""

    halo: int
    level: int = 4
    suite: str = "Original"
    snapshot: int = 127

    @property
    def release_dir(self) -> PurePosixPath:
        """The halo directory relative to the root of a collection."""
        return PurePosixPath(f"level{self.level}") / self.suite / f"halo_{self.halo}"

    @property
    def subdirs(self) -> tuple[str, str]:
        """The snapshot and group-catalog directories of the snapshot."""
        return f"snapdir_{self.snapshot:03d}", f"groups_{self.snapshot:03d}"

    @property
    def label(self) -> str:
        """A Globus task label naming the project and the snapshot."""
        return f"jr-hz-mono-age Au{self.halo} level{self.level} {self.suite} snap{self.snapshot:03d}"

    def local_dir(self, root: Path) -> Path:
        """Return the halo directory of the local copy under *root*."""
        return root / self.release_dir

    def manifest_path(self, root: Path) -> Path:
        """Return the ``transfer.json`` of the halo under *root*."""
        return self.local_dir(root) / MANIFEST_NAME


@dataclass(frozen=True)
class RemoteFile:
    """A file of the release: its path relative to the halo directory, and its size in bytes."""

    path: str
    size: int


@dataclass(frozen=True)
class Endpoint:
    """One end of a transfer: a Globus collection and a directory on it."""

    collection_id: str
    display_name: str
    directory: str


@dataclass(frozen=True)
class TaskCounts:
    """The running totals of a Globus transfer task."""

    files: int
    files_transferred: int
    files_skipped: int
    bytes_transferred: int
    faults: int


@dataclass(frozen=True)
class TaskStatus:
    """The state of a Globus transfer task, as the Transfer API reports it."""

    status: str
    nice_status: str | None
    request_time: str
    completion_time: str | None
    bytes_per_second: int
    counts: TaskCounts

    @property
    def finished(self) -> bool:
        """Whether the task has ended, successfully or not."""
        return self.status in ("SUCCEEDED", "FAILED")


@dataclass(frozen=True)
class Submission:
    """The Globus task of a transfer, when it was submitted, and with which SDK and sync level."""

    task_id: str
    submitted: str
    globus_sdk: str
    sync_level: str


@dataclass(frozen=True)
class TransferRecord:
    """What was requested from Globus for one snapshot, and the last known state of the task."""

    snapshot: HaloSnapshot
    source: Endpoint
    destination: Endpoint
    files: tuple[RemoteFile, ...]
    submission: Submission
    task: TaskStatus | None = None

    @property
    def total_bytes(self) -> int:
        """The summed size of the listed files."""
        return sum(item.size for item in self.files)


def write_record(record: TransferRecord, path: Path) -> None:
    """Write *record* to *path* as indented JSON, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")


def read_record(path: Path) -> TransferRecord:
    """Read a record written by :func:`write_record`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    task = data["task"]
    if task is not None:
        task["counts"] = TaskCounts(**task["counts"])
    return TransferRecord(
        snapshot=HaloSnapshot(**data["snapshot"]),
        source=Endpoint(**data["source"]),
        destination=Endpoint(**data["destination"]),
        files=tuple(RemoteFile(**item) for item in data["files"]),
        submission=Submission(**data["submission"]),
        task=None if task is None else TaskStatus(**task),
    )
