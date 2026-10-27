"""Shared fixtures of the test suite.

The Auriga fixtures build a small copy of one snapshot in the layout of the release: two
snapshot chunks with gas and stars, one star exactly on the center of the main subhalo,
and a two-chunk group catalog whose second chunk holds no subhalo. The Globus fixtures
answer the Transfer API with ``responses``, so the real ``TransferClient`` runs without a
network or a login. The mock fixtures build one small mock galaxy per session
(:mod:`experiments.mock`), prepare it and analyze it, because each step takes seconds.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import globus_sdk
import h5py
import numpy as np
import pytest
import responses
from click.testing import CliRunner, Result
from responses import matchers

from experiments.analysis import HaloAnalysis, PreparedHalo, analyze_halo, prepare_halo
from experiments.auriga import Endpoint, HaloSnapshot, RemoteFile, Submission, TransferRecord, write_record
from experiments.cli import main
from experiments.mock import MockGalaxy, MockSizes, build_mock
from experiments.palicio import PALICIO_WINDOWS
from experiments.populations import Method

TRANSFER_API = "https://transfer.api.globus.org/v0.10"
AURIGA_ID = "02a2dbb8-f64d-4440-bafe-44b60b964501"
GCP_ID = "11111111-2222-3333-4444-555555555555"
TASK_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CENTER = (100.0, 100.0, 100.0)
# Gas and star particles per snapshot chunk.
GAS = (5, 3)
STARS = (4, 2)
# Groups and subhalos per catalog chunk.
GROUPS = (3, 1)
SUBHALOS = (2, 0)
# A mock small enough to build in seconds, with about 110 stars per 1 Gyr bin of the annulus.
MOCK_SIZES = MockSizes(n_disk=20_000, n_halo=20_000, n_stars=3_000)
MOCK_SNAPSHOT = HaloSnapshot(0)
MOCK_BOOTSTRAP = 4


def run(*args: str) -> Result:
    """Invoke the CLI with *args* and return the result."""
    return CliRunner().invoke(main, list(args))


def invoke(*args: str) -> str:
    """Invoke the CLI with *args*, assert that it succeeded and return its output."""
    result = run(*args)
    if result.exc_info is not None and not isinstance(result.exception, SystemExit):
        traceback.print_exception(*result.exc_info)
    assert result.exit_code == 0, result.output
    return result.output


def _particle_counts(n_gas: int, n_stars: int) -> list[int]:
    """Return a ``NumPart`` vector with gas and stars only."""
    return [n_gas, 0, 0, 0, n_stars, 0]


def write_copy(root: Path, snapshot: HaloSnapshot, *, manifest: bool = True) -> Path:
    """Write a synthetic local copy of *snapshot* under *root*, and return its halo directory."""
    halo_dir = snapshot.local_dir(root)
    snapdir, groupdir = (halo_dir / name for name in snapshot.subdirs)
    snapdir.mkdir(parents=True)
    groupdir.mkdir()
    rng = np.random.default_rng(3)
    total = _particle_counts(sum(GAS), sum(STARS))
    for i, (n_gas, n_stars) in enumerate(zip(GAS, STARS, strict=True)):
        with h5py.File(snapdir / f"snapshot_{snapshot.snapshot:03d}.{i}.hdf5", "w") as snap:
            header = snap.create_group("Header")
            header.attrs["NumFilesPerSnapshot"] = len(GAS)
            header.attrs["NumPart_ThisFile"] = np.array(_particle_counts(n_gas, n_stars), dtype=np.uint32)
            header.attrs["NumPart_Total"] = np.array(total, dtype=np.uint32)
            header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
            header.attrs["Redshift"] = 0.0
            for ptype, n in ((0, n_gas), (4, n_stars)):
                positions = rng.uniform(90.0, 110.0, size=(n, 3))
                if ptype == 4 and i == 0:
                    positions[0] = CENTER
                group = snap.create_group(f"PartType{ptype}")
                group["Coordinates"] = positions
                group["Masses"] = np.full(n, 1e-5)
    for i, (n_groups, n_subhalos) in enumerate(zip(GROUPS, SUBHALOS, strict=True)):
        with h5py.File(groupdir / f"fof_subhalo_tab_{snapshot.snapshot:03d}.{i}.hdf5", "w") as cat:
            header = cat.create_group("Header")
            header.attrs["NumFiles"] = len(GROUPS)
            header.attrs["Ngroups_ThisFile"] = n_groups
            header.attrs["Nsubgroups_ThisFile"] = n_subhalos
            header.attrs["Ngroups_Total"] = sum(GROUPS)
            header.attrs["Nsubgroups_Total"] = sum(SUBHALOS)
            cat.create_group("Group")["GroupPos"] = rng.uniform(0.0, 200.0, size=(n_groups, 3))
            if n_subhalos:
                positions = rng.uniform(0.0, 200.0, size=(n_subhalos, 3))
                positions[0] = CENTER
                cat.create_group("Subhalo")["SubhaloPos"] = positions
    if manifest:
        files = tuple(
            RemoteFile(str(path.relative_to(halo_dir)), path.stat().st_size)
            for path in sorted((*snapdir.iterdir(), *groupdir.iterdir()))
        )
        write_record(make_record(snapshot, files), snapshot.manifest_path(root))
    return halo_dir


def make_record(snapshot: HaloSnapshot, files: tuple[RemoteFile, ...]) -> TransferRecord:
    """Return the record of a transfer of *files* that has not been polled."""
    return TransferRecord(
        snapshot=snapshot,
        source=Endpoint(AURIGA_ID, "The Auriga Simulations", f"/{snapshot.release_dir}"),
        destination=Endpoint(GCP_ID, "jr-hz-mono-age-auriga", f"/data/auriga/{snapshot.release_dir}"),
        files=files,
        submission=Submission(TASK_ID, "2026-10-26T12:00:00+00:00", globus_sdk.__version__, "checksum"),
    )


@pytest.fixture(name="au24")
def fixture_au24() -> HaloSnapshot:
    """Return snapshot 127 of Au24 at Level 4."""
    return HaloSnapshot(24)


@pytest.fixture(name="copy_root")
def fixture_copy_root(tmp_path: Path, au24: HaloSnapshot) -> Path:
    """Return the root of a synthetic local copy of Au24, with its ``transfer.json``."""
    root = tmp_path / "auriga"
    write_copy(root, au24)
    return root


@pytest.fixture(name="bare_root")
def fixture_bare_root(tmp_path: Path, au24: HaloSnapshot) -> Path:
    """Return the root of a synthetic local copy of Au24 without ``transfer.json``, free to modify."""
    root = tmp_path / "auriga"
    write_copy(root, au24, manifest=False)
    return root


@dataclass
class TransferAPI:
    """Canned answers of the Transfer API, registered on a ``responses`` mock."""

    mock: responses.RequestsMock
    submitted: list[str] = field(default_factory=list)

    def collection(self, collection_id: str, display_name: str, entity_type: str, **extra: object) -> None:
        """Answer ``GET /endpoint/<collection_id>`` with a collection document."""
        doc = {"id": collection_id, "display_name": display_name, "entity_type": entity_type, **extra}
        self.mock.get(f"{TRANSFER_API}/endpoint/{collection_id}", json=doc)

    def search(self, *docs: dict[str, object]) -> None:
        """Answer ``GET /endpoint_search`` with *docs*."""
        self.mock.get(f"{TRANSFER_API}/endpoint_search", json={"DATA": list(docs)})

    def listing(self, collection_id: str, path: str, *entries: tuple[str, str, int]) -> None:
        """Answer ``ls`` of *path* with entries of name, type and size."""
        data = [{"name": name, "type": kind, "size": size} for name, kind, size in entries]
        self.mock.get(
            f"{TRANSFER_API}/operation/endpoint/{collection_id}/ls",
            json={"DATA": data},
            match=[matchers.query_param_matcher({"path": path}, strict_match=False)],
        )

    def release(self, snapshot: HaloSnapshot, *sizes: int) -> None:
        """Answer ``ls`` of the two directories of *snapshot*, with one chunk per size in each."""
        stems = (f"snapshot_{snapshot.snapshot:03d}", f"fof_subhalo_tab_{snapshot.snapshot:03d}")
        for subdir, stem in zip(snapshot.subdirs, stems, strict=True):
            entries = [(f"{stem}.{i}.hdf5", "file", size) for i, size in enumerate(sizes)]
            self.listing(AURIGA_ID, f"/{snapshot.release_dir}/{subdir}/", *entries)

    def accept_transfers(self) -> None:
        """Answer the submission ID and transfer requests, numbering the tasks in order."""
        self.mock.get(f"{TRANSFER_API}/submission_id", json={"value": "submission"})
        self.mock.add_callback(responses.POST, f"{TRANSFER_API}/transfer", callback=self._submit)

    def _submit(self, _request: object) -> tuple[int, dict[str, str], str]:
        """Accept one transfer and return its task ID."""
        task_id = f"task-{len(self.submitted)}"
        self.submitted.append(task_id)
        return 202, {"Content-Type": "application/json"}, f'{{"task_id": "{task_id}", "code": "Accepted"}}'

    def task(self, task_id: str, status: str, **extra: object) -> None:
        """Answer ``GET /task/<task_id>`` once with a task document in *status*."""
        doc: dict[str, object] = {
            "status": status,
            "nice_status": None,
            "request_time": "2026-10-26T12:00:00+00:00",
            "completion_time": None,
            "files": 16,
            "files_transferred": 4,
            "files_skipped": 0,
            "bytes_transferred": 2_000_000_000,
            "effective_bytes_per_second": 490_000,
            "faults": 0,
        }
        self.mock.get(f"{TRANSFER_API}/task/{task_id}", json={**doc, **extra})


@pytest.fixture(name="transfer_api")
def fixture_transfer_api() -> Iterator[TransferAPI]:
    """Yield the canned Transfer API; a registered answer must not be left unused."""
    with responses.RequestsMock(assert_all_requests_are_fired=True) as mock:
        yield TransferAPI(mock)


@pytest.fixture(name="client")
def fixture_client() -> globus_sdk.TransferClient:
    """Return a Transfer client that sends no token."""
    return globus_sdk.TransferClient(authorizer=globus_sdk.NullAuthorizer())


@dataclass(frozen=True)
class MockRun:
    """A mock galaxy written under *root*, its truth, the prepared halo and its analysis."""

    root: Path
    truth: MockGalaxy
    prepared: PreparedHalo
    analysis: HaloAnalysis


@pytest.fixture(name="mock_run", scope="session")
def fixture_mock_run(tmp_path_factory: pytest.TempPathFactory) -> MockRun:
    """Return a small mock galaxy as halo 0, prepared and analyzed with Au24's window."""
    root = tmp_path_factory.mktemp("mock")
    truth = build_mock(MOCK_SNAPSHOT, root, MOCK_SIZES)
    prepared = prepare_halo(MOCK_SNAPSHOT, root)
    analysis = analyze_halo(prepared, Method(PALICIO_WINDOWS[24], 2.0, MOCK_BOOTSTRAP))
    return MockRun(root, truth, prepared, analysis)
