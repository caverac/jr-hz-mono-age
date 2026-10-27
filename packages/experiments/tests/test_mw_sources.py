"""Tests of fetching and checking the Milky Way inputs, without the network."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from experiments import mw_sources
from experiments.mw_sources import (
    FILE_SOURCES,
    GAIA_CHUNKS,
    GAIA_MAXREC,
    SOURCE_ID_END,
    FileSource,
    byte_ranges,
    check_file,
    check_gaia,
    fetch_file,
    fetch_gaia,
    gaia_chunks,
    md5sum,
    run_gaia_chunk,
)

BLOB = bytes(range(256)) * 400


def blob_source(parts: int, md5: str | None = None) -> FileSource:
    """Return a source of ``BLOB`` fetched in *parts* ranges."""
    checksum = md5 if md5 is not None else hashlib.md5(BLOB, usedforsecurity=False).hexdigest()
    return FileSource("blob", "blob/file.bin", "https://example.org/file.bin", len(BLOB), checksum, parts, "agent")


class Opener:
    """A urlopen that serves ``BLOB``, or a short read, and records the requests."""

    def __init__(self, short: bool = False) -> None:
        """Serve the whole of each range, or all but its last byte if *short*."""
        self.requests: list[urllib.request.Request] = []
        self.short = short

    def __call__(self, request: urllib.request.Request) -> io.BytesIO:
        """Return the bytes of the range that *request* asks for."""
        self.requests.append(request)
        span = request.get_header("Range")
        if span is None:
            data = BLOB
        else:
            lo, hi = (int(v) for v in span.removeprefix("bytes=").split("-"))
            stop = hi + 1
            data = BLOB[lo:stop]
        return io.BytesIO(data[:-1] if self.short else data)


def test_byte_ranges_cover_the_file() -> None:
    """The ranges are consecutive, inclusive, and end at the last byte."""
    assert byte_ranges(10, 3) == [(0, 2), (3, 5), (6, 9)]
    assert byte_ranges(2, 4) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("parts", [1, 4])
def test_fetch_file_joins_its_ranges_and_skips_a_good_copy(tmp_path: Path, parts: int) -> None:
    """The ranges join into the file, with the user agent; a second fetch sends no request."""
    opener = Opener()
    path = fetch_file(blob_source(parts), tmp_path, opener)
    assert path.read_bytes() == BLOB
    assert len(opener.requests) == parts
    assert opener.requests[0].get_header("User-agent") == "agent"
    fetch_file(blob_source(parts), tmp_path, opener)
    assert len(opener.requests) == parts


def test_fetch_file_rejects_a_bad_checksum_and_a_short_read(tmp_path: Path) -> None:
    """A wrong MD5 or a missing byte is an error, and no file is left behind."""
    with pytest.raises(OSError, match="MD5"):
        fetch_file(blob_source(2, md5="0" * 32), tmp_path, Opener())
    with pytest.raises(OSError, match="bytes of"):
        fetch_file(blob_source(2), tmp_path, Opener(short=True))
    assert not (tmp_path / "blob" / "file.bin").exists()


def test_check_file(tmp_path: Path) -> None:
    """A copy is missing, of the wrong size, of the wrong checksum, or good."""
    source = blob_source(1)
    assert check_file(source, tmp_path).detail == "missing"
    path = tmp_path / source.path
    path.parent.mkdir(parents=True)
    path.write_bytes(BLOB[:-1])
    assert "bytes instead of" in check_file(source, tmp_path).detail
    path.write_bytes(BLOB[::-1])
    assert "MD5" in check_file(source, tmp_path).detail
    path.write_bytes(BLOB)
    assert check_file(source, tmp_path).ok
    assert md5sum(path) == source.md5


def test_the_sources_pin_their_files() -> None:
    """Every file has a URL, a size and an MD5 checksum; the two APOGEE files come in byte ranges."""
    names = {s.name: s for s in FILE_SOURCES}
    assert set(names) == {"astronn", "allstar", "lamost", "imig"}
    assert all(len(s.md5) == 32 and s.size > 0 and s.url.startswith("https://") for s in FILE_SOURCES)
    assert names["astronn"].parts > 1 and names["lamost"].user_agent is not None


def test_gaia_chunks_cover_every_source_id() -> None:
    """The chunks are consecutive and span [0, SOURCE_ID_END)."""
    chunks = gaia_chunks()
    assert len(chunks) == GAIA_CHUNKS
    assert chunks[0][0] == 0 and chunks[-1][1] == SOURCE_ID_END
    assert all(a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:]))


class Job:
    """A TAP job that completes after a number of polls, with a table of a given length."""

    def __init__(self, polls: int, rows: int, fail: bool = False) -> None:
        """Complete after *polls* polls with *rows* rows, or fail."""
        self.job_id = "job"
        self.polls, self.rows, self.fail = polls, rows, fail

    @property
    def phase(self) -> str:
        """Return EXECUTING until the polls run out."""
        self.polls -= 1
        return "EXECUTING" if self.polls >= 0 else ("ERROR" if self.fail else "COMPLETED")

    def run(self) -> object:
        """Start nothing."""
        return self

    def raise_if_error(self) -> None:
        """Raise if the job failed."""
        if self.fail:
            raise RuntimeError("job failed")

    def fetch_result(self) -> Job:
        """Return itself as the result."""
        return self

    def to_table(self) -> Table:
        """Return a table of the given length."""
        return Table({"source_id": np.arange(self.rows)})


class Service:
    """A TAP service whose jobs take *polls* polls and return *rows* rows."""

    def __init__(self, polls: int = 1, rows: int = 2, fail: bool = False) -> None:
        """Make jobs of *polls* polls and *rows* rows, which fail if *fail*."""
        self.polls, self.rows, self.fail = polls, rows, fail
        self.queries: list[str] = []

    def submit_job(self, query: str, *, maxrec: int) -> Job:
        """Record the query and return a job."""
        assert maxrec == GAIA_MAXREC
        self.queries.append(query)
        return Job(self.polls, self.rows, self.fail)


def test_run_gaia_chunk_polls_and_checks() -> None:
    """A chunk is polled until it completes; a failure, a timeout and a truncation are errors."""
    slept: list[float] = []
    assert len(run_gaia_chunk(Service(polls=2), 0, 10, slept.append)) == 2
    assert len(slept) == 2
    with pytest.raises(RuntimeError, match="job failed"):
        run_gaia_chunk(Service(fail=True), 0, 10, slept.append)
    with pytest.raises(TimeoutError):
        run_gaia_chunk(Service(polls=10**6), 0, 10, slept.append)
    with pytest.raises(RuntimeError, match="truncated"):
        run_gaia_chunk(Service(rows=GAIA_MAXREC), 0, 10, slept.append)


def test_fetch_gaia_stacks_the_chunks_and_counts_the_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chunks stack into the file; the wrong count of rows is an error; a good copy is kept."""
    monkeypatch.setattr(mw_sources, "GAIA_ROWS", 2 * GAIA_CHUNKS)
    service = Service(polls=0, rows=2)
    path = fetch_gaia(tmp_path, service, lambda _: None)
    assert len(Table.read(path)) == 2 * GAIA_CHUNKS
    assert "source_id >= 0 AND source_id < " in service.queries[0]
    assert fetch_gaia(tmp_path, Service(rows=99), lambda _: None) == path
    assert check_gaia(tmp_path).ok
    monkeypatch.setattr(mw_sources, "GAIA_ROWS", 7)
    assert "rows instead of" in check_gaia(tmp_path).detail
    with pytest.raises(OSError, match="rows instead of"):
        fetch_gaia(tmp_path, Service(polls=0, rows=2), lambda _: None)
    assert check_gaia(tmp_path / "empty").detail == "missing"


def test_source_with_a_changed_size_is_refetched(tmp_path: Path) -> None:
    """A copy of the wrong size is downloaded again."""
    source = dataclasses.replace(blob_source(1))
    path = tmp_path / source.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old")
    opener = Opener()
    fetch_file(source, tmp_path, opener)
    assert len(opener.requests) == 1 and path.read_bytes() == BLOB
