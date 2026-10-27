"""The Milky Way inputs: where each one comes from, where it lives, and how to fetch and check it.

Five inputs, none of them committed, live under ``staging/milky-way/``:

- ``astronn``: the astroNN value-added catalog of APOGEE DR17, with the abundances of
  Leung & Bovy (2019), the distances, Galactocentric coordinates and actions, and the
  ages of Mackereth et al. (2019);
- ``allstar``: APOGEE DR17 allStarLite, row-matched to the astroNN catalog, for the flag
  ``EXTRATARG`` that separates the main survey from the special programs;
- ``lamost``: the 137,448 primary red-clump stars of LAMOST of Huang et al. (2020, ApJS
  249, 29), from Zenodo record 3875974;
- ``imig``: the density fits of mono-age, mono-abundance populations of Imig et al.
  (2025, ApJ 990, 203), at a fixed commit of their repository;
- ``gaia``: the Gaia DR3 sources of a cylinder of radius 1.55 kpc and half-height 1.3 kpc
  around the Sun whose Wesenheit absolute magnitude admits a red-clump giant
  (:data:`GAIA_QUERY`), from the ESA archive in ``GAIA_CHUNKS`` ranges of source_id,
  because the archive aborts anonymous jobs after 60 s.

The sizes and MD5 checksums pin the files the analysis was run on, downloaded on
2026-10-22. The SDSS servers throttle each connection, so the two APOGEE files come in
``parts`` byte ranges fetched at once; Zenodo refuses the default user agent of urllib.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

from astropy.table import Table, vstack

BROWSER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15"
BLOCK_BYTES = 1 << 20
MD5_BLOCK_BYTES = 1 << 24


@dataclass(frozen=True)
class FileSource:
    """A file fetched over HTTP, its place under the root, its size (bytes) and MD5 checksum."""

    name: str
    path: str
    url: str
    size: int
    md5: str
    parts: int = 1
    user_agent: str | None = None


FILE_SOURCES = (
    FileSource(
        "astronn",
        "apogee/apogee_astroNN-DR17.fits",
        "https://dr17.sdss.org/sas/dr17/env/APOGEE_ASTRO_NN/apogee_astroNN-DR17.fits",
        647_328_960,
        "3bd18896c2624437001a1fa96b3be9b6",
        parts=24,
    ),
    FileSource(
        "allstar",
        "apogee/allStarLite-dr17-synspec_rev1.fits",
        "https://dr17.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStarLite-dr17-synspec_rev1.fits",
        1_834_963_200,
        "e878b29a139c2bd50ed786f913045414",
        parts=24,
    ),
    FileSource(
        "lamost",
        "lamost/LMRCV1.fits",
        "https://zenodo.org/records/3875974/files/LMRCV1.fits?download=1",
        23_653_440,
        "c2b3afc4d9fbbd6ea383ab723a475f40",
        user_agent=BROWSER_AGENT,
    ),
    FileSource(
        "imig",
        "imig2025/mw_density_params.fits",
        "https://raw.githubusercontent.com/astrojimig/mw_density_imig2025/"
        "18a367aab9f1c0ddd7233187263354571b451e5d/results/mw_density_params.fits",
        27_881_280,
        "01a651a2fc3712de22bb71a241ab4bd2",
    ),
)

GAIA_PATH = "gaia/rc_cylinder_1p5kpc_dr3.fits"
GAIA_ROWS = 888_966
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
GAIA_CHUNKS = 16
GAIA_MAXREC = 3_000_000  # the archive truncates at 50,000 rows unless told otherwise
GAIA_POLL_S = 10.0
GAIA_TIMEOUT_S = 1800.0
SOURCE_ID_END = 12 * 4**12 * 2**35  # one past the largest DR3 source_id (HEALPix level 12)
# The window of the Wesenheit magnitude is written as bounds on the parallax, because the
# archive may evaluate LOG10(parallax) before the filter parallax > 0.5.
GAIA_QUERY = """
SELECT source_id, ra, dec, l, b, parallax, parallax_error, pmra, pmdec,
       phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE source_id >= {lo} AND source_id < {hi}
  AND parallax > 0.5 AND parallax_over_error > 5 AND phot_g_mean_mag < 16.0
  AND bp_rp > 0.9
  AND parallax > POWER(10.0, (7.2 - phot_g_mean_mag + 1.9*bp_rp) / 5.0)
  AND parallax < POWER(10.0, (9.3 - phot_g_mean_mag + 1.9*bp_rp) / 5.0)
  AND COS(RADIANS(b))/parallax < 1.55
  AND ABS(SIN(RADIANS(b))/parallax) < 1.3
"""

Opener = Callable[[urllib.request.Request], IO[bytes]]


def md5sum(path: Path) -> str:
    """Return the MD5 checksum of the file at *path*."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(MD5_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def byte_ranges(size: int, parts: int) -> list[tuple[int, int]]:
    """Return *parts* consecutive inclusive byte ranges that cover a file of *size* bytes."""
    edges = [size * k // parts for k in range(parts + 1)]
    return [(lo, hi - 1) for lo, hi in zip(edges[:-1], edges[1:], strict=True) if hi > lo]


def _fetch_range(source: FileSource, span: tuple[int, int] | None, target: Path, opener: Opener) -> None:
    """Write the bytes *span* of *source* (all of it if None) to *target*."""
    headers = {} if source.user_agent is None else {"User-Agent": source.user_agent}
    if span is not None:
        headers["Range"] = f"bytes={span[0]}-{span[1]}"
    with opener(urllib.request.Request(source.url, headers=headers)) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, BLOCK_BYTES)
    expected = source.size if span is None else span[1] - span[0] + 1
    if target.stat().st_size != expected:
        raise OSError(f"{source.name}: got {target.stat().st_size} bytes of {expected}")


def fetch_file(source: FileSource, root: Path, opener: Opener = urllib.request.urlopen) -> Path:
    """Download *source* under *root* in its byte ranges and check its MD5; skip it if it is already there."""
    target = root / source.path
    if target.exists() and target.stat().st_size == source.size and md5sum(target) == source.md5:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    spans: list[tuple[int, int] | None] = [None] if source.parts == 1 else list(byte_ranges(source.size, source.parts))
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        pieces = [Path(scratch) / f"part{k:03d}" for k in range(len(spans))]
        with ThreadPoolExecutor(max_workers=len(spans)) as pool:
            list(pool.map(lambda job: _fetch_range(source, job[0], job[1], opener), zip(spans, pieces)))
        joined = Path(scratch) / "joined"
        with joined.open("wb") as out:
            for piece in pieces:
                with piece.open("rb") as part:
                    shutil.copyfileobj(part, out, BLOCK_BYTES)
        if (checksum := md5sum(joined)) != source.md5:
            raise OSError(f"{source.name}: MD5 {checksum} instead of {source.md5}")
        joined.replace(target)
    return target


class TapJob(Protocol):
    """The part of a pyvo asynchronous TAP job that the Gaia fetch uses."""

    job_id: str

    @property
    def phase(self) -> str:
        """The phase of the job, such as EXECUTING or COMPLETED."""

    def run(self) -> object:
        """Start the job."""

    def raise_if_error(self) -> None:
        """Raise if the job failed."""

    def fetch_result(self) -> TapResult:
        """Return the result of the finished job."""


class TapResult(Protocol):  # pylint: disable=too-few-public-methods
    """The result of a TAP job."""

    def to_table(self) -> Table:
        """Return the result as an astropy table."""


class TapService(Protocol):  # pylint: disable=too-few-public-methods
    """The part of a pyvo TAP service that the Gaia fetch uses."""

    def submit_job(self, query: str, *, maxrec: int) -> TapJob:
        """Submit an asynchronous job."""


def gaia_chunks() -> list[tuple[int, int]]:
    """Return the ``GAIA_CHUNKS`` ranges of source_id [lo, hi) that cover Gaia DR3."""
    step = SOURCE_ID_END // GAIA_CHUNKS
    return [(k * step, SOURCE_ID_END if k == GAIA_CHUNKS - 1 else (k + 1) * step) for k in range(GAIA_CHUNKS)]


def run_gaia_chunk(service: TapService, lo: int, hi: int, sleep: Callable[[float], None] = time.sleep) -> Table:
    """Run the Gaia query over source_id in [lo, hi) as one asynchronous job and return its table.

    The job is polled rather than waited for, because the archive drops long-poll
    connections; a result as long as ``GAIA_MAXREC`` would be truncated, which is an error.
    """
    job = service.submit_job(GAIA_QUERY.format(lo=lo, hi=hi), maxrec=GAIA_MAXREC)
    job.run()
    waited = 0.0
    while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
        if waited > GAIA_TIMEOUT_S:
            raise TimeoutError(f"Gaia job {job.job_id} still {job.phase}")
        sleep(GAIA_POLL_S)
        waited += GAIA_POLL_S
    job.raise_if_error()
    table = job.fetch_result().to_table()
    if len(table) >= GAIA_MAXREC:
        raise RuntimeError(f"the Gaia chunk {lo}-{hi} is truncated at {GAIA_MAXREC} rows")
    return table


def fetch_gaia(root: Path, service: TapService, sleep: Callable[[float], None] = time.sleep) -> Path:
    """Run the Gaia query chunk by chunk, stack the chunks and write them under *root*; skip it if it is there.

    The file has no fixed checksum, because astropy writes the date into its header, so
    the count of rows, ``GAIA_ROWS``, checks it instead.
    """
    target = root / GAIA_PATH
    if target.exists() and len(Table.read(target)) == GAIA_ROWS:
        return target
    table = vstack([run_gaia_chunk(service, lo, hi, sleep) for lo, hi in gaia_chunks()])
    if len(table) != GAIA_ROWS:
        raise OSError(f"gaia: {len(table)} rows instead of {GAIA_ROWS}")
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write(target, overwrite=True)
    return target


@dataclass(frozen=True)
class Check:
    """The state of one local input: whether it exists and matches its pinned size and checksum, or rows."""

    name: str
    path: Path
    ok: bool
    detail: str


def check_file(source: FileSource, root: Path) -> Check:
    """Return whether *source* is under *root* with its size and MD5."""
    path = root / source.path
    if not path.exists():
        return Check(source.name, path, False, "missing")
    if (size := path.stat().st_size) != source.size:
        return Check(source.name, path, False, f"{size} bytes instead of {source.size}")
    if (checksum := md5sum(path)) != source.md5:
        return Check(source.name, path, False, f"MD5 {checksum} instead of {source.md5}")
    return Check(source.name, path, True, f"{size:,} bytes, MD5 {checksum}")


def check_gaia(root: Path) -> Check:
    """Return whether the Gaia file is under *root* with its ``GAIA_ROWS`` rows."""
    path = root / GAIA_PATH
    if not path.exists():
        return Check("gaia", path, False, "missing")
    if (rows := len(Table.read(path))) != GAIA_ROWS:
        return Check("gaia", path, False, f"{rows} rows instead of {GAIA_ROWS}")
    return Check("gaia", path, True, f"{rows:,} rows")
