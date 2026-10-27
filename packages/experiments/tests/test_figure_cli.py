"""Tests of ``experiments auriga render-data``, ``plot`` and ``render``, and of ``experiments figures``.

The mock galaxy of the session is linked as Au5 and Au6 (and as Au24 for ``figures``,
which draws the halos of Palicio's table), so that the commands run on real snapshot
chunks without a download.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run

from experiments.auriga import HaloSnapshot
from experiments.commands._report import write_json


def link_mock(mock_run: MockRun, root: Path, halo: int) -> None:
    """Link the mock's halo directory as *halo* under *root*."""
    halo_dir = HaloSnapshot(halo).local_dir(root)
    halo_dir.parent.mkdir(parents=True, exist_ok=True)
    halo_dir.symlink_to(MOCK_SNAPSHOT.local_dir(mock_run.root))


@pytest.fixture(name="dirs", scope="module")
def fixture_dirs(mock_run: MockRun, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Return the snapshot root with the mock as Au5, Au6 and Au24, and the data written for them."""
    base = tmp_path_factory.mktemp("figures")
    dirs = {"root": base / "auriga", "data": base / "data", "figures": base / "figures"}
    for halo in (5, 6, 24):
        link_mock(mock_run, dirs["root"], halo)
    output = invoke(
        "auriga", "render-data", "5", "6", "24", "--root", str(dirs["root"]), "--out-dir", str(dirs["data"]),
        "--pixels", "40", "--sample", "300",
    )  # fmt: skip
    assert output.count("wrote") == 3
    for halo in (5, 24):
        write_json(mock_run.analysis, dirs["data"] / f"{HaloSnapshot(halo).stem}.json")
    return dirs


def test_render_data_writes_maps_and_samples(dirs: dict[str, Path]) -> None:
    """Each halo gets its maps, its star sample and the sample's description."""
    stem = HaloSnapshot(5).stem
    for suffix in ("-maps.npz", "-stars.bin", "-stars.json"):
        assert (dirs["data"] / f"{stem}{suffix}").is_file()
    assert (dirs["data"] / f"{stem}-stars.bin").stat().st_size == 300 * 8


def test_render_draws_portraits(dirs: dict[str, Path], tmp_path: Path) -> None:
    """``render`` draws a JPEG portrait per halo, without a PDF; a second run changes nothing."""
    args = ("auriga", "render", "5", "6", "--data-dir", str(dirs["data"]), "--figures-dir", str(tmp_path))
    output = invoke(*args)
    assert output.count("wrote") == 2
    assert (tmp_path / f"{HaloSnapshot(6).stem}-portrait.jpg").is_file()
    assert not (tmp_path / f"{HaloSnapshot(6).stem}-portrait.pdf").exists()
    assert invoke(*args).count("unchanged") == 2


def test_plot_draws_the_mono_age_figure(dirs: dict[str, Path], tmp_path: Path) -> None:
    """``plot`` draws the mono-age figure from the record of ``analyze``."""
    output = invoke("auriga", "plot", "5", "--data-dir", str(dirs["data"]), "--figures-dir", str(tmp_path))
    assert "wrote" in output
    assert (tmp_path / f"{HaloSnapshot(5).stem}-mono-age.png").is_file()


def test_figure_commands_need_their_data(dirs: dict[str, Path], tmp_path: Path) -> None:
    """Without the data, the commands say which data command writes it."""
    result = run("auriga", "plot", "6", "--data-dir", str(dirs["data"]), "--figures-dir", str(tmp_path))
    assert result.exit_code == 1
    assert "`experiments auriga analyze`" in result.output
    result = run("auriga", "render", "7", "--data-dir", str(dirs["data"]), "--figures-dir", str(tmp_path))
    assert "`experiments auriga render-data`" in result.output
    result = run("auriga", "render-data", "7", "--root", str(dirs["root"]), "--out-dir", str(tmp_path))
    assert "download Au7" in result.output


def test_figures_draws_what_is_cached(dirs: dict[str, Path], tmp_path: Path) -> None:
    """``figures`` draws the halos of Palicio's table that have data, and names the others."""
    output = invoke("figures", "--data-dir", str(dirs["data"]), "--figures-dir", str(tmp_path))
    assert "no analysis for Au2, Au3, Au16, Au19, Au23, Au27" in output
    assert "no maps for Au2, Au3, Au16, Au19, Au23, Au27" in output
    assert (tmp_path / f"{HaloSnapshot(24).stem}-mono-age.png").is_file()
    assert (tmp_path / f"{HaloSnapshot(24).stem}-portrait.jpg").is_file()
