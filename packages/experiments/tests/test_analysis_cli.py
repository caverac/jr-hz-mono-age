"""Tests of ``experiments auriga analyze`` and ``experiments auriga mock-check``."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from conftest import MOCK_SNAPSHOT, MockRun, invoke, run

from experiments.analysis import HaloAnalysis, PreparedHalo
from experiments.auriga import HaloSnapshot
from experiments.commands import auriga_mock_check
from experiments.commands.auriga_analyze import palicio_window, result_name
from experiments.mock import MockGalaxy, MockSizes
from experiments.palicio import PalicioWindow
from experiments.populations import Method


def test_palicio_window_comes_from_their_table_unless_given() -> None:
    """Palicio's Table 1 gives the window; either value can be replaced, and both for a halo not in it."""
    assert palicio_window(19, None, None) == PalicioWindow(15.0, 0.5)
    assert palicio_window(19, 0.8, None) == PalicioWindow(15.0, 0.8)
    assert palicio_window(19, None, 12.0) == PalicioWindow(12.0, 0.5)
    assert palicio_window(5, 1.0, 20.0) == PalicioWindow(20.0, 1.0)
    with pytest.raises(click.ClickException, match="no window for Au5"):
        palicio_window(5, 1.0, None)


def test_result_name() -> None:
    """The record of a snapshot is named after the halo, the level, the suite and the snapshot."""
    assert result_name(HaloSnapshot(24)) == "au24-level4-Original-snap127.json"


def test_analyze_writes_one_record_per_halo(mock_run: MockRun, tmp_path: Path) -> None:
    """``analyze`` measures a halo (the mock, linked as Au5) with a given window and writes its record."""
    root = tmp_path / "auriga"
    halo_dir = HaloSnapshot(5).local_dir(root)
    halo_dir.parent.mkdir(parents=True)
    halo_dir.symlink_to(MOCK_SNAPSHOT.local_dir(mock_run.root))
    out_dir = tmp_path / "results"
    output = invoke(
        "auriga", "analyze", "5", "--root", str(root), "--z-lim", "1", "--l-kpc", "20",
        "--n-bootstrap", "3", "--out-dir", str(out_dir),
    )  # fmt: skip
    assert output.startswith("Au5 level 4 Original snapshot 127")
    record = json.loads((out_dir / result_name(HaloSnapshot(5))).read_text())
    assert record["snapshot"]["halo"] == 5
    assert record["mono_age"]["method"] == {"window": {"l_kpc": 20.0, "z_lim": 1.0}, "k": 2.0, "n_bootstrap": 3}


def test_analyze_needs_the_snapshot(tmp_path: Path) -> None:
    """Without a local copy, ``analyze`` says how to get one."""
    result = run("auriga", "analyze", "24", "--root", str(tmp_path))
    assert result.exit_code == 1
    assert "download Au24 with `experiments auriga download`" in result.output


def test_analyze_needs_a_window(tmp_path: Path) -> None:
    """A halo outside Palicio's table needs both window options."""
    result = run("auriga", "analyze", "5", "--root", str(tmp_path), "--z-lim", "1")
    assert result.exit_code == 1
    assert "pass --z-lim and --l-kpc" in result.output


def test_mock_check_builds_analyzes_and_compares(
    mock_run: MockRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mock-check`` builds the mock with the given sizes, analyzes it with Au24's window, and records the check."""
    built: list[MockSizes] = []

    def build(snapshot: HaloSnapshot, root: Path, sizes: MockSizes) -> MockGalaxy:
        assert (snapshot, root) == (MOCK_SNAPSHOT, tmp_path / "mock")
        built.append(sizes)
        return mock_run.truth

    def analyze(prepared: PreparedHalo, method: Method) -> HaloAnalysis:
        assert prepared is mock_run.prepared
        assert method == Method(PalicioWindow(20.0, 1.0), 2.0, 7)
        return mock_run.analysis

    monkeypatch.setattr(auriga_mock_check, "build_mock", build)
    monkeypatch.setattr(auriga_mock_check, "prepare_halo", lambda snapshot, root: mock_run.prepared)
    monkeypatch.setattr(auriga_mock_check, "analyze_halo", analyze)
    output = invoke(
        "auriga", "mock-check", "--root", str(tmp_path / "mock"), "--out-dir", str(tmp_path / "results"),
        "--n-disk", "11", "--n-halo", "12", "--n-stars", "13", "--n-bootstrap", "7",
    )  # fmt: skip
    assert built == [MockSizes(11, 12, 13)]
    assert "mock check" in output
    assert "spin axis error" in output
    record = json.loads((tmp_path / "results" / "mock-check.json").read_text())
    assert len(record["bins"]) == 10
