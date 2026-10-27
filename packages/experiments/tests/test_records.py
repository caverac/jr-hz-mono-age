"""Tests of reading the JSON records back into their dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
from conftest import MockRun

from experiments.commands._report import write_json
from experiments.mock import check_mock
from experiments.records import from_dict, load_analysis, load_mock_check


@dataclass(frozen=True)
class Pair:
    """A dataclass with a fixed-length tuple, an int and a string."""

    values: tuple[float, int]
    count: int
    name: str


def test_analysis_round_trip(mock_run: MockRun, tmp_path: Path) -> None:
    """A record written by ``write_json`` reads back into an equal analysis."""
    path = write_json(mock_run.analysis, tmp_path / "record.json")
    loaded = load_analysis(path)
    assert json.dumps(asdict(loaded)) == json.dumps(asdict(mock_run.analysis))


def test_mock_check_round_trip(mock_run: MockRun, tmp_path: Path) -> None:
    """The record of ``mock-check`` reads back into an equal check."""
    check = check_mock(mock_run.truth, mock_run.prepared, mock_run.analysis)
    assert load_mock_check(write_json(check, tmp_path / "mock-check.json")) == check


def test_from_dict_reads_tuples_ints_and_strings() -> None:
    """Fixed-length tuples are read element by element; ints and strings are kept."""
    assert from_dict(Pair, {"values": [1, 2], "count": 3, "name": "a"}) == Pair((1.0, 2), 3, "a")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "expected a JSON object"),
        ({"values": {"a": 1}, "count": 3, "name": "a"}, "expected a JSON array"),
        ({"values": [1, 2], "count": 3.5, "name": "a"}, "expected <class 'int'>"),
        ({"values": [1, 2], "count": True, "name": "a"}, "expected <class 'int'>"),
        ({"values": ["x", 2], "count": 3, "name": "a"}, "expected <class 'float'>"),
        ({"values": [1, 2], "count": 3, "name": 4}, "expected <class 'str'>"),
    ],
)
def test_from_dict_refuses_the_wrong_json_types(data: object, message: str) -> None:
    """A value of the wrong JSON type is an error, not a silent conversion."""
    with pytest.raises(TypeError, match=message):
        from_dict(Pair, data)


def test_from_dict_needs_a_dataclass() -> None:
    """Only dataclasses can be read."""
    with pytest.raises(TypeError, match="int is not a dataclass"):
        from_dict(int, {})
