"""Read the JSON records of the data commands back into their dataclasses.

The data commands write a dataclass with :func:`dataclasses.asdict`. :func:`from_dict`
inverts that from the type hints of the dataclass: nested dataclasses, lists, tuples and
dicts of them, and floats, ints and strings. A value of the wrong JSON type is an error,
so a record written by another version of the code fails loudly instead of plotting
something else.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TypeVar, cast, get_args, get_origin, get_type_hints

from experiments.analysis import HaloAnalysis
from experiments.mock import MockCheck

T = TypeVar("T")


def _mapping(value: object) -> Mapping[str, object]:
    """Return *value* as a JSON object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object, found {type(value).__name__}")
    return cast(Mapping[str, object], value)


def _sequence(value: object) -> Sequence[object]:
    """Return *value* as a JSON array."""
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON array, found {type(value).__name__}")
    return value


def _scalar(hint: object, value: object) -> object:
    """Return the float, int or string *value* as the scalar type *hint*."""
    if hint is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if hint is str and isinstance(value, str):
        return value
    raise TypeError(f"expected {hint}, found {type(value).__name__}")


def _convert(hint: object, value: object) -> object:
    """Return the JSON *value* as an instance of the type *hint*."""
    if isinstance(hint, type) and is_dataclass(hint):
        return from_dict(hint, value)
    origin, args = get_origin(hint), get_args(hint)
    if origin is list:
        return [_convert(args[0], item) for item in _sequence(value)]
    if origin is tuple:
        items = _sequence(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in items)
        return tuple(_convert(arg, item) for arg, item in zip(args, items, strict=True))
    if origin is dict:
        return {str(key): _convert(args[1], item) for key, item in _mapping(value).items()}
    return _scalar(hint, value)


def from_dict(cls: type[T], data: object) -> T:
    """Return the dataclass *cls* built from the JSON object *data*."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    mapping, hints = _mapping(data), get_type_hints(cls)
    values = {field.name: _convert(hints[field.name], mapping[field.name]) for field in fields(cls)}
    return cast(Callable[..., T], cls)(**values)


def load_analysis(path: Path) -> HaloAnalysis:
    """Return the analysis record of one halo written by ``experiments auriga analyze``."""
    return from_dict(HaloAnalysis, json.loads(path.read_text()))


def load_mock_check(path: Path) -> MockCheck:
    """Return the record written by ``experiments auriga mock-check``."""
    return from_dict(MockCheck, json.loads(path.read_text()))
