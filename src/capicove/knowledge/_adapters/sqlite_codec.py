"""Canonical JSON encoding for the SQLite ledger."""

import json
import sqlite3
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from capicove.knowledge.errors import MemoryError


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


P = ParamSpec("P")
R = TypeVar("R")


def backend_errors(operation: Callable[P, R]) -> Callable[P, R]:
    @wraps(operation)
    def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return operation(*args, **kwargs)
        except sqlite3.Error as exc:
            raise MemoryError("memory_backend_unavailable") from exc

    return invoke
