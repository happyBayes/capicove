"""Source-owned validation of bounded immutable text payloads."""

import sqlite3
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from capicove.contracts.knowledge import text
from capicove.source.errors import EvidenceUnavailable


def validate_text_payload(value: str) -> str:
    return text(value, 1024 * 1024)


P = ParamSpec("P")
R = TypeVar("R")


def evidence_errors(operation: Callable[P, R]) -> Callable[P, R]:
    @wraps(operation)
    def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return operation(*args, **kwargs)
        except sqlite3.Error as exc:
            raise EvidenceUnavailable("evidence_unavailable") from exc

    return invoke
