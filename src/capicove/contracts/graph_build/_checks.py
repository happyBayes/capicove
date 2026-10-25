"""Private shared validation for closed graph records."""

import math
import re
from typing import TypeVar


T = TypeVar("T")


def text(value: object, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise TypeError("expected a string")
    if not value.strip():
        raise ValueError("expected a nonblank string")


def choice(value: object, choices: frozenset[str]) -> None:
    text(value)
    if value not in choices:
        raise ValueError("unsupported vocabulary value")


def score(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("score must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("score must be finite and within [0, 1]")
    return number


def digest(value: object) -> None:
    text(value)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256")


def values(value: object, cls: type[T], *, required: bool = False) -> tuple[T, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("expected a sequence")
    if any(not isinstance(item, cls) for item in value):
        raise TypeError("unexpected sequence element type")
    result = tuple(value)
    if required and not result:
        raise ValueError("empty sequence is not allowed")
    return result


def refs(value: object, *, required: bool = False) -> tuple[str, ...]:
    result = values(value, str, required=required)
    for item in result:
        text(item)
    if len(set(result)) != len(result):
        raise ValueError("duplicate references")
    return result


def qualifiers(value: object) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    if not isinstance(value, (tuple, list)):
        raise TypeError("expected qualifier pairs")
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("expected qualifier key and value")
        key, item = pair
        choice(key, frozenset({"raw_predicate", "occurrence_ref", "time_text", "condition_text"}))
        text(item)
        result.append((key, item))
    refs([key for key, _ in result])
    return tuple(sorted(result))


def features(value: object) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    if not isinstance(value, (tuple, list)):
        raise TypeError("expected feature pairs")
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("expected feature key and score")
        key, item = pair
        choice(key, frozenset({"name_similarity", "embedding_similarity", "shared_context_score"}))
        result.append((key, score(item)))
    refs([key for key, _ in result])
    return tuple(sorted(result))
