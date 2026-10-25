"""Closed temporal values and deterministic conflict semantics."""

from datetime import UTC, datetime

import pytest

from capicove.contracts.knowledge import Authority, TimeInterval, TypedLiteral
from capicove.knowledge.conflicts import Assertion, resolve_slot


T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)


def test_literal_types_are_distinct_and_canonical() -> None:
    assert TypedLiteral("boolean", True) != TypedLiteral("int64", 1)
    assert TypedLiteral("decimal", "1.00").value == "1"
    for kind, value in (
        ("int64", True),
        ("string", {}),
        ("int64", 2**63),
        ("datetime", "2026-01-01"),
    ):
        with pytest.raises(ValueError):
            TypedLiteral(kind, value)  # type: ignore[arg-type]


def test_utc_intervals_reject_local_offsets() -> None:
    with pytest.raises(ValueError):
        TimeInterval(datetime.fromisoformat("2026-01-01T00:00:00+08:00"))
    assert TimeInterval(T0, T1).contains(T0)
    assert not TimeInterval(T0, T1).contains(T1)


def test_ordered_backfill_closes_old_interval_and_tie_is_disputed() -> None:
    a = Assertion("a", TypedLiteral("string", "Paris"), TimeInterval(T0), Authority.USER)
    b = Assertion("b", TypedLiteral("string", "London"), TimeInterval(T1), Authority.USER)
    resolved = resolve_slot((b, a), cardinality="one", temporal_kind="interval")
    assert resolved["a"].validity == TimeInterval(T0, T1)
    assert resolved["b"].state == "accepted"
    c = Assertion("c", TypedLiteral("string", "Berlin"), TimeInterval(T1), Authority.USER)
    resolved = resolve_slot((a, b, c), cardinality="one", temporal_kind="interval")
    assert resolved["b"].state == resolved["c"].state == "disputed"
    assert all(
        v.state == "accepted"
        for v in resolve_slot((a, b, c), cardinality="many", temporal_kind="interval").values()
    )


def test_memory_hit_rejects_nonfinite_score_and_mutable_citations() -> None:
    from capicove.contracts.evidence import EvidenceRef, EvidenceSpan
    from capicove.contracts.memory import MemoryHit

    span = EvidenceSpan(EvidenceRef("a" * 64), 0, 1)
    with pytest.raises(ValueError):
        MemoryHit("x", "semantic", float("nan"), "x", (span,), TimeInterval(), TimeInterval())
    with pytest.raises(ValueError):
        MemoryHit("x", "semantic", 0.5, "x", [span], TimeInterval(), TimeInterval())  # type: ignore[arg-type]
