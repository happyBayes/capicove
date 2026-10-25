"""Order-independent world-time clipping and unresolved conflict classification."""

from dataclasses import dataclass

from capicove.contracts.knowledge import Authority, EntityRef, TimeInterval, TypedLiteral
from capicove.knowledge.temporal import later_start, overlaps


@dataclass(frozen=True, slots=True)
class Assertion:
    fact_id: str
    object: EntityRef | TypedLiteral
    validity: TimeInterval
    authority: Authority


@dataclass(frozen=True, slots=True)
class Resolution:
    validity: TimeInterval
    state: str


def resolve_slot(
    assertions: tuple[Assertion, ...], *, cardinality: str, temporal_kind: str
) -> dict[str, Resolution]:
    if cardinality not in {"one", "many"} or temporal_kind not in {"static", "interval"}:
        raise ValueError("invalid predicate policy")
    if temporal_kind == "static" and any(a.validity != TimeInterval() for a in assertions):
        raise ValueError("static predicate cannot have validity bounds")
    result = {a.fact_id: Resolution(a.validity, "accepted") for a in assertions}
    if cardinality == "many":
        return result
    for a in assertions:
        # Any later explicit world-time change clips the older open interval.
        later = [
            b.validity.start
            for b in assertions
            if b.object != a.object
            and b.validity.start is not None
            and later_start(b.validity.start, a.validity.start)
        ]
        end = a.validity.end
        if temporal_kind == "interval" and later:
            bound = min(later)
            end = bound if end is None else min(end, bound)
        result[a.fact_id] = Resolution(TimeInterval(a.validity.start, end), "accepted")
    for a in assertions:
        interval = result[a.fact_id].validity
        peers = [
            b
            for b in assertions
            if b.fact_id != a.fact_id
            and b.object != a.object
            and overlaps(interval, result[b.fact_id].validity)
        ]
        highest = max((b.authority for b in peers), default=a.authority)
        if highest > a.authority:
            state = "rejected"
        elif any(b.authority == a.authority for b in peers):
            state = "disputed"
        else:
            state = "accepted"
        result[a.fact_id] = Resolution(interval, state)
    return result
