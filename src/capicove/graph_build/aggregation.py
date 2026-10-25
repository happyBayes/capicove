"""Pure occurrence binding and conservative aggregation of immutable claims.

The host supplies already authorized, source-bound mentions and memberships.
``support_groups`` keys come from ``source_occurrence_ref(SourceSpan)``; values
are explicitly established independent event/source groups. Build IDs, hashes,
representations, dates, and changed mention IDs never establish independence.
Unmapped evidence yields one reserved ``unknown-support:`` marker per complete
semantic group. This marker is uncertainty, not proven recurrence: consumers
must exclude it from counts of established independent support. A multi-source
claim with several established groups gets one ``composite-support:`` marker;
it must not be counted as independent of its constituent groups without further
occurrence resolution. These markers are not ordinary independent event IDs.

Assertion identity uses exact semantic fields and sorted unique representation
ranges including block/page lineage. Regenerated evidence, mention, and chunk IDs
do not affect identity; differing source ranges remain separate assertions.
Bindings with equal identity combine evidence and original relationship-mention
references without discarding provenance. Persisted mentions retain the full
source spans, descriptions, and extraction details behind those references.
No raw RelationshipAssertion passed to aggregation is changed or overwritten.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

from capicove.contracts.graph_build import (
    EntityMembership,
    EntityMention,
    RelationshipAggregate,
    RelationshipAssertion,
    RelationshipMention,
    SourceSpan,
)
from capicove.contracts.graph_build.vocabulary import PREDICATES
from capicove.graph_build.codec import fingerprint


_STATUSES = frozenset({"explicit", "inferred", "hypothesized", "disputed"})
_POLARITIES = frozenset({"positive", "negative"})
_RESERVED_SUPPORT_PREFIXES = ("unknown-support:", "composite-support:")
_T = TypeVar("_T")


def source_occurrence_ref(span: SourceSpan) -> str:
    """Identify exact source coordinates, independently of regenerated evidence IDs.

    Use this capability-local reference as a support_groups key. It locates an
    occurrence; it does not establish event independence or authorize evidence.
    Text-unit, mention and evidence reference labels are deliberately excluded.
    """
    if not isinstance(span, SourceSpan):
        raise TypeError("span must be SourceSpan")
    return "source-occurrence:" + fingerprint(
        {
            "representation_ref": span.representation_ref,
            "start_char": span.start_char,
            "end_char": span.end_char,
            "block_id": span.block_id,
            "page_index": span.page_index,
        }
    )


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("reference must be a string")
    if not value.strip():
        raise ValueError("reference must not be blank")
    return value


def _snapshot_mapping(value: Mapping[str, _T], name: str) -> dict[str, _T]:
    """Take one item snapshot; never collapse conflicting custom-mapping rows."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    entries = tuple(value.items())
    result: dict[str, _T] = {}
    for pair in entries:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{name} items must be key/value tuples")
        key, item = pair
        _text(key)
        if key in result:
            raise ValueError(f"duplicate or conflicting {name} key")
        result[key] = item
    return result


def _qualifiers(value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise TypeError("qualifiers must be a tuple")
    result = []
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("qualifiers must contain key/value tuples")
        result.append((_text(pair[0]), _text(pair[1])))
    if len({key for key, _ in result}) != len(result):
        raise ValueError("qualifier keys must be unique")
    return tuple(sorted(result))


def assertion_group_key(
    source: str,
    predicate: str,
    target: str,
    status: str,
    polarity: str,
    attribution: str | None,
    qualifiers: tuple[tuple[str, str], ...],
) -> str:
    """Hash every exact semantic field, sorting qualifiers without text folding."""
    for value in (source, predicate, target, status, polarity):
        _text(value)
    if attribution is not None:
        _text(attribution)
    if predicate not in PREDICATES or status not in _STATUSES or polarity not in _POLARITIES:
        raise ValueError("unsupported claim vocabulary")
    return fingerprint(
        {
            "source": source,
            "predicate": predicate,
            "target": target,
            "status": status,
            "polarity": polarity,
            "attribution": attribution,
            "qualifiers": _qualifiers(qualifiers),
        }
    )


def _group(row: RelationshipAssertion) -> str:
    return assertion_group_key(
        row.source_entity_id,
        row.predicate,
        row.target_entity_id,
        row.epistemic_status,
        row.polarity,
        row.attribution_ref,
        row.qualifiers,
    )


def _relevance(row: RelationshipAssertion) -> tuple[object, ...]:
    return (
        row.source_entity_id,
        row.predicate,
        row.target_entity_id,
        row.attribution_ref,
        _qualifiers(row.qualifiers),
    )


def aggregate_assertions(
    assertions: tuple[RelationshipAssertion, ...],
) -> tuple[RelationshipAggregate, ...]:
    """Reference all rows unchanged; validate explicit counter links globally.

    Counter evidence is permitted for any status by the canonical contract, but is
    never inferred. Targets must have identical directed endpoints, predicate,
    attribution, and the complete sorted qualifier inventory. Status and polarity
    may differ. Unequal or missing time/occurrence qualifiers fail closed; this
    operator cannot establish that different events contradict one another.
    Repeated identical immutable rows deduplicate by ID; conflicting rows sharing
    an ID fail rather than silently overwriting evidence.
    """
    if not isinstance(assertions, tuple) or any(
        not isinstance(row, RelationshipAssertion) for row in assertions
    ):
        raise TypeError("assertions must be a tuple of RelationshipAssertion values")
    by_id: dict[str, RelationshipAssertion] = {}
    for row in assertions:
        if row.assertion_id in by_id and by_id[row.assertion_id] != row:
            raise ValueError("conflicting duplicate assertion ID")
        by_id[row.assertion_id] = row
    groups: dict[str, list[RelationshipAssertion]] = {}
    for row in by_id.values():
        for ref in row.counter_assertion_ids:
            target = by_id.get(ref)
            if target is None or ref == row.assertion_id or _relevance(row) != _relevance(target):
                raise ValueError("counter assertion is missing, self, or semantically irrelevant")
        groups.setdefault(_group(row), []).append(row)
    return tuple(
        RelationshipAggregate(
            key,
            tuple(sorted(row.assertion_id for row in group)),
            tuple(sorted({row.support_group_id for row in group})),
            tuple(sorted({ref for row in group for ref in row.counter_assertion_ids})),
        )
        for key, group in sorted(groups.items())
    )


@dataclass(frozen=True, slots=True)
class _Binding:
    mention: RelationshipMention
    source: str
    target: str
    attribution: str | None
    group_key: str
    assertion_id: str
    occurrence_refs: frozenset[str]
    derivation_ref: str | None

    @property
    def relevance(self) -> tuple[object, ...]:
        return (
            self.source,
            self.mention.predicate,
            self.target,
            self.attribution,
            _qualifiers(self.mention.qualifiers),
        )


def _binding(
    mention: RelationshipMention,
    memberships: dict[str, str],
    entities: dict[str, EntityMention],
    derivation_ref: str | None,
) -> _Binding:
    if mention.source_mention_id not in memberships or mention.target_mention_id not in memberships:
        raise ValueError("relationship endpoint membership is missing")
    attribution = None
    if mention.attribution_ref is not None:
        speaker = entities.get(mention.attribution_ref)
        if (
            speaker is None
            or speaker.entity_type != "PERSON"
            or speaker.mention_id not in memberships
        ):
            raise ValueError("attribution requires a resolved PERSON mention membership")
        attribution = memberships[speaker.mention_id]
    source, target = memberships[mention.source_mention_id], memberships[mention.target_mention_id]
    key = assertion_group_key(
        source,
        mention.predicate,
        target,
        mention.epistemic_status,
        mention.polarity,
        attribution,
        mention.qualifiers,
    )
    coordinates = {
        (span.representation_ref, span.start_char, span.end_char, span.block_id, span.page_index)
        for span in mention.source_spans
    }
    # Optional block/page pairs need an explicit sort key; None and str/int
    # cannot be ordered directly when other source coordinates coincide.
    ordered = sorted(coordinates, key=lambda row: (row[:3], row[3] or "", row[4] or 0))
    assertion_id = "assertion:" + fingerprint(
        {
            "semantics": key,
            "source_occurrences": ordered,
            "derivation_ref": derivation_ref,
        }
    )
    refs = frozenset(source_occurrence_ref(span) for span in mention.source_spans)
    return _Binding(mention, source, target, attribution, key, assertion_id, refs, derivation_ref)


def bind_assertions(
    mentions: tuple[EntityMention | RelationshipMention, ...],
    entity_memberships: tuple[EntityMembership, ...],
    support_groups: Mapping[str, str],
    *,
    counter_mentions: Mapping[str, tuple[str, ...]] | None = None,
    derivation_refs: Mapping[str, str] | None = None,
) -> tuple[RelationshipAssertion, ...]:
    """Bind resolved claims, retaining immutable occurrence evidence and lineage.

    ``mentions`` may include EntityMention values to attest PERSON attribution:
    memberships alone carry no entity type. Unattributed relationships need only
    their endpoint memberships. All mention and membership mention IDs are unique.

    ``support_groups`` uses source_occurrence_ref(span) keys, not evidence refs:
    extraction can share one evidence ref across different source ranges. Both
    caller mappings are snapshotted once before value validation and subsequent
    reads. Conflicting duplicate snapshot keys fail before support reduction.
    Duplicate assertions first union stable occurrence refs; only then is the
    complete occurrence-to-group mapping reduced. Any unmapped coordinate keeps
    the assertion unknown, so partial support cannot be hidden by a copied label.

    ``counter_mentions`` explicitly maps relationship mention IDs to relationship
    mention IDs. Disputed claims require nonempty valid links. Other statuses may
    carry explicitly supplied counter evidence, as permitted by the canonical
    contract. No polarity/date comparison creates a link. Qualifiers must match
    exactly (after ordering); cross-event contradiction requires a richer future
    evidence contract. Links translate after occurrence IDs are computed, so even
    aliases that become self-links fail. Counter links are not part of occurrence
    identity, permitting cycles and additional explicit evidence without changing
    the source claim ID. Duplicate bound claims union evidence and counter refs;
    conflicting established support assignments fail instead of overwriting.
    """
    if not isinstance(mentions, tuple) or any(
        not isinstance(item, (EntityMention, RelationshipMention)) for item in mentions
    ):
        raise TypeError("mentions must be a tuple of entity/relationship mentions")
    if not isinstance(entity_memberships, tuple) or any(
        not isinstance(item, EntityMembership) for item in entity_memberships
    ):
        raise TypeError("entity_memberships must be a tuple of EntityMembership values")
    if len({item.mention_id for item in mentions}) != len(mentions):
        raise ValueError("mention IDs must be unique")
    memberships = {item.mention_id: item.entity_id for item in entity_memberships}
    if len(memberships) != len(entity_memberships):
        raise ValueError("membership mention IDs must be unique")
    support = _snapshot_mapping(support_groups, "support_groups")
    counters = _snapshot_mapping(
        {} if counter_mentions is None else counter_mentions, "counter_mentions"
    )
    derivations = _snapshot_mapping(
        {} if derivation_refs is None else derivation_refs, "derivation_refs"
    )
    for support_group in support.values():
        if _text(support_group).startswith(_RESERVED_SUPPORT_PREFIXES):
            raise ValueError("support group uses a reserved uncertainty prefix")
    for targets in counters.values():
        if not isinstance(targets, tuple):
            raise TypeError("counter targets must be a tuple")
        for target_ref in targets:
            _text(target_ref)
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate counter targets")
    for derivation_ref in derivations.values():
        _text(derivation_ref)
    entities = {item.mention_id: item for item in mentions if isinstance(item, EntityMention)}
    bound = {
        item.mention_id: _binding(item, memberships, entities, derivations.get(item.mention_id))
        for item in mentions
        if isinstance(item, RelationshipMention)
    }
    if not counters.keys() <= bound.keys():
        raise ValueError("counter source mention is missing")
    if not derivations.keys() <= bound.keys():
        raise ValueError("derivation source mention is missing")
    grouped: dict[str, list[_Binding]] = {}
    links: dict[str, set[str]] = {}
    for ref, item in bound.items():
        targets = counters.get(ref, ())
        if item.mention.epistemic_status == "disputed" and not targets:
            raise ValueError("disputed mention requires explicit counter targets")
        resolved = links.setdefault(item.assertion_id, set())
        for target_ref in targets:
            target = bound.get(target_ref)
            if (
                target is None
                or target.assertion_id == item.assertion_id
                or target.relevance != item.relevance
            ):
                raise ValueError("counter target is missing, self, or semantically irrelevant")
            resolved.add(target.assertion_id)
        grouped.setdefault(item.assertion_id, []).append(item)
    result = []
    for assertion_id, group in sorted(grouped.items()):
        item = group[0]
        occurrences = {ref for entry in group for ref in entry.occurrence_refs}
        if not occurrences <= support.keys():
            support_group = "unknown-support:" + item.group_key
        else:
            established = sorted({support[ref] for ref in occurrences})
            support_group = (
                established[0]
                if len(established) == 1
                else "composite-support:" + fingerprint(established)
            )
        evidence = {
            ref
            for entry in group
            for ref in (
                entry.mention.mention_id,
                *(span.evidence_ref for span in entry.mention.source_spans),
            )
        }
        result.append(
            RelationshipAssertion(
                assertion_id,
                item.source,
                item.mention.predicate,
                item.target,
                item.mention.epistemic_status,
                item.mention.polarity,
                item.attribution,
                _qualifiers(item.mention.qualifiers),
                tuple(sorted(evidence)),
                support_group,
                item.derivation_ref,
                tuple(sorted(links[assertion_id])),
            )
        )
    return tuple(result)
