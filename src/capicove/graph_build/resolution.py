"""Pure, conservative snapshot-local grouping of already scoped mention evidence.

Reserved capability-local IdentityKey namespaces ``capicove:subject_ref`` and
``capicove:occurrence_ref`` denote explicit source-bound subject/occurrence refs,
never names, narrator attribution, descriptions, or dates. The caller must bind
them from explicit evidence. They constrain merging even when non-authoritative,
but cannot establish authoritative identity or authorize a match themselves.
All reserved context keys are emitted as non-authoritative Entity keys, even
when input mentions claim authority; immutable source mentions retain that claim.
``domain`` retains Task 11's explicit concept/skill context semantics. Other
namespaces have no implicit exclusivity/cardinality policy: unequal inventories
are not conflicts. Established identity incompatibility arrives as a persisted
hard-conflict reason or distinct decision.

Persisted decisions use mention IDs here (Task 11's Entity-based pair decisions
must be explicitly mapped by its caller). Evidence binding, scope authorization,
and rejection of model-claimed authority belong to the upstream host. This pure
operator performs no evidence reads, model calls, or authority promotion.
"""

from dataclasses import asdict, dataclass, replace

from capicove.contracts.graph_build import (
    Entity,
    EntityMembership,
    EntityMention,
    GraphScope,
    PairDecision,
)
from capicove.contracts.graph_build.vocabulary import DIAGNOSTICS, ENTITY_TYPES, EVENTS
from capicove.graph_build.candidates import normalize_name
from capicove.graph_build.codec import fingerprint
from capicove.graph_build.profiles import ResolutionProfile


_HARD_REASONS = frozenset(
    {
        "cannot_link",
        "conflicting_authoritative_identity",
        "incompatible_entity_type",
        "incompatible_domain_or_subject",
        "conflicting_occurrence",
    }
)
_CONTEXT_NAMESPACES = frozenset({"domain", "capicove:subject_ref", "capicove:occurrence_ref"})


@dataclass(frozen=True, slots=True)
class ResolutionIssue:
    """A deterministic abstention/conflict with the affected mention endpoints."""

    code: str
    mention_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Capability-local immutable grouping, retaining all input decisions."""

    entities: tuple[Entity, ...]
    memberships: tuple[EntityMembership, ...]
    decisions: tuple[PairDecision, ...]
    issues: tuple[ResolutionIssue, ...]
    resolution_fingerprint: str

    @property
    def issue_codes(self) -> tuple[str, ...]:
        """Return the stable distinct codes while issues retain affected values."""
        return tuple(sorted({issue.code for issue in self.issues}))


def can_merge(
    left: frozenset[str],
    right: frozenset[str],
    cannot_links: frozenset[frozenset[str]],
) -> bool:
    """Check every cross-cluster pair against accumulated cannot-links."""
    return not any(frozenset((a, b)) in cannot_links for a in left for b in right)


def _context(mention: EntityMention, namespace: str) -> frozenset[str]:
    return frozenset(key.value for key in mention.identity_keys if key.namespace == namespace)


def _keys(mention: EntityMention) -> frozenset[tuple[str, str]]:
    return frozenset(
        (key.namespace, key.value)
        for key in mention.identity_keys
        if key.authoritative and key.namespace not in _CONTEXT_NAMESPACES
    )


def _compatible(left: EntityMention, right: EntityMention) -> bool:
    if left.entity_type != right.entity_type:
        return False
    namespaces = []
    if left.entity_type in DIAGNOSTICS:
        namespaces.append("capicove:subject_ref")
    if left.entity_type in EVENTS:
        namespaces.append("capicove:occurrence_ref")
    if left.entity_type in {"SKILL", "CONCEPT"}:
        namespaces.append("domain")
    for namespace in namespaces:
        first, second = _context(left, namespace), _context(right, namespace)
        if namespace != "domain" and len(first | second) > 1:
            return False
        if first and second and first.isdisjoint(second):
            return False
    return True


def _match_issue(left: EntityMention, right: EntityMention, pair: PairDecision) -> str | None:
    refs = set(pair.evidence_refs)
    # Both endpoint occurrences must be cited. A reason string plus an arbitrary
    # unrelated citation does not bind a strong decision to these mentions.
    for item in (left, right):
        if not refs.intersection({item.mention_id, *(s.evidence_ref for s in item.source_spans)}):
            return "insufficient_identity_evidence"
    if left.entity_type in DIAGNOSTICS:
        if not all(_context(item, "capicove:subject_ref") for item in (left, right)):
            return "insufficient_domain_or_subject_context"
    if left.entity_type in {"SKILL", "CONCEPT"}:
        if not all(_context(item, "domain") for item in (left, right)):
            return "insufficient_domain_or_subject_context"
    endpoint_refs = {
        ref
        for item in (left, right)
        for ref in (item.mention_id, *(span.evidence_ref for span in item.source_spans))
    }
    # Ordinary endpoint citations establish binding, not separate identity
    # evidence. A cited common explicit occurrence ref can be separate evidence
    # when it is outside these endpoint citations, just like an alias source.
    explicit_refs = refs - endpoint_refs
    if left.entity_type in EVENTS:
        if "explicit_occurrence_identity" in pair.reason_codes and explicit_refs:
            return None
        return "insufficient_occurrence_evidence"
    if "authoritative_identity_key" in pair.reason_codes and _keys(left) & _keys(right):
        return None
    if (
        "explicit_alias_evidence" in pair.reason_codes
        and explicit_refs
        and (
            left.entity_type not in {"SKILL", "CONCEPT"}
            or _context(left, "domain") == _context(right, "domain")
        )
    ):
        return None
    return "insufficient_identity_evidence"


def _display_name(items: tuple[EntityMention, ...]) -> str:
    """Rank raw names by unique source occurrences, core quality, and confidence.

    An occurrence is the sorted set of (representation, start, end) source ranges;
    changing evidence IDs/text units or repeating context cannot add votes. Each
    name gets one vote per occurrence and its best (core, confidence) quality.
    Ties use normalized name then exact raw text, both ascending. No descriptions
    are synthesized; source snippets remain on immutable input mentions.
    """
    occurrences: dict[str, set[tuple[tuple[str, int, int], ...]]] = {}
    quality: dict[str, tuple[bool, float]] = {}
    for item in items:
        occurrence = tuple(
            sorted({(s.representation_ref, s.start_char, s.end_char) for s in item.source_spans})
        )
        occurrences.setdefault(item.name, set()).add(occurrence)
        tier = (item.anchor_role == "core", item.extraction_confidence)
        quality[item.name] = max(quality.get(item.name, tier), tier)
    return min(
        occurrences,
        key=lambda name: (
            -len(occurrences[name]),
            -quality[name][0],
            -quality[name][1],
            normalize_name(name),
            name,
        ),
    )


def _entity(items: tuple[EntityMention, ...], scope: GraphScope, policy: str) -> Entity:
    mention_ids = tuple(item.mention_id for item in items)
    display_name = _display_name(items)
    keys = tuple(
        sorted(
            {
                replace(key, authoritative=False) if key.namespace in _CONTEXT_NAMESPACES else key
                for item in items
                for key in item.identity_keys
            },
            key=lambda key: (key.namespace, key.value, key.authoritative),
        )
    )

    def reference(namespace: str) -> str | None:
        refs = {key.value for key in keys if key.namespace == namespace}
        return next(iter(refs)) if len(refs) == 1 else None

    return Entity(
        fingerprint({"scope": asdict(scope), "mention_ids": mention_ids, "resolution": policy}),
        scope,
        items[0].entity_type,
        display_name,
        tuple(sorted({item.name for item in items} - {display_name})),
        mention_ids,
        keys,
        reference("capicove:occurrence_ref") if items[0].entity_type in EVENTS else None,
        reference("capicove:subject_ref") if items[0].entity_type in DIAGNOSTICS else None,
    )


def resolve_entities(
    mentions: tuple[EntityMention, ...],
    decisions: tuple[PairDecision, ...],
    profile: ResolutionProfile,
    *,
    scope: GraphScope,
) -> ResolutionResult:
    """Greedily merge strong edges in canonical endpoint order within one scope.

    The original three-argument plan omitted a scope source: neither mentions,
    decisions nor ResolutionProfile carry scope. The required keyword explicitly
    binds canonical Entity output without inventing or inferring scope. The host
    must authorize and bind all inputs to it; possession does not grant access.

    Scores never rank identity strength or supply an unimplemented fuzzy policy.
    Stable unordered endpoint order breaks ties among accepted strong rules.
    Every distinct decision and recognized hard reason is installed before any
    match, including contradictory matched/adjudicated input. Decisions retain
    their content with ascending endpoints, sorted set-valued reasons/evidence,
    and features sorted by key; issues explain unused matches. Input values are
    never mutated.
    """
    if not isinstance(scope, GraphScope) or not isinstance(profile, ResolutionProfile):
        raise TypeError("scope and profile must be GraphScope and ResolutionProfile")
    if not isinstance(mentions, tuple) or not isinstance(decisions, tuple):
        raise TypeError("mentions and decisions must be tuples")
    if any(not isinstance(item, EntityMention) for item in mentions):
        raise TypeError("mentions must contain EntityMention values")
    if any(not isinstance(pair, PairDecision) for pair in decisions):
        raise TypeError("decisions must contain PairDecision values")
    if len(mentions) > profile.identity_pool_limit:
        raise ValueError("identity_pool_limit exceeded")
    by_id = {item.mention_id: item for item in mentions}
    if len(by_id) != len(mentions):
        raise ValueError("mention IDs must be unique")
    if any(not _compatible(item, item) for item in mentions):
        raise ValueError("mention contains conflicting explicit subject or occurrence references")
    seen: set[frozenset[str]] = set()
    for pair in decisions:
        endpoints = frozenset((pair.left_id, pair.right_id))
        if not endpoints <= by_id.keys():
            raise ValueError("decision endpoint is missing")
        if endpoints in seen:
            raise ValueError("duplicate decision endpoints")
        seen.add(endpoints)
    canonical = (
        PairDecision(
            min(pair.left_id, pair.right_id),
            max(pair.left_id, pair.right_id),
            pair.decision,
            tuple(sorted(set(pair.reason_codes))),
            tuple(sorted(pair.features)),
            tuple(sorted(set(pair.evidence_refs))),
        )
        for pair in decisions
    )
    ordered = tuple(sorted(canonical, key=lambda pair: (pair.left_id, pair.right_id)))
    cannot_links = frozenset(
        frozenset((pair.left_id, pair.right_id))
        for pair in ordered
        if pair.decision == "distinct" or _HARD_REASONS.intersection(pair.reason_codes)
    )
    clusters = {mention_id: frozenset((mention_id,)) for mention_id in by_id}
    issues = []
    for pair in ordered:
        if pair.decision != "matched":
            continue
        left, right = clusters[pair.left_id], clusters[pair.right_id]
        issue_endpoints = tuple(sorted((pair.left_id, pair.right_id)))
        if left is not right and (
            not can_merge(left, right, cannot_links)
            or not all(_compatible(by_id[a], by_id[b]) for a in left for b in right)
        ):
            issues.append(ResolutionIssue("hard_cluster_conflict", issue_endpoints))
            continue
        issue = _match_issue(by_id[pair.left_id], by_id[pair.right_id], pair)
        if issue is not None:
            issues.append(ResolutionIssue(issue, issue_endpoints))
            continue
        if left is right:
            continue
        merged = left | right
        for mention_id in merged:
            clusters[mention_id] = merged
    policy = fingerprint(
        {
            "profile": asdict(profile),
            "algorithm": "constrained-strong-mention-resolution-v2",
            "ontology": "personal-growth-v1",
            "entity_types": sorted(ENTITY_TYPES),
            "events": sorted(EVENTS),
            "diagnostics": sorted(DIAGNOSTICS),
        }
    )
    entities = tuple(
        _entity(tuple(by_id[item] for item in group), scope, policy)
        for group in sorted({tuple(sorted(cluster)) for cluster in clusters.values()})
    )
    return ResolutionResult(
        entities,
        tuple(
            EntityMembership(entity.entity_id, item)
            for entity in entities
            for item in entity.mention_ids
        ),
        ordered,
        tuple(issues),
        policy,
    )
