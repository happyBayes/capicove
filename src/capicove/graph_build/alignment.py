"""Conservative correspondence to one baseline; active identity stays current-only.

Alignment never merges current clusters or assigns historical IDs to the analysis
view. All qualified competing matches abstain together, without score ordering.
Identical entities shared across snapshots have an implicit exact correspondence;
they reserve both endpoints and cannot be displaced by non-self matches.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from capicove.contracts.graph_build import (
    Entity,
    EntityAlignment,
    GraphScope,
    GraphSnapshotManifest,
    GraphSnapshotRef,
    PairDecision,
    ResolutionMembership,
)
from capicove.contracts.graph_build.vocabulary import DIAGNOSTICS, EVENTS
from capicove.graph_build.codec import manifest_bytes
from capicove.graph_build.ports import AccessGate
from capicove.graph_build.profiles import GraphBuildProfile, graph_build_profile_fingerprint


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
class AlignmentResult:
    """Retained correspondences and an immutable identity view of current rows."""

    alignments: tuple[EntityAlignment, ...]
    resolved_entities: tuple[Entity, ...]
    resolution_memberships: tuple[ResolutionMembership, ...]
    issue_codes: tuple[str, ...]
    baseline_status: str


@dataclass(frozen=True, slots=True)
class LoadedBaseline:
    """Verified reader output, carrying the profile bound by the manifest digest.

    This capability-local value is not an authorization credential. The reader
    must verify entity table integrity and resolution-stage provenance binding
    the resolver fingerprint; manifest bytes alone cannot attest dataset rows.
    """

    manifest: GraphSnapshotManifest
    profile: GraphBuildProfile
    resolver_fingerprint: str
    entities: tuple[Entity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, GraphSnapshotManifest):
            raise TypeError("baseline manifest must be GraphSnapshotManifest")
        if not isinstance(self.profile, GraphBuildProfile):
            raise TypeError("baseline profile must be GraphBuildProfile")
        _digest(self.resolver_fingerprint)
        _entities(self.entities)


class BaselineReader(Protocol):
    """Read one exact snapshot after the host scope access gate.

    Implementations must authorize its full persisted source lineage under current
    policy, reject unavailable sources, and verify every loaded entity row against
    the pinned entity dataset's schema, inventory and content hashes. Read the
    profile and resolver fingerprint from verified stage provenance, never caller
    claims. Return complete entities from the requested immutable snapshot; never
    resolve a moving latest alias. Scope authorization alone is insufficient;
    the reader must fail closed on denied source lineage before returning data.
    Its authorization and immutable read must share a consistent policy/read
    boundary. Catalog implementation belongs to Task 20.
    """

    def read(self, ref: GraphSnapshotRef, scope: GraphScope) -> LoadedBaseline:
        """Return verified immutable data or raise on denial/integrity failure."""
        ...


def align_with_baseline(
    current: tuple[Entity, ...],
    decisions: tuple[PairDecision, ...],
    resolver_fingerprint: str,
    *,
    baseline: GraphSnapshotRef | None,
    scope: GraphScope,
    profile: GraphBuildProfile,
    access_gate: AccessGate,
    reader: BaselineReader,
    schema_version: int = 1,
) -> AlignmentResult:
    """Authorize and read one pinned baseline once, then align the whole build.

    The build host selects the immutable reference once before candidate work and
    passes all persisted decisions here, never calls this per candidate batch.
    Compatibility requires exact schema, ontology, profile and resolver identity.
    No-baseline builds still require an authorized scope. Current entities and
    decisions must already have passed upstream evidence authorization/binding.
    """
    if not isinstance(scope, GraphScope) or not isinstance(profile, GraphBuildProfile):
        raise TypeError("scope and profile must be GraphScope and GraphBuildProfile")
    if baseline is not None and not isinstance(baseline, GraphSnapshotRef):
        raise TypeError("baseline must be an immutable GraphSnapshotRef")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("unsupported alignment schema version")
    _digest(resolver_fingerprint)
    now = _entities(current)
    if any(item.scope != scope for item in now.values()):
        raise ValueError("current entities must match requested scope")
    access_gate.require_scope(scope)
    if baseline is None:
        return align_entities(current, None, decisions, resolver_fingerprint)
    loaded = reader.read(baseline, scope)
    if not isinstance(loaded, LoadedBaseline):
        raise TypeError("reader must return LoadedBaseline")
    manifest = loaded.manifest
    if hashlib.sha256(manifest_bytes(manifest)).hexdigest() != baseline.manifest_sha256:
        raise ValueError("baseline manifest digest mismatch")
    if manifest.scope != scope or any(item.scope != scope for item in loaded.entities):
        raise ValueError("baseline scope mismatch")
    if manifest.completeness != "complete":
        raise ValueError("baseline must be complete")
    if manifest.schema_version != schema_version:
        raise ValueError("baseline schema mismatch")
    if loaded.profile.ontology_version != profile.ontology_version:
        raise ValueError("baseline ontology mismatch")
    if manifest.profile_fingerprint != graph_build_profile_fingerprint(loaded.profile):
        raise ValueError("baseline profile integrity mismatch")
    if manifest.profile_fingerprint != graph_build_profile_fingerprint(profile):
        raise ValueError("baseline profile mismatch")
    if loaded.resolver_fingerprint != resolver_fingerprint:
        raise ValueError("baseline resolver fingerprint mismatch")
    return align_entities(current, loaded.entities, decisions, resolver_fingerprint)


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("resolver fingerprint must be lowercase SHA-256")


def _entities(items: tuple[Entity, ...]) -> dict[str, Entity]:
    if not isinstance(items, tuple) or any(not isinstance(item, Entity) for item in items):
        raise TypeError("entities must be a tuple of Entity values")
    result = {item.entity_id: item for item in items}
    if len(result) != len(items):
        raise ValueError("entity IDs must be unique within each snapshot")
    return result


def _context(item: Entity, namespace: str) -> frozenset[str]:
    values = {key.value for key in item.identity_keys if key.namespace == namespace}
    ref = {
        "capicove:subject_ref": item.subject_ref,
        "capicove:occurrence_ref": item.occurrence_ref,
    }.get(namespace)
    if ref is not None:
        values.add(ref)
    return frozenset(values)


def _match_issue(left: Entity, right: Entity, pair: PairDecision) -> str | None:
    if left.entity_type != right.entity_type or _HARD_REASONS.intersection(pair.reason_codes):
        return "hard_alignment_conflict"
    namespaces = []
    if left.entity_type in DIAGNOSTICS:
        namespaces.append("capicove:subject_ref")
    if left.entity_type in EVENTS:
        namespaces.append("capicove:occurrence_ref")
    if left.entity_type in {"SKILL", "CONCEPT"}:
        namespaces.append("domain")
    for namespace in namespaces:
        first, second = _context(left, namespace), _context(right, namespace)
        if (namespace != "domain" and len(first | second) > 1) or (
            first and second and first.isdisjoint(second)
        ):
            return "hard_alignment_conflict"
        if namespace != "capicove:occurrence_ref" and not (first and second):
            return "insufficient_domain_or_subject_context"
    refs = set(pair.evidence_refs)
    if any(not refs.intersection(item.mention_ids) for item in (left, right)):
        return "insufficient_identity_evidence"
    extra_refs = (
        refs - set(left.mention_ids + right.mention_ids) - {left.entity_id, right.entity_id}
    )
    if left.entity_type in EVENTS:
        if "explicit_occurrence_identity" in pair.reason_codes and extra_refs:
            return None
        return "insufficient_occurrence_evidence"
    keys = [
        {
            (key.namespace, key.value)
            for key in item.identity_keys
            if key.authoritative and key.namespace not in _CONTEXT_NAMESPACES
        }
        for item in (left, right)
    ]
    if "authoritative_identity_key" in pair.reason_codes and keys[0] & keys[1]:
        return None
    if (
        "explicit_alias_evidence" in pair.reason_codes
        and extra_refs
        and (
            left.entity_type not in {"SKILL", "CONCEPT"}
            or _context(left, "domain") == _context(right, "domain")
        )
    ):
        return None
    return "insufficient_identity_evidence"


def align_entities(
    current: tuple[Entity, ...],
    previous: tuple[Entity, ...] | None,
    decisions: tuple[PairDecision, ...],
    resolver_fingerprint: str,
) -> AlignmentResult:
    """Apply persisted entity-ID decisions without importing historical facts.

    Inputs must already be authorized and evidence-bound by the host. Mention
    citations bind both endpoints; separate alias/occurrence evidence or shared
    authoritative keys establish strong identity. Distinct decisions require
    both endpoint mention citations and a recognized assessed hard-conflict
    reason. Scores alone establish neither identity nor non-identity.
    Decisions are canonicalized to current/previous orientation. Every assessed
    pair is retained, with unsafe matches downgraded and explained by issue codes.
    Equal IDs on both sides are allowed only for identical full Entity values.
    """
    _digest(resolver_fingerprint)
    now, old = _entities(current), _entities(previous if previous is not None else ())
    if not isinstance(decisions, tuple) or any(not isinstance(p, PairDecision) for p in decisions):
        raise TypeError("decisions must be a tuple of PairDecision values")
    if len({item.scope for item in (*now.values(), *old.values())}) > 1:
        raise ValueError("all entities must share one scope")
    shared = now.keys() & old.keys()
    if any(now[key] != old[key] for key in shared):
        raise ValueError("shared entity IDs must have unchanged content")
    pairs: dict[tuple[str, str], PairDecision] = {}
    for pair in decisions:
        orientations = [
            (a, b)
            for a, b in ((pair.left_id, pair.right_id), (pair.right_id, pair.left_id))
            if a in now and b in old
        ]
        if len(orientations) != 1:
            raise ValueError("decision requires exactly one current/previous orientation")
        endpoints = orientations[0]
        if endpoints in pairs:
            raise ValueError("duplicate decision endpoints")
        pairs[endpoints] = PairDecision(
            *endpoints,
            pair.decision,
            tuple(sorted(set(pair.reason_codes))),
            tuple(sorted(pair.features)),
            tuple(sorted(set(pair.evidence_refs))),
        )
    issues: set[str] = set()
    assessed: dict[tuple[str, str], str] = {}
    for endpoints, pair in pairs.items():
        status = pair.decision
        if status == "matched":
            issue = _match_issue(now[endpoints[0]], old[endpoints[1]], pair)
            if issue is not None:
                issues.add(issue)
                status = "ambiguous"
        elif status == "distinct" and (
            not _HARD_REASONS.intersection(pair.reason_codes)
            or any(
                not set(pair.evidence_refs).intersection(item.mention_ids)
                for item in (now[endpoints[0]], old[endpoints[1]])
            )
        ):
            issues.add("insufficient_distinct_evidence")
            status = "ambiguous"
        assessed[endpoints] = status
    strong = [pair for pair, status in assessed.items() if status == "matched"]
    current_counts = Counter(a for a, _ in strong)
    previous_counts = Counter(b for _, b in strong)
    for endpoints in strong:
        if (
            current_counts[endpoints[0]] > 1
            or previous_counts[endpoints[1]] > 1
            or endpoints[0] in shared
            or endpoints[1] in shared
        ):
            assessed[endpoints] = "ambiguous"
            issues.add("conflicting_correspondence")
    alignments = [
        EntityAlignment(
            *endpoints, assessed[endpoints], pair.features, pair.evidence_refs, resolver_fingerprint
        )
        for endpoints, pair in pairs.items()
    ]
    alignments.extend(
        EntityAlignment(
            key, key, "matched", (), tuple(sorted(now[key].mention_ids)), resolver_fingerprint
        )
        for key in shared
    )
    ordered = tuple(sorted(current, key=lambda item: item.entity_id))
    return AlignmentResult(
        tuple(
            sorted(alignments, key=lambda item: (item.current_entity_id, item.previous_entity_id))
        ),
        ordered,
        tuple(ResolutionMembership(item.entity_id, item.entity_id) for item in ordered),
        tuple(sorted(issues)),
        "no_baseline" if previous is None else "aligned",
    )
