"""Deterministic, scoped identity candidates without inferred identity authority."""

import heapq
import math
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations, islice
from typing import TypeVar

from capicove.contracts.graph_build import Entity, GraphScope, PairDecision
from capicove.contracts.graph_build.vocabulary import DIAGNOSTICS, EVENTS
from capicove.graph_build.profiles import ResolutionProfile


_T = TypeVar("_T")


class IdentityStageLimitError(ValueError):
    """Candidate generation exceeded its pool budget; no partial result is complete."""


def _nonblank(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identity context references must be nonblank strings")


def _unit_vectors(vectors: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    normalized = []
    dimensions = len(vectors[0]) if vectors else 0
    for vector in vectors:
        if not isinstance(vector, tuple) or not dimensions or len(vector) != dimensions:
            raise ValueError("identity vector dimensions must be equal and nonempty")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            raise ValueError("identity vector components must be finite numbers")
        scale = max(abs(value) for value in vector)
        if scale == 0:
            raise ValueError("identity vector must be nonzero")
        scaled = tuple(value / scale for value in vector)
        norm = math.hypot(*scaled)
        normalized.append(tuple(value / norm for value in scaled))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class IdentityContext:
    """Scoped explicit pair evidence and vectors in one pinned compatible space.

    Alias/occurrence triples are (left ID, right ID, source evidence reference),
    supplied by the caller from explicit evidence, never inferred from names.
    Vectors and neighbors are keyed by entity ID; vectors share vector_space.
    This capability-local value grants no scope access or identity authority.
    """

    scope: GraphScope
    explicit_aliases: tuple[tuple[str, str, str], ...] = ()
    occurrence_evidence: tuple[tuple[str, str, str], ...] = ()
    vector_space: str | None = None
    vectors: tuple[tuple[str, tuple[float, ...]], ...] = ()
    neighbor_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, GraphScope):
            raise TypeError("identity context scope must be GraphScope")
        for records in (self.explicit_aliases, self.occurrence_evidence):
            if not isinstance(records, tuple):
                raise TypeError("identity evidence must be immutable tuples")
            for record in records:
                if not isinstance(record, tuple) or len(record) != 3:
                    raise ValueError("identity evidence must contain pair and source reference")
                for value in record:
                    _nonblank(value)
                if record[0] == record[1]:
                    raise ValueError("identity evidence pair endpoints must differ")
        for entries in (self.vectors, self.neighbor_ids):
            if not isinstance(entries, tuple):
                raise TypeError("identity context entries must be immutable tuples")
            ids = []
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValueError("identity context entries must contain ID and values")
                _nonblank(entry[0])
                ids.append(entry[0])
            if len(set(ids)) != len(ids):
                raise ValueError("identity context IDs must be unique")
        if self.vector_space is not None:
            _nonblank(self.vector_space)
        if self.vectors and self.vector_space is None:
            raise ValueError("identity vectors require a pinned compatible vector space")
        _unit_vectors(tuple(vector for _, vector in self.vectors))
        for _, neighbors in self.neighbor_ids:
            if not isinstance(neighbors, tuple):
                raise TypeError("neighbor IDs must be immutable tuples")
            for neighbor in neighbors:
                _nonblank(neighbor)


def candidate_batches(pairs: Iterable[_T], profile: ResolutionProfile) -> Iterator[tuple[_T, ...]]:
    """Consume pair work lazily in configured batches of at most 2,000."""
    if not isinstance(profile, ResolutionProfile):
        raise TypeError("profile must be ResolutionProfile")
    iterator = iter(pairs)
    while batch := tuple(islice(iterator, profile.pair_batch_size)):
        yield batch


def normalize_name(name: str) -> str:
    """Derive a candidate key without changing retained source text."""
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def _names(entity: Entity) -> frozenset[str]:
    return frozenset(filter(None, map(normalize_name, (entity.display_name, *entity.aliases))))


def _block_key(entity: Entity) -> tuple[str, frozenset[str], str | None]:
    # The canonical Entity has no domain field. Explicit domain keys retain this
    # context without extending the canonical contract or inventing a domain.
    domains = frozenset(key.value for key in entity.identity_keys if key.namespace == "domain")
    return (
        entity.entity_type,
        domains if entity.entity_type in {"SKILL", "CONCEPT"} else frozenset(),
        entity.subject_ref if entity.entity_type in DIAGNOSTICS else None,
    )


def _keys(entity: Entity, *, authoritative: bool = False) -> frozenset[tuple[str, str]]:
    return frozenset(
        (key.namespace, key.value)
        for key in entity.identity_keys
        if key.namespace != "domain" and (not authoritative or key.authoritative)
    )


def _conflicts(left: Entity, right: Entity) -> tuple[str, ...]:
    reasons = []
    if left.entity_type != right.entity_type:
        reasons.append("incompatible_entity_type")
    else:
        _, left_domains, left_subject = _block_key(left)
        _, right_domains, right_subject = _block_key(right)
        if (left_domains and right_domains and left_domains.isdisjoint(right_domains)) or (
            left_subject is not None and right_subject is not None and left_subject != right_subject
        ):
            reasons.append("incompatible_domain_or_subject")
    if (
        left.entity_type in EVENTS
        and left.occurrence_ref is not None
        and right.occurrence_ref is not None
        and left.occurrence_ref != right.occurrence_ref
    ):
        reasons.append("conflicting_occurrence")
    return tuple(reasons)


def _required_context_present(left: Entity, right: Entity) -> bool:
    if left.entity_type in {"SKILL", "CONCEPT"}:
        return bool(_block_key(left)[1] and _block_key(right)[1])
    if left.entity_type in DIAGNOSTICS:
        return left.subject_ref is not None and right.subject_ref is not None
    return True


def _eligible_pairs(
    indices: list[int], ordered: list[tuple[Entity, tuple[float, ...]]]
) -> Iterator[tuple[int, int]]:
    """Yield domain-compatible unordered pairs without a pairwise matrix.

    Domain postings allow overlapping inventories to meet across all lanes.
    Each row deduplicates its neighbors before yielding; unknown domains can
    retrieve candidates but cannot authorize an automatic identity match.
    """
    if not indices or ordered[indices[0]][0].entity_type not in {"SKILL", "CONCEPT"}:
        yield from combinations(indices, 2)
        return
    postings: dict[str, list[int]] = {}
    domains_by_index: dict[int, frozenset[str]] = {}
    unknown = []
    for index in indices:
        domains = _block_key(ordered[index][0])[1]
        domains_by_index[index] = domains
        if not domains:
            unknown.append(index)
        for domain in domains:
            postings.setdefault(domain, []).append(index)
    for index in indices:
        domains = domains_by_index[index]
        neighbors = set(unknown) if domains else set(indices)
        for domain in domains:
            neighbors.update(postings[domain])
        for neighbor in sorted(neighbors):
            if neighbor > index:
                yield index, neighbor


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


def identity_candidates(
    entities: tuple[Entity, ...],
    vectors: tuple[tuple[float, ...], ...],
    profile: ResolutionProfile,
) -> tuple[tuple[str, str], ...]:
    """Return canonical pairs from keys, names/aliases, and exact cosine top-k.

    Inputs must be aligned, unique, and in one scope. Explicit disjoint domains
    block concepts/skills; overlapping or missing domains retain candidates.
    Missing required context cannot authorize a match in pair_features.
    Comparisons use pair blocks and per-entity heaps, never an N by N matrix.
    All candidate pairs are retained, so dense exact-key/name groups can produce
    quadratic output even though temporary comparison work remains bounded.
    """
    if not isinstance(profile, ResolutionProfile):
        raise TypeError("profile must be ResolutionProfile")
    if not isinstance(entities, tuple) or not isinstance(vectors, tuple):
        raise TypeError("entities and aligned vectors must be tuples")
    if len(entities) > profile.identity_pool_limit:
        raise IdentityStageLimitError("identity_pool_limit exceeded")
    if len(entities) != len(vectors):
        raise ValueError("entity and vector lengths must match")
    if any(not isinstance(item, Entity) for item in entities):
        raise TypeError("identity pool must contain Entity values")
    if len({item.entity_id for item in entities}) != len(entities):
        raise ValueError("entity IDs must be unique")
    if len({item.scope for item in entities}) > 1:
        raise ValueError("identity pool must have one consistent scope")
    normalized = _unit_vectors(vectors)
    ordered = sorted(zip(entities, normalized, strict=True), key=lambda item: item[0].entity_id)
    groups: dict[tuple[str, str | None], list[int]] = {}
    for index, (item, _) in enumerate(ordered):
        entity_type, _, subject = _block_key(item)
        groups.setdefault((entity_type, subject), []).append(index)
    pairs: set[tuple[str, str]] = set()
    for indices in groups.values():
        for key_function in (_keys, _names):
            buckets: dict[object, list[int]] = {}
            for index in indices:
                for key in key_function(ordered[index][0]):
                    buckets.setdefault(key, []).append(index)
            for bucket in buckets.values():
                for batch in candidate_batches(_eligible_pairs(bucket, ordered), profile):
                    for left_index, right_index in batch:
                        left, right = ordered[left_index][0], ordered[right_index][0]
                        if not _conflicts(left, right):
                            pairs.add((left.entity_id, right.entity_id))
        heaps: dict[int, list[tuple[float, int]]] = {index: [] for index in indices}
        for batch in candidate_batches(_eligible_pairs(indices, ordered), profile):
            for left_index, right_index in batch:
                left, left_vector = ordered[left_index]
                right, right_vector = ordered[right_index]
                if _conflicts(left, right):
                    continue
                score = _cosine(left_vector, right_vector)
                for index, neighbor in ((left_index, right_index), (right_index, left_index)):
                    heap = heaps[index]
                    ranked = (score, -neighbor)
                    if len(heap) < profile.top_k:
                        heapq.heappush(heap, ranked)
                    elif ranked > heap[0]:
                        heapq.heapreplace(heap, ranked)
        for index, heap in heaps.items():
            for _, negative_neighbor in heap:
                first, second = sorted((index, -negative_neighbor))
                pairs.add((ordered[first][0].entity_id, ordered[second][0].entity_id))
    return tuple(sorted(pairs))


def cross_identity_candidates(
    current: tuple[Entity, ...],
    current_vectors: tuple[tuple[float, ...], ...],
    baseline: tuple[Entity, ...],
    baseline_vectors: tuple[tuple[float, ...], ...],
    profile: ResolutionProfile,
) -> tuple[tuple[str, str], ...]:
    """Recall deterministic current-to-baseline identity candidates.

    Exact name, alias, and identity-key matches are retained independently of
    embedding rank. Embedding top-k is computed in both bipartite directions
    and unioned, so same-snapshot neighbors cannot consume recall and either
    side's nearest neighbors survive. Pair work is streamed in bounded batches;
    candidate absence has no identity-decision meaning.
    """
    if not isinstance(profile, ResolutionProfile):
        raise TypeError("profile must be ResolutionProfile")
    if not all(
        isinstance(items, tuple) for items in (current, current_vectors, baseline, baseline_vectors)
    ):
        raise TypeError("entities and aligned vectors must be tuples")
    if len(current) + len(baseline) > profile.identity_pool_limit:
        raise IdentityStageLimitError("identity_pool_limit exceeded")
    if len(current) != len(current_vectors) or len(baseline) != len(baseline_vectors):
        raise ValueError("entity and vector lengths must match")
    entities = current + baseline
    if any(not isinstance(item, Entity) for item in entities):
        raise TypeError("identity pools must contain Entity values")
    if len({item.entity_id for item in entities}) != len(entities):
        raise ValueError("entity IDs must be unique across current and baseline pools")
    if len({item.scope for item in entities}) > 1:
        raise ValueError("identity pools must have one consistent scope")

    normalized = _unit_vectors(current_vectors + baseline_vectors)
    current_count = len(current)
    current_ordered = sorted(
        zip(current, normalized[:current_count], strict=True), key=lambda item: item[0].entity_id
    )
    baseline_ordered = sorted(
        zip(baseline, normalized[current_count:], strict=True), key=lambda item: item[0].entity_id
    )
    if not current_ordered or not baseline_ordered:
        return ()

    def eligible_pairs() -> Iterator[tuple[int, int]]:
        for current_index, (current_entity, _) in enumerate(current_ordered):
            for baseline_index, (baseline_entity, _) in enumerate(baseline_ordered):
                if not _conflicts(current_entity, baseline_entity):
                    yield current_index, baseline_index

    pairs: set[tuple[str, str]] = set()
    current_heaps: dict[int, list[tuple[float, int]]] = {
        index: [] for index in range(len(current_ordered))
    }
    baseline_heaps: dict[int, list[tuple[float, int]]] = {
        index: [] for index in range(len(baseline_ordered))
    }
    for batch in candidate_batches(eligible_pairs(), profile):
        for current_index, baseline_index in batch:
            current_entity, current_vector = current_ordered[current_index]
            baseline_entity, baseline_vector = baseline_ordered[baseline_index]
            pair = (current_entity.entity_id, baseline_entity.entity_id)
            if _names(current_entity) & _names(baseline_entity) or _keys(current_entity) & _keys(
                baseline_entity
            ):
                pairs.add(pair)
            score = _cosine(current_vector, baseline_vector)
            for heap, ranked in (
                (current_heaps[current_index], (score, -baseline_index)),
                (baseline_heaps[baseline_index], (score, -current_index)),
            ):
                if len(heap) < profile.top_k:
                    heapq.heappush(heap, ranked)
                elif ranked > heap[0]:
                    heapq.heapreplace(heap, ranked)
    for current_index, heap in current_heaps.items():
        for _, negative_baseline_index in heap:
            pairs.add(
                (
                    current_ordered[current_index][0].entity_id,
                    baseline_ordered[-negative_baseline_index][0].entity_id,
                )
            )
    for baseline_index, heap in baseline_heaps.items():
        for _, negative_current_index in heap:
            pairs.add(
                (
                    current_ordered[-negative_current_index][0].entity_id,
                    baseline_ordered[baseline_index][0].entity_id,
                )
            )
    return tuple(sorted(pairs))


def pair_features(left: Entity, right: Entity, context: IdentityContext) -> PairDecision:
    """Assess evidence independently; fuzzy scores alone always abstain.

    Explicit occurrence identity is required for events even when other strong
    keys or explicit aliases agree. Candidate absence and low cosine never imply
    distinct identity. Qualification/calibration and cluster merging are separate.
    """
    if not isinstance(left, Entity) or not isinstance(right, Entity):
        raise TypeError("pair endpoints must be Entity values")
    if not isinstance(context, IdentityContext):
        raise TypeError("context must be IdentityContext")
    if left.scope != context.scope or right.scope != context.scope:
        raise ValueError("pair and context scope must agree")
    if left.entity_id == right.entity_id:
        raise ValueError("pair endpoints must differ")
    if right.entity_id < left.entity_id:
        left, right = right, left
    conflicts = _conflicts(left, right)
    if conflicts or not _required_context_present(left, right):
        return PairDecision(
            left.entity_id,
            right.entity_id,
            "distinct" if conflicts else "ambiguous",
            conflicts or ("insufficient_domain_or_subject_context",),
            (),
            tuple(sorted(set(left.mention_ids + right.mention_ids))),
        )
    endpoints = {left.entity_id, right.entity_id}
    alias_refs = {ref for a, b, ref in context.explicit_aliases if {a, b} == endpoints}
    occurrence_refs = {ref for a, b, ref in context.occurrence_evidence if {a, b} == endpoints}
    same_occurrence = (
        left.occurrence_ref is not None and left.occurrence_ref == right.occurrence_ref
    )
    strong_match = bool(_keys(left, authoritative=True) & _keys(right, authoritative=True))
    scores = {"name_similarity": float(bool(_names(left) & _names(right)))}
    vectors = dict(context.vectors)
    if left.entity_id in vectors and right.entity_id in vectors:
        unit_left, unit_right = _unit_vectors((vectors[left.entity_id], vectors[right.entity_id]))
        # Canonical feature scores are bounded to [0, 1]. This affine cosine is
        # an uncalibrated similarity, never an identity probability.
        scores["embedding_similarity"] = (1.0 + _cosine(unit_left, unit_right)) / 2.0
    neighbors = dict(context.neighbor_ids)
    if left.entity_id in neighbors and right.entity_id in neighbors:
        left_neighbors, right_neighbors = (
            set(neighbors[left.entity_id]),
            set(neighbors[right.entity_id]),
        )
        union = left_neighbors | right_neighbors
        scores["shared_context_score"] = (
            len(left_neighbors & right_neighbors) / len(union) if union else 0.0
        )
    evidence = set(left.mention_ids + right.mention_ids) | alias_refs | occurrence_refs
    if same_occurrence and left.occurrence_ref is not None:
        evidence.add(left.occurrence_ref)
    if left.entity_type in EVENTS:
        decision = "matched" if same_occurrence or occurrence_refs else "ambiguous"
        reasons = (
            ("explicit_occurrence_identity",)
            if decision == "matched"
            else ("insufficient_occurrence_evidence",)
        )
    elif strong_match:
        decision = "matched"
        reasons = ("authoritative_identity_key",)
    elif _block_key(left) != _block_key(right):
        decision, reasons = "ambiguous", ("incomplete_domain_context",)
    elif alias_refs:
        decision, reasons = "matched", ("explicit_alias_evidence",)
    else:
        decision, reasons = "ambiguous", ("unqualified_identity_evidence",)
    support = []
    if _keys(left) & _keys(right):
        support.append("shared_identity_key")
    if scores["name_similarity"]:
        support.append("normalized_name_or_alias")
    if alias_refs:
        support.append("explicit_alias_evidence")
    if same_occurrence or occurrence_refs:
        support.append("explicit_occurrence_identity")
    if "embedding_similarity" in scores:
        support.append("uncalibrated_affine_cosine")
    return PairDecision(
        left.entity_id,
        right.entity_id,
        decision,
        tuple(sorted(set(reasons) | set(support))),
        tuple(sorted(scores.items())),
        tuple(sorted(evidence)),
    )
