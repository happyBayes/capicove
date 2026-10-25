"""Immutable source-only association projection, never a claim of verified truth.

Direction and predicate are collapsed only in this view. Original assertions and
their evidence remain untouched, including excluded owner hubs and self-loops.
No graph provider is loaded by projection.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from capicove.contracts.graph_build import Community, Entity, RelationshipAssertion
from capicove.graph_build.codec import fingerprint
from capicove.graph_build.community_registry import CommunityPartitionerRegistry
from capicove.graph_build.ports import CommunityPartitioner, CommunityRuntimeIdentity
from capicove.graph_build.profiles import CommunityProfile


_GENERATED_PREFIXES = (
    "report:",
    "finding:",
    "diagnostic-finding:",
    "generated:",
    "generated_report:",
    "generated-report:",
    "community-report:",
    "hypothesis:",
)
_UNCERTAIN_SUPPORT_PREFIXES = ("unknown-support:", "composite-support:")


def _text(value: object) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("projection references must be nonblank strings")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("projection references must be valid UTF-8") from None


def _sorted_refs(value: tuple[str, ...], *, required: bool = False) -> None:
    if type(value) is not tuple:
        raise TypeError("projection references must be immutable tuples")
    for ref in value:
        _text(ref)
    if (required and not value) or value != tuple(sorted(set(value))):
        raise ValueError("projection references must be unique and sorted")


def _count(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("projection counts must be nonnegative integers")


@dataclass(frozen=True, slots=True)
class WeightedEdge:
    """One undirected association with sorted assertion and established support IDs."""

    source_id: str
    target_id: str
    weight: float
    assertion_ids: tuple[str, ...]
    support_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.source_id)
        _text(self.target_id)
        if self.source_id >= self.target_id:
            raise ValueError("edge endpoints must be canonical and distinct")
        if type(self.weight) not in (int, float):
            raise TypeError("edge weight must be numeric")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("edge weight must be finite and positive")
        _sorted_refs(self.assertion_ids, required=True)
        _sorted_refs(self.support_group_ids, required=True)
        if len(self.support_group_ids) > len(self.assertion_ids):
            raise ValueError("edge support groups cannot outnumber assertions")
        if any(ref.startswith(_UNCERTAIN_SUPPORT_PREFIXES) for ref in self.support_group_ids):
            raise ValueError("edge support groups must be established")


@dataclass(frozen=True, slots=True)
class ProjectionDiagnostics:
    """Disjoint assertion counts; each exclusion receives its first matching reason.

    Precedence is epistemic status, polarity, generated source, self-loop, known
    owner participation, then uncertain independent support. Only nonzero reason
    counts are stored, sorted lexicographically. Invalid inventories raise rather
    than appearing as exclusions.
    """

    total_assertions: int
    included_assertions: int
    excluded_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _count(self.total_assertions)
        _count(self.included_assertions)
        if type(self.excluded_counts) is not tuple:
            raise TypeError("exclusion counts must be an immutable tuple")
        for item in self.excluded_counts:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("exclusion entries must be immutable reason/count pairs")
            reason, count = item
            _text(reason)
            _count(count)
            if count == 0:
                raise ValueError("exclusion counts must be positive")
        _sorted_refs(tuple(reason for reason, _ in self.excluded_counts))
        if self.total_assertions != self.included_assertions + sum(
            count for _, count in self.excluded_counts
        ):
            raise ValueError("projection diagnostics must account for every assertion")


@dataclass(frozen=True, slots=True)
class WeightedGraph:
    """Capability-local projection retaining isolated nodes and replay identity."""

    node_ids: tuple[str, ...]
    edges: tuple[WeightedEdge, ...]
    diagnostics: ProjectionDiagnostics
    projection_fingerprint: str

    def __post_init__(self) -> None:
        _sorted_refs(self.node_ids)
        if type(self.edges) is not tuple or any(
            type(edge) is not WeightedEdge for edge in self.edges
        ):
            raise TypeError("edges must be an immutable tuple of WeightedEdge values")
        pairs = tuple((edge.source_id, edge.target_id) for edge in self.edges)
        if pairs != tuple(sorted(set(pairs))):
            raise ValueError("edges must be unique and sorted")
        nodes = set(self.node_ids)
        if any(source not in nodes or target not in nodes for source, target in pairs):
            raise ValueError("edge endpoints must exist in the graph")
        if type(self.diagnostics) is not ProjectionDiagnostics:
            raise TypeError("diagnostics must be ProjectionDiagnostics")
        assertion_ids = tuple(ref for edge in self.edges for ref in edge.assertion_ids)
        if len(set(assertion_ids)) != len(assertion_ids):
            raise ValueError("an assertion cannot belong to multiple edges")
        if self.diagnostics.included_assertions != len(assertion_ids):
            raise ValueError("included assertion count must match the edges")
        if (
            not isinstance(self.projection_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.projection_fingerprint) is None
        ):
            raise ValueError("projection fingerprint must be a lowercase SHA-256 digest")


def edge_weight(independent_support_count: int, cap: int = 20) -> float:
    """Return bounded logarithmic support weight for strict integer inputs.

    Zero has mathematical weight one, but projection excludes assertions without
    established independent support before creating any edge.
    """
    if type(independent_support_count) is not int or type(cap) is not int:
        raise TypeError("support count and cap must be integers")
    if independent_support_count < 0 or cap < 1:
        raise ValueError("support count must be nonnegative and cap positive")
    return 1.0 + math.log1p(min(independent_support_count, cap))


def _validate_assertion_text(row: RelationshipAssertion) -> None:
    """Validate source text even when the assertion will be excluded from edges."""
    for value in (
        row.assertion_id,
        row.source_entity_id,
        row.target_entity_id,
        row.predicate,
        row.epistemic_status,
        row.polarity,
        row.support_group_id,
        *row.evidence_refs,
        *row.counter_assertion_ids,
    ):
        _text(value)
    for optional_ref in (row.attribution_ref, row.derivation_ref):
        if optional_ref is not None:
            _text(optional_ref)
    for key, value in row.qualifiers:
        _text(key)
        _text(value)


def _exclusion(row: RelationshipAssertion, owner_entity_id: str | None) -> str | None:
    if row.epistemic_status != "explicit":
        return "not_explicit"
    if row.polarity != "positive":
        return "not_positive"
    if row.derivation_ref is not None or any(
        ref.startswith(_GENERATED_PREFIXES) for ref in (row.assertion_id, *row.evidence_refs)
    ):
        return "generated_source"
    if row.source_entity_id == row.target_entity_id:
        return "self_loop"
    if row.predicate == "PARTICIPATED_IN" and owner_entity_id in (
        row.source_entity_id,
        row.target_entity_id,
    ):
        return "owner_participation"
    if row.support_group_id.startswith(_UNCERTAIN_SUPPORT_PREFIXES):
        return "unknown_independent_support"
    return None


def community_projection(
    assertions: Iterable[RelationshipAssertion],
    entities: Iterable[Entity],
    owner_entity_id: str | None,
    profile: CommunityProfile,
) -> WeightedGraph:
    """Project explicit source associations from one current, same-scope inventory.

    Assertions have no separate scope field: both endpoints must belong to the
    supplied entity inventory. The owner is explicit and must resolve to a PERSON.
    Unknown/composite support markers cannot establish independent observations.
    """
    if type(profile) is not CommunityProfile:
        raise TypeError("profile must be CommunityProfile")
    rows, nodes = tuple(assertions), tuple(entities)
    if any(type(row) is not RelationshipAssertion for row in rows):
        raise TypeError("assertions must contain only RelationshipAssertion values")
    if any(type(node) is not Entity for node in nodes):
        raise TypeError("entities must contain only Entity values")
    for row in rows:
        _validate_assertion_text(row)
    by_id = {node.entity_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate entity IDs")
    if len({row.assertion_id for row in rows}) != len(rows):
        raise ValueError("duplicate assertion IDs")
    if len({node.scope for node in nodes}) > 1:
        raise ValueError("entity inventory must have one scope")
    if owner_entity_id is not None and (
        not isinstance(owner_entity_id, str)
        or owner_entity_id not in by_id
        or by_id[owner_entity_id].entity_type != "PERSON"
    ):
        raise ValueError("owner must identify an existing PERSON")
    if any(row.source_entity_id not in by_id or row.target_entity_id not in by_id for row in rows):
        raise ValueError("assertion endpoints must exist in the entity inventory")

    grouped: dict[tuple[str, str], list[RelationshipAssertion]] = {}
    exclusions: list[tuple[str, str]] = []
    included: list[RelationshipAssertion] = []
    for row in sorted(rows, key=lambda item: item.assertion_id):
        reason = _exclusion(row, owner_entity_id)
        if reason is not None:
            exclusions.append((row.assertion_id, reason))
            continue
        source, target = sorted((row.source_entity_id, row.target_entity_id))
        grouped.setdefault((source, target), []).append(row)
        included.append(row)

    edges = []
    for (source, target), members in sorted(grouped.items()):
        supports = tuple(sorted({row.support_group_id for row in members}))
        edges.append(
            WeightedEdge(
                source,
                target,
                edge_weight(len(supports), profile.support_cap),
                tuple(row.assertion_id for row in members),
                supports,
            )
        )
    diagnostics = ProjectionDiagnostics(
        len(rows),
        len(included),
        tuple(
            sorted(
                Counter(reason for _, reason in exclusions).items(),
            )
        ),
    )
    node_ids = tuple(sorted(by_id))
    # Tuple-valued citation and qualifier collections are sets for replay identity.
    included_payloads = []
    for row in included:
        payload = asdict(row)
        for name in ("evidence_refs", "counter_assertion_ids", "qualifiers"):
            payload[name] = sorted(payload[name])
        included_payloads.append(payload)
    identity = fingerprint(
        {
            "projection_version": 1,
            "scope": asdict(nodes[0].scope) if nodes else None,
            "node_ids": node_ids,
            "edges": [asdict(edge) for edge in edges],
            "assertions": included_payloads,
            "profile": asdict(profile),
            "owner_entity_id": owner_entity_id,
            "diagnostics": asdict(diagnostics),
            "exclusions": exclusions,
        }
    )
    return WeightedGraph(node_ids, tuple(edges), diagnostics, identity)


@dataclass(frozen=True, slots=True)
class CommunityExecutionStep:
    """One preorder visit, including the exact ordered induced graph and output."""

    path: tuple[int, ...]
    graph: WeightedGraph
    partition: tuple[tuple[str, ...], ...] | None

    def __post_init__(self) -> None:
        if type(self.path) is not tuple or not self.path:
            raise TypeError("community path must be a nonempty immutable tuple")
        for index in self.path:
            _count(index)
        if type(self.graph) is not WeightedGraph:
            raise TypeError("execution graph must be immutable WeightedGraph")
        if self.partition is not None:
            if type(self.partition) is not tuple:
                raise TypeError("recorded partition must be immutable")
            for members in self.partition:
                _sorted_refs(members, required=True)
            if self.partition != tuple(sorted(self.partition)):
                raise ValueError("recorded partition must be sorted")
            _validate_partition(tuple(frozenset(members) for members in self.partition), self.graph)


@dataclass(frozen=True, slots=True)
class CommunityExecutionRecord:
    """Replay evidence for structural associations, never a claim of source truth.

    Graphs retain exact root/recursive node and weighted-edge iteration order.
    The profile pins all partition parameters and hierarchy limits. Runtime
    configuration declares backend settings; replay is only qualified within
    this exact runtime/version/configuration, not across library versions.
    """

    algorithm_id: str
    algorithm_version: str
    runtime_identity: CommunityRuntimeIdentity
    profile: CommunityProfile
    graph: WeightedGraph
    steps: tuple[CommunityExecutionStep, ...]
    communities: tuple[Community, ...]
    hierarchy_version: int = 1

    def __post_init__(self) -> None:
        _text(self.algorithm_id)
        _text(self.algorithm_version)
        if type(self.runtime_identity) is not CommunityRuntimeIdentity:
            raise TypeError("runtime identity must be immutable")
        if type(self.profile) is not CommunityProfile or type(self.graph) is not WeightedGraph:
            raise TypeError("execution requires immutable graph and community profile")
        for values, cls in ((self.steps, CommunityExecutionStep), (self.communities, Community)):
            if type(values) is not tuple or any(type(value) is not cls for value in values):
                raise TypeError("execution values must be immutable typed tuples")
        if type(self.hierarchy_version) is not int or self.hierarchy_version != 1:
            raise ValueError("unsupported hierarchy version")
        _validate_execution(self)

    @property
    def projection_fingerprint(self) -> str:
        """Return the source projection identity retained by every induced graph."""
        return self.graph.projection_fingerprint

    @property
    def execution_fingerprint(self) -> str:
        """Bind configuration, ordered execution inputs, and the complete result."""
        return fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class CommunityHierarchyResult:
    """Return communities and their immutable execution record atomically."""

    execution_record: CommunityExecutionRecord

    def __post_init__(self) -> None:
        if type(self.execution_record) is not CommunityExecutionRecord:
            raise TypeError("execution_record must be CommunityExecutionRecord")

    @property
    def communities(self) -> tuple[Community, ...]:
        """Return preorder communities, with parents retaining whole-subtree members."""
        return self.execution_record.communities


def should_split(size: int, depth: int, max_size: int, max_depth: int) -> bool:
    """Split only oversized communities below the independent hierarchy depth limit."""
    for value in (size, depth, max_size, max_depth):
        if type(value) is not int:
            raise TypeError("hierarchy limits and counts must be integers")
        if value < 0:
            raise ValueError("hierarchy limits and counts must be nonnegative")
    if max_size == 0:
        raise ValueError("max_size must be positive")
    return size > max_size and depth < max_depth


def _components(graph: WeightedGraph) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = {node: set() for node in graph.node_ids}
    for edge in graph.edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    unseen = set(graph.node_ids)
    components = []
    for node in graph.node_ids:
        if node not in unseen:
            continue
        members, pending = set(), [node]
        unseen.remove(node)
        while pending:
            current = pending.pop()
            members.add(current)
            for neighbor in sorted(adjacency[current] & unseen):
                unseen.remove(neighbor)
                pending.append(neighbor)
        components.append(tuple(sorted(members)))
    return tuple(sorted(components))


def _induced(graph: WeightedGraph, members: tuple[str, ...]) -> WeightedGraph:
    nodes = frozenset(members)
    edges = tuple(e for e in graph.edges if e.source_id in nodes and e.target_id in nodes)
    count = sum(len(e.assertion_ids) for e in edges)
    return WeightedGraph(
        members, edges, ProjectionDiagnostics(count, count, ()), graph.projection_fingerprint
    )


def _validate_partition(
    partition: tuple[frozenset[str], ...], graph: WeightedGraph
) -> tuple[tuple[str, ...], ...]:
    if type(partition) is not tuple:
        raise TypeError("partition must be an immutable tuple")
    seen: set[str] = set()
    for members in partition:
        if type(members) is not frozenset:
            raise TypeError("partition members must be immutable frozensets")
        if not members:
            raise ValueError("partition members cannot be empty")
        for member in members:
            _text(member)
        if seen.intersection(members):
            raise ValueError("partition members overlap")
        seen.update(members)
    if seen != set(graph.node_ids):
        raise ValueError("partition must cover every input node exactly once")
    return tuple(sorted(tuple(sorted(members)) for members in partition))


def _execution_identity(
    graph: WeightedGraph,
    profile: CommunityProfile,
    algorithm_id: str,
    algorithm_version: str,
    runtime_identity: CommunityRuntimeIdentity,
) -> dict[str, object]:
    return {
        "hierarchy_version": 1,
        "projection_fingerprint": graph.projection_fingerprint,
        "profile": asdict(profile),
        "algorithm_id": algorithm_id,
        "algorithm_version": algorithm_version,
        "runtime_identity": asdict(runtime_identity),
    }


def _community_id(
    identity: dict[str, object],
    members: tuple[str, ...],
    path: tuple[int, ...],
    parent_id: str | None,
) -> str:
    return "community:" + fingerprint(
        {
            **identity,
            "path": path,
            "member_ids": members,
            "parent_id": parent_id,
            "depth": len(path) - 1,
        }
    )


def _validate_execution(record: CommunityExecutionRecord) -> None:
    """Reconstruct a complete preorder from recorded partitions without execution.

    A flat work stack validates arbitrarily deep bounded trees without invoking
    a partitioner or recursively constructing another execution record.
    """
    if record.algorithm_id != record.profile.partition_profile.tag:
        raise ValueError("execution algorithm must match the partition profile")
    if len(record.steps) != len(record.communities):
        raise ValueError("execution steps and communities must align")
    identity = _execution_identity(
        record.graph,
        record.profile,
        record.algorithm_id,
        record.algorithm_version,
        record.runtime_identity,
    )
    pending: list[tuple[tuple[str, ...], tuple[int, ...], str | None]] = [
        (members, (index,), None)
        for index, members in reversed(tuple(enumerate(_components(record.graph))))
    ]
    index = 0
    while pending:
        if index >= len(record.steps):
            raise ValueError("execution is missing a required visit")
        members, path, parent_id = pending.pop()
        depth = len(path) - 1
        step, row = record.steps[index], record.communities[index]
        induced = _induced(record.graph, members)
        if step.path != path or step.graph != induced:
            raise ValueError("execution visit path or induced graph does not match")
        eligible = should_split(
            len(members), depth, record.profile.max_size, record.profile.max_depth
        )
        if eligible != (step.partition is not None):
            raise ValueError("execution partition is missing or forbidden by the profile")
        issues: tuple[str, ...] = ()
        if step.partition is not None:
            _validate_partition(tuple(frozenset(group) for group in step.partition), induced)
            if len(step.partition) == 1:
                issues = ("partition_no_progress",)
        elif len(members) > record.profile.max_size:
            issues = ("max_depth_reached",)
        community_id = _community_id(identity, members, path, parent_id)
        if row != Community(community_id, parent_id, depth, members, issues):
            raise ValueError(
                "execution community identity, membership, or stop issues do not match"
            )
        if step.partition is not None and len(step.partition) > 1:
            pending.extend(
                (child, (*path, child_index), community_id)
                for child_index, child in reversed(tuple(enumerate(step.partition)))
            )
        index += 1
    if index != len(record.steps):
        raise ValueError("execution contains extra or reordered visits")


def hierarchical_community_run(
    graph: WeightedGraph, partitioner: CommunityPartitioner, profile: CommunityProfile
) -> CommunityHierarchyResult:
    """Build bounded association communities and immutable replay evidence.

    Connected components form depth-zero roots. Internal member_ids contain the
    whole subtree; consumers may derive all memberships or only leaf memberships
    by inspecting parent IDs. Neither membership nor explicit source association
    establishes truth. All mutable working state belongs to this invocation.
    """
    if type(graph) is not WeightedGraph or type(profile) is not CommunityProfile:
        raise TypeError("hierarchy requires immutable WeightedGraph and CommunityProfile")
    adapter = CommunityPartitionerRegistry(((profile.partition_profile.tag, partitioner),)).resolve(
        profile.partition_profile
    )
    algorithm_id, algorithm_version = adapter.algorithm_id, adapter.algorithm_version
    runtime_identity = adapter.runtime_identity
    identity = _execution_identity(
        graph, profile, algorithm_id, algorithm_version, runtime_identity
    )
    steps: list[CommunityExecutionStep] = []
    rows: list[Community] = []
    # Explicit stack avoids Python recursion limits even for a large declared depth.
    pending: list[tuple[tuple[str, ...], tuple[int, ...], str | None]] = [
        (members, (index,), None)
        for index, members in reversed(tuple(enumerate(_components(graph))))
    ]
    while pending:
        members, path, parent_id = pending.pop()
        depth = len(path) - 1
        induced = _induced(graph, members)
        partition = None
        issues: tuple[str, ...] = ()
        if should_split(len(members), depth, profile.max_size, profile.max_depth):
            partition = _validate_partition(
                adapter.partition(induced, profile.partition_profile), induced
            )
            if len(partition) == 1:
                issues = ("partition_no_progress",)
        elif len(members) > profile.max_size:
            issues = ("max_depth_reached",)
        community_id = _community_id(identity, members, path, parent_id)
        rows.append(Community(community_id, parent_id, depth, members, issues))
        steps.append(CommunityExecutionStep(path, induced, partition))
        if partition is not None and len(partition) > 1:
            pending.extend(
                (child, (*path, index), community_id)
                for index, child in reversed(tuple(enumerate(partition)))
            )
    if (adapter.algorithm_id, adapter.algorithm_version, adapter.runtime_identity) != (
        algorithm_id,
        algorithm_version,
        runtime_identity,
    ):
        raise ValueError("partition adapter changed its execution identity during run")
    return CommunityHierarchyResult(
        CommunityExecutionRecord(
            algorithm_id,
            algorithm_version,
            runtime_identity,
            profile,
            graph,
            tuple(steps),
            tuple(rows),
        )
    )


def hierarchical_communities(
    graph: WeightedGraph, partitioner: CommunityPartitioner, profile: CommunityProfile
) -> tuple[Community, ...]:
    """Compatibility view; use hierarchical_community_run to retain replay metadata."""
    return hierarchical_community_run(graph, partitioner, profile).communities
