"""Authored association graphs exercise bounded hierarchy and replay identity."""

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, replace
from itertools import pairwise
from typing import Any

import pytest

from capicove.graph_build import communities
from capicove.graph_build.communities import ProjectionDiagnostics, WeightedEdge, WeightedGraph
from capicove.graph_build.profiles import CommunityProfile, LouvainPartitionProfile


def graph(nodes: str = "abcdz", pairs: tuple[str, ...] = ("ab", "bc", "cd")) -> WeightedGraph:
    """Build small authored connected chains with optional isolated nodes."""
    edges = tuple(WeightedEdge(p[0], p[1], 1.0, (p,), (p,)) for p in sorted(pairs))
    return WeightedGraph(
        tuple(sorted(nodes)), edges, ProjectionDiagnostics(len(pairs), len(pairs), ()), "a" * 64
    )


@dataclass(frozen=True)
class Scripted:
    """Pure test partitioner with explicitly declared execution identity."""

    output: Any = None
    algorithm_id: str = "louvain"
    algorithm_version: str = "script-v1"

    @property
    def runtime_identity(self) -> Any:
        from capicove.graph_build.ports import CommunityRuntimeIdentity

        return CommunityRuntimeIdentity("authored-script", "1", (("mode", "halves"),))

    def partition(self, graph: WeightedGraph, profile: LouvainPartitionProfile) -> Any:
        if self.output is not None:
            return self.output
        # The profile must arrive unchanged; all tests deliberately use this seed.
        assert profile.seed == 17
        midpoint = len(graph.node_ids) // 2
        return (frozenset(graph.node_ids[midpoint:]), frozenset(graph.node_ids[:midpoint]))


def profile(**kwargs: Any) -> CommunityProfile:
    return CommunityProfile(partition_profile=LouvainPartitionProfile(seed=17), **kwargs)


def test_hierarchy_stops_at_declared_depth() -> None:
    assert hasattr(communities, "should_split"), "bounded hierarchy is missing"
    assert communities.should_split(51, 0, 50, 3)
    assert not communities.should_split(50, 0, 50, 3)
    assert not communities.should_split(51, 3, 50, 3)


@pytest.mark.parametrize(
    "args",
    [
        (True, 0, 1, 2),
        (2, 0.0, 1, 2),
        (2, 0, False, 2),
        (2, 0, 1, "2"),
        (-1, 0, 1, 2),
        (2, -1, 1, 2),
        (2, 0, 0, 2),
        (2, 0, 1, -1),
    ],
)
def test_split_rejects_invalid_counts(args: tuple[Any, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        communities.should_split(*args)


def test_roots_children_and_execution_record_are_complete() -> None:
    source = graph()
    policy = profile(max_size=1, max_depth=1)
    run = communities.hierarchical_community_run(source, Scripted(), policy)
    rows = run.communities
    assert [(r.member_ids, r.depth) for r in rows] == [
        (tuple("abcd"), 0),
        (tuple("ab"), 1),
        (tuple("cd"), 1),
        (("z",), 0),
    ]
    assert rows[0].parent_id is None and rows[3].parent_id is None
    assert rows[1].parent_id == rows[2].parent_id == rows[0].community_id
    assert rows[1].issue_codes == rows[2].issue_codes == ("max_depth_reached",)
    record = run.execution_record
    assert record.algorithm_id == "louvain"
    assert record.algorithm_version == "script-v1"
    assert record.runtime_identity.runtime_version == "1"
    assert record.profile == policy
    assert record.graph == source
    assert record.projection_fingerprint == source.projection_fingerprint
    assert record.communities == rows
    assert [step.path for step in record.steps] == [(0,), (0, 0), (0, 1), (1,)]
    assert [step.graph.node_ids for step in record.steps] == [
        tuple("abcd"),
        tuple("ab"),
        tuple("cd"),
        ("z",),
    ]
    assert [len(step.graph.edges) for step in record.steps] == [3, 1, 1, 0]
    assert record.steps[0].partition == (tuple("ab"), tuple("cd"))
    assert all(step.partition is None for step in record.steps[1:])
    assert communities.hierarchical_communities(source, Scripted(), policy) == rows
    assert len(record.execution_fingerprint) == 64
    assert source == graph()
    with pytest.raises(FrozenInstanceError):
        record.algorithm_version = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("policy", [profile(max_size=4), profile(max_size=1, max_depth=0)])
def test_small_components_and_depth_zero_do_not_partition(policy: CommunityProfile) -> None:
    # Invalid output would fail if the partitioner were unnecessarily called.
    run = communities.hierarchical_community_run(graph(), Scripted(output=[]), policy)
    assert [r.member_ids for r in run.communities] == [tuple("abcd"), ("z",)]
    assert all(step.partition is None for step in run.execution_record.steps)


def test_empty_graph_is_a_recorded_empty_result() -> None:
    run = communities.hierarchical_community_run(graph("", ()), Scripted(output=[]), profile())
    assert run.communities == ()
    assert run.execution_record.steps == ()


@pytest.mark.parametrize(
    "output",
    [
        [],
        [frozenset("ab"), frozenset("cd")],
        (set("ab"), frozenset("cd")),
        (),
        (frozenset(), frozenset("abcd")),
        (frozenset("abc"), frozenset("cd")),
        (frozenset("ab"), frozenset("ce")),
        (frozenset("ab"),),
        (frozenset((1,)), frozenset("abcd")),
    ],
)
def test_invalid_partition_is_rejected(output: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        communities.hierarchical_community_run(graph(), Scripted(output), profile(max_size=1))


def test_no_progress_is_one_oversized_leaf() -> None:
    run = communities.hierarchical_community_run(
        graph(), Scripted((frozenset("abcd"),)), profile(max_size=1, max_depth=99)
    )
    assert len(run.communities) == 2
    assert run.communities[0].issue_codes == ("partition_no_progress",)
    assert run.execution_record.steps[0].partition == (tuple("abcd"),)


def test_replay_is_stable_reentrant_and_binds_inputs() -> None:
    source, adapter, policy = graph(), Scripted(), profile(max_size=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = tuple(
            pool.map(
                lambda _: communities.hierarchical_community_run(source, adapter, policy), range(8)
            )
        )
    assert all(run == runs[0] for run in runs)
    baseline = runs[0].execution_record.execution_fingerprint
    for changed_source, changed_adapter, changed_policy in [
        (replace(source, projection_fingerprint="b" * 64), adapter, policy),
        (source, replace(adapter, algorithm_version="v2"), policy),
        (source, adapter, replace(policy, support_cap=7)),
    ]:
        changed = communities.hierarchical_community_run(
            changed_source, changed_adapter, changed_policy
        )
        assert changed.execution_record.execution_fingerprint != baseline
        assert changed.communities[0].community_id != runs[0].communities[0].community_id


def test_registry_requires_explicit_immutable_compatible_bindings() -> None:
    from capicove.graph_build.community_registry import CommunityPartitionerRegistry

    adapter = Scripted()
    registry = CommunityPartitionerRegistry((("louvain", adapter),))
    assert registry.resolve(LouvainPartitionProfile()) is adapter
    with pytest.raises(FrozenInstanceError):
        registry.bindings = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="missing"):
        CommunityPartitionerRegistry(()).resolve(LouvainPartitionProfile())
    for invalid in [
        [],
        [("louvain", adapter)],
        (["louvain", adapter],),
        (("louvain", adapter), ("louvain", adapter)),
        (("unknown", adapter),),
        (("louvain", replace(adapter, algorithm_id="leiden")),),
        (("louvain", None),),
    ]:
        with pytest.raises((TypeError, ValueError)):
            CommunityPartitionerRegistry(invalid)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        registry.resolve({"tag": "louvain"})  # type: ignore[arg-type]


def test_hierarchy_rejects_missing_runtime_identity_and_mismatch() -> None:
    for adapter in (object(), Scripted(algorithm_id="leiden"), Scripted(algorithm_version="")):
        with pytest.raises((TypeError, ValueError)):
            communities.hierarchical_community_run(graph(), adapter, profile())  # type: ignore[arg-type]


def test_partition_order_is_normalized_before_ids_and_recursion() -> None:
    policy = profile(max_size=2)
    forward = Scripted((frozenset("ab"), frozenset("cd")))
    reverse = Scripted((frozenset("dc"), frozenset("ba")))
    assert communities.hierarchical_community_run(
        graph(), forward, policy
    ) == communities.hierarchical_community_run(graph(), reverse, policy)


def test_runtime_identity_is_required_even_without_partition_calls() -> None:
    class Unrecorded:
        algorithm_id = "louvain"
        algorithm_version = "1"

        def partition(
            self, graph: WeightedGraph, profile: LouvainPartitionProfile
        ) -> tuple[frozenset[str], ...]:
            raise AssertionError("should not partition a small graph")

    with pytest.raises(TypeError, match="runtime identity"):
        communities.hierarchical_community_run(graph(), Unrecorded(), profile())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "configuration",
    [
        [("backend", "script")],
        (["backend", "script"],),
        (("backend", []),),
        (("x", "a"), ("x", "b")),
        (("z", "a"), ("a", "b")),
        (("", "x"),),
        (("x", "\ud800"),),
    ],
)
def test_runtime_identity_rejects_mutable_or_ambiguous_configuration(configuration: Any) -> None:
    from capicove.graph_build.ports import CommunityRuntimeIdentity

    with pytest.raises((TypeError, ValueError)):
        CommunityRuntimeIdentity("script", "1", configuration)


def test_runtime_version_changes_record_and_community_identity() -> None:
    from capicove.graph_build.ports import CommunityRuntimeIdentity

    @dataclass(frozen=True)
    class RuntimeVariant(Scripted):
        runtime_version: str = "1"

        @property
        def runtime_identity(self) -> CommunityRuntimeIdentity:
            return CommunityRuntimeIdentity("script", self.runtime_version, ())

    first = communities.hierarchical_community_run(graph(), RuntimeVariant(), profile())
    second = communities.hierarchical_community_run(
        graph(), RuntimeVariant(runtime_version="2"), profile()
    )
    assert (
        first.execution_record.execution_fingerprint
        != second.execution_record.execution_fingerprint
    )
    assert first.communities[0].community_id != second.communities[0].community_id


def test_record_rejects_mutable_execution_containers() -> None:
    record = communities.hierarchical_community_run(graph(), Scripted(), profile()).execution_record
    kwargs: dict[str, Any]
    for kwargs in (
        {"steps": list(record.steps)},
        {"communities": list(record.communities)},
        {"runtime_identity": {}},
        {"profile": {}},
        {"graph": {}},
        {"hierarchy_version": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(record, **kwargs)
    for kwargs in (
        {"path": [0]},
        {"partition": [tuple("abcd")]},
        {"partition": (tuple("ab"),)},
        {"graph": {}},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(record.steps[0], **kwargs)


def test_hierarchy_detects_an_adapter_changing_its_declared_identity() -> None:
    from capicove.graph_build.ports import CommunityRuntimeIdentity

    class Unstable:
        algorithm_id = "louvain"
        algorithm_version = "1"
        runtime_identity = CommunityRuntimeIdentity("script", "1", ())

        def partition(
            self, graph: WeightedGraph, profile: LouvainPartitionProfile
        ) -> tuple[frozenset[str], ...]:
            self.algorithm_version = "2"
            return (frozenset(graph.node_ids),)

    with pytest.raises(ValueError, match="changed its execution identity"):
        communities.hierarchical_community_run(graph(), Unstable(), profile(max_size=1))


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "missing_steps",
        "missing_communities",
        "foreign_graph",
        "algorithm",
        "version",
        "reverse",
        "parent",
        "id",
        "members",
        "depth",
        "issues",
        "partition",
        "missing_partition",
        "extra_partition",
        "step_graph",
        "step_path",
        "extra_visit",
    ],
)
def test_execution_record_rejects_inconsistent_structure(mutation: str) -> None:
    record = communities.hierarchical_community_run(
        graph(), Scripted(), profile(max_size=2)
    ).execution_record
    steps, rows = list(record.steps), list(record.communities)
    replacements: dict[str, Any] = {}
    if mutation == "empty":
        replacements.update(steps=(), communities=())
    elif mutation == "missing_steps":
        replacements["steps"] = ()
    elif mutation == "missing_communities":
        replacements["communities"] = ()
    elif mutation == "foreign_graph":
        replacements["graph"] = graph("xy", ("xy",))
    elif mutation == "algorithm":
        replacements["algorithm_id"] = "leiden"
    elif mutation == "version":
        replacements["algorithm_version"] = "changed-version"
    elif mutation == "reverse":
        replacements.update(steps=tuple(reversed(steps)), communities=tuple(reversed(rows)))
    elif mutation in ("parent", "id", "members", "depth", "issues"):
        changes = {
            "parent": {"parent_id": "foreign"},
            "id": {"community_id": "foreign"},
            "members": {"member_ids": ("a",)},
            "depth": {"depth": 2},
            "issues": {"issue_codes": ("partition_no_progress",)},
        }
        rows[1] = replace(rows[1], **changes[mutation])  # type: ignore[arg-type]
        replacements["communities"] = tuple(rows)
    elif mutation in (
        "partition",
        "missing_partition",
        "extra_partition",
        "step_graph",
        "step_path",
    ):
        if mutation == "partition":
            steps[0] = replace(steps[0], partition=(tuple("abcd"),))
        elif mutation == "missing_partition":
            steps[0] = replace(steps[0], partition=None)
        elif mutation == "extra_partition":
            steps[1] = replace(steps[1], partition=(tuple("ab"),))
        elif mutation == "step_graph":
            steps[1] = replace(steps[1], graph=graph("ab", ()))
        else:
            steps[1] = replace(steps[1], path=(0, 7))
        replacements["steps"] = tuple(steps)
    else:
        replacements.update(steps=(*steps, steps[-1]), communities=(*rows, rows[-1]))
    with pytest.raises(ValueError, match="execution"):
        replace(record, **replacements)


@pytest.mark.parametrize(
    "policy,output",
    [(profile(max_size=1, max_depth=0), None), (profile(max_size=1), (frozenset("abcd"),))],
)
def test_execution_record_requires_exact_stop_issues(policy: CommunityProfile, output: Any) -> None:
    record = communities.hierarchical_community_run(
        graph(), Scripted(output), policy
    ).execution_record
    with pytest.raises(ValueError, match="execution"):
        replace(
            record,
            communities=(replace(record.communities[0], issue_codes=()), *record.communities[1:]),
        )


def test_empty_execution_cannot_claim_nonempty_visits() -> None:
    nonempty = communities.hierarchical_community_run(
        graph(), Scripted(), profile()
    ).execution_record
    with pytest.raises(ValueError, match="execution"):
        replace(nonempty, graph=graph("", ()))


def test_deep_execution_validation_is_iterative() -> None:
    @dataclass(frozen=True)
    class PeelOne(Scripted):
        def partition(
            self, graph: WeightedGraph, profile: LouvainPartitionProfile
        ) -> tuple[frozenset[str], ...]:
            return (frozenset(graph.node_ids[:1]), frozenset(graph.node_ids[1:]))

    # Peeling is quadratic in depth, so prove iteration by lowering the recursion
    # limit below the hierarchy depth instead of exceeding the default limit.
    depth = 200
    nodes = tuple(f"node-{index:04d}" for index in range(depth + 1))
    edges = tuple(
        WeightedEdge(left, right, 1.0, (left,), (left,)) for left, right in pairwise(nodes)
    )
    source = WeightedGraph(nodes, edges, ProjectionDiagnostics(depth, depth, ()), "a" * 64)
    stack_depth = sum(1 for _ in traceback.walk_stack(None))
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(stack_depth + depth // 2)
    try:
        record = communities.hierarchical_community_run(
            source, PeelOne(), profile(max_size=1, max_depth=depth)
        ).execution_record
        assert replace(record) == record
    finally:
        sys.setrecursionlimit(original_limit)
    assert len(record.steps) == len(record.communities) == 2 * depth + 1
    assert max(row.depth for row in record.communities) == depth
