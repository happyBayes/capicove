"""Real optional NetworkX integration on project-authored association graphs."""

import importlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.version import Version

from capicove.graph_build import communities
from capicove.graph_build.profiles import CommunityProfile, LouvainPartitionProfile
from tests.unit.graph_build.test_communities import graph


def adapter_module() -> Any:
    assert importlib.util.find_spec("capicove.graph_build._adapters.louvain") is not None, (
        "real Louvain adapter is missing"
    )
    return importlib.import_module("capicove.graph_build._adapters.louvain")


def optional_runtime() -> Any:
    adapter_module()
    nx = pytest.importorskip("networkx", reason="graph-community extra unavailable")
    installed = Version(nx.__version__)
    if not Version("3.6") <= installed < Version("4") or installed.is_prerelease:
        pytest.skip(f"graph-community extra incompatible: NetworkX {installed}")
    return nx


def test_real_louvain_clusters_replay_and_record_actual_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nx = optional_runtime()
    adapter = adapter_module().NetworkXLouvainPartitioner()
    source = graph("abcdefghz", ("ab", "ac", "bc", "cd", "de", "df", "ef", "gh"))
    # Strong triangles with a deliberately weak bridge have two clear clusters.
    source = replace(
        source,
        edges=tuple(
            replace(e, weight=0.01 if e.source_id + e.target_id == "cd" else 10.0)
            for e in source.edges
        ),
    )
    policy = CommunityProfile(
        max_size=3,
        partition_profile=LouvainPartitionProfile(
            resolution=1.1, threshold=1e-8, max_level=7, seed=17
        ),
    )
    actual = nx.community.louvain_communities
    calls: list[tuple[Any, dict[str, Any]]] = []

    def observe(graph: Any, **kwargs: Any) -> Any:
        calls.append((graph.copy(), kwargs))
        return actual(graph, **kwargs)

    monkeypatch.setattr(nx.community, "louvain_communities", observe)
    first = communities.hierarchical_community_run(source, adapter, policy)
    second = communities.hierarchical_community_run(source, adapter, policy)
    assert first == second
    assert [(r.member_ids, r.depth) for r in first.communities] == [
        (tuple("abcdef"), 0),
        (tuple("abc"), 1),
        (tuple("def"), 1),
        (tuple("gh"), 0),
        (("z",), 0),
    ]
    assert len(calls) == 2
    native, kwargs = calls[0]
    assert type(native) is nx.Graph and not native.is_directed()
    assert tuple(native.nodes) == tuple("abcdef")
    assert tuple(native.edges) == (
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("d", "f"),
        ("e", "f"),
    )
    assert native["c"]["d"]["weight"] == 0.01
    assert kwargs == {
        "weight": "weight",
        "resolution": 1.1,
        "threshold": 1e-8,
        "max_level": 7,
        "seed": 17,
        "backend": "networkx",
    }
    record = first.execution_record
    assert record.runtime_identity.runtime_id == "networkx"
    assert record.runtime_identity.runtime_version == nx.__version__
    assert record.algorithm_version != nx.__version__
    assert dict(record.runtime_identity.configuration)["backend"] == "networkx"
    assert record.profile == policy
    assert record.steps[0].partition == (tuple("abc"), tuple("def"))
    assert adapter.partition(graph("abc", ("ab", "ac", "bc")), policy.partition_profile) == (
        frozenset("abc"),
    )


def test_real_adapter_no_progress_and_disconnected_singletons() -> None:
    optional_runtime()
    adapter = adapter_module().NetworkXLouvainPartitioner()
    run = communities.hierarchical_community_run(
        graph("abcdz", ("ab", "ac", "ad", "bc", "bd", "cd")), adapter, CommunityProfile(max_size=2)
    )
    assert run.communities[0].member_ids == tuple("abcd")
    assert run.communities[0].issue_codes == ("partition_no_progress",)
    assert run.communities[1].member_ids == ("z",)
    assert adapter.partition(graph("az", ()), LouvainPartitionProfile()) == (
        frozenset("a"),
        frozenset("z"),
    )
    assert adapter.partition(graph("", ()), LouvainPartitionProfile()) == ()


def test_importing_adapter_and_registry_does_not_import_networkx() -> None:
    adapter_module()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.graph_build.communities; import capicove.graph_build.community_registry; import capicove.graph_build._adapters.louvain; import sys; assert 'networkx' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_real_replay_does_not_depend_on_process_hash_order() -> None:
    optional_runtime()
    script = """
from capicove.graph_build._adapters.louvain import NetworkXLouvainPartitioner
from capicove.graph_build.communities import hierarchical_community_run
from capicove.graph_build.profiles import CommunityProfile
from tests.unit.graph_build.test_communities import graph
source = graph("abcdefz", ("ab", "ac", "bc", "cd", "de", "df", "ef"))
run = hierarchical_community_run(source, NetworkXLouvainPartitioner(), CommunityProfile(max_size=2))
print(run.execution_record.execution_fingerprint)
"""
    fingerprints = []
    for seed in ("1", "123", "random"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        fingerprints.append(result.stdout.strip())
    assert len(fingerprints[0]) == 64
    assert len(set(fingerprints)) == 1


def test_missing_networkx_fails_without_acquisition(monkeypatch: pytest.MonkeyPatch) -> None:
    module = adapter_module()

    def unavailable(name: str) -> Any:
        assert name == "networkx"
        raise ModuleNotFoundError("networkx unavailable")

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="graph-community requires installed"):
        module.NetworkXLouvainPartitioner()


@pytest.mark.parametrize(
    "version", ["3.5", "3.6rc1", "3.6.dev0", "4.0", "4.0rc1", "1!3.6", "3.6junk", "3.6.1.2junk"]
)
def test_adapter_rejects_incompatible_or_malformed_runtime(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nx = optional_runtime()
    monkeypatch.setattr(nx, "__version__", version)
    with pytest.raises(ImportError, match=r"3\.6"):
        adapter_module().NetworkXLouvainPartitioner()


@pytest.mark.parametrize("version", ["3.6", "3.6.1", "3.6.1.post1", "3.6.1+local.1", "3.10.2"])
def test_adapter_records_supported_release_versions(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nx = optional_runtime()
    monkeypatch.setattr(nx, "__version__", version)
    assert adapter_module().NetworkXLouvainPartitioner().runtime_identity.runtime_version == version


def test_adapter_rejects_untyped_profile_and_changed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nx = optional_runtime()
    adapter = adapter_module().NetworkXLouvainPartitioner()
    with pytest.raises(TypeError):
        adapter.partition(graph(), {"tag": "louvain"})
    monkeypatch.setattr(nx, "__version__", "3.7.0")
    with pytest.raises(ValueError, match="runtime"):
        adapter.partition(graph(), LouvainPartitionProfile())


@pytest.mark.parametrize("priority", ["algos", "generators", "classes", "louvain_partitions"])
def test_real_dispatcher_cannot_select_an_automatic_backend(
    priority: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nx = optional_runtime()
    module = adapter_module()
    adapter = module.NetworkXLouvainPartitioner()
    native = nx.Graph()
    native.add_edge("a", "b", weight=1)
    backend_name = "capicove-test-foreign"
    foreign_calls: list[str] = []

    def foreign_partition(native: Any, *args: Any, **kwargs: Any) -> Any:
        foreign_calls.append("louvain_partitions")
        return iter([[set(native)]])

    backend = SimpleNamespace(
        louvain_partitions=foreign_partition,
        convert_from_nx=lambda native, **kwargs: native,
        can_run=lambda *args, **kwargs: True,
        should_run=lambda *args, **kwargs: True,
    )
    dispatch = nx.utils.backends
    # Install a representative backend through the real dispatcher's in-memory
    # registration surfaces. Monkeypatch restores config before unregistering it.
    monkeypatch.setitem(
        dispatch.backend_info, backend_name, {"functions": {"louvain_partitions": {}}}
    )
    monkeypatch.setitem(dispatch.backends, backend_name, SimpleNamespace(load=lambda: backend))
    monkeypatch.setitem(dispatch._loaded_backends, backend_name, backend)
    monkeypatch.setattr(
        dispatch._dispatchable, "__call__", dispatch._dispatchable._call_if_any_backends_installed
    )
    monkeypatch.setitem(nx.config.backend_priority, priority, [backend_name])
    if priority in ("algos", "louvain_partitions"):
        # An explicit outer backend does not constrain nested real dispatch.
        nx.community.louvain_communities(native, seed=17, backend="networkx")
        assert foreign_calls == ["louvain_partitions"]
        foreign_calls.clear()
    with pytest.raises(ValueError, match="backend priorit"):
        module.NetworkXLouvainPartitioner()
    with pytest.raises(ValueError, match="backend priorit"):
        adapter.partition(graph(), LouvainPartitionProfile())
    assert foreign_calls == []
    assert nx.config.backend_priority[priority] == [backend_name]


@pytest.mark.parametrize(
    "change", ["version", "algos", "generators", "classes", "louvain_partitions"]
)
def test_adapter_rejects_runtime_drift_during_real_partition(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nx = optional_runtime()
    adapter = adapter_module().NetworkXLouvainPartitioner()
    actual = nx.community.louvain_communities

    def changing_runtime(native: Any, **kwargs: Any) -> Any:
        partition = actual(native, **kwargs)
        if change == "version":
            monkeypatch.setattr(nx, "__version__", "3.7.0")
        else:
            monkeypatch.setitem(nx.config.backend_priority, change, ["networkx"])
        return partition

    monkeypatch.setattr(nx.community, "louvain_communities", changing_runtime)
    with pytest.raises(ValueError, match=r"runtime|backend priorit"):
        adapter.partition(graph(), LouvainPartitionProfile())


def test_adapter_records_all_empty_priority_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    nx = optional_runtime()
    monkeypatch.setitem(nx.config.backend_priority, "louvain_partitions", [])
    adapter = adapter_module().NetworkXLouvainPartitioner()
    configuration = dict(adapter.runtime_identity.configuration)
    assert {
        key: value for key, value in configuration.items() if key.startswith("backend_priority.")
    } == {
        "backend_priority.algos": "[]",
        "backend_priority.generators": "[]",
        "backend_priority.classes": "[]",
        "backend_priority.louvain_partitions": "[]",
    }
    # Even a harmless added key is configuration drift and cannot reuse metadata.
    monkeypatch.setitem(nx.config.backend_priority, "is_empty", [])
    with pytest.raises(ValueError, match="runtime"):
        adapter.partition(graph(), LouvainPartitionProfile())
