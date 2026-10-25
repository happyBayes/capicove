"""Explicit optional NetworkX Louvain adapter; imports alone load no graph runtime."""

import importlib
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import ModuleType

from capicove.graph_build.communities import WeightedGraph
from capicove.graph_build.ports import CommunityRuntimeIdentity
from capicove.graph_build.profiles import LouvainPartitionProfile, PartitionProfile


def _networkx() -> ModuleType:
    """Load an installed supported final/post release, without acquisition.

    Parse release components numerically, including PEP 440 post/local suffixes.
    Development/pre releases and nonzero epochs are intentionally unsupported;
    they are not qualified releases within the declared optional requirement.
    No packaging dependency or optional import is needed at module import time.
    """
    try:
        nx = importlib.import_module("networkx")
    except ImportError as exc:
        raise ImportError("graph-community requires installed networkx>=3.6,<4") from exc
    version = getattr(nx, "__version__", None)
    match = re.fullmatch(
        r"v?(?:0!)?(?P<release>[0-9]+(?:\.[0-9]+)*)(?:[-_.]?post[0-9]+)?(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?",
        version if type(version) is str else "",
        re.IGNORECASE,
    )
    release = tuple(int(part) for part in match["release"].split(".")) if match else ()
    if not release or not (3, 6) <= release < (4,):
        raise ImportError("graph-community requires a released networkx>=3.6,<4")
    return nx


def _runtime_identity(nx: ModuleType) -> CommunityRuntimeIdentity:
    """Snapshot a supported runtime with automatic dispatch completely disabled.

    An explicit outer backend does not constrain dispatch in nested NetworkX
    calls. Every priority list must be empty, including per-function overrides
    and future categories. Global configuration is inspected, never modified.
    """
    priorities = nx.config.backend_priority
    if not isinstance(priorities, Mapping) or not {"algos", "generators", "classes"}.issubset(
        priorities
    ):
        raise ValueError("NetworkX backend priorities require all base categories")
    snapshot = []
    for name, value in priorities.items():
        if type(name) is not str or not name or type(value) is not list:
            raise ValueError("NetworkX backend priorities must be named lists")
        values = tuple(value)
        if values:
            raise ValueError("NetworkX backend priorities must all be empty")
        snapshot.append(("backend_priority." + name, json.dumps(values, separators=(",", ":"))))
    return CommunityRuntimeIdentity(
        "networkx",
        nx.__version__,
        tuple(
            sorted(
                (
                    ("backend", "networkx"),
                    *snapshot,
                    ("edge_order", "lexicographic-endpoints"),
                    ("graph_type", "undirected-Graph"),
                    ("node_order", "lexicographic-id"),
                    ("python_implementation", platform.python_implementation()),
                    ("python_version", platform.python_version()),
                    ("weight", "weight"),
                )
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class NetworkXLouvainPartitioner:
    """Reentrant CPU NetworkX implementation with immutable runtime metadata.

    Construction explicitly selects and verifies the installed optional runtime.
    Each call creates its own native graph and random generator seeded by the
    exact closed profile. Algorithm version describes this adapter, separately
    from the actual package version recorded in runtime_identity.
    """

    algorithm_id: str = field(default="louvain", init=False)
    algorithm_version: str = field(default="capicove-louvain-v1", init=False)
    runtime_identity: CommunityRuntimeIdentity = field(init=False)

    def __post_init__(self) -> None:
        nx = _networkx()
        object.__setattr__(self, "runtime_identity", _runtime_identity(nx))

    def _checked_runtime(self) -> ModuleType:
        nx = _networkx()
        if _runtime_identity(nx) != self.runtime_identity:
            raise ValueError("NetworkX runtime identity changed after adapter construction")
        return nx

    def partition(
        self, graph: WeightedGraph, profile: PartitionProfile
    ) -> tuple[frozenset[str], ...]:
        """Return stable immutable memberships without exposing a native graph."""
        if type(graph) is not WeightedGraph:
            raise TypeError("Louvain requires immutable WeightedGraph")
        if type(profile) is not LouvainPartitionProfile or profile.tag != "louvain":
            raise TypeError("Louvain requires the exact louvain profile variant")
        nx = self._checked_runtime()
        native = nx.Graph()
        native.add_nodes_from(graph.node_ids)
        native.add_weighted_edges_from(
            (edge.source_id, edge.target_id, edge.weight) for edge in graph.edges
        )
        self._checked_runtime()
        partition = nx.community.louvain_communities(
            native,
            weight="weight",
            resolution=profile.resolution,
            threshold=profile.threshold,
            max_level=profile.max_level,
            seed=profile.seed,
            backend="networkx",
        )
        result = tuple(
            sorted(
                (frozenset(members) for members in partition),
                key=lambda members: tuple(sorted(members)),
            )
        )
        self._checked_runtime()
        return result
