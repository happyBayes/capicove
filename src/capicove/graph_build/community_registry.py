"""Explicit immutable partitioner bindings; no plugin or entry-point discovery."""

from dataclasses import dataclass

from capicove.graph_build.ports import CommunityPartitioner, CommunityRuntimeIdentity
from capicove.graph_build.profiles import LouvainPartitionProfile, PartitionProfile


def _profile_tag(profile: PartitionProfile) -> str:
    if type(profile) is not LouvainPartitionProfile:
        raise TypeError("unsupported partition profile")
    return profile.tag


def _validate_partitioner(tag: str, partitioner: CommunityPartitioner) -> None:
    if type(tag) is not str or tag != "louvain":
        raise ValueError("unknown partition profile tag")
    if not callable(getattr(partitioner, "partition", None)):
        raise TypeError("missing partition adapter")
    if getattr(partitioner, "algorithm_id", None) != tag:
        raise ValueError("partition adapter algorithm does not match profile tag")
    version = getattr(partitioner, "algorithm_version", None)
    if type(version) is not str or not version.strip():
        raise ValueError("partition adapter must declare algorithm_version")
    try:
        version.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("algorithm_version must be valid UTF-8") from None
    if type(getattr(partitioner, "runtime_identity", None)) is not CommunityRuntimeIdentity:
        raise TypeError("partition adapter must declare immutable runtime identity")


@dataclass(frozen=True, slots=True)
class CommunityPartitionerRegistry:
    """Bind reviewed tags at composition time; injected adapters own their lifetime.

    Immutability covers bindings. Adapter implementations must independently obey
    the port's reentrancy and stable identity requirements.
    """

    bindings: tuple[tuple[str, CommunityPartitioner], ...]

    def __post_init__(self) -> None:
        if type(self.bindings) is not tuple:
            raise TypeError("partition bindings must be an immutable tuple")
        seen: set[str] = set()
        for binding in self.bindings:
            if type(binding) is not tuple or len(binding) != 2:
                raise TypeError("partition bindings must be immutable tag/adapter pairs")
            tag, adapter = binding
            _validate_partitioner(tag, adapter)
            if tag in seen:
                raise ValueError("duplicate partition binding")
            seen.add(tag)

    def resolve(self, profile: PartitionProfile) -> CommunityPartitioner:
        """Resolve only an explicitly bound adapter for a closed reviewed profile."""
        tag = _profile_tag(profile)
        for bound_tag, adapter in self.bindings:
            if bound_tag == tag:
                _validate_partitioner(tag, adapter)
                return adapter
        raise ValueError("missing partition adapter for profile")
