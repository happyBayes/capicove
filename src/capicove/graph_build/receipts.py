"""Backend-independent identities and immutable operational inference receipts."""

import re
from dataclasses import dataclass

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.codec import fingerprint


class ReceiptStoreError(RuntimeError):
    """Report unavailable or corrupt receipt metadata without source content."""


class ReceiptConflictError(ReceiptStoreError):
    """Reject completion by an expired, displaced, or conflicting attempt."""


def require_content_id(value: str, prefix: str) -> None:
    """Accept only an opaque prefixed lowercase SHA-256 identifier."""
    if not isinstance(value, str) or re.fullmatch(prefix + r":[0-9a-f]{64}", value) is None:
        raise ValueError("invalid content identifier")


@dataclass(frozen=True, slots=True)
class InferenceReceipt:
    """Keep terminal result references and bounded machine metadata, never payloads.

    Every terminal outcome has a protected result, including failures/cancellation.
    Usage is an immutable sequence of unique counter-name/integer pairs.
    """

    key: str
    status: str
    result_artifact_ref: str
    result_digest: str
    call_count: int
    repair_count: int
    usage: tuple[tuple[str, int], ...]
    rejected_count: int
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        require_content_id(self.key, "receipt")
        require_content_id(self.result_artifact_ref, "artifact")
        require_content_id("sha256:" + self.result_digest, "sha256")
        if self.status not in ("succeeded", "failed", "cancelled"):
            raise ValueError("receipt status must be terminal")
        for value in (self.call_count, self.repair_count, self.rejected_count):
            if type(value) is not int or value < 0:
                raise ValueError("receipt counters must be nonnegative integers")
        if self.repair_count > self.call_count:
            raise ValueError("repair_count exceeds call_count")
        usage = tuple(self.usage)
        if len(usage) > 32:
            raise ValueError("too many usage counters")
        names = set()
        for pair in usage:
            if len(pair) != 2:
                raise ValueError("usage must contain name/count pairs")
            name, count = pair
            _machine_code(name)
            if name in names or type(count) is not int or count < 0:
                raise ValueError("invalid usage counter")
            names.add(name)
        raw_codes: object = self.issue_codes
        if isinstance(raw_codes, str) or len(self.issue_codes) > 128:
            raise ValueError("invalid issue codes")
        codes = tuple(self.issue_codes)
        for code in codes:
            _machine_code(code)
        object.__setattr__(self, "usage", tuple(sorted((name, count) for name, count in usage)))
        object.__setattr__(self, "issue_codes", codes)


def _machine_code(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
        raise ValueError("expected a bounded machine code")


def receipt_key(scope: GraphScope, role: str, request_sha256: str) -> str:
    """Bind a request identity to both workspace and scope and the producing role."""
    if not isinstance(scope, GraphScope):
        raise TypeError("scope must be a GraphScope")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("role must not be blank")
    if not isinstance(request_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
        raise ValueError("request_sha256 must be a lowercase SHA-256")
    return "receipt:" + fingerprint([1, scope.workspace_ref, scope.scope_ref, role, request_sha256])
