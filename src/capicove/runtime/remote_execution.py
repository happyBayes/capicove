"""Execute claimed remote operations inside isolated workers (R3.02 with R3.04).

The gateway never runs an isolated target's handler itself. Each claimed attempt
runs once in a fresh sandbox of the operation's execution domain, and its result
is published only under the claim's fencing token. Outcomes that may follow
partial effects (crash, deadline, oversized output, an envelope or peer identity
that cannot be trusted) are recorded as ``outcome_unknown`` and never retried;
only work that never started, or a failure the worker itself reported, is
recorded as ``failed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.runtime.remote_operations import OperationService
from capicove.runtime.remote_workers import (
    DEFAULT_REGISTRY,
    CgroupLimits,
    WorkerDomain,
    WorkerError,
    run_isolated,
)


_KNOWN_FAILURES = frozenset({"operation_failed", "unknown_operation", "isolation_unavailable"})


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """The recorded state of one executed attempt."""

    operation_id: str
    state: str


class IsolatedExecutor:
    """Claim, sandbox and publish one operation at a time for one domain."""

    def __init__(
        self,
        service: OperationService,
        domain: WorkerDomain,
        operations: Mapping[str, str],
        *,
        registry: str = DEFAULT_REGISTRY,
        import_roots: Sequence[Path] = (),
        cgroup: CgroupLimits | None = None,
        lease_margin_seconds: float = 5.0,
    ) -> None:
        self._service = service
        self._domain = domain
        self._operations = dict(operations)
        self._registry = registry
        self._roots = tuple(import_roots)
        self._cgroup = cgroup
        self._margin = lease_margin_seconds

    def run_next(self, worker: str, *, deadline_seconds: float) -> ExecutionOutcome | None:
        """Execute the next startable operation, or return ``None`` when idle."""
        claim = self._service.claim(worker, lease_seconds=deadline_seconds + self._margin)
        if claim is None:
            return None
        operation = self._operations.get(claim.tool_id)
        if operation is None:
            failed = self._service.fail(
                claim.operation_id, claim.fence, "no isolated operation is registered"
            )
            return ExecutionOutcome(failed.operation_id, failed.state)
        try:
            result = run_isolated(
                self._domain,
                operation,
                dict(claim.arguments),
                deadline=datetime.now(UTC) + timedelta(seconds=deadline_seconds),
                registry=self._registry,
                import_roots=self._roots,
                cgroup=self._cgroup,
            )
        except WorkerError as error:
            if error.code in _KNOWN_FAILURES:
                reference = self._service.fail(claim.operation_id, claim.fence, error.code)
            else:
                reference = self._service.mark_unknown(claim.operation_id, claim.fence, error.code)
            return ExecutionOutcome(reference.operation_id, reference.state)
        completed = self._service.complete(claim.operation_id, claim.fence, result.output)
        return ExecutionOutcome(completed.operation_id, completed.state)
