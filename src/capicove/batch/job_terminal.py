"""Content-free terminal receipts for durable batch jobs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from capicove.batch.checkpoint import BatchCheckpoint
from capicove.batch.models import BatchPlan, BatchState


_JOB_ID = re.compile(r"job:[0-9a-f]{24}\Z")
_PLAN_ID = re.compile(r"plan:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL_STATES = frozenset(
    {BatchState.SUCCEEDED, BatchState.FAILED, BatchState.SKIPPED, BatchState.SOURCE_CHANGED}
)


@dataclass(frozen=True, slots=True)
class TerminalReceipt:
    """Summarize checkpoint identity without copying item results or source data."""

    job_id: str
    plan_id: str
    total_items: int
    completed_items: int
    failed_items: int
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if _JOB_ID.fullmatch(self.job_id) is None or _PLAN_ID.fullmatch(self.plan_id) is None:
            raise ValueError("invalid terminal receipt identity")
        if (
            type(self.total_items) is not int
            or type(self.completed_items) is not int
            or type(self.failed_items) is not int
            or not 0 <= self.failed_items <= self.completed_items <= self.total_items <= 1_000_000
        ):
            raise ValueError("invalid terminal receipt counts")
        if _SHA256.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("invalid terminal checkpoint digest")


def build_terminal_receipt(
    job_id: str,
    plan: BatchPlan,
    checkpoints: tuple[BatchCheckpoint, ...],
) -> TerminalReceipt:
    """Require one terminal checkpoint per planned item and hash their states."""
    expected = {
        planned.item.item_id: (planned.cache_key, planned.item.sha256) for planned in plan.items
    }
    if len(checkpoints) != len(expected):
        raise ValueError("terminal checkpoints are incomplete")
    observed: dict[str, BatchCheckpoint] = {}
    for checkpoint in checkpoints:
        if (
            checkpoint.item_id in observed
            or checkpoint.plan_id != plan.plan_id
            or expected.get(checkpoint.item_id) != (checkpoint.cache_key, checkpoint.source_sha256)
            or checkpoint.state not in _TERMINAL_STATES
        ):
            raise ValueError("terminal checkpoints are incomplete or changed")
        observed[checkpoint.item_id] = checkpoint
    items = tuple(
        (
            item_id,
            checkpoint.state.value,
            checkpoint.attempts,
            checkpoint.cache_key,
            checkpoint.source_sha256,
            checkpoint.failure.code if checkpoint.failure else None,
        )
        for item_id, checkpoint in sorted(observed.items())
    )
    digest = hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()
    failed = sum(
        checkpoint.state in {BatchState.FAILED, BatchState.SOURCE_CHANGED}
        for checkpoint in checkpoints
    )
    return TerminalReceipt(
        job_id=job_id,
        plan_id=plan.plan_id,
        total_items=len(plan.items),
        completed_items=len(checkpoints),
        failed_items=failed,
        checkpoint_sha256=digest,
    )


def encode_terminal_receipt(receipt: TerminalReceipt) -> tuple[str, str]:
    """Encode a closed canonical receipt and its integrity digest."""
    encoded = _canonical_json(
        {
            "schema_version": 1,
            "job_id": receipt.job_id,
            "plan_id": receipt.plan_id,
            "total_items": receipt.total_items,
            "completed_items": receipt.completed_items,
            "failed_items": receipt.failed_items,
            "checkpoint_sha256": receipt.checkpoint_sha256,
        }
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def decode_terminal_receipt(encoded: str) -> TerminalReceipt:
    """Reject noncanonical, changed, or unreviewed terminal receipt fields."""
    try:
        encoded_size = len(encoded.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("invalid terminal receipt") from error
    if encoded_size > 2048:
        raise ValueError("terminal receipt exceeds size limit")
    try:
        body = json.loads(encoded)
        if not isinstance(body, dict) or set(body) != {
            "schema_version",
            "job_id",
            "plan_id",
            "total_items",
            "completed_items",
            "failed_items",
            "checkpoint_sha256",
        }:
            raise ValueError("invalid terminal receipt fields")
        if body["schema_version"] != 1:
            raise ValueError("unsupported terminal receipt version")
        receipt = TerminalReceipt(
            job_id=body["job_id"],
            plan_id=body["plan_id"],
            total_items=body["total_items"],
            completed_items=body["completed_items"],
            failed_items=body["failed_items"],
            checkpoint_sha256=body["checkpoint_sha256"],
        )
        if encode_terminal_receipt(receipt)[0] != encoded:
            raise ValueError("noncanonical terminal receipt")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid terminal receipt") from error
    return receipt


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
