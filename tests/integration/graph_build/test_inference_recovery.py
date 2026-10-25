"""Recover abandoned inference claims without claiming exactly-once execution."""

from collections.abc import Callable
from pathlib import Path

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
from capicove.graph_build.receipts import InferenceReceipt
from tests.unit.graph_build.test_inference import GOOD, SCOPE, SOURCES, Harness, raw


def test_crash_after_artifact_before_receipt_retains_bytes_and_recovers(tmp_path: Path) -> None:
    harness = Harness(tmp_path, [raw(), raw()])
    captured: list[InferenceReceipt] = []

    def crash(
        scope: GraphScope,
        key: str,
        attempt_id: str,
        receipt: InferenceReceipt,
        *,
        success_guard: Callable[[], None] | None = None,
    ) -> None:
        assert harness.artifacts.read(scope, receipt.result_artifact_ref) == GOOD
        captured.append(receipt)
        raise RuntimeError("simulated crash before receipt commit")

    harness.receipts.complete = crash  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated crash"):
        harness.run()
    pending = captured[0]
    assert harness.receipts.get(SCOPE, pending.key) is None
    assert harness.artifacts.read(SCOPE, pending.result_artifact_ref) == GOOD
    harness.clock.now += 4000
    harness.artifacts = FilesystemProtectedArtifacts(
        tmp_path / "artifacts", harness.gate, evidence_refs=SOURCES
    )
    harness.receipts = SqliteReceiptStore(
        tmp_path / "ledger" / "receipts.sqlite",
        harness.gate,
        harness.artifacts,
        clock=harness.clock,
    )
    # Restart has a new monotonic build budget, but the persisted claim has expired.
    harness.runner = type(harness.runner)(
        generator=harness.generator,
        tokenizer=harness.tokenizer,
        gate=harness.gate,
        artifact_factory=lambda refs: harness.artifacts,
        receipts=harness.receipts,
        clock=harness.clock,
        wall_clock=harness.clock,
        sleeper=harness.clock.sleep,
        cancelled=lambda: False,
    )
    recovered = harness.run()
    assert recovered.status == "succeeded"
    assert recovered.result_artifact_ref == pending.result_artifact_ref
    assert len(harness.generator.requests) == 2
    assert harness.run(mode="replay") == recovered


def test_crash_after_provider_before_artifact_reexecutes_after_lease_expiry(tmp_path: Path) -> None:
    harness = Harness(tmp_path, [raw(), raw()])
    put = harness.artifacts.put
    payloads: list[bytes] = []

    def crash(scope: GraphScope, payload: bytes) -> str:
        payloads.append(payload)
        raise RuntimeError("simulated crash before artifact commit")

    harness.artifacts.put = crash  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated crash"):
        harness.run()
    assert payloads == [GOOD]
    assert len(harness.generator.requests) == 1
    harness.clock.now += 4000
    harness.artifacts.put = put  # type: ignore[method-assign]
    harness.runner = type(harness.runner)(
        generator=harness.generator,
        tokenizer=harness.tokenizer,
        gate=harness.gate,
        artifact_factory=lambda refs: harness.artifacts,
        receipts=harness.receipts,
        clock=harness.clock,
        wall_clock=harness.clock,
        sleeper=harness.clock.sleep,
        cancelled=lambda: False,
    )
    recovered = harness.run()
    assert recovered.status == "succeeded"
    assert harness.artifacts.read(SCOPE, recovered.result_artifact_ref) == GOOD
    assert len(harness.generator.requests) == 2


@pytest.mark.parametrize("stop,expected", [("cancel", "cancelled"), ("deadline", "failed")])
def test_success_is_not_published_if_verification_stops_the_run(
    tmp_path: Path, stop: str, expected: str
) -> None:
    harness = Harness(tmp_path, [raw()])
    read = harness.artifacts.read
    verified: list[str] = []

    def stop_during_verification(scope: GraphScope, artifact_ref: str) -> bytes:
        payload = read(scope, artifact_ref)
        verified.append(artifact_ref)
        if stop == "cancel":
            harness.cancelled = True
        else:
            harness.clock.now = 1001.0
        return payload

    harness.artifacts.read = stop_during_verification  # type: ignore[method-assign]
    result = harness.run(deadline=1000.5)
    assert verified
    assert result.status == expected
    assert result.issue_codes == (("cancelled",) if stop == "cancel" else ("deadline_exceeded",))
    assert harness.receipts.get(SCOPE, result.key) == result
    assert read(SCOPE, result.result_artifact_ref) == GOOD
    assert len(harness.generator.requests) == 1
