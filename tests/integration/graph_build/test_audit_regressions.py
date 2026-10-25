"""Behavior-level audit regressions with authored data and no model downloads."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import Any

import pytest

from tests.graph_build_fixtures import SCOPE
from tests.integration.graph_build.test_build_service import authored
from tests.integration.graph_build.test_service_retrieval import configure_service_retrieval
from tests.unit.graph_build.test_extraction import response
from tests.unit.graph_build.test_inference import Harness, raw


def test_direct_calls_share_provider_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = Harness(tmp_path, [])
    start = Barrier(3)
    release = Event()
    lock = Lock()
    active = peak = 0

    def generate(request: Any, *, control: Any) -> Any:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 3:
                release.set()
        release.wait(0.25)
        with lock:
            active -= 1
        return raw()

    monkeypatch.setattr(h.generator, "generate", generate)

    def invoke(i: int) -> Any:
        start.wait(timeout=5)
        return h.run(generation_id=f"concurrent-{i}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(invoke, range(3)))
    assert all(result.status == "succeeded" for result in results)
    assert peak <= h.request.inference_policy.concurrent_units


@pytest.mark.slow
def test_service_publishes_after_bounded_report_repair(tmp_path: Path) -> None:
    """The inference unit matrix owns exhaustion; this checks successful assembly."""
    service, request, _, _ = authored(tmp_path)
    generator = service.inference_runner._generator
    original = generator.generate
    calls: list[str] = []

    def malformed_once(req: Any, *, control: Any) -> Any:
        calls.append(req.role)
        if req.role == "report" and calls.count("report") == 1:
            return raw(b"{")
        return original(req, control=control)

    generator.generate = malformed_once
    result = service.build(request)
    # A successful leaf report is followed by a distinct overview request.
    assert calls.count("report") == 3
    assert result.status == "succeeded"
    assert result.graph_ref is not None
    assert result.completeness == "complete"


@pytest.mark.slow
def test_service_semantic_failure_does_not_repair(tmp_path: Path) -> None:
    service, request, _, _ = authored(tmp_path)
    generator = service.inference_runner._generator
    original = generator.generate
    calls: list[str] = []

    def semantic(req: Any, *, control: Any) -> Any:
        calls.append(req.role)
        if req.role == "user":
            payload = response()
            payload["entities"].append(dict(payload["entities"][0]))
            return raw(json.dumps(payload).encode())
        return original(req, control=control)

    generator.generate = semantic
    result = service.build(request)
    assert calls.count("user") == 1
    assert result.status == "partial"
    assert "extraction_failed" in result.issue_codes


@pytest.mark.slow
@pytest.mark.parametrize(
    ("stop", "boundary"),
    [("cancel", "write"), ("deadline", "encode")],
)
def test_cancellation_stops_retrieval_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str, boundary: str
) -> None:
    from tests.unit.retrieval.test_text_embedding import Encoder

    service, request, catalog, _ = authored(tmp_path)
    encoder = Encoder()
    configured, _, index = configure_service_retrieval(tmp_path, service, catalog, encoder=encoder)
    target = index if boundary == "write" else encoder
    method = "add" if boundary == "write" else "encode"
    original = getattr(target, method)
    count = 0

    def cancel_after_first_call(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == 1:
            if stop == "cancel":
                with sqlite3.connect(service.runs.path) as connection:
                    run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
                service.cancel(run_id, SCOPE)
            else:
                service.monotonic_clock = lambda: float("inf")
        return result

    monkeypatch.setattr(target, method, cancel_after_first_call)
    result = service.build(request)
    assert result.status == ("cancelled" if stop == "cancel" else "failed")
    assert count == 1, index.events
    if boundary == "write":
        assert index.events[-1] == "write"
        state = configured.bindings.attempt(result.run_id, SCOPE)
        assert state is not None and state.status == "failed"
    else:
        assert index.events == []
    assert result.retrieval_binding_ref is None
