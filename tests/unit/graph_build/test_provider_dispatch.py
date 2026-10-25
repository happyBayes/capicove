"""Shared provider admission covers independent runners and live run controls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import Any

import pytest

from capicove.graph_build.extraction import extraction_request
from capicove.graph_build.inference import InferenceJob, InferenceRunControl
from tests.unit.graph_build.test_extraction import build_profile, unit
from tests.unit.graph_build.test_inference import SCOPE, SOURCES, Harness, raw, validate


def request_for(limit: int = 2, offset: int = 100) -> Any:
    profile = build_profile()
    return extraction_request(
        unit(offset),
        replace(
            profile,
            generation_profile_id=f"dispatch-{offset}",
            inference=replace(
                profile.inference, concurrent_units=limit, context_window_tokens=30000
            ),
        ),
    )


@pytest.mark.parametrize("batched", [False, True])
def test_independent_runners_share_one_provider_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batched: bool
) -> None:
    runners = [Harness(tmp_path / str(i), []) for i in range(3)]
    lock = Lock()
    start = Barrier(3)
    active = peak = 0

    def generate(request: Any, *, control: Any) -> Any:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        Event().wait(0.05)
        with lock:
            active -= 1
        return raw()

    for h in runners:
        monkeypatch.setattr(h.generator, "generate", generate)

    def invoke(i: int) -> list[Any]:
        h = runners[i]
        start.wait(timeout=5)
        if batched:
            return list(
                h.runner.run_many(
                    InferenceJob(SCOPE, request_for(offset=100 + j), SOURCES, validate)
                    for j in range(3)
                )
            )
        return [h.run(generation_id=str(i))]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(invoke, range(3)))
    assert all(row.status == "succeeded" for batch in results for row in batch)
    assert peak == 2


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_waiting_for_provider_obeys_run_control_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    owner, waiter = (Harness(tmp_path / name, []) for name in ("owner", "waiter"))
    owner.request = request_for(limit=1)
    entered, release = Event(), Event()

    def generate(request: Any, *, control: Any) -> Any:
        entered.set()
        assert release.wait(5)
        return raw()

    monkeypatch.setattr(owner.generator, "generate", generate)
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        if checks > 8:
            if stop == "cancel":
                return True
            monkeypatch.setattr(waiter.runner, "_now", lambda: 100000.0)
        return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.run)
        try:
            assert entered.wait(5)
            queued = pool.submit(waiter.run, run_control=InferenceRunControl(cancelled=cancelled))
            result = queued.result(timeout=3)
            assert result.status == ("cancelled" if stop == "cancel" else "failed")
            assert result.call_count == 0
            assert waiter.generator.requests == []
        finally:
            release.set()
        assert first.result(timeout=3).status == "succeeded"


def test_provider_exception_releases_slot(tmp_path: Path) -> None:
    h = Harness(tmp_path, [RuntimeError("provider failed"), raw()])
    h.request = request_for(limit=1)
    with pytest.raises(RuntimeError, match="provider failed"):
        h.run(generation_id="failed")
    assert h.run(generation_id="next").status == "succeeded"
