"""Acceptance receipts require a live, runner-owned observation channel."""

import importlib
from types import ModuleType
from typing import Any

import pytest


def api() -> ModuleType:
    from tests.remote_support import receipts

    assert hasattr(receipts, "RunObserver"), "live observation harness is missing"
    return importlib.import_module("tests.remote_support.receipts")


def test_receipt_cannot_be_replaced_by_expected_json() -> None:
    module = api()
    with module.RunObserver("case-a", "synthetic-cpu") as observer:
        with pytest.raises(ValueError):
            observer.verify({"status": "passed", "provider_calls": 1})
        with pytest.raises(ValueError):
            observer.finish(required_events={"provider-request", "worker-exit"})


def test_live_worker_request_and_exit_are_independently_witnessed() -> None:
    module = api()
    with module.RunObserver("case-a", "synthetic-cpu") as observer:
        outcome = observer.run_worker()
        assert outcome.returncode == 0
        receipt = observer.finish(required_events={"provider-request", "worker-exit"})
        observer.verify(receipt)
        assert receipt["provider_calls"] == 1
        changed = dict(receipt, provider_calls=20)
        with pytest.raises(ValueError):
            observer.verify(changed)
    with module.RunObserver("case-a", "synthetic-cpu") as other:
        with pytest.raises(ValueError):
            other.verify(receipt)


def test_worker_failure_is_not_a_success_receipt() -> None:
    module = api()
    with module.RunObserver("case-a", "synthetic-cpu") as observer:
        assert observer.run_worker(fail=True).returncode != 0
        with pytest.raises(ValueError):
            observer.finish(required_events={"worker-exit"})


def test_observer_cannot_reopen_a_completed_run() -> None:
    module = api()
    observer = module.RunObserver("case-a", "synthetic-cpu")
    with observer:
        observer.run_worker()
    with pytest.raises(ValueError):
        with observer:
            pass


def test_observation_identity_cannot_be_relabelled() -> None:
    observer = api().RunObserver("case-a", "synthetic-cpu")
    with pytest.raises(AttributeError):
        observer.test_id = "case-b"
    with pytest.raises(AttributeError):
        observer.profile = "production"


def test_concurrent_worker_admission_is_reserved_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()
    original = subprocess.Popen

    def paused_spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        entered.set()
        assert release.wait(timeout=3)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", paused_spawn)
    with api().RunObserver("case-a", "synthetic-cpu") as observer:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(observer.run_worker)
            assert entered.wait(timeout=3)
            try:
                with pytest.raises(ValueError, match="one worker"):
                    observer.run_worker()
            finally:
                release.set()
            assert first.result(timeout=5).returncode == 0
