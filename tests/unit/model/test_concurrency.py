"""Leases serialize inference and keep waiting callers safe during eviction."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from tests.unit.model.test_factory import factory


def test_shared_instance_is_serialized_and_waiting_lease_is_retained(tmp_path: Path) -> None:
    runtime, made = factory(tmp_path)
    waiting, entered = Event(), Event()

    def work() -> None:
        waiting.set()
        with runtime.text_encoder("graph.embedding"):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with runtime.text_encoder("graph.embedding"):
            future = executor.submit(work)
            assert waiting.wait(5)
            assert not entered.wait(0.05)
        future.result(timeout=5)
    assert entered.is_set() and len(made) == 1
    runtime.close()
