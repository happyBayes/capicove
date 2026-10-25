"""Bounded background review never grants approval authority."""

import threading

from capicove.evolution.background import BackgroundReview


def test_single_flight_and_cooperative_cancel() -> None:
    entered = threading.Event()
    finished = threading.Event()

    def review(cancel: threading.Event) -> None:
        entered.set()
        cancel.wait(1)
        finished.set()

    with BackgroundReview() as worker:
        assert worker.submit(review)
        assert entered.wait(1)
        assert not worker.submit(review)
        assert worker.cancel(timeout=1)
        assert finished.is_set()


def test_failed_callback_is_observable_and_close_rejects_work() -> None:
    import pytest

    def fail(cancel: threading.Event) -> None:
        raise RuntimeError("private text is not exposed")

    worker = BackgroundReview()
    assert worker.cancel()
    assert worker.submit(fail)
    assert worker.cancel(timeout=1)
    assert worker.failed
    with pytest.raises(ValueError, match="timeout"):
        worker.cancel(timeout=10)
    worker.close()
    assert not worker.submit(fail)


def test_cancel_discards_coalesced_followup() -> None:
    entered = threading.Event()
    followup = threading.Event()

    def review(cancel: threading.Event) -> None:
        entered.set()
        cancel.wait(2)

    with BackgroundReview() as worker:
        assert worker.submit(review)
        assert entered.wait(2)
        assert worker.submit(lambda cancel: followup.set(), coalesce=True)
        assert worker.cancel(timeout=2)
        assert not followup.is_set()
