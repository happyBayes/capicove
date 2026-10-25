"""Tests for runtime-checkable processing ports."""

from __future__ import annotations

from capicove.processing.ports import ExecutionBackend


class _CompleteBackend:
    backend_id = "fake"

    def submit(self, plan: object, context: object) -> object:
        return object()

    def poll(self, reference: object) -> object:
        return object()

    def cancel(self, reference: object) -> None:
        return None


class _BackendWithoutCancel:
    backend_id = "fake"

    def submit(self, plan: object, context: object) -> object:
        return object()

    def poll(self, reference: object) -> object:
        return object()


def test_execution_backend_protocol_accepts_complete_fake() -> None:
    assert isinstance(_CompleteBackend(), ExecutionBackend)


def test_execution_backend_protocol_rejects_missing_cancel() -> None:
    assert not isinstance(_BackendWithoutCancel(), ExecutionBackend)
