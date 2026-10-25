"""Persistent epochs fence effects after revocation acknowledgement."""

import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from capicove.runtime.remote_epochs import EpochStore


def store(path: Path) -> EpochStore:
    assert importlib.util.find_spec("capicove.runtime.remote_epochs") is not None
    return EpochStore(path)


@pytest.mark.parametrize("phase", ["start", "send", "commit", "release"])
def test_revoked_epoch_cannot_enter_effect_phase(tmp_path: Path, phase: str) -> None:
    epochs = store(tmp_path / "epochs.sqlite")
    epoch = epochs.current("workspace")
    assert epochs.revoke("workspace") == epoch + 1
    observed = []
    with pytest.raises(PermissionError):
        with epochs.permit("workspace", epoch, phase=phase):
            observed.append("effect")
    assert observed == []
    assert store(tmp_path / "epochs.sqlite").current("workspace") == epoch + 1


def test_acknowledgement_waits_for_existing_permit(tmp_path: Path) -> None:
    epochs = store(tmp_path / "epochs.sqlite")
    initial = epochs.current("workspace")
    entered, release, attempted, acknowledged = (Event() for _ in range(4))

    def effect() -> None:
        with epochs.permit("workspace", initial, phase="release"):
            entered.set()
            assert release.wait(3)

    def revoke() -> int:
        attempted.set()
        result = epochs.revoke("workspace")
        acknowledged.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(effect)
        assert entered.wait(3)
        second = pool.submit(revoke)
        assert attempted.wait(3)
        try:
            assert not acknowledged.wait(0.05)
        finally:
            release.set()
        first.result(3)
        assert second.result(3) == initial + 1
    with pytest.raises(PermissionError):
        with epochs.permit("workspace", initial, phase="release"):
            pytest.fail("stale release after acknowledgement")


def test_concurrent_permits_share_the_fence_and_revocation_still_waits(tmp_path: Path) -> None:
    epochs = store(tmp_path / "epochs.sqlite")
    initial = epochs.current("workspace")
    holders = 8
    entered = [Event() for _ in range(holders)]
    release, acknowledged = Event(), Event()

    def effect(index: int) -> None:
        with epochs.permit("workspace", initial, phase="send"):
            entered[index].set()
            assert release.wait(5)

    def revoke() -> int:
        result = epochs.revoke("workspace")
        acknowledged.set()
        return result

    with ThreadPoolExecutor(max_workers=holders + 1) as pool:
        effects = [pool.submit(effect, index) for index in range(holders)]
        # Every holder enters while the others still hold their permits.
        assert all(event.wait(1.5) for event in entered)
        revocation = pool.submit(revoke)
        try:
            assert not acknowledged.wait(0.2)
        finally:
            release.set()
        for future in effects:
            future.result(5)
        assert revocation.result(10) == initial + 1
    with pytest.raises(PermissionError):
        with epochs.permit("workspace", initial, phase="send"):
            pytest.fail("stale permit after acknowledgement")
