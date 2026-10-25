"""Cross-process publication serialization and explicit storage failures."""

import multiprocessing
from pathlib import Path

import pytest

from tests.integration.wiki.test_maintenance_publication import prepared
from tests.unit.wiki.maintenance.test_capture import experimental_workspace


def _apply(root: Path, proposal: str, review: str) -> None:
    from capicove.wiki.maintenance.service import MaintenanceService

    result = MaintenanceService(root).apply(proposal, review)
    assert result.wiki_revision == 1


def test_two_processes_apply_once(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    processes = [
        multiprocessing.get_context("spawn").Process(
            target=_apply, args=(tmp_path, proposal.proposal_id, review.review_id)
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert not process.is_alive()
        assert process.exitcode == 0
    assert len(tuple((service.wiki / "maintenance/publications").glob("*.json"))) == 1


def test_disk_full_keeps_recoverable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove.wiki.maintenance.transaction as transaction

    service = experimental_workspace(tmp_path)
    before = (service.wiki / "maintenance/state.json").read_bytes()
    original = transaction._replace

    def fail_state(wiki: Path, relative: str, content: bytes | None) -> None:
        if relative == "maintenance/state.json":
            raise OSError(28, "No space left on device")
        original(wiki, relative, content)

    monkeypatch.setattr(transaction, "_replace", fail_state)
    with pytest.raises(OSError):
        transaction.transact(
            service.wiki,
            "b" * 64,
            {"pages/new.md": b"new"},
            state_bytes=b"after",
            policy_revision=0,
        )
    assert (service.wiki / "_migration/maintenance.pending").exists()
    assert (service.wiki / "maintenance/state.json").read_bytes() == before
    monkeypatch.setattr(transaction, "_replace", original)
    service.recover(abort=True)
    assert not (service.wiki / "pages/new.md").exists()
    assert (service.wiki / "maintenance/state.json").read_bytes() == before


def test_stale_lock_file_is_not_a_lease(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    (service.wiki / "_migration/writer.lock").write_bytes(b"stale owner")
    assert service.apply(proposal.proposal_id, review.review_id).wiki_revision == 1
