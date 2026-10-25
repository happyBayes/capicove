"""Real process interruptions of bounded maintenance transactions."""

import multiprocessing
from pathlib import Path

import pytest

from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.wiki_maintenance_support.faults import BARRIERS, crash_at


def _publish(root: Path, barrier: str) -> None:
    from capicove.wiki.maintenance.transaction import transact

    wiki = root / "wiki"
    transact(
        wiki,
        "a" * 64,
        {"pages/example.md": b"new"},
        state_bytes=b"new state",
        policy_revision=0,
        barrier=lambda name: crash_at(root, name) if name == barrier else None,
    )


@pytest.mark.parametrize("barrier", sorted(BARRIERS))
def test_process_crash_recovery(tmp_path: Path, barrier: str) -> None:
    from capicove.wiki.maintenance.transaction import recover

    experimental_workspace(tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "pages/example.md").write_bytes(b"old")
    process = multiprocessing.get_context("spawn").Process(
        target=_publish, args=(tmp_path, barrier)
    )
    process.start()
    process.join(15)
    assert not process.is_alive()
    assert process.exitcode == 73
    if barrier == "before_journal":
        assert (wiki / "pages/example.md").read_bytes() == b"old"
    else:
        recover(wiki)
        assert (wiki / "pages/example.md").read_bytes() == b"new"
        assert (wiki / "maintenance/state.json").read_bytes() == b"new state"
    assert not (wiki / "_migration/maintenance.pending").exists()


def test_recovery_preserves_manual_edit_and_policy(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.transaction import recover

    experimental_workspace(tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "pages/example.md").write_bytes(b"old")
    process = multiprocessing.get_context("spawn").Process(
        target=_publish, args=(tmp_path, "after_destination")
    )
    process.start()
    process.join(15)
    assert process.exitcode == 73
    (wiki / "pages/example.md").write_bytes(b"manual")
    with pytest.raises(ValueError, match="recovery_conflict"):
        recover(wiki)
    assert (wiki / "pages/example.md").read_bytes() == b"manual"
    assert (wiki / "_migration/maintenance.pending").exists()
    (wiki / "pages/example.md").write_bytes(b"new")
    policy = wiki / "maintenance/source-policy.json"
    policy.write_text('{"schema_version":1,"policy_revision":1,"withdrawn":[]}')
    with pytest.raises(ValueError, match="policy_changed"):
        recover(wiki)
    recover(wiki, abort=True)
    assert (wiki / "pages/example.md").read_bytes() == b"old"
    assert '"policy_revision":1' in policy.read_text()


def _apply_crashing(root: Path, proposal_id: str, review_id: str, barrier: str) -> None:
    import capicove.wiki.maintenance.publication as publication
    from capicove.wiki.maintenance.service import MaintenanceService
    from capicove.wiki.maintenance.transaction import transact_locked

    def interrupted(*args: object, **kwargs: object) -> None:
        kwargs["barrier"] = lambda name: crash_at(root, name) if name == barrier else None
        transact_locked(*args, **kwargs)  # type: ignore[arg-type]

    publication.__dict__["transact_locked"] = interrupted
    MaintenanceService(root).apply(proposal_id, review_id)


@pytest.mark.parametrize("barrier", sorted(BARRIERS))
def test_reviewed_publication_recovers_in_new_process(tmp_path: Path, barrier: str) -> None:
    from tests.integration.wiki.test_maintenance_publication import prepared

    service, proposal, review = prepared(tmp_path)
    process = multiprocessing.get_context("spawn").Process(
        target=_apply_crashing, args=(tmp_path, proposal.proposal_id, review.review_id, barrier)
    )
    process.start()
    process.join(20)
    assert not process.is_alive()
    assert process.exitcode == 73
    if barrier != "before_journal":
        with pytest.raises(ValueError, match="publication_pending"):
            service.answer_task("Plants")
        service.recover()
    receipt = service.apply(proposal.proposal_id, review.review_id)
    assert receipt.wiki_revision == 1
    assert len(service._inventory()) == 1
    assert not (service.wiki / "_migration/maintenance.pending").exists()


def test_transaction_never_writes_through_swapped_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from capicove.wiki.maintenance.transaction import transact

    service = experimental_workspace(tmp_path)
    pages = service.wiki / "pages"
    (pages / "example.md").write_bytes(b"old")
    external = tmp_path / "external"
    external.mkdir()
    victim = external / "example.md"
    victim.write_bytes(b"external")
    identity = pages.stat().st_ino
    original_open = os.open
    swapped = False

    def redirect(
        path: str | bytes | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal swapped
        target_parent = (dir_fd is not None and os.fstat(dir_fd).st_ino == identity) or (
            dir_fd is None and str(path).startswith(str(pages) + "/")
        )
        if flags & os.O_CREAT and target_parent and not swapped:
            pages.rename(service.wiki / "held-pages")
            pages.symlink_to(external, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", redirect)
    with pytest.raises((ValueError, OSError)):
        transact(
            service.wiki,
            "e" * 64,
            {"pages/example.md": b"replacement"},
            state_bytes=b"state",
            policy_revision=0,
        )
    assert swapped
    assert victim.read_bytes() == b"external"
    assert (service.wiki / "_migration/maintenance.pending").exists()


def test_resume_rejects_changed_unchanged_evidence(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.transaction import recover

    experimental_workspace(tmp_path)
    wiki = tmp_path / "wiki"
    (wiki / "pages/example.md").write_bytes(b"old")
    source = wiki / "raw" / ("b" * 64 + ".md")
    source.write_bytes(b"original evidence")
    process = multiprocessing.get_context("spawn").Process(
        target=_publish, args=(tmp_path, "after_destination")
    )
    process.start()
    process.join(15)
    assert process.exitcode == 73
    source.write_bytes(b"manual evidence change")
    with pytest.raises(ValueError, match="recovery_conflict"):
        recover(wiki)
    assert (wiki / "_migration/maintenance.pending").exists()
    recover(wiki, abort=True)
    assert source.read_bytes() == b"manual evidence change"
    assert (wiki / "_migration" / ("maintenance-" + "a" * 64) / "outcome.json").exists()


def test_transaction_accepts_bounded_large_derived_index(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.transaction import transact

    service = experimental_workspace(tmp_path)
    payload = b"x" * (17 * 1024 * 1024)
    transact(
        service.wiki,
        "d" * 64,
        {"_derived/maintenance.sqlite": payload},
        state_bytes=b"state",
        policy_revision=0,
    )
    assert (service.wiki / "_derived/maintenance.sqlite").read_bytes() == payload
