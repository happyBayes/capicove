"""Source owns canonical body, authorized spans and content-addressed identity."""

from pathlib import Path

import pytest

from capicove.contracts.evidence import EvidenceRef, EvidenceSpan
from capicove.source import EvidenceUnavailable, SQLiteTextEvidenceStore


def test_evidence_identity_scope_and_exact_span(tmp_path: Path) -> None:
    store = SQLiteTextEvidenceStore(tmp_path / "source.sqlite3", tmp_path / "privacy")
    a = store.accept_text(workspace="w", scope="s", text="Alice lives in Paris")
    b = store.accept_text(workspace="w", scope="s", text="Alice lives in Paris")
    assert a == b and len(a.evidence_id) == len("evidence:") + 64
    assert store.read_text(workspace="w", scope="s", span=EvidenceSpan(a, 15, 20)) == "Paris"
    with pytest.raises(EvidenceUnavailable):
        store.read_text(workspace="w", scope="other", span=EvidenceSpan(a, 0, 20))
    with pytest.raises(EvidenceUnavailable):
        store.read_text(workspace="w", scope="s", span=EvidenceSpan(a, 0, 100))


def test_source_erasure_denies_restored_payload(tmp_path: Path) -> None:
    import sqlite3

    store = SQLiteTextEvidenceStore(tmp_path / "source.sqlite3", tmp_path / "privacy")
    ref = store.accept_text(workspace="w", scope="s", text="private")
    with (
        sqlite3.connect(tmp_path / "source.sqlite3") as db,
        sqlite3.connect(tmp_path / "backup") as backup,
    ):
        db.backup(backup)
    store.erase_scope(workspace="w", scope="s")
    with (
        sqlite3.connect(tmp_path / "backup") as backup,
        sqlite3.connect(tmp_path / "source.sqlite3") as db,
    ):
        backup.backup(db)
    with pytest.raises(EvidenceUnavailable):
        store.read_text(workspace="w", scope="s", span=EvidenceSpan(ref, 0, 7))


def test_missing_privacy_authority_fails_closed_after_restart(tmp_path: Path) -> None:
    store = SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy")
    store.accept_text(workspace="w", scope="s", text="private")
    (tmp_path / "privacy" / "erased-scopes.sqlite3").unlink()
    with pytest.raises(EvidenceUnavailable, match="privacy_authority_unavailable"):
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy")


def test_erasure_between_evidence_accept_and_memory_commit_blocks_derivations(
    tmp_path: Path,
) -> None:
    import sqlite3
    from datetime import UTC, datetime

    from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService

    class ErasingStore(SQLiteTextEvidenceStore):
        def accept_text(self, *, workspace: str, scope: str, text: str) -> EvidenceRef:
            ref = super().accept_text(workspace=workspace, scope=scope, text=text)
            self.erase_scope(workspace=workspace, scope=scope)
            return ref

    s = TemporalKnowledgeService(
        SQLiteMemoryRepository(tmp_path / "memory.db"),
        ErasingStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(),
    )
    with pytest.raises(EvidenceUnavailable):
        s.remember_episode(
            workspace="w",
            scope="s",
            content="must not derive",
            event_time=datetime.now(UTC),
            idempotency_key="r",
        )
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0


def test_batch_read_checks_all_spans_and_never_returns_partial_data(tmp_path: Path) -> None:
    store = SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy")
    ref = store.accept_text(workspace="w", scope="s", text="Alice lives in Paris")
    assert store.read_spans(
        workspace="w", scope="s", spans=(EvidenceSpan(ref, 0, 5), EvidenceSpan(ref, 15, 20))
    ) == ("Alice", "Paris")
    with pytest.raises(EvidenceUnavailable):
        store.read_spans(
            workspace="w",
            scope="s",
            spans=(EvidenceSpan(ref, 0, 5), EvidenceSpan(EvidenceRef("a" * 64), 0, 1)),
        )


def test_source_authorization_is_enforced_even_without_tool_layer(tmp_path: Path) -> None:
    owner = SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy")
    ref = owner.accept_text(workspace="w", scope="private", text="owner only")
    limited = SQLiteTextEvidenceStore(
        tmp_path / "source.db",
        tmp_path / "privacy",
        authorize=lambda workspace, scope: workspace == "w" and scope == "public",
    )
    with pytest.raises(EvidenceUnavailable, match="permission_denied"):
        limited.read_text(workspace="w", scope="private", span=EvidenceSpan(ref, 0, 5))
    with pytest.raises(EvidenceUnavailable, match="permission_denied"):
        limited.erase_scope(workspace="w", scope="private")
