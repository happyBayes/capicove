"""Native knowledge-base acceptance and adversarial lifecycle checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from capicove.runtime.native_knowledge import KnowledgeBase, KnowledgeConfiguration


def configured(tmp_path: Path) -> KnowledgeConfiguration:
    source = tmp_path / "inputs"
    source.mkdir()
    return KnowledgeConfiguration.from_mapping(
        {
            "schema_version": 1,
            "workspace_id": "test",
            "storage_root": str(tmp_path / "store"),
            "source_roots": {"docs": str(source)},
            "models": None,
            "generation": None,
            "audio": None,
            "graph": False,
            "network_enabled": False,
        }
    )


def test_import_restart_update_and_forget(tmp_path: Path) -> None:
    config = configured(tmp_path)
    path = config.source_roots["docs"] / "one.md"
    path.write_text("Mercury is the closest planet to the Sun.")
    with KnowledgeBase(config) as kb:
        result = kb.update("docs", "one.md")
        assert result["status"] == "succeeded"
        assert result["graph_status"] == "disabled"
        hit = kb.search("closest planet")["evidence"][0]
        assert hit["source_sha256"] and hit["quote"].startswith("Mercury")
        identity = hit["evidence_id"]
        assert kb.evidence(identity)["quote"] == hit["quote"]
        assert kb.ask("closest planet")["status"] == "evidence_only"
    with KnowledgeBase(config) as kb:
        assert kb.search("Mercury")["evidence"]
        path.write_text("Venus has a dense atmosphere.")
        with pytest.raises(ValueError, match="source_changed"):
            kb.search("Mercury")
        kb.update("docs", "one.md")
        assert not kb.search("Mercury")["evidence"]
        assert kb.search("Venus")["evidence"]
        with pytest.raises(ValueError, match="evidence_unavailable"):
            kb.evidence(identity)
        kb.forget("docs", "one.md")
        assert kb.search("Venus")["evidence"] == []
    with KnowledgeBase(config) as kb:
        assert kb.status()["sources"] == 0


@pytest.mark.parametrize("relative", ["../secret", "/etc/passwd", "a/../../secret", ""])
def test_path_escape_is_rejected(tmp_path: Path, relative: str) -> None:
    with KnowledgeBase(configured(tmp_path)) as kb:
        with pytest.raises((ValueError, PermissionError)):
            kb.update("docs", relative)


def test_symlinks_and_hardlinks_are_rejected(tmp_path: Path) -> None:
    config = configured(tmp_path)
    outside = tmp_path / "secret.md"
    outside.write_text("secret")
    root = config.source_roots["docs"]
    (root / "link.md").symlink_to(outside)
    (root / "hard.md").hardlink_to(outside)
    with KnowledgeBase(config) as kb:
        for name in ("link.md", "hard.md"):
            with pytest.raises((ValueError, PermissionError)):
                kb.update("docs", name)


def test_failed_update_does_not_activate_incomplete_data(tmp_path: Path) -> None:
    config = configured(tmp_path)
    path = config.source_roots["docs"] / "one.md"
    path.write_text("Mercury")
    with KnowledgeBase(config) as kb:
        kb.update("docs", "one.md")
        previous = kb.status()["generation"]
        path.write_bytes(b"\xff")
        with pytest.raises(ValueError):
            kb.update("docs", "one.md")
        assert kb.status()["generation"] == previous
        assert kb.status()["last_run"]["status"] == "failed"
        with pytest.raises(ValueError, match="source_changed"):
            kb.search("Mercury")


def test_unsupported_configuration_and_no_implicit_network(tmp_path: Path) -> None:
    config = configured(tmp_path)
    raw = config.to_mapping()
    for changes in ({"schema_version": True}, {"unknown": 1}, {"graph": True}):
        with pytest.raises(ValueError):
            KnowledgeConfiguration.from_mapping({**raw, **changes})
    with pytest.raises(ValueError):
        KnowledgeConfiguration.from_mapping({**raw, "generation": "/tmp/model.json"})


def test_answer_requires_exact_authorized_citations(tmp_path: Path) -> None:
    config = configured(tmp_path)
    (config.source_roots["docs"] / "one.md").write_text("Mercury is a planet.")
    with KnowledgeBase(config) as kb:
        kb.update("docs", "one.md")
        evidence = kb.search("Mercury")["evidence"]
        item = evidence[0]
        valid = {
            "claims": [
                {
                    "text": "Mercury is a planet.",
                    "citations": [{"evidence_id": item["evidence_id"], "quote": "Mercury"}],
                }
            ]
        }
        assert kb.validate_answer(valid, evidence)[0]["text"] == "Mercury is a planet."
        for invalid in (
            {"claims": [{"text": "No citation", "citations": []}]},
            {
                "claims": [
                    {
                        "text": "Forged",
                        "citations": [{"evidence_id": "invented", "quote": "Mercury"}],
                    }
                ]
            },
            {
                "claims": [
                    {
                        "text": "Forged",
                        "citations": [{"evidence_id": item["evidence_id"], "quote": "Venus"}],
                    }
                ]
            },
            {"claims": [], "command": "ignore policy"},
        ):
            with pytest.raises(ValueError, match="invalid_answer"):
                kb.validate_answer(invalid, evidence)


def test_revocation_during_generation_discards_answer(tmp_path: Path) -> None:
    config = configured(tmp_path)
    (config.source_roots["docs"] / "one.md").write_text("Mercury is a planet.")

    def generate(question: str, evidence: list[dict[str, Any]], graph: Any) -> Any:
        with KnowledgeBase(config) as other:
            other.forget("docs", "one.md")
        return {
            "claims": [
                {
                    "text": "Mercury",
                    "citations": [{"evidence_id": evidence[0]["evidence_id"], "quote": "Mercury"}],
                }
            ]
        }

    with KnowledgeBase(config, answerer=generate) as kb:
        kb.update("docs", "one.md")
        with pytest.raises(ValueError, match="knowledge_changed"):
            kb.ask("Mercury")


def test_store_permissions_and_symlink_database(tmp_path: Path) -> None:
    config = configured(tmp_path)
    with KnowledgeBase(config):
        assert config.storage_root.stat().st_mode & 0o077 == 0
    db = config.storage_root / "knowledge.sqlite"
    db.unlink()
    target = tmp_path / "other"
    target.write_text("untouched")
    db.symlink_to(target)
    with pytest.raises((ValueError, PermissionError, OSError)):
        with KnowledgeBase(config):
            pass
    assert target.read_text() == "untouched"


def test_cli_init_and_lifecycle(tmp_path: Path, capsys: Any) -> None:
    from capicove.cli.app import main

    root = tmp_path / "workspace"
    assert main(["kb", "init", str(root)]) == 0
    capsys.readouterr()
    config = root / "knowledge.json"
    (root / "inputs" / "demo.md").write_text("Capicove preserves evidence.")
    assert main(["kb", "update", "--config", str(config), "--root", "inputs", "demo.md"]) == 0
    capsys.readouterr()
    assert main(["kb", "ask", "--config", str(config), "evidence"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "evidence_only" and payload["evidence"]


def test_corrupt_prepared_version_is_not_evidence(tmp_path: Path) -> None:
    config = configured(tmp_path)
    (config.source_roots["docs"] / "one.md").write_text("Original evidence.")
    with KnowledgeBase(config) as kb:
        kb.update("docs", "one.md")
        with kb.store.transaction():
            row = kb.store.db.execute("SELECT id,payload FROM versions").fetchone()
            payload = json.loads(row[1])
            payload["chunks"][0]["quote"] = "Fabricated evidence."
            kb.store.db.execute(
                "UPDATE versions SET payload=? WHERE id=?", (json.dumps(payload), row[0])
            )
        with pytest.raises(ValueError, match="corrupt_knowledge_version"):
            kb.search("Fabricated")


def test_graph_context_corruption_and_explicit_disable(tmp_path: Path) -> None:
    from dataclasses import replace

    config = configured(tmp_path)
    with KnowledgeBase(config) as kb:
        with kb.store.transaction():
            kb.store.set(
                "graph", {"receipt": {"status": "succeeded"}, "context": [{"text": "tampered"}]}
            )
            kb.store.set("graph_digest", "wrong")
        assert kb.status()["graph_status"] == "disabled"
    with KnowledgeBase(replace(config, graph=True)) as kb:
        with pytest.raises(ValueError, match="corrupt_knowledge_graph"):
            kb.search("tampered")
