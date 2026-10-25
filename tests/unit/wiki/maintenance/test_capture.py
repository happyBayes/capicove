"""Independent capture oracles for experimental maintenance workspaces."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from capicove.wiki import WikiService
from capicove.wiki.maintenance.service import MaintenanceService
from tests.wiki_maintenance_support.oracle import snapshot


def experimental_workspace(root: Path) -> MaintenanceService:
    """Prepare a disposable v3 fixture without enabling production migration."""
    WikiService(root).initialize(created_on=date(2026, 10, 22))
    wiki = root / "wiki"
    manifest = wiki / "wiki.toml"
    manifest.write_text(manifest.read_text().replace("schema_version = 1", "schema_version = 3"))
    for name in ("maintenance", "representations"):
        (wiki / name).mkdir(exist_ok=True)
    (wiki / "maintenance" / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": "workspace:test",
                "wiki_revision": 0,
                "publication_id": None,
            }
        )
    )
    (wiki / "maintenance" / "source-policy.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_revision": 0,
                "withdrawn": [],
            }
        )
    )
    return MaintenanceService(root, source_roots=(root,))


def test_capture_preserves_bytes_and_deduplicates_without_pages(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    raw = b"\xef\xbb\xbf# Same\r\nAlpha evidence.\r"
    source.write_bytes(raw)
    before = snapshot(tmp_path / "wiki" / "pages")
    captured = service.capture(source)
    digest = hashlib.sha256(raw).hexdigest()
    assert captured.source_id == "source:sha256:" + digest
    assert captured.created is True
    assert (tmp_path / "wiki" / "raw" / (digest + ".md")).read_bytes() == raw
    assert service.read_markdown(captured.source_id).text == "# Same\nAlpha evidence.\n"
    assert snapshot(tmp_path / "wiki" / "pages") == before
    other = tmp_path / "renamed.md"
    other.write_bytes(raw)
    saved = snapshot(tmp_path / "wiki")
    assert service.capture(other).created is False
    assert snapshot(tmp_path / "wiki") == saved
    other.write_text("# Same\nDifferent bytes")
    assert service.capture(other).source_id != captured.source_id


@pytest.mark.parametrize("raw", [b"\xff", b"x" * 2_000_001, b"x" * (16 * 1024 * 1024 + 1)])
def test_invalid_capture_has_zero_writes(tmp_path: Path, raw: bytes) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "bad.md"
    source.write_bytes(raw)
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError):
        service.capture(source)
    assert snapshot(tmp_path / "wiki") == before


def test_capture_rejects_symlink_and_unauthorized_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    service = experimental_workspace(root)
    outside = tmp_path / "outside.md"
    outside.write_text("Private")
    link = root / "link.md"
    link.symlink_to(outside)
    before = snapshot(root / "wiki")
    for source in (outside, link, root / ".." / "outside.md"):
        with pytest.raises(ValueError):
            service.capture(source)
    assert snapshot(root / "wiki") == before


def test_destination_collision_is_not_overwritten(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Evidence")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (tmp_path / "wiki" / "raw" / (digest + ".md")).write_bytes(b"tampered")
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match=r"digest|collision"):
        service.capture(source)
    assert snapshot(tmp_path / "wiki") == before


def test_v1_is_not_silently_migrated(tmp_path: Path) -> None:
    WikiService(tmp_path).initialize(created_on=date(2026, 10, 22))
    source = tmp_path / "a.md"
    source.write_text("Evidence")
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="schema"):
        MaintenanceService(tmp_path, source_roots=(tmp_path,)).capture(source)
    assert snapshot(tmp_path / "wiki") == before


def test_source_mutation_during_read_has_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    service = experimental_workspace(tmp_path)
    source = tmp_path / "race.md"
    source.write_text("original")
    before = snapshot(tmp_path / "wiki")
    original = os.fstat
    calls = 0

    def changed(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            source.write_text("mutated after read")
        return original(descriptor)

    monkeypatch.setattr(os, "fstat", changed)
    with pytest.raises(ValueError, match="changed"):
        service.capture(source)
    assert snapshot(tmp_path / "wiki") == before


def test_representation_collision_precedes_any_raw_write(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.representations import markdown_representation

    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_bytes(b"Source")
    representation = markdown_representation(b"Source")
    target = (
        tmp_path / "wiki" / "representations" / (representation.representation_sha256 + ".json")
    )
    target.write_bytes(b"bad")
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError):
        service.capture(source)
    assert snapshot(tmp_path / "wiki") == before


def test_withdrawal_and_representation_tamper_fail_closed(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Evidence")
    result = service.capture(source)
    policy = tmp_path / "wiki" / "maintenance" / "source-policy.json"
    policy.write_text(
        json.dumps({"schema_version": 1, "policy_revision": 1, "withdrawn": [result.source_id]})
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="withdrawn"):
        service.read_markdown(result.source_id)
    with pytest.raises(ValueError, match="withdrawn"):
        service.capture(source)
    assert snapshot(tmp_path / "wiki") == before


def test_duplicate_policy_keys_are_rejected(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Evidence")
    policy = tmp_path / "wiki" / "maintenance" / "source-policy.json"
    policy.write_bytes(
        b'{"schema_version":1,"policy_revision":0,"withdrawn":["source:sha256:'
        + b"a" * 64
        + b'"],"withdrawn":[]}'
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="duplicate"):
        service.capture(source)
    assert snapshot(tmp_path / "wiki") == before
