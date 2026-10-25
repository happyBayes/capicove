"""Local image host startup rejects unsafe files before runtime composition."""

import json
import os
from pathlib import Path

import pytest

from capicove.runtime.image_search_host import load_image_search_host, read_json_object


@pytest.mark.parametrize("payload", ["[]", '{"a":1,"a":2}', '{"nested":{"a":1,"a":2}}'])
def test_json_requires_unambiguous_object(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "input.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        read_json_object(path, maximum_bytes=100)


def test_json_rejects_oversize_symlink_fifo_and_writable_config(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"a":1}')
    with pytest.raises(ValueError, match="size limit"):
        read_json_object(path, maximum_bytes=2)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        read_json_object(link, maximum_bytes=100)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        read_json_object(fifo, maximum_bytes=100)
    path.chmod(0o666)
    with pytest.raises(ValueError, match="permissions"):
        read_json_object(path, maximum_bytes=100, trusted=True)


def _config(root: Path) -> dict[str, object]:
    root = root.resolve()
    for name in ("generations", "qualifications", "models", "photos"):
        (root / name).mkdir()
    (root / "lifecycle.sqlite3").touch()
    for name, payload in (("cursor.key", b"c" * 32), ("qualification.key", b"q" * 32)):
        (root / name).write_bytes(payload)
        (root / name).chmod(0o600)
    return {
        "schema_version": 1,
        "workspace_id": "test",
        "actor_id": "owner",
        "generation_root": str(root / "generations"),
        "qualification_root": str(root / "qualifications"),
        "lifecycle_database_path": str(root / "lifecycle.sqlite3"),
        "model_root": str(root / "models"),
        "provisioning_manifest": "provisioning.json",
        "provisioning_sha256": "a" * 64,
        "source_roots": {"photos": str(root / "photos")},
        "cursor_key_file": str(root / "cursor.key"),
        "qualification_key_file": str(root / "qualification.key"),
        "permissions": ["retrieval:search"],
        "cpu_threads": 2,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("permissions", []),
        ("permissions", ["retrieval:search", "admin"]),
        ("cpu_threads", True),
        ("cpu_threads", 0),
        ("cpu_threads", 33),
        ("source_roots", {}),
        ("actor_id", " "),
        ("model_root", "relative"),
        ("extra", "untrusted"),
    ],
)
def test_startup_rejects_invalid_closed_config(tmp_path: Path, field: str, value: object) -> None:
    config = _config(tmp_path)
    config[field] = value
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_image_search_host(path)
    assert (tmp_path / "lifecycle.sqlite3").stat().st_size == 0


@pytest.mark.parametrize("case", ["short", "long", "shared", "broad", "missing-storage"])
def test_startup_rejects_signing_key_and_storage_misconfiguration(
    tmp_path: Path, case: str
) -> None:
    config = _config(tmp_path)
    if case == "short":
        (tmp_path / "cursor.key").write_bytes(b"c" * 31)
    elif case == "long":
        (tmp_path / "cursor.key").write_bytes(b"c" * 65)
    elif case == "shared":
        config["cursor_key_file"] = config["qualification_key_file"]
    elif case == "broad":
        (tmp_path / "cursor.key").chmod(0o644)
    else:
        config["generation_root"] = str(tmp_path.resolve() / "missing")
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_image_search_host(path)
    assert not (tmp_path / "missing").exists()
    assert (tmp_path / "lifecycle.sqlite3").stat().st_size == 0
