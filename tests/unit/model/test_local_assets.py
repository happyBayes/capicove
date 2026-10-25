"""Verified local model assets never trigger model acquisition."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from capicove.model.local_assets import (
    LocalModelAssets,
    require_local_model_directory,
)


def assets_at(root: Path, contents: dict[str, bytes]) -> LocalModelAssets:
    """Provision synthetic authored files with independently calculated digests."""
    root.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    files = tuple(
        sorted((name, hashlib.sha256(data).hexdigest()) for name, data in contents.items())
    )
    digest = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return LocalModelAssets(root, files, digest)


def test_missing_model_does_not_become_a_hub_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_unavailable"):
        require_local_model_directory(tmp_path / "absent")


@pytest.mark.parametrize("root", [Path("relative"), Path("org/model")])
def test_only_explicit_absolute_roots_are_accepted(root: Path) -> None:
    with pytest.raises(ValueError, match="model_unavailable"):
        require_local_model_directory(root)


def test_closed_manifest_verifies_every_file(tmp_path: Path) -> None:
    assets = assets_at(tmp_path, {"config.json": b"{}", "nested/weights.safetensors": b"weights"})
    assets.verify()
    (tmp_path / "nested/weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset_digest_mismatch"):
        assets.verify()


def test_missing_and_unlisted_files_fail_closed(tmp_path: Path) -> None:
    assets = assets_at(tmp_path, {"config.json": b"{}"})
    (tmp_path / "unlisted.json").write_text("{}")
    with pytest.raises(ValueError, match="asset_inventory_mismatch"):
        assets.verify()
    (tmp_path / "unlisted.json").unlink()
    (tmp_path / "config.json").unlink()
    with pytest.raises(ValueError, match="asset_inventory_mismatch"):
        assets.verify()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "./a", "a//b", "a\\b", ""])
def test_manifest_rejects_noncanonical_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid_asset_manifest"):
        LocalModelAssets(tmp_path, ((name, "a" * 64),), "b" * 64).verify()


def test_manifest_digest_and_file_hash_format_are_checked(tmp_path: Path) -> None:
    assets = assets_at(tmp_path, {"config.json": b"{}"})
    with pytest.raises(ValueError, match="manifest_digest_mismatch"):
        replace(assets, manifest_sha256="a" * 64).verify()
    with pytest.raises(ValueError, match="invalid_asset_manifest"):
        replace(assets, files=(("config.json", "A" * 64),)).verify()
    with pytest.raises(ValueError, match="invalid_asset_manifest"):
        replace(assets, files=assets.files + assets.files).verify()


@pytest.mark.parametrize("kind", ["root", "file", "directory", "hardlink", "fifo"])
def test_linked_or_nonregular_assets_are_rejected(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "assets"
    assets = assets_at(root, {"config.json": b"{}"})
    if kind == "root":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        assets = replace(assets, root=alias)
    elif kind == "directory":
        (root / "nested").symlink_to(tmp_path, target_is_directory=True)
    else:
        (root / "config.json").unlink()
        target = tmp_path / "target.json"
        target.write_text("{}")
        if kind == "file":
            (root / "config.json").symlink_to(target)
        elif kind == "hardlink":
            os.link(target, root / "config.json")
        else:
            os.mkfifo(root / "config.json")
    with pytest.raises(ValueError, match="unsafe_asset_path"):
        assets.verify()


@pytest.mark.parametrize(
    "name,content",
    [
        ("config.json", b'{"auto_map":{"AutoModel":"custom.Model"}}'),
        ("tokenizer_config.json", b'{"trust_remote_code":true}'),
        ("config.json", b'{"text_config":{"auto_map":{"AutoConfig":"custom.Config"}}}'),
        ("custom.py", b"print('unsafe')"),
        ("pytorch_model.bin", b"pickle"),
        ("model.pt", b"pickle"),
        ("adapter_config.json", b'{"base_model_name_or_path":"/unverified/model"}'),
        ("tokenizer_config.json", b'{"tokenizer_file":"/unverified/tokenizer.json"}'),
        ("tokenizer_config.json", b'{"vocab_file":"../vocab.json"}'),
        ("config.json", b'{"configuration_files":["unlisted.json"]}'),
        ("model.safetensors.index.json", b'{"weight_map":{"a":"../escape.safetensors"}}'),
        ("model.safetensors.index.json", b'{"weight_map":{"a":"pytorch_model.bin"}}'),
    ],
)
def test_remote_code_and_unsafe_weights_are_rejected(
    tmp_path: Path, name: str, content: bytes
) -> None:
    assets = assets_at(tmp_path, {name: content})
    with pytest.raises(ValueError, match="unsafe_model_assets"):
        assets.verify()


def test_manifest_can_verify_safe_sharded_weights(tmp_path: Path) -> None:
    assets = assets_at(
        tmp_path,
        {
            "model.safetensors.index.json": b'{"weight_map":{"layer":"part.safetensors"}}',
            "part.safetensors": b"safe synthetic weights",
        },
    )
    assets.verify()


def test_token_vocabulary_keys_are_data_not_loader_configuration(tmp_path: Path) -> None:
    assets = assets_at(
        tmp_path,
        {
            "tokenizer.json": b'{"model":{"vocab":{"auto_map":42,"tokenizer_file":7}}}',
            "vocab.json": b'{"auto_map":42,"tokenizer_file":7}',
        },
    )
    assets.verify()


@pytest.mark.parametrize("error_type", [PermissionError, FileNotFoundError, OSError])
def test_descendant_enumeration_failure_cannot_hide_unlisted_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[OSError]
) -> None:
    assets = assets_at(tmp_path, {"config.json": b"{}"})
    hidden = tmp_path / "unreadable"
    hidden.mkdir()
    (hidden / "unlisted.json").write_text("{}")
    original_scandir = os.scandir

    def failing_scandir(path: Any) -> Any:
        if Path(path) == hidden:
            raise error_type("private directory metadata")
        return original_scandir(path)

    # Undo the patch before teardown: pytest's tmp_path cleanup also uses scandir.
    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", failing_scandir)
        with pytest.raises(ValueError, match=r"^unsafe_model_assets$") as error:
            assets.verify()
    assert error.value.__suppress_context__


def test_walk_entry_type_failure_cannot_skip_a_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = assets_at(tmp_path, {"config.json": b"{}"})
    hidden = tmp_path / "unreadable"
    hidden.mkdir()
    (hidden / "unlisted.json").write_text("{}")
    original_scandir = os.scandir

    def unreadable_type() -> bool:
        raise PermissionError("private directory metadata")

    class FailingEntryScandir:
        def __init__(self, path: Any) -> None:
            self.entries = original_scandir(path)

        def __enter__(self) -> "FailingEntryScandir":
            return self

        def __exit__(self, *args: Any) -> None:
            self.entries.close()

        def __iter__(self) -> "FailingEntryScandir":
            return self

        def __next__(self) -> Any:
            entry = next(self.entries)
            if Path(entry.path) == hidden:
                return SimpleNamespace(name=entry.name, path=entry.path, is_dir=unreadable_type)
            return entry

    monkeypatch.setattr(os, "scandir", FailingEntryScandir)
    with pytest.raises(ValueError, match=r"^unsafe_model_assets$"):
        assets.verify()
