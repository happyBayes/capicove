"""Closed offline startup configuration and immutable model identities."""

import json
from pathlib import Path
from typing import Any

import pytest

from capicove.model import ModelCapability, ModelConfiguration, ModelProfile, load_configuration


def configuration(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "max_instances": 1,
        "profiles": {
            "dense": {
                "provider": "bge-m3",
                "capability": "text_embedding",
                "root": str(root),
                "manifest_sha256": "a" * 64,
                "device": "cpu",
                "options": {"max_tokens": 512, "batch_size": 1},
            }
        },
        "bindings": {"graph.embedding": "dense"},
    }


def test_profile_is_frozen_and_identity_covers_effective_configuration(tmp_path: Path) -> None:
    raw = configuration(tmp_path)
    config = ModelConfiguration.from_mapping(raw)
    profile = config.profiles["dense"]
    assert isinstance(profile, ModelProfile)
    identity = profile.identity
    raw["profiles"]["dense"]["options"]["max_tokens"] = 128
    assert profile.options["max_tokens"] == 512
    assert ModelConfiguration.from_mapping(raw).profiles["dense"].identity != identity
    with pytest.raises(TypeError):
        profile.options["max_tokens"] = 128  # type: ignore[index]
    same = ModelProfile(
        "other-name",
        "bge-m3",
        ModelCapability.TEXT_EMBEDDING,
        tmp_path,
        "a" * 64,
        "cpu",
        {"batch_size": 1, "max_tokens": 512},
    )
    assert same.identity == identity


@pytest.mark.parametrize(
    "change",
    [
        {"network": True},
        {"schema_version": True},
        {"schema_version": 3},
        {"max_instances": 0},
        {"max_instances": True},
        {"bindings": {"graph.embedding": "missing"}},
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelConfiguration.from_mapping({**configuration(tmp_path), **change})


@pytest.mark.parametrize(
    "change",
    [
        {"root": "BAAI/bge-m3"},
        {"provider": "package.module:Class"},
        {"manifest_sha256": "bad"},
        {"capability": "arbitrary"},
        {"device": "auto"},
        {"network": True},
        {"options": {"nested": {"a": 1}}},
        {"options": {"bad": float("nan")}},
    ],
)
def test_invalid_profile_is_rejected(tmp_path: Path, change: dict[str, Any]) -> None:
    raw = configuration(tmp_path)
    raw["profiles"]["dense"].update(change)
    with pytest.raises(ValueError):
        ModelConfiguration.from_mapping(raw)


def test_config_loader_reads_json_and_toml_and_rejects_links(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(configuration(tmp_path)))
    assert load_configuration(path).bindings["graph.embedding"] == "dense"
    toml = tmp_path / "models.toml"
    toml.write_text(
        'schema_version = 1\n[profiles.dense]\nprovider = "bge-m3"\n'
        'capability = "text_embedding"\ndevice = "cpu"\n'
        f'root = "{tmp_path}"\nmanifest_sha256 = "{"a" * 64}"\n'
        '[bindings]\n"graph.embedding" = "dense"\n'
    )
    assert load_configuration(toml).max_instances == 1
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        load_configuration(link)


def test_profiles_reject_symlinked_roots(tmp_path: Path) -> None:
    link = tmp_path / "linked"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        ModelConfiguration.from_mapping(configuration(link))


def test_duplicate_json_keys_and_oversized_config_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="duplicate"):
        load_configuration(path)
    path.write_bytes(b" " * 1_048_577)
    with pytest.raises(ValueError, match="size"):
        load_configuration(path)
