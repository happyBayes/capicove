"""Pinned reranker install plans require explicit, allowlisted downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import main
from capicove.model.installers import load_model_installer
from capicove.model.local_assets import LocalModelAssets
from capicove.model.reranker_assets import (
    FILES,
    MANIFEST_SHA256,
    MODEL_ID,
    REVISION,
    BgeRerankerInstaller,
    reranker_assets,
)


def test_builtin_installer_has_a_pinned_allowlist(tmp_path: Path) -> None:
    installer = load_model_installer("bge-reranker-v2-m3")
    assert isinstance(installer, BgeRerankerInstaller)
    plan = installer.model_plan("default", tmp_path)
    assert plan.models[0].repository == MODEL_ID
    assert plan.models[0].revision == REVISION
    assert plan.total_size_bytes == 2288173057
    assert plan.manifest_sha256 == MANIFEST_SHA256
    assert "model.safetensors" in {f.path for f in FILES}
    assert not any(f.path.endswith((".bin", ".py")) for f in FILES)
    assert reranker_assets(tmp_path).manifest_sha256 == MANIFEST_SHA256


def test_confirmation_and_profile_are_required(tmp_path: Path) -> None:
    installer = BgeRerankerInstaller()
    with pytest.raises(ValueError, match="confirmation"):
        installer.install_models("default", tmp_path, confirmed=False)
    with pytest.raises(ValueError, match="profile"):
        installer.model_plan("missing", tmp_path)


def test_cli_can_plan_without_any_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "models",
                "install",
                "bge-reranker-v2-m3",
                "--profile",
                "default",
                "--destination",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    assert REVISION in capsys.readouterr().out


def test_installer_downloads_only_pinned_files_then_verifies_before_exposing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.write_text("fake test artifact")
    calls: list[tuple[str, str, dict[str, Any]]] = []
    verifications: list[Path] = []

    def download(repository: str, file: str, **kwargs: Any) -> str:
        calls.append((repository, file, kwargs))
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(LocalModelAssets, "verify", lambda self: verifications.append(self.root))
    destination = tmp_path / "installed"
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    bundle = destination / "bge-reranker-v2-m3"
    assert {p.name for p in bundle.iterdir()} == {f.path for f in FILES}
    assert len(verifications) == 1 and verifications[0] != destination
    assert all(
        repo == MODEL_ID and options == {"revision": REVISION, "token": False}
        for repo, _, options in calls
    )
    assert destination.with_name("installed.receipt.json").is_file()


def test_failed_verification_never_exposes_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.write_text("wrong bytes")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *a, **kw: str(source)),
    )
    destination = tmp_path / "installed"
    with pytest.raises(ValueError):
        BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert not destination.exists()


@pytest.fixture
def tiny_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Install real SHA-verified tiny files without any network or model payload."""
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    from capicove.model import reranker_assets as assets

    source = tmp_path / "upstream"
    source.mkdir()
    files = []
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        payload = b"{}" if name.endswith(".json") else b"synthetic weights"
        (source / name).write_bytes(payload)
        files.append(assets.RerankerFile(name, len(payload), hashlib.sha256(payload).hexdigest()))
    pairs = tuple((file.path, file.sha256) for file in files)
    digest = hashlib.sha256(
        json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(assets, "FILES", tuple(files))
    monkeypatch.setattr(assets, "_PAIRS", pairs)
    monkeypatch.setattr(assets, "MANIFEST_SHA256", digest)
    downloads: list[str] = []

    def download(repository: str, name: str, **kwargs: Any) -> str:
        assert repository == MODEL_ID and kwargs == {"revision": REVISION, "token": False}
        downloads.append(name)
        return str(source / name)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return downloads


@pytest.mark.parametrize("link_kind", ["symbolic", "hard"])
def test_receipt_links_never_overwrite_unrelated_files(
    tmp_path: Path, tiny_model: list[str], link_kind: str
) -> None:
    import os

    destination = tmp_path / "installed"
    receipt = tmp_path / "installed.receipt.json"
    victim = tmp_path / "unrelated.json"
    victim.write_text("original content")
    if link_kind == "symbolic":
        receipt.symlink_to(victim)
    else:
        os.link(victim, receipt)
    with pytest.raises(ValueError, match="receipt"):
        BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert victim.read_text() == "original content"
    assert tiny_model == []


def test_receipt_swap_during_publication_cannot_overwrite_link_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_model: list[str]
) -> None:
    import os

    destination = tmp_path / "installed"
    receipt = destination.with_name(destination.name + ".receipt.json")
    victim = tmp_path / "unrelated.json"
    victim.write_text("original content")
    replace = os.replace

    def swap(source: Any, target: Any, **kwargs: Any) -> None:
        receipt.symlink_to(victim)
        replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", swap)
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert victim.read_text() == "original content"
    assert receipt.is_file() and not receipt.is_symlink()


def test_partial_receipt_write_leaves_previous_receipt_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_model: list[str]
) -> None:
    import json

    destination = tmp_path / "installed"
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    receipt = destination.with_name(destination.name + ".receipt.json")
    before = receipt.read_bytes()

    def fail(value: Any, stream: Any, **kwargs: Any) -> None:
        stream.write('{"partial":')
        raise OSError("simulated write failure")

    with monkeypatch.context() as context:
        context.setattr(json, "dump", fail)
        with pytest.raises(OSError):
            BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert receipt.read_bytes() == before
    assert not list(tmp_path.glob(".capicove-receipt-*"))
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert receipt.read_bytes() == before


def test_install_into_precreated_directory_matches_model_tools(
    tmp_path: Path, tiny_model: list[str]
) -> None:
    from dataclasses import asdict

    from capicove.runtime.model_mcp import _verification

    destination = tmp_path / "installed"
    destination.mkdir(mode=0o700)
    plan = BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    body = {
        "models": [asdict(model) for model in plan.models],
        "plan_digest": "a" * 64,
        "destination": "models",
        "manifest_sha256": plan.manifest_sha256,
    }
    result = _verification(body, destination)
    assert result["complete"] and result["verified_files"] == 3, result
    reranker_assets(destination).verify()
    assert reranker_assets(destination).root == destination / "bge-reranker-v2-m3"


def test_receipt_failure_is_recoverable_without_republishing_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_model: list[str]
) -> None:
    import os

    destination = tmp_path / "installed"
    receipt = destination.with_name(destination.name + ".receipt.json")
    replace = os.replace

    def fail_receipt(source: Any, target: Any, **kwargs: Any) -> None:
        if Path(target).name == receipt.name:
            raise OSError("simulated receipt publication failure")
        replace(source, target, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(os, "replace", fail_receipt)
        with pytest.raises(OSError):
            BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert receipt.is_file()
    reranker_assets(destination).verify()
    receipt.unlink()
    calls = len(tiny_model)
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert len(tiny_model) == calls
    assert receipt.is_file()


def test_legacy_flat_install_migrates_without_downloads(
    tmp_path: Path, tiny_model: list[str]
) -> None:
    from capicove.model import reranker_assets as assets

    destination = tmp_path / "installed"
    destination.mkdir()
    for item in assets.FILES:
        (destination / item.path).write_bytes((tmp_path / "upstream" / item.path).read_bytes())
    reranker_assets(destination).verify()
    BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert tiny_model == []
    assert reranker_assets(destination).root == destination / "bge-reranker-v2-m3"
    reranker_assets(destination).verify()
    assert not (destination / "model.safetensors").exists()
    assert destination.with_name(destination.name + ".receipt.json").is_file()


def test_corrupt_existing_install_is_never_overwritten(
    tmp_path: Path, tiny_model: list[str]
) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    unrelated = destination / "private.txt"
    unrelated.write_text("preserve this")
    with pytest.raises(ValueError):
        BgeRerankerInstaller().install_models("default", destination, confirmed=True)
    assert unrelated.read_text() == "preserve this"
    assert tiny_model == []


def test_configured_model_tools_install_and_verify_the_same_inventory(
    tmp_path: Path, tiny_model: list[str]
) -> None:
    import json

    from tests.unit.runtime.test_model_mcp import ModelSpace

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    space = ModelSpace(workspace, workspace / "host.json")
    space.write()
    selectors = {"plugin": "bge-reranker-v2-m3", "profile": "default", "destination": "models"}
    planned = space.call("plan", selectors)
    assert planned["status"] == "succeeded", planned
    plan = planned["result"]
    assert space.call("verify", selectors)["result"]["missing_files"] == 3
    config = json.loads(space.config.read_text())
    config["profile"] = "admin"
    config["permissions"] = ["model:read", "model:install"]
    config["enabled_tools"] = [
        "capicove.model.plan",
        "capicove.model.verify",
        "capicove.model.install",
    ]
    config["network"] = {"enabled": True, "source_hosts": plan["source_hosts"]}
    config["capabilities"]["model"]["install_allowlist"] = [
        {"plan_digest": plan["plan_digest"], **selectors, "source_hosts": plan["source_hosts"]}
    ]
    space.config.write_text(json.dumps(config))
    result = space.call("install", {"plan_digest": plan["plan_digest"], "destination": "models"})
    assert result["status"] == "succeeded", result
    assert result["result"]["complete"] and result["result"]["verified_files"] == 3
    assert space.call("verify", selectors)["result"]["complete"]
    assert space.call("plan", selectors)["result"]["plan_digest"] == plan["plan_digest"]
    assert len(tiny_model) == 3
    assert (
        space.call("install", {"plan_digest": plan["plan_digest"], "destination": "models"})[
            "status"
        ]
        == "succeeded"
    )
    assert len(tiny_model) == 3
