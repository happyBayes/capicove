"""Pinned BGE-M3 acquisition keeps pickle weights out of runtime bundles."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from capicove.model.bge_m3 import load_download_lock, verify_upstream


def test_official_dense_download_is_pinned_and_excludes_other_models() -> None:
    lock = load_download_lock()
    assert lock["repository"] == "BAAI/bge-m3"
    assert len(lock["revision"]) == 40
    assert "pytorch_model.bin" in lock["files"]
    assert "colbert_linear.pt" not in lock["files"]
    assert "sentencepiece.bpe.model" not in lock["files"]


def test_modified_upstream_is_rejected_before_conversion(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"edited")
    lock = {
        "files": {"weights.bin": {"size_bytes": 6, "sha256": hashlib.sha256(b"source").hexdigest()}}
    }
    with pytest.raises(ValueError, match="digest"):
        verify_upstream(tmp_path, lock)


def test_safetensors_profile_requires_operator_trusted_manifest(tmp_path: Path) -> None:
    from capicove.retrieval.bge import load_bge_m3_profile

    (tmp_path / "bge-m3.receipt.json").write_text(json.dumps({"asset_manifest_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="manifest"):
        load_bge_m3_profile(tmp_path / "bge-m3", expected_manifest_sha256="b" * 64)


def test_local_m3_loader_selects_the_bge_provider(tmp_path: Path) -> None:
    from capicove.retrieval.bge import load_bge_m3_profile
    from tests.unit.retrieval.test_bge_adapter import local_profile

    root = tmp_path / "bge-m3"
    root.mkdir()
    authored = local_profile(root)
    (tmp_path / "bge-m3.receipt.json").write_text(
        json.dumps(
            {
                "asset_manifest_sha256": authored.assets.manifest_sha256,
                "asset_files": authored.assets.files,
            }
        )
    )
    profile = load_bge_m3_profile(root, expected_manifest_sha256=authored.assets.manifest_sha256)
    assert profile.space.provider_id == "local-bge"
    assert profile.space.dimension == 1024
    assert profile.model_family == "bge-m3"


def test_acquisition_requires_opt_in_before_filesystem_effects(tmp_path: Path) -> None:
    from capicove.model.bge_m3 import provision_bge_m3

    destination = tmp_path / "not-created"
    with pytest.raises(PermissionError, match="confirmation"):
        provision_bge_m3(destination, confirmed=False)
    assert not destination.exists()


def test_provisioner_verifies_then_converts_and_publishes_closed_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove.model.bge_m3 as module

    payloads = {"pytorch_model.bin": b"authored weights", "config.json": b"{}"}
    lock = {
        "repository": "BAAI/bge-m3",
        "revision": "a" * 40,
        "license": "MIT",
        "files": {
            name: {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        },
    }
    monkeypatch.setattr(module, "load_download_lock", lambda: lock)
    monkeypatch.setattr("capicove.model.bge_m3.shutil.which", lambda name: "/authorized/hf")

    def download(command: list[str], *, check: bool) -> None:
        assert check is True
        assert command[1:3] == ["download", "BAAI/bge-m3"]
        assert command[command.index("--revision") + 1] == "a" * 40
        root = Path(command[command.index("--local-dir") + 1])
        root.mkdir()
        for name, payload in payloads.items():
            (root / name).write_bytes(payload)

    class Tensor:
        pass

    def load(path: Path, **kwargs: Any) -> dict[str, Tensor]:
        assert path.read_bytes() == payloads["pytorch_model.bin"]
        assert kwargs == {"map_location": "cpu", "weights_only": True, "mmap": True}
        return {"weight": Tensor()}

    def save(weights: Any, path: str, *, metadata: Any) -> None:
        assert metadata == {"format": "pt"}
        Path(path).write_bytes(b"authored safetensors")

    modules = {
        "torch": SimpleNamespace(load=load, Tensor=Tensor, __version__="qualified-test"),
        "safetensors": SimpleNamespace(__version__="qualified-test"),
        "safetensors.torch": SimpleNamespace(save_file=save),
    }
    monkeypatch.setattr("capicove.model.bge_m3.subprocess.run", download)
    monkeypatch.setattr("capicove.model.bge_m3.importlib.import_module", lambda name: modules[name])
    receipt = module.provision_bge_m3(tmp_path, confirmed=True)
    assert receipt["conversion"]["weights_only"] is True
    assert sorted(path.name for path in (tmp_path / "bge-m3").iterdir()) == [
        "config.json",
        "model.safetensors",
    ]
    assert json.loads((tmp_path / "bge-m3.receipt.json").read_text()) == json.loads(
        json.dumps(receipt)
    )
    with pytest.raises(ValueError, match="already exists"):
        module.provision_bge_m3(tmp_path, confirmed=True)
