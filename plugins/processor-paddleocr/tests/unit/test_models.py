from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from capicove_processor_paddleocr.models import (
    PaddleModelFileSpec,
    PaddleModelManifest,
    PaddleModelSpec,
    install_models,
    load_book_balanced_manifest,
    plan_model_installation,
    verify_installed_models,
)
from pytest import MonkeyPatch


def _tiny_manifest() -> PaddleModelManifest:
    models = []
    for key, parameter in (
        ("pp-doclayout-l", "layout_detection_model_dir"),
        ("pp-ocrv5-server-det", "text_detection_model_dir"),
        ("pp-ocrv5-server-rec", "text_recognition_model_dir"),
    ):
        payload = f"{key}\n".encode()
        models.append(
            PaddleModelSpec(
                key,
                f"PaddlePaddle/{key}",
                "a" * 40,
                len(payload),
                parameter,
                (
                    PaddleModelFileSpec(
                        "inference.json",
                        len(payload),
                        hashlib.sha256(payload).hexdigest(),
                    ),
                ),
            )
        )
    return PaddleModelManifest(
        "book-balanced",
        2,
        ("en", "ja", "zh"),
        (("name", "PP-StructureV3"),),
        (("device", "$requested_device"),),
        tuple(models),
        "b" * 64,
    )


def _install_tiny_models(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    *,
    include_hf_cache: bool = False,
) -> PaddleModelManifest:
    manifest = _tiny_manifest()
    monkeypatch.setattr("capicove_processor_paddleocr.models.shutil.which", lambda name: "/bin/hf")

    def fake_run(
        command: tuple[str, ...], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        del check, env
        local_dir = Path(command[command.index("--local-dir") + 1])
        if command[1] == "download":
            model = next(item for item in manifest.models if item.repository == command[2])
            if include_hf_cache:
                (local_dir / ".cache/huggingface").mkdir(parents=True)
                (local_dir / ".cache/huggingface/CACHEDIR.TAG").write_text(
                    "Signature: 8a477f597d28d172789f06886806bc55\n",
                    encoding="utf-8",
                )
            (local_dir / "inference.json").write_bytes(f"{model.key}\n".encode())
        else:
            assert command[1:3] == ("cache", "verify")
            assert "--fail-on-missing-files" in command
            assert "--fail-on-extra-files" in command
            assert not (local_dir / ".cache").exists()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("capicove_processor_paddleocr.models.subprocess.run", fake_run)
    install_models(manifest, tmp_path, confirmed=True)
    return manifest


def test_book_balanced_manifest_locks_three_official_models(tmp_path: Path) -> None:
    manifest = load_book_balanced_manifest()
    plan = plan_model_installation(manifest, tmp_path)

    # Expected public model repository commit SHAs, not credentials.
    assert [(item.key, item.revision) for item in plan.models] == [
        ("pp-doclayout-l", "ca760c3336d922ee63cdbff85003181222e3911c"),  # pragma: allowlist secret
        (
            "pp-ocrv5-server-det",
            "ca867c897ecbca8873081573a802ad70d499cb94",  # pragma: allowlist secret
        ),
        (
            "pp-ocrv5-server-rec",
            "b26c3587fda8da3c8ec0ce357214b4d661ff1558",  # pragma: allowlist secret
        ),
    ]
    assert plan.total_size_bytes > 300_000_000
    assert plan.destination == tmp_path.resolve()
    assert plan.profile_version == 2
    assert plan.supported_languages == ("en", "ja", "zh")
    assert dict(plan.inference_engine) == {
        "name": "PP-StructureV3",
        "paddleocr": ">=3.7,<3.8",
        "paddlepaddle": ">=3.3,<4",
        "paddlex": ">=3.7,<3.8",
        "pypdf": ">=5,<7",
        "pypdfium2": ">=4,<6",
    }
    assert dict(plan.pipeline_configuration)["text_det_limit_side_len"] == 736
    assert all(len(model.files) == 6 for model in manifest.models)
    assert sum(file.size_bytes for model in manifest.models for file in model.files) == 303_710_330


def test_model_installation_requires_explicit_confirmation(tmp_path: Path) -> None:
    manifest = load_book_balanced_manifest()

    with pytest.raises(PermissionError, match="confirmation"):
        install_models(manifest, tmp_path, confirmed=False)


def test_model_installation_ignores_hf_local_metadata_and_removes_it(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    manifest = _install_tiny_models(tmp_path, monkeypatch, include_hf_cache=True)

    for model in manifest.models:
        installed = tmp_path / model.key
        assert (installed / "inference.json").is_file()
        assert not (installed / ".cache").exists()
    verify_installed_models(manifest, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("missing_receipt", "wrong_receipt", "missing_file", "tampered_file", "extra_file"),
)
def test_runtime_model_verification_fails_closed(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    mutation: str,
) -> None:
    manifest = _install_tiny_models(tmp_path, monkeypatch)
    receipt = tmp_path / "book-balanced.receipt.json"
    model_file = tmp_path / manifest.models[0].key / "inference.json"
    if mutation == "missing_receipt":
        receipt.unlink()
    elif mutation == "wrong_receipt":
        value = json.loads(receipt.read_text(encoding="utf-8"))
        value["manifest_sha256"] = "f" * 64
        receipt.write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "missing_file":
        model_file.unlink()
    elif mutation == "tampered_file":
        model_file.write_bytes(b"x" * model_file.stat().st_size)
    else:
        (model_file.parent / "extra.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match=r"receipt|model file"):
        verify_installed_models(manifest, tmp_path)


def test_existing_invalid_models_do_not_receive_a_new_receipt(tmp_path: Path) -> None:
    manifest = load_book_balanced_manifest()
    for model in manifest.models:
        (tmp_path / model.key).mkdir()
    receipt = tmp_path / "book-balanced.receipt.json"
    receipt.write_text('{"previous":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="model file set"):
        install_models(manifest, tmp_path, confirmed=True)
    assert receipt.read_text(encoding="utf-8") == '{"previous":true}\n'


def test_existing_model_symlink_is_rejected(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    manifest = load_book_balanced_manifest()
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / manifest.models[0].key).symlink_to(external, target_is_directory=True)
    for model in manifest.models[1:]:
        (tmp_path / model.key).mkdir()
    monkeypatch.setattr("capicove_processor_paddleocr.models.shutil.which", lambda name: "/bin/hf")

    with pytest.raises(ValueError, match="symbolic link"):
        install_models(manifest, tmp_path, confirmed=True)
    assert not (tmp_path / "book-balanced.receipt.json").exists()
