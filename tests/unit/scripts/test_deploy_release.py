"""Delivery plans must reject changed archives, traversal and network requirements."""

import hashlib
import importlib.util
import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest


def module() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "deploy_release.py"
    spec = importlib.util.spec_from_file_location("capicove_deploy_test", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def bundle(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "wheelhouse").mkdir()
    wheel = root / "wheelhouse" / "example-1-py3-none-any.whl"
    wheel.write_bytes(b"authored wheel placeholder")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (root / "requirements.lock").write_text(f"example==1 --hash=sha256:{digest}\n")
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (wheel, root / "requirements.lock")
    }
    manifest = {
        "schema_version": 1,
        "profiles": ["core"],
        "device": "cpu",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sys.platform,
        "files": files,
    }
    (root / "bundle.json").write_text(json.dumps(manifest))
    return root, hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest()


def test_valid_closed_bundle_and_pinned_plan(tmp_path: Path) -> None:
    root, digest = bundle(tmp_path)
    assert module().verify_bundle(root, digest)["profiles"] == ["core"]
    with pytest.raises(ValueError):
        module().verify_bundle(root, "0" * 64)


@pytest.mark.parametrize("attack", ["changed", "extra", "symlink", "traversal", "url"])
def test_reject_unsafe_delivery_before_install(tmp_path: Path, attack: str) -> None:
    root, digest = bundle(tmp_path)
    if attack == "changed":
        (root / "requirements.lock").write_text("changed")
    elif attack == "extra":
        (root / "wheelhouse" / "injected.whl").write_bytes(b"extra")
    elif attack == "symlink":
        file = root / "requirements.lock"
        target = tmp_path / "outside"
        target.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(target)
    else:
        data = json.loads((root / "bundle.json").read_text())
        if attack == "traversal":
            data["files"]["../outside"] = "0" * 64
        else:
            lock = root / "requirements.lock"
            lock.write_text("example @ https://example.invalid/exfil.whl\n")
            data["files"]["requirements.lock"] = hashlib.sha256(lock.read_bytes()).hexdigest()
        (root / "bundle.json").write_text(json.dumps(data))
        digest = hashlib.sha256((root / "bundle.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module().verify_bundle(root, digest)


def test_profiles_are_closed_and_do_not_select_a_model() -> None:
    deployment = module()
    assert deployment.PROFILES["graph-generation"][0] == "capicove"
    with pytest.raises(ValueError):
        deployment.selected_packages(["unknown"])


def test_native_configuration_verifies_assets_before_enabling_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    original = {
        "capabilities": {},
        "permissions": [],
        "enabled_tools": [],
        "network": {"enabled": False, "source_hosts": []},
    }
    host = workspace / "host.json"
    host.write_text(json.dumps(original))
    native = tmp_path / "native.json"
    payload = tmp_path / "asset"
    payload.write_bytes(b"authored native asset")
    files = {
        name: {"path": str(payload), "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
        for name in ("ffmpeg", "ffprobe", "model", "vad_model", "whisper_cli", "whisper_vad_cli")
    }
    native.write_text(json.dumps({"files": files}))
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        module().configure_native_media(workspace, inputs, native, ["audio-asr", "media"])
    assert json.loads(host.read_text()) == original
    payload.write_bytes(b"authored native asset")
    module().configure_native_media(workspace, inputs, native, ["audio-asr", "media"])
    configured = json.loads(host.read_text())
    assert "capicove.audio.transcribe" in configured["enabled_tools"]
    assert "capicove.video.split" in configured["enabled_tools"]
    assert configured["network"]["enabled"] is False


def test_install_uses_private_verified_snapshot_when_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = module()
    root, checksum = bundle(tmp_path)
    destination = tmp_path / "runtime"

    class ReachedInstaller(Exception):
        pass

    def run(command: list[str], working: Path, **options: object) -> str:
        if command[1:3] == ["-m", "venv"]:
            (root / "requirements.lock").write_text("changed after initial verification")
            return ""
        assert command[1:4] == ["-m", "pip", "install"]
        assert str(destination / "installation-bundle" / "wheelhouse") in command
        assert str(root / "wheelhouse") not in command
        raise ReachedInstaller

    monkeypatch.setattr(deployment, "_run", run)
    with pytest.raises(ReachedInstaller):
        deployment.install(Namespace(bundle=root, destination=destination, sha256=checksum))


def test_knowledge_profile_selects_installed_dependencies() -> None:
    assert module().selected_packages(["knowledge"])["capicove"] >= {"knowledge", "mcp", "api"}


@pytest.mark.skipif(sys.platform != "linux", reason="native knowledge requires Linux secure paths")
def test_knowledge_profile_and_real_cli_lifecycle(tmp_path: Path) -> None:
    from contextlib import redirect_stdout
    from io import StringIO

    from capicove.cli.app import main

    deployment = module()

    def run(command: list[str]) -> str:
        output = StringIO()
        with redirect_stdout(output):
            code = main(command[1:])
        assert code == 0, output.getvalue()
        return output.getvalue()

    receipt = deployment.qualify_knowledge("capicove", tmp_path, run)
    assert receipt["status"] == "passed"
    assert receipt["mode"] == "lexical"
    assert receipt["revocation"] == "passed"


def test_knowledge_acceptance_rejects_wrong_installed_result(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="knowledge"):
        module().qualify_knowledge("capicove", tmp_path, lambda command: '{"status":"failed"}')


def test_wheel_identity_uses_top_level_metadata_with_vendored_dependencies(tmp_path: Path) -> None:
    wheel = tmp_path / "example-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example-1.dist-info/METADATA", "Name: example\nVersion: 1\n")
        archive.writestr("example/_vendor/other-2.dist-info/METADATA", "Name: other\nVersion: 2\n")
    assert module().wheel_identity(wheel) == ("example", "1")


def test_wheel_identity_rejects_multiple_top_level_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "example-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("example", "other"):
            archive.writestr(f"{name}-1.dist-info/METADATA", f"Name: {name}\nVersion: 1\n")
    with pytest.raises(ValueError, match="ambiguous"):
        module().wheel_identity(wheel)


def test_spark_preparation_builds_only_the_resolved_pyspark_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = module()
    source = tmp_path / "release"
    source.mkdir()
    checksums = []
    for name in ("capicove", "capicove_execution_spark"):
        wheel = source / f"{name}-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"authored release wheel")
        checksums.append(f"{deployment.digest(wheel)}  {wheel.name}\n")
    (source / "SHA256SUMS").write_text("".join(checksums))
    (source / "release-manifest.json").write_text('[{"version":"0.1.0"}]')

    class ReachedAcquisition(Exception):
        pass

    def run(command: list[str], working: Path) -> str:
        if command[0] == "uv" or command[1:3] == ["-m", "venv"]:
            return ""
        assert command[1:4] == ["-m", "pip", "wheel"]
        assert "--only-binary=:all:" in command
        assert "--no-binary=pyspark" in command
        assert "--no-deps" in command
        assert "--wheel-dir" in command
        raise ReachedAcquisition

    monkeypatch.setattr(deployment, "_run", run)
    with pytest.raises(ReachedAcquisition):
        deployment.prepare(
            Namespace(
                dist=source,
                output=tmp_path / "bundle",
                profile=["spark"],
                device="cpu",
                allow_network=True,
            )
        )
