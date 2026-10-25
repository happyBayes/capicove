"""Prepare a selected, hash-locked wheel bundle and install it offline.

Run ``prepare --help`` on a connected build machine, then ``install --help`` on
the matching target platform. Model weights and credentials are never bundled.
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import wave
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


PROFILES = {
    "core": ("capicove", ""),
    "interfaces": ("capicove", "api,mcp,wiki"),
    "knowledge": ("capicove", "knowledge,mcp,api"),
    "media": ("capicove", "video,video-dedup,video-filter"),
    "audio-asr": ("capicove", ""),
    "semantic-text": ("capicove", "text-embedding"),
    "semantic-vision": ("capicove", "video-filter-models,model-serving"),
    "graph-generation": ("capicove", "graph-remote,graph-community,graph-data"),
    "local-generation": ("capicove", "graph-models,graph-community,graph-data"),
    "reverse-image": ("capicove", "retrieval-cpu"),
    "background-removal": ("capicove", "image-rembg"),
    "video-ocr": ("capicove", "video-filter-ocr"),
    "video-tags": ("capicove", "video-filter-tags"),
    "ray": ("capicove", "image-ray"),
    "ocr-book": ("capicove-processor-paddleocr", ""),
    "elasticsearch": ("capicove-retriever-elasticsearch", ""),
    "opensearch": ("capicove-retriever-opensearch", ""),
    "faiss": ("capicove-retriever-faiss", ""),
    "spark": ("capicove-execution-spark", ""),
}
_LOCK_LINE = re.compile(r"[A-Za-z0-9_.-]+==[A-Za-z0-9.!+_-]+ --hash=sha256:[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_packages(profiles: list[str]) -> dict[str, set[str]]:
    """Resolve closed profile names to package extras, without model selection."""
    packages: dict[str, set[str]] = {"capicove": set()}
    if not profiles or set(profiles) - set(PROFILES):
        raise ValueError("select one or more documented profiles")
    for profile in profiles:
        name, extras = PROFILES[profile]
        packages.setdefault(name, set()).update(filter(None, extras.split(",")))
    return packages


def qualify_knowledge(
    executable: str, root: Path, run: Callable[[list[str]], str]
) -> dict[str, str]:
    """Exercise installed evidence, restart, replacement and revocation without models."""
    workspace = root / "knowledge"
    result = json.loads(run([executable, "kb", "init", str(workspace)]))
    if result.get("status") != "succeeded":
        raise RuntimeError("installed knowledge initialization failed")
    config = str(workspace / "knowledge.json")
    source = workspace / "inputs" / "acceptance.md"
    source.write_text("Mercury is the closest planet to the Sun.\n", encoding="utf-8")

    def query(command: str, *arguments: str) -> dict[str, Any]:
        return json.loads(run([executable, "kb", command, "--config", config, *arguments]))

    result = query("update", "--root", "inputs", source.name)
    if result.get("activated") is not True:
        raise RuntimeError("installed knowledge import failed")
    answer = query("ask", "closest planet")
    if answer.get("status") != "evidence_only" or not answer.get("evidence"):
        raise RuntimeError("installed knowledge evidence-only answer failed")
    item = answer["evidence"][0]
    resolved = query("evidence", item["evidence_id"])
    if (
        resolved.get("quote") != source.read_text()
        or resolved.get("locator", {}).get("kind") != "text"
    ):
        raise RuntimeError("installed knowledge citation resolution failed")
    source.write_text("Venus has a dense atmosphere.\n", encoding="utf-8")
    if query("update", "--root", "inputs", source.name).get("activated") is not True:
        raise RuntimeError("installed knowledge replacement failed")
    changed = query("search", "Venus").get("evidence", [])
    if not changed or changed[0]["evidence_id"] == item["evidence_id"]:
        raise RuntimeError("installed knowledge replacement retained stale evidence")
    catalog = json.loads(
        run([executable, "mcp", "catalog", "--config", str(workspace / "host.json")])
    )
    expected = {f"capicove.knowledge.{name}" for name in ("status", "search", "ask", "evidence")}
    if {tool["tool_id"] for tool in catalog.get("tools", [])} != expected:
        raise RuntimeError("installed knowledge MCP catalog failed")
    query("forget", "--root", "inputs", source.name)
    if query("ask", "Venus").get("status") != "no_evidence" or query("status").get("sources") != 0:
        raise RuntimeError("installed knowledge revocation failed")
    return {"status": "passed", "mode": "lexical", "revocation": "passed", "config": config}


def _environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PIP_", "UV_", "CAPICOVE_"))
        and key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_PRELOAD", "LD_LIBRARY_PATH"}
    }
    environment.update(
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        RAY_USAGE_STATS_ENABLED="0",
    )
    return environment


def _run(
    command: list[str], root: Path, *, timeout: int = 1800, credential_env: str | None = None
) -> str:
    environment = _environment()
    if credential_env and credential_env in os.environ:
        environment[credential_env] = os.environ[credential_env]
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        # Never echo arbitrary dependency/server output or environment credentials.
        step = " ".join([Path(command[0]).name, *command[1:3]])
        raise RuntimeError(f"deployment step failed ({step}, exit {result.returncode})")
    return result.stdout


def _private_new_directory(path: Path) -> None:
    if not path.is_absolute() or path.resolve() != path or not path.parent.is_dir():
        raise ValueError("destination requires an existing unlinked absolute parent")
    path.mkdir(mode=0o700)


def _scripts(environment: Path) -> tuple[str, str]:
    directory = environment / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    return str(directory / ("python" + suffix)), str(directory / ("capicove" + suffix))


def verify_bundle(root: Path, expected: str) -> dict[str, Any]:
    """Verify the closed inventory and inert offline requirements before execution."""
    if root.resolve() != root.absolute() or not root.is_dir():
        raise ValueError("bundle must be an unlinked local directory")
    manifest_path = root / "bundle.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 1_048_576:
        raise ValueError("unsafe bundle manifest")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(manifest_path) != expected:
        raise ValueError("bundle manifest digest mismatch")
    data = json.loads(manifest_path.read_text())
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get("files"), dict)
        or not 1 <= len(data["files"]) <= 2048
        or data.get("python") != f"{sys.version_info.major}.{sys.version_info.minor}"
        or data.get("platform") != sys.platform
    ):
        raise ValueError("bundle schema or target platform mismatch")
    selected_packages(data["profiles"])
    actual = set()
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("bundle links are not allowed")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("bundle requires regular unlinked files")
        actual.add(path.relative_to(root).as_posix())
    if actual != set(data["files"]) | {"bundle.json"}:
        raise ValueError("bundle inventory mismatch")
    for name, checksum in data["files"].items():
        if (
            not isinstance(name, str)
            or "\\" in name
            or Path(name).is_absolute()
            or any(part in {".", ".."} for part in name.split("/"))
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or digest(root / name) != checksum
        ):
            raise ValueError("bundle file digest or path mismatch")
    lines = (root / "requirements.lock").read_text().splitlines()
    if not lines or any(not _LOCK_LINE.fullmatch(line) for line in lines):
        raise ValueError("requirements must contain only exact versions and SHA-256 hashes")
    return data


def configure_native_media(
    workspace: Path,
    inputs: Path,
    configuration: Path,
    profiles: list[str],
) -> dict[str, Any]:
    """Generate existing host fields from explicitly provisioned, hash-pinned assets."""
    native = json.loads(configuration.read_text())
    files = native["files"]
    required = {"ffmpeg", "ffprobe"}
    if "audio-asr" in profiles:
        required.update({"whisper_cli", "whisper_vad_cli", "model", "vad_model"})
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("native configuration is missing required files")
    for name in required | ({"speech"} if "speech" in files else set()):
        record = files[name]
        path = Path(record["path"])
        if (
            not path.is_absolute()
            or path.resolve() != path
            or not path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            or digest(path) != record["sha256"]
        ):
            raise ValueError("native asset integrity check failed")
    host_path = workspace / "host.json"
    host = json.loads(host_path.read_text())

    def owned(name: str) -> str:
        path = workspace / name
        path.mkdir(mode=0o700)
        return str(path)

    permissions, tools = set(host["permissions"]), set(host["enabled_tools"])
    if "audio-asr" in profiles:
        host["capabilities"]["audio"] = {
            "schema_version": 2,
            "read_roots": [str(inputs)],
            "staging_root": owned("audio-staging"),
            "evidence_root": owned("audio-evidence"),
            "output_root": owned("audio-output"),
            **{
                name: files[name]["path"]
                for name in ("ffmpeg", "ffprobe", "whisper_cli", "whisper_vad_cli")
            },
            **{name: files[name] for name in ("model", "vad_model")},
            "max_input_bytes": 1_073_741_824,
            "max_duration_seconds": 3600,
        }
        permissions.update({"artifact:write", "audio:read", "audio:write", "subprocess:execute"})
        tools.update({"capicove.audio.import", "capicove.audio.transcribe", "capicove.audio.clip"})
    if "media" in profiles:
        host["capabilities"]["video"] = {
            "schema_version": 1,
            "read_roots": [str(inputs)],
            "staging_root": owned("video-staging"),
            "output_root": owned("video-output"),
            "backend": "ffmpeg",
            "ffmpeg": files["ffmpeg"]["path"],
            "ffprobe": files["ffprobe"]["path"],
            "max_input_bytes": 1_073_741_824,
        }
        permissions.update({"artifact:write", "video:read", "subprocess:execute"})
        tools.update(
            {"capicove.video.split", "capicove.video.filter", "capicove.video.deduplicate"}
        )
    host["permissions"], host["enabled_tools"] = sorted(permissions), sorted(tools)
    host_path.write_text(json.dumps(host, indent=2, sort_keys=True) + "\n")
    return native


def _qualify_media(
    capicove: str,
    destination: Path,
    native: dict[str, Any],
    profiles: list[str],
) -> dict[str, str]:
    inputs, config = destination / "inputs", str(destination / "workspace" / "host.json")
    result = {}
    if "audio-asr" in profiles:
        speech = inputs / "acceptance.wav"
        if "speech" in native["files"]:
            shutil.copyfile(native["files"]["speech"]["path"], speech)
        else:
            with wave.open(str(speech), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\0\0" * 16000 * 3)
        response = json.loads(
            _run(
                [
                    capicove,
                    "audio",
                    "transcribe",
                    "--runtime-config",
                    config,
                    "--input",
                    str(speech),
                    "--language",
                    "en",
                    "--output",
                    "json",
                ],
                destination,
                timeout=300,
            )
        )
        if response.get("status") != "succeeded":
            raise RuntimeError("installed native transcription failed")
        result["audio-asr"] = "qualified"
    if "media" in profiles:
        video = inputs / "acceptance.mp4"
        _run(
            [
                native["files"]["ffmpeg"]["path"],
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=160x120:r=10:d=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            destination,
            timeout=30,
        )
        response = json.loads(
            _run(
                [
                    capicove,
                    "video",
                    "split",
                    "--runtime-config",
                    config,
                    "--input",
                    str(video),
                    "--strategy",
                    "duration",
                    "--split-duration",
                    "1",
                    "--output",
                    "json",
                ],
                destination,
            )
        )
        if response.get("status") != "succeeded":
            raise RuntimeError("installed native video splitting failed")
        result["media"] = "qualified_split"
    return result


def wheel_identity(wheel: Path) -> tuple[str, str]:
    """Read the distribution's metadata, excluding nested vendored distributions."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = [
            name
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise ValueError("wheel metadata is ambiguous")
        record = email.message_from_bytes(archive.read(metadata[0]))
    name, version = record["Name"], record["Version"]
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("wheel identity is missing")
    return re.sub(r"[-_.]+", "-", name).lower(), version


def prepare(args: argparse.Namespace) -> None:
    """Resolve once with explicit acquisition; produce portable offline wheel locks."""
    if not args.allow_network:
        raise ValueError("prepare requires --allow-network for dependency acquisition")
    packages = selected_packages(args.profile)
    source, output = args.dist.absolute(), args.output.absolute()
    release = json.loads((source / "release-manifest.json").read_text())
    if not isinstance(release, list) or not release:
        raise ValueError("release manifest must contain inspected distributions")
    versions = {entry["version"] for entry in release}
    if len(versions) != 1:
        raise ValueError("release distribution versions must agree")
    release_version = versions.pop()
    # The release builder's SHA256SUMS is the authority for selected source wheels.
    checksums = {}
    for line in (source / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        checksums[name.lstrip("*")] = checksum
    selected = []
    for name, extras in packages.items():
        wheels = list(source.glob(name.replace("-", "_") + "-*-py3-none-any.whl"))
        if len(wheels) != 1 or checksums.get(wheels[0].name) != digest(wheels[0]):
            raise ValueError("release wheel is missing, ambiguous or changed")
        selected.append(str(wheels[0]) + ("[" + ",".join(sorted(extras)) + "]" if extras else ""))
    _private_new_directory(output)
    wheelhouse = output / "wheelhouse"
    wheelhouse.mkdir()
    with tempfile.TemporaryDirectory(prefix="capicove-delivery-build-") as temporary:
        work = Path(temporary)
        inputs = work / "requirements.in"
        inputs.write_text("\n".join(selected) + "\n")
        resolved = work / "resolved.txt"
        _run(
            [
                "uv",
                "pip",
                "compile",
                str(inputs),
                "--output-file",
                str(resolved),
                "--torch-backend",
                args.device,
                "--python",
                sys.executable,
                "--default-index",
                "https://pypi.org/simple",
                "--no-header",
                "--no-annotate",
            ],
            work,
        )
        _run([sys.executable, "-m", "venv", str(work / "builder")], work)
        python, _ = _scripts(work / "builder")
        build_spark = "spark" in args.profile
        command = [
            python,
            "-m",
            "pip",
            "wheel" if build_spark else "download",
            "--only-binary=:all:",
            "--wheel-dir" if build_spark else "--dest",
            str(wheelhouse),
            "--requirement",
            str(resolved),
            "--index-url",
            "https://pypi.org/simple",
        ]
        if build_spark:
            # PySpark publishes source archives. Build only that resolved package
            # during explicit acquisition; target installation remains wheel-only.
            command.extend(["--no-binary=pyspark", "--no-deps"])
        # Exact resolver versions include the backend suffix; this is acquisition only.
        command.extend(["--extra-index-url", "https://download.pytorch.org/whl/" + args.device])
        _run(command, work)
    locks = []
    names = set()
    for wheel in sorted(wheelhouse.glob("*.whl")):
        name, version = wheel_identity(wheel)
        if name in names:
            raise ValueError("multiple wheels for one distribution")
        names.add(name)
        line = f"{name}=={version} --hash=sha256:{digest(wheel)}"
        if not _LOCK_LINE.fullmatch(line):
            raise ValueError("wheel metadata cannot form an offline lock")
        locks.append(line)
    (output / "requirements.lock").write_text("\n".join(sorted(locks)) + "\n")
    shutil.copyfile(Path(__file__), output / "deploy_release.py")
    manifest = {
        "schema_version": 1,
        "profiles": sorted(set(args.profile)),
        "device": args.device,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sys.platform,
        "release_version": release_version,
        "release_manifest_sha256": digest(source / "release-manifest.json"),
        "files": {
            path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum = digest(output / "bundle.json")
    verify_bundle(output, checksum)
    print(
        json.dumps(
            {
                "bundle": str(output),
                "sha256": checksum,
                "profiles": manifest["profiles"],
                "wheel_count": len(locks),
                "models_included": False,
            },
            sort_keys=True,
        )
    )


def install(args: argparse.Namespace) -> None:
    """Install only verified local wheels and qualify the core workflow."""
    bundle, destination = args.bundle.absolute(), args.destination.absolute()
    manifest = verify_bundle(bundle, args.sha256)
    _private_new_directory(destination)
    environment = destination / "venv"
    report: dict[str, Any] = {
        "schema_version": 1,
        "bundle_sha256": args.sha256,
        "status": "installing",
        "profiles": manifest["profiles"],
    }
    report_path = destination / "delivery.json"
    try:
        snapshot = destination / "installation-bundle"
        shutil.copytree(bundle, snapshot, symlinks=True)
        verify_bundle(snapshot, args.sha256)
        bundle = snapshot
        _run([sys.executable, "-m", "venv", str(environment)], destination)
        python, capicove = _scripts(environment)
        verify_bundle(bundle, args.sha256)
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--require-hashes",
                "--find-links",
                str(bundle / "wheelhouse"),
                "-r",
                str(bundle / "requirements.lock"),
            ],
            destination,
        )
        _run([python, "-m", "pip", "check"], destination)
        inputs = destination / "inputs"
        inputs.mkdir(mode=0o700)
        (inputs / "welcome.md").write_text(
            "# Capicove\nCapicove delivery verifies local evidence search.\n"
        )
        workspace = destination / "workspace"
        command = [capicove, "setup", "init", str(workspace), "--source-root", f"delivery={inputs}"]
        for source_root in args.source_root:
            command.extend(["--source-root", source_root])
        _run(command, destination)
        config = str(workspace / "host.json")
        native = (
            configure_native_media(workspace, inputs, args.native_config, manifest["profiles"])
            if args.native_config
            else None
        )
        _run([capicove, "setup", "doctor", "--config", config], destination)
        _run(
            [
                capicove,
                "source",
                "import",
                "--runtime-config",
                config,
                "--root",
                "delivery",
                "--relative-path",
                "welcome.md",
            ],
            destination,
        )
        _run([capicove, "index", "build", "--runtime-config", config], destination)
        result = json.loads(
            _run([capicove, "search", "--runtime-config", config, "delivery"], destination)
        )
        if result.get("status") != "succeeded" or not any(
            hit.get("snippet", "").strip()
            == "# Capicove\nCapicove delivery verifies local evidence search."
            for hit in result.get("result", {}).get("hits", [])
        ):
            raise RuntimeError("installed search did not retrieve the authored document")
        report.update(
            status="installed",
            core_acceptance="passed",
            config=config,
            optional_runtime_status={
                name: "requires_explicit_runtime_configuration"
                for name in manifest["profiles"]
                if name not in {"core", "interfaces"}
            },
        )
        if "knowledge" in manifest["profiles"]:
            report["knowledge_acceptance"] = qualify_knowledge(
                capicove, destination, lambda command: _run(command, destination)
            )
        if native is not None:
            (destination / "native-assets.json").write_text(json.dumps(native, indent=2) + "\n")
            report["optional_runtime_status"].update(
                _qualify_media(capicove, destination, native, manifest["profiles"])
            )
        if args.llm_config:
            from_config = json.loads(args.llm_config.read_text())
            config_path = destination / "generation.json"
            config_path.write_text(json.dumps(from_config, indent=2) + "\n")
            config_path.chmod(0o600)
            result = _run(
                [capicove, "setup", "check-llm", "--config", str(config_path)],
                destination,
                credential_env=from_config.get("api_key_env"),
            )
            report["generation"] = json.loads(result)
            report["optional_runtime_status"]["graph-generation"] = "qualified"
        report["packages"] = json.loads(
            _run([python, "-m", "pip", "list", "--format=json"], destination)
        )
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_path.chmod(0o600)
    print(
        json.dumps(
            {
                "delivery": str(destination),
                "report": str(report_path),
                "core_acceptance": "passed",
                "status": "installed",
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="Acquire dependencies and seal an offline bundle.")
    build.add_argument("--dist", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--profile", choices=sorted(PROFILES), action="append", required=True)
    build.add_argument("--device", choices=("cpu", "cu130"), default="cpu")
    build.add_argument("--allow-network", action="store_true")
    apply = commands.add_parser("install", help="Install a reviewed bundle without network access.")
    apply.add_argument("--bundle", type=Path, required=True)
    apply.add_argument("--sha256", required=True, help="Trusted SHA-256 of bundle.json.")
    apply.add_argument("--destination", type=Path, required=True)
    apply.add_argument("--source-root", action="append", default=[])
    apply.add_argument(
        "--native-config",
        type=Path,
        help="Explicit paths and SHA-256 hashes for native executables/models.",
    )
    apply.add_argument(
        "--llm-config",
        type=Path,
        help="Explicitly qualify the selected remote LLM after offline installation.",
    )
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else install)(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Deployment failed: {error}\n")


if __name__ == "__main__":
    main()
