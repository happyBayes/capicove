"""Explicit acquisition and local configuration for the pinned speech profile."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import secrets
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from capicove.processing.audio.synthesis import file_digest, open_regular
from capicove.runtime.speech import (
    MAX_SPEECH_CONFIG_BYTES,
    SpeechRuntime,
    read_speech_configuration,
)


MODEL_REPOSITORY = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
_ORIGINAL_MODEL_URL = "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REVISION = (
    "ca27d74bc954b73dadab5b71ca265d87fc861a7c"  # pragma: allowlist secret -- Public revision.
)
BACKEND_REVISION = (
    "e0f142d5ddc1461ce721644d92af33db23545105"  # pragma: allowlist secret -- Public revision.
)
_SOURCE_URL = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{BACKEND_REVISION}"
_REFERENCE_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    """Identify one immutable download by name, length, and content hash."""

    name: str
    size_bytes: int
    sha256: str


MODEL_FILES = (
    Asset(
        "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
        1847874400,
        "ac7931aeb2e7aad1a6ed6602d353a5679c9d096b18ce8204ac730a8408d572e1",  # pragma: allowlist secret -- Public model hash.
    ),
    Asset(
        "mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf",
        669081472,
        "b9503e95e44705739cf82c15ce909b040b96a31ad79d02cf4dc5e1484399265d",  # pragma: allowlist secret -- Public model hash.
    ),
)
_BACKEND_SOURCE = Asset(
    "llama.cpp.tar.gz",
    38829291,
    "0145e79796786bd8b699ad296efa31c40ba499a44af299c022d8a5866ff0dc6b",  # pragma: allowlist secret -- Public source archive hash.
)


def setup_plan(root: Path, assets_dir: Path | None, executable: Path | None) -> dict[str, object]:
    """Show the exact scope of acquisition before asking for confirmation."""
    reuse = root.exists()
    return {
        "root": str(root),
        "reuse": reuse,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "original_model": _ORIGINAL_MODEL_URL,
        "original_model_license": "Apache-2.0",
        "converted_repository_license": "Not declared in the pinned repository metadata",
        "usage": "Local speech synthesis with reference recordings you are authorized to use",
        "models": [asdict(asset) for asset in MODEL_FILES],
        "model_download": not reuse and assets_dir is None,
        "model_download_bytes": sum(asset.size_bytes for asset in MODEL_FILES)
        if not reuse and assets_dir is None
        else 0,
        "backend_download": not reuse and executable is None,
        "backend_source": {
            "url": _SOURCE_URL,
            "revision": BACKEND_REVISION,
            "license": "MIT",
            **asdict(_BACKEND_SOURCE),
        },
    }


def _approved_url(url: str) -> bool:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    return (
        parsed.scheme == "https"
        and parsed.username is None
        and parsed.password is None
        and parsed.port in (None, 443)
        and (
            host in {"codeload.github.com", "huggingface.co", "hf.co"}
            or host.endswith((".hf.co", ".huggingface.co"))
        )
    )


class _ApprovedRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if not _approved_url(newurl):
            raise ValueError("download redirected outside the approved HTTPS hosts")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, asset: Asset, path: Path) -> None:
    """Stream only the named asset and verify it before making it available."""
    if not _approved_url(url):
        raise ValueError("download URL is outside the approved HTTPS hosts")
    with build_opener(_ApprovedRedirects()).open(url, timeout=60) as response:
        _write_verified_asset(response, asset, path)


def _write_verified_asset(source: BinaryIO, asset: Asset, path: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > asset.size_bytes:
                raise ValueError(f"asset exceeds pinned size: {asset.name}")
            target.write(chunk)
            digest.update(chunk)
    if size != asset.size_bytes or digest.hexdigest() != asset.sha256:
        raise ValueError(f"asset failed size or SHA-256 verification: {asset.name}")


def _copy_asset(source: Path, destination: Path, asset: Asset) -> None:
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != asset.size_bytes:
        raise ValueError(f"local file differs from pinned size: {asset.name}")
    with open_regular(source) as reader:
        _write_verified_asset(reader, asset, destination)


def _extract_source(path: Path, source: Path) -> None:
    """Extract the verified archive without links or paths outside the build directory."""
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (
                not parts
                or parts[0] != f"llama.cpp-{BACKEND_REVISION}"
                or any(part == ".." or "\\" in part or ":" in part for part in parts)
            ):
                raise ValueError("unexpected path in the llama.cpp source archive")
            output = source.joinpath(*parts[1:])
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output.parent.mkdir(parents=True, exist_ok=True)
                reader = archive.extractfile(member)
                if reader is None:
                    raise ValueError("unreadable llama.cpp archive member")
                with reader, output.open("xb") as target:
                    shutil.copyfileobj(reader, target)
                output.chmod(0o600 | (member.mode & 0o100))
            else:
                raise ValueError("unexpected link or device in the llama.cpp source archive")


def _build_backend(stage: Path, target: Path, *, metal: bool) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise ValueError("install CMake and a C++ toolchain, or pass --llama-tts")
    with tempfile.TemporaryDirectory(prefix=".build-", dir=stage) as directory:
        work = Path(directory)
        archive = work / _BACKEND_SOURCE.name
        _download(_SOURCE_URL, _BACKEND_SOURCE, archive)
        source = work / "source"
        _extract_source(archive, source)
        build = work / "build"
        with (work / "build.log").open("wb+") as log:
            for command in (
                [
                    cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=OFF",
                    f"-DGGML_METAL={'ON' if metal else 'OFF'}",
                    "-DGGML_METAL_EMBED_LIBRARY=ON",
                    "-DGGML_CUDA=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DLLAMA_OPENSSL=OFF",
                ],
                [
                    cmake,
                    "--build",
                    str(build),
                    "--config",
                    "Release",
                    "--target",
                    "llama-tts",
                    "-j",
                    "2",
                ],
            ):
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=1800,
                )
                if result.returncode:
                    log.seek(max(0, log.tell() - 4000))
                    details = log.read(4000).decode(errors="replace").strip()
                    raise ValueError(f"llama-tts build failed.\n{details}")
        candidates = (build / "bin" / "Release" / target.name, build / "bin" / target.name)
        binary = next((path for path in candidates if path.is_file()), None)
        if binary is None:
            raise ValueError("llama-tts build did not produce the expected executable")
        shutil.copyfile(binary, target)
        target.chmod(0o700)


def _check_backend(binary: Path) -> None:
    """Check that the installed executable starts and exposes the required flags."""
    result = subprocess.run(
        [str(binary), "--help"],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    help_text = result.stdout + result.stderr
    if any(flag not in help_text for flag in ("--offline", "--mmproj", "--tts-speaker-file")):
        raise ValueError("llama-tts does not expose the required offline speech options")


def _private_directory(path: Path) -> None:
    if os.name == "nt":
        from capicove._windows_files import validate_windows_path

        validate_windows_path(path, directory=True, private_acl=True)
    else:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("speech directory must be owner-private")


def _protect_new_directory(path: Path) -> None:
    if os.name == "nt":
        system = Path(os.environ["SystemRoot"]) / "System32"
        identity = subprocess.run(
            [str(system / "whoami.exe"), "/user", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        sid = next(csv.reader(identity.stdout.splitlines()))[-1].strip()
        if not sid.startswith("S-1-") or any(not part.isdecimal() for part in sid[2:].split("-")):
            raise ValueError("could not determine the Windows account SID")
        for args in (["/grant:r", f"*{sid}:(OI)(CI)F"], ["/inheritance:r"]):
            subprocess.run(
                [str(system / "icacls.exe"), str(path), *args],
                check=True,
                capture_output=True,
                timeout=30,
            )
    _private_directory(path)


@contextmanager
def setup_lock(root: Path) -> Iterator[None]:
    """Serialize local setup and grant updates for one destination."""
    if root == root.parent or root.resolve() != root:
        raise ValueError("use an absolute speech directory without links")
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / f".{root.name}.setup-lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError(f"another setup holds the destination lock: {lock}") from error
    try:
        yield
    finally:
        lock.rmdir()


def _write_json(path: Path, document: dict[str, object]) -> None:
    _write_private(path, (json.dumps(document, indent=2) + "\n").encode())


def _write_private(path: Path, payload: bytes) -> None:
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_speech(
    root: Path, assets_dir: Path | None, executable: Path | None, *, download_authorized: bool
) -> None:
    """Install a complete configuration after the CLI has obtained consent."""
    if root.exists():
        if assets_dir is not None or executable is not None:
            raise ValueError("use a new destination to select different input files")
        _private_directory(root)
        return
    if (assets_dir is None or executable is None) and not download_authorized:
        raise ValueError("confirm downloads before acquiring speech assets")
    if platform.system() not in {"Darwin", "Linux", "Windows"}:
        raise ValueError("speech setup supports macOS, Linux, and Windows")
    if shutil.disk_usage(root.parent).free < sum(asset.size_bytes for asset in MODEL_FILES):
        raise ValueError("not enough free disk space for the two speech model files")
    if executable is not None and (
        not stat.S_ISREG(executable.lstat().st_mode) or not os.access(executable, os.X_OK)
    ):
        raise ValueError("the supplied llama-tts must be a regular executable file")

    metal = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    with tempfile.TemporaryDirectory(prefix=f".{root.name}.staging-", dir=root.parent) as directory:
        stage = Path(directory)
        _protect_new_directory(stage)
        for name in ("models", "bin", "references", "outputs"):
            (stage / name).mkdir(mode=0o700)
        binary = stage / "bin" / ("llama-tts.exe" if os.name == "nt" else "llama-tts")
        if executable is None:
            _build_backend(stage, binary, metal=metal)
        else:
            with open_regular(executable) as source, binary.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            binary.chmod(0o700)
        _check_backend(binary)
        for asset in MODEL_FILES:
            target = stage / "models" / asset.name
            if assets_dir is None:
                url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{asset.name}"
                _download(url, asset, target)
            else:
                _copy_asset(assets_dir / asset.name, target, asset)

        _write_configuration(
            root,
            stage,
            binary,
            metal=metal,
            external_backend=executable is not None,
            local_models=assets_dir is not None,
        )
        if root.exists():
            raise ValueError("speech destination appeared during setup")
        stage.rename(root)


def _write_configuration(
    root: Path,
    stage: Path,
    binary: Path,
    *,
    metal: bool,
    external_backend: bool,
    local_models: bool,
) -> None:
    model, mmproj = MODEL_FILES
    manifest = json.dumps(
        sorted((asset.name, asset.sha256) for asset in MODEL_FILES), separators=(",", ":")
    ).encode()
    binary_hash = file_digest(binary)
    backend_version = "operator-supplied" if external_backend else BACKEND_REVISION
    _write_json(
        stage / "model.json",
        {
            "schema_version": 1,
            "max_instances": 1,
            "bindings": {"tts": "qwen3-tts"},
            "profiles": {
                "qwen3-tts": {
                    "provider": "llama-cpp-tts",
                    "capability": "audio_synthesis",
                    "root": str(root / "models"),
                    "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                    "device": "mps" if metal else "cpu",
                    "options": {
                        "executable": str(root / binary.relative_to(stage)),
                        "executable_sha256": binary_hash,
                        "model": model.name,
                        "model_sha256": model.sha256,
                        "mmproj": mmproj.name,
                        "mmproj_sha256": mmproj.sha256,
                        "backend_version": backend_version,
                        "threads": 2,
                    },
                }
            },
        },
    )
    _write_json(
        stage / "speech.json",
        {
            "schema_version": 1,
            "model_config": str(root / "model.json"),
            "binding": "tts",
            "read_roots": [str(root / "references")],
            "output_root": str(root / "outputs"),
            "authorizations": {},
            "signing_key_env": "CAPICOVE_SPEECH_SIGNING_KEY",
        },
    )
    _write_private(stage / "signing.key", secrets.token_hex(32).encode("ascii"))
    _write_json(
        stage / "provisioning.json",
        {
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "model_acquisition": "verified-local-files" if local_models else "download",
            "original_model": _ORIGINAL_MODEL_URL,
            "original_model_license": "Apache-2.0",
            "models": [asdict(asset) for asset in MODEL_FILES],
            "backend_revision": backend_version,
            "backend_archive_sha256": None if external_backend else _BACKEND_SOURCE.sha256,
            "executable_sha256": binary_hash,
        },
    )


def read_reference(path: Path) -> bytes:
    """Read and validate the exact PCM bytes to be authorized."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("reference must be a regular WAV file")
    with open_regular(path) as stream:
        data = stream.read(_REFERENCE_LIMIT + 1)
    if len(data) > _REFERENCE_LIMIT:
        raise ValueError("reference WAV exceeds 16 MiB")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            frames = wav.getnframes()
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or not 3 <= frames / wav.getframerate() <= 30
                or len(wav.readframes(frames)) != frames * 2
            ):
                raise ValueError("reference must be a complete 3-30 second mono PCM16 WAV")
    except (wave.Error, EOFError) as error:
        raise ValueError("reference must be a complete mono PCM16 WAV") from error
    return data


def grant_reference(root: Path, runtime: SpeechRuntime, data: bytes) -> tuple[Path, str]:
    """Store an exact reference copy and add a grant to the existing runtime config."""
    config = read_speech_configuration(runtime.path)
    if config["read_roots"] != [str(root / "references")] or config["output_root"] != str(
        root / "outputs"
    ):
        raise ValueError("reference authorization requires a managed speech configuration")
    directory = root / "references"
    _private_directory(directory)
    digest = hashlib.sha256(data).hexdigest()
    path = directory / f"{digest}.wav"
    grants = runtime.authorizations()
    grant = grants.setdefault(digest, uuid.uuid4().hex)
    config["authorizations"] = grants
    payload = (json.dumps(config, indent=2) + "\n").encode()
    if len(payload) > MAX_SPEECH_CONFIG_BYTES:
        raise ValueError("speech configuration has reached its grant size limit")
    if path.exists() or path.is_symlink():
        if file_digest(path) != digest:
            raise ValueError("the existing managed reference differs from its recorded hash")
    else:
        _write_private(path, data)
    _write_private(runtime.path, payload)
    return path, grant
