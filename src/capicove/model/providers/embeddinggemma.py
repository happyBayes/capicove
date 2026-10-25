"""Offline EmbeddingGemma 2 GGUF provider served by a pinned local llama.cpp binary.

The provider owns one ``llama-server`` child bound to a Unix socket inside a private
directory: no TCP listener, no network access (``--offline``) and no implicit
download. Model files, the multimodal projector and the server binary are verified
against pinned SHA-256 digests before the child starts. Vectors from this backend
form their own space; they are not claimed interchangeable with a Transformers
checkpoint of the same upstream model.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess  # nosec B404
import tempfile
import time
import wave
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.configuration import ModelProfile
from capicove.model.multimodal_embedding import (
    MultimodalEmbeddingError,
    PcmAudio,
    TextDocument,
    Vector,
    require_batch,
    validate_vectors,
)


PROVIDER_ID = "embeddinggemma2-llamacpp"
PROMPT_VERSION = "embeddinggemma2-search-prompts-v1"
QUERY_PROMPT = "task: search result | query: {query}"
DOCUMENT_PROMPT = "title: {title} | text: {text}"
DIMENSION = 768
CODE_PROVIDER_ID = "embeddinggemma2-llamacpp-code"
CODE_PROMPT_VERSION = "embeddinggemma2-code-retrieval-prompts-v1"
CODE_QUERY_PROMPT = "task: code retrieval | query: {query}"
_TEXT_DEADLINE_SECONDS = 600
_REQUIRED = {
    "model",
    "model_sha256",
    "mmproj",
    "mmproj_sha256",
    "llama_server",
    "llama_server_sha256",
    "backend_version",
    "model_revision",
}
_OPTIONAL = {"threads", "context_tokens", "startup_timeout_seconds", "max_batch"}
_RESPONSE_LIMIT = 64 * 1024 * 1024


def _int_option(options: object, key: str, default: int, low: int, high: int) -> int:
    value = options.get(key, default)  # type: ignore[attr-defined]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid {key}")
    return value


def manifest_digest(options: object) -> str:
    """Hash the exact role/name/digest inventory pinned by a profile."""
    inventory = sorted(
        (role, Path(str(options[role])).name, str(options[role + "_sha256"]))  # type: ignore[index]
        for role in ("model", "mmproj", "llama_server")
    )
    return hashlib.sha256(
        json.dumps(inventory, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def validate_embeddinggemma(profile: ModelProfile) -> None:
    """Reject unknown settings, implicit executables, devices and unpinned assets."""
    options = profile.options
    if (
        profile.device != "cpu"
        or not _REQUIRED <= set(options)
        or set(options) - _REQUIRED - _OPTIONAL
    ):
        raise ValueError("invalid EmbeddingGemma 2 options")
    for key in _REQUIRED:
        value = options[key]
        if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
            raise ValueError("invalid EmbeddingGemma 2 option")
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid EmbeddingGemma 2 digest")
    for key in ("model", "mmproj"):
        name = str(options[key])
        if Path(name).name != name or name in {".", ".."} or not name.endswith(".gguf"):
            raise ValueError("GGUF assets must be direct children of the model root")
    if not Path(str(options["llama_server"])).is_absolute():
        raise ValueError("explicit llama-server path required")
    _int_option(options, "threads", 4, 1, 64)
    _int_option(options, "context_tokens", 8192, 512, 8192)
    _int_option(options, "startup_timeout_seconds", 120, 1, 600)
    _int_option(options, "max_batch", 8, 1, 64)
    if manifest_digest(options) != profile.manifest_sha256:
        raise ValueError("EmbeddingGemma 2 manifest digest mismatch")


def create_embeddinggemma(profile: ModelProfile) -> object:
    """Verify pinned files, then start the local server child explicitly."""
    validate_embeddinggemma(profile)
    return _start(profile, space_for(profile))


def _start(profile: ModelProfile, space: EmbeddingSpaceRef) -> LlamaCppEmbeddingGemma:
    options = profile.options
    model = profile.root / str(options["model"])
    mmproj = profile.root / str(options["mmproj"])
    server = Path(str(options["llama_server"]))
    for path, digest in (
        (model, options["model_sha256"]),
        (mmproj, options["mmproj_sha256"]),
        (server, options["llama_server_sha256"]),
    ):
        _verify_file(path, str(digest))
    if not os.access(server, os.X_OK):
        raise ValueError("llama-server is not executable")
    return LlamaCppEmbeddingGemma(
        model=model,
        mmproj=mmproj,
        server=server,
        space=space,
        threads=_int_option(options, "threads", 4, 1, 64),
        context_tokens=_int_option(options, "context_tokens", 8192, 512, 8192),
        startup_timeout=_int_option(options, "startup_timeout_seconds", 120, 1, 600),
        max_batch=_int_option(options, "max_batch", 8, 1, 64),
    )


def validate_embeddinggemma_code(profile: ModelProfile) -> None:
    """Apply the shared asset/runtime policy to the code text-embedding binding."""
    validate_embeddinggemma(profile)


def create_embeddinggemma_code(profile: ModelProfile) -> object:
    """Start the pinned runtime behind the provider-neutral text batch encoder port."""
    validate_embeddinggemma_code(profile)
    return LlamaCppCodeTextEncoder(_start(profile, code_space_for(profile)))


def code_space_for(profile: ModelProfile) -> EmbeddingSpaceRef:
    """Derive the text-only code space; prompts differ, so it never equals the shared space."""
    options = profile.options
    return EmbeddingSpaceRef(
        space_id=f"{CODE_PROVIDER_ID}:{profile.manifest_sha256[:32]}",
        provider_id=CODE_PROVIDER_ID,
        model_revision=f"{options['model_revision']}#{options['backend_version']}",
        preprocessing_version=CODE_PROMPT_VERSION,
        dimension=DIMENSION,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def space_for(profile: ModelProfile) -> EmbeddingSpaceRef:
    """Derive the complete space from pinned assets, backend and prompts."""
    options = profile.options
    return EmbeddingSpaceRef(
        space_id=f"{PROVIDER_ID}:{profile.manifest_sha256[:32]}",
        provider_id=PROVIDER_ID,
        model_revision=f"{options['model_revision']}#{options['backend_version']}",
        preprocessing_version=PROMPT_VERSION,
        dimension=DIMENSION,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset(
            {QueryModality.TEXT, QueryModality.IMAGE, QueryModality.AUDIO_CLIP}
        ),
        target_modalities=frozenset(
            {
                TargetType.TEXT_CHUNK,
                TargetType.IMAGE,
                TargetType.KEYFRAME,
                TargetType.AUDIO_SEGMENT,
            }
        ),
    )


def _verify_file(path: Path, expected: str) -> None:
    if path.resolve() != path:
        raise ValueError("unsafe EmbeddingGemma 2 asset path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("EmbeddingGemma 2 asset must be a regular file")
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("EmbeddingGemma 2 asset digest mismatch")
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("EmbeddingGemma 2 asset changed while hashing")


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def wav_bytes(clip: PcmAudio) -> bytes:
    """Wrap prepared PCM in a minimal WAV container without altering samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(clip.sample_rate_hz)
        stream.writeframes(clip.pcm)
    return buffer.getvalue()


class LlamaCppEmbeddingGemma:
    """Serialize requests to one owned ``llama-server --embedding`` child."""

    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        model: Path,
        mmproj: Path,
        server: Path,
        space: EmbeddingSpaceRef,
        threads: int = 4,
        context_tokens: int = 8192,
        startup_timeout: float = 120,
        max_batch: int = 8,
    ) -> None:
        self.space = space
        self.max_batch = max_batch
        self.load_seconds: float | None = None
        self._directory = Path(tempfile.mkdtemp(prefix="capicove-eg2-"))
        os.chmod(self._directory, 0o700)
        self._socket = str(self._directory / "server.sock")
        # llama-server randomizes its media marker unless pinned; a fresh secret
        # marker per instance also prevents corpus text from smuggling media slots.
        self.media_marker = f"<__media_capicove_{secrets.token_hex(16)}__>"
        started = time.monotonic()
        try:
            self._process: subprocess.Popen[bytes] | None = subprocess.Popen(  # nosec B603
                [
                    str(server),
                    "--model",
                    str(model),
                    "--mmproj",
                    str(mmproj),
                    "--embedding",
                    "--pooling",
                    "mean",
                    "--host",
                    self._socket,
                    "--offline",
                    "--no-webui",
                    "--threads",
                    str(threads),
                    "--ctx-size",
                    str(context_tokens),
                    "--batch-size",
                    str(context_tokens),
                    "--ubatch-size",
                    str(context_tokens),
                    "--parallel",
                    "1",
                    "--n-gpu-layers",
                    "0",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": str(self._directory),
                    "LLAMA_MEDIA_MARKER": self.media_marker,
                },
                cwd=self._directory,
                start_new_session=True,
                shell=False,
            )
        except OSError:
            shutil.rmtree(self._directory, ignore_errors=True)
            raise MultimodalEmbeddingError("unavailable", "llama-server could not start") from None
        try:
            self._wait_ready(started + startup_timeout)
        except BaseException:
            self.close()
            raise
        self.load_seconds = time.monotonic() - started

    @property
    def pid(self) -> int | None:
        """Return the owned child process ID while it is running."""
        return None if self._process is None else self._process.pid

    def _wait_ready(self, limit: float) -> None:
        while time.monotonic() < limit:
            assert self._process is not None
            if self._process.poll() is not None:
                raise MultimodalEmbeddingError("unavailable", "llama-server exited during load")
            try:
                connection = _UnixConnection(self._socket, timeout=2)
                try:
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    response.read(4096)
                    if response.status == 200:
                        return
                finally:
                    connection.close()
            except OSError:
                pass
            time.sleep(0.1)
        raise MultimodalEmbeddingError("unavailable", "llama-server did not become ready")

    def encode_text_queries(
        self, texts: Sequence[str], *, deadline: datetime
    ) -> tuple[Vector, ...]:
        """Apply the fixed search-query instruction to each text."""
        require_batch(texts, self.max_batch)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise MultimodalEmbeddingError("invalid_input", "queries must be non-blank text")
        self._reject_marker(texts)
        return self._embed([QUERY_PROMPT.format(query=text) for text in texts], deadline)

    def encode_text_documents(
        self, documents: Sequence[TextDocument], *, deadline: datetime
    ) -> tuple[Vector, ...]:
        """Apply ``title: none`` unless the document carries a real title."""
        require_batch(documents, self.max_batch)
        if any(not isinstance(document, TextDocument) for document in documents):
            raise MultimodalEmbeddingError("invalid_input", "documents must be TextDocument")
        self._reject_marker([d.text for d in documents] + [d.title or "" for d in documents])
        return self._embed(
            [DOCUMENT_PROMPT.format(title=d.title or "none", text=d.text) for d in documents],
            deadline,
        )

    def encode_images(self, images: Sequence[bytes], *, deadline: datetime) -> tuple[Vector, ...]:
        """Send encoded image bytes with only the media marker; no text prefix."""
        require_batch(images, self.max_batch)
        if any(not isinstance(image, bytes) or not image for image in images):
            raise MultimodalEmbeddingError("invalid_input", "images must be non-empty bytes")
        return self._embed([self._media(image) for image in images], deadline)

    def encode_audio(self, clips: Sequence[PcmAudio], *, deadline: datetime) -> tuple[Vector, ...]:
        """Send prepared waveform directly; no ASR, VAD or text prefix."""
        require_batch(clips, self.max_batch)
        if any(not isinstance(clip, PcmAudio) for clip in clips):
            raise MultimodalEmbeddingError("invalid_input", "audio must be PcmAudio")
        return self._embed([self._media(wav_bytes(clip)) for clip in clips], deadline)

    def _reject_marker(self, texts: Sequence[str]) -> None:
        if any(self.media_marker in text for text in texts):
            raise MultimodalEmbeddingError("invalid_input", "text contains a media marker")

    def _media(self, payload: bytes) -> dict[str, object]:
        return {
            "prompt_string": self.media_marker,
            "multimodal_data": [base64.b64encode(payload).decode("ascii")],
        }

    def _embed(self, content: list[object], deadline: datetime) -> tuple[Vector, ...]:
        if deadline.tzinfo is None:
            raise MultimodalEmbeddingError("invalid_input", "deadline must be timezone-aware")
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise MultimodalEmbeddingError("deadline_exceeded", "encoder deadline expired")
        if self._process is None or self._process.poll() is not None:
            raise MultimodalEmbeddingError("unavailable", "llama-server is not running")
        body = json.dumps({"content": content, "embd_normalize": 2}).encode()
        connection = _UnixConnection(self._socket, timeout=remaining)
        try:
            connection.request(
                "POST", "/embedding", body=body, headers={"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            payload = response.read(_RESPONSE_LIMIT + 1)
            status = response.status
        except TimeoutError:
            raise MultimodalEmbeddingError(
                "deadline_exceeded", "encoder deadline expired"
            ) from None
        except OSError:
            raise MultimodalEmbeddingError("unavailable", "llama-server request failed") from None
        finally:
            connection.close()
        if status != 200:
            code = "invalid_input" if 400 <= status < 500 else "unavailable"
            raise MultimodalEmbeddingError(code, f"llama-server returned HTTP {status}")
        if len(payload) > _RESPONSE_LIMIT:
            raise MultimodalEmbeddingError("invalid_output", "embedding response is too large")
        try:
            rows = json.loads(payload)
        except ValueError:
            raise MultimodalEmbeddingError(
                "invalid_output", "embedding response is not JSON"
            ) from None
        return validate_vectors(
            _pooled_rows(rows, len(content)), count=len(content), space=self.space
        )

    def close(self) -> None:
        """Terminate and reap the child before removing its private directory."""
        process, self._process = getattr(self, "_process", None), None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        shutil.rmtree(self._directory, ignore_errors=True)


def _pooled_rows(rows: object, count: int) -> list[object]:
    """Require exactly one pooled row per input index, in any response order."""
    if not isinstance(rows, list) or len(rows) != count:
        raise MultimodalEmbeddingError("invalid_output", "embedding count mismatch")
    ordered: dict[int, object] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"index", "embedding"}:
            raise MultimodalEmbeddingError("invalid_output", "malformed embedding row")
        index, embedding = row.get("index"), row.get("embedding")
        if (
            type(index) is not int
            or index in ordered
            or not 0 <= index < count
            or not isinstance(embedding, list)
            or len(embedding) != 1
        ):
            raise MultimodalEmbeddingError("invalid_output", "malformed embedding row")
        ordered[index] = embedding[0]
    return [ordered[index] for index in range(count)]


class LlamaCppCodeTextEncoder:
    """Provider-neutral ``TextBatchEncoder`` with EmbeddingGemma code-retrieval prompts.

    Documents use ``title: none``; callers that want path context put it in the
    document text so that context stays provider-neutral and identity-bound.
    """

    provider_id = CODE_PROVIDER_ID

    def __init__(self, runtime: LlamaCppEmbeddingGemma) -> None:
        self._runtime = runtime
        self.max_batch = runtime.max_batch

    @property
    def space(self) -> EmbeddingSpaceRef:
        """Return the complete code text space."""
        return self._runtime.space

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[Any, ...]:
        """Encode each text at its index; overlong input fails rather than truncating."""
        from capicove.retrieval.models import TextEmbedding

        if space != self.space:
            raise ValueError("requested space does not match this encoder")
        if role not in {"query", "document"}:
            raise ValueError("role must be query or document")
        if not isinstance(texts, tuple) or any(
            not isinstance(t, str) or not t.strip() for t in texts
        ):
            raise ValueError("texts must be a tuple of non-blank strings")
        self._runtime._reject_marker(texts)
        deadline = datetime.now(UTC) + timedelta(seconds=_TEXT_DEADLINE_SECONDS)
        prompts = [
            CODE_QUERY_PROMPT.format(query=text)
            if role == "query"
            else DOCUMENT_PROMPT.format(title="none", text=text)
            for text in texts
        ]
        vectors: list[Vector] = []
        for start in range(0, len(prompts), self.max_batch):
            vectors.extend(
                self._runtime._embed(list(prompts[start : start + self.max_batch]), deadline)
            )
        return tuple(TextEmbedding(space, vector) for vector in vectors)

    def close(self) -> None:
        """Stop the owned runtime."""
        self._runtime.close()
