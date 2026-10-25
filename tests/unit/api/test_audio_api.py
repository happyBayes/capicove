"""Exercise uploaded audio without optional native runtimes."""

from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from capicove.api.audio import create_app
from capicove.processing.audio.models import (
    AudioBackendLineage,
    AudioTranscript,
    AudioTranscriptionSegment,
)


class Provider:
    model_id = "test-asr"

    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.closed = 0

    def transcribe(self, source: Path, *, language: str, deadline: datetime) -> AudioTranscript:
        self.paths.append(source)
        assert source.read_bytes() == b"audio"
        return AudioTranscript(
            "hello",
            language,
            (AudioTranscriptionSegment(0, 0, 1, "hello"),),
            (),
            AudioBackendLineage("test", "1", "model", "a" * 64, "cpu", "f32"),
            1.0,
        )

    def close(self) -> None:
        self.closed += 1


def test_upload_formats_cleanup_and_unknown_fields(tmp_path: Path) -> None:
    provider = Provider()
    with TestClient(create_app(provider, model="speech", temp_directory=tmp_path)) as client:
        for fmt in ("json", "text", "verbose_json"):
            result = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("../../secret.wav", b"audio")},
                data={"model": "speech", "response_format": fmt, "language": "en"},
            )
            assert result.status_code == 200, result.text
            assert (result.text if fmt == "text" else result.json()["text"]) == "hello"
            assert not list(tmp_path.iterdir())
        assert (
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", b"audio")},
                data={"model": "speech", "prompt": "ignore"},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("a.wav", b"audio")},
                data={"model": "missing"},
            ).status_code
            == 404
        )
    assert provider.closed == 1
    assert all(not path.exists() for path in provider.paths)


def test_auth_and_limit_before_inference() -> None:
    provider = Provider()
    key = "secret"  # pragma: allowlist secret
    with TestClient(
        create_app(provider, model="speech", api_key=key, max_upload_bytes=4)
    ) as client:
        assert client.post("/v1/audio/transcriptions").status_code == 401
        result = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer secret"},
            files={"file": ("a.wav", b"audio")},
            data={"model": "speech"},
        )
        assert result.status_code == 413
    assert not provider.paths


def test_failures_do_not_retain_uploads(tmp_path: Path) -> None:
    from capicove.processing.audio.models import AudioFailureCode
    from capicove.processing.audio.transcription import AudioTranscriptionError

    class Failed(Provider):
        def transcribe(self, source: Path, *, language: str, deadline: datetime) -> AudioTranscript:
            assert source.exists()
            raise AudioTranscriptionError(AudioFailureCode.TIMEOUT, "private diagnostic")

    with TestClient(create_app(Failed(), model="speech", temp_directory=tmp_path)) as client:
        result = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("x.wav", b"audio")},
            data={"model": "speech"},
        )
        assert result.status_code == 504
        assert "private" not in result.text
        assert not list(tmp_path.iterdir())


def test_duplicate_fields_and_multiple_files_are_rejected() -> None:
    provider = Provider()
    with TestClient(create_app(provider, model="speech")) as client:
        result = client.post(
            "/v1/audio/transcriptions",
            files=[
                ("model", (None, "speech")),
                ("model", (None, "speech")),
                ("file", ("x.wav", b"audio")),
            ],
        )
        assert result.status_code == 400
        result = client.post(
            "/v1/audio/transcriptions",
            files=[
                ("model", (None, "speech")),
                ("file", ("x.wav", b"audio")),
                ("file", ("y.wav", b"audio")),
            ],
        )
        assert result.status_code == 400
    assert not provider.paths


def test_cancellation_keeps_files_until_native_work_finishes(tmp_path: Path) -> None:
    import asyncio
    import threading

    import httpx
    import pytest

    entered, release = threading.Event(), threading.Event()

    class Blocking(Provider):
        staged: Path | None = None

        def transcribe(self, source: Path, *, language: str, deadline: datetime) -> AudioTranscript:
            self.staged = source
            entered.set()
            assert release.wait(5)
            assert source.exists()
            return super().transcribe(source, language=language, deadline=deadline)

    async def exercise() -> None:
        provider = Blocking()
        app = create_app(provider, model="speech", temp_directory=tmp_path)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                pending = asyncio.create_task(
                    client.post(
                        "/v1/audio/transcriptions",
                        files={"file": ("x", b"audio")},
                        data={"model": "speech"},
                    )
                )
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    pending.cancel()
                    await asyncio.sleep(0.02)
                    assert provider.staged is not None and provider.staged.exists()
                    assert (await client.get("/health")).status_code == 200
                    assert (await client.post("/v1/audio/transcriptions")).status_code == 429
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert not list(tmp_path.iterdir())

    asyncio.run(exercise())
