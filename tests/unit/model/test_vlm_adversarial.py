"""Adversarial VLM stop, resolver lifetime and upstream protocol regressions."""

import json
import shutil
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from capicove.model.providers.openai_vlm import VlmError
from tests.unit.model.test_openai_vlm import client, request
from tests.unit.model.test_openai_vlm import upstream as upstream


@pytest.mark.parametrize("content", [None, "", "\n "])
@pytest.mark.parametrize("reason", ["stop", "length"])
def test_empty_completed_text_preserves_stop_and_usage(
    upstream: tuple[str, dict[str, Any]], content: str | None, reason: str
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0].update(
        message={"role": "assistant", "content": content}, finish_reason=reason
    )
    with client(endpoint) as provider:
        result = provider.generate(replace(request(), stop=("The",)))
    assert result.text == (content or "")
    assert result.finish_reason == reason
    assert result.usage is not None and result.usage.completion_tokens == 4


@pytest.mark.parametrize("text", [None, "", '{"color":'])
def test_custom_stop_can_end_json_before_a_complete_object(
    upstream: tuple[str, dict[str, Any]], text: str | None
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0]["message"]["content"] = text
    with client(endpoint) as provider:
        result = provider.generate(replace(request(), stop=("red",), json_object=True))
    assert result.text == (text or "") and result.finish_reason == "stop"


def test_dns_timeout_cancels_resolution_without_late_request(
    upstream: tuple[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    import aiohttp

    endpoint, state = upstream
    cancelled = []

    async def slow(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(aiohttp.AsyncResolver, "resolve", slow)
    start = time.monotonic()
    with client(endpoint.replace("127.0.0.1", "localhost"), timeout_seconds=0.05) as provider:
        with pytest.raises(VlmError, match="upstream_timeout"):
            provider.generate(request())
    assert time.monotonic() - start < 0.3
    assert cancelled == [True]
    assert "request" not in state
    with client(endpoint) as provider:
        assert provider.generate(request()).text


@pytest.mark.parametrize("field", ["content", "reasoning", "reasoning_content"])
def test_unpaired_surrogates_are_rejected_at_upstream_boundary(
    upstream: tuple[str, dict[str, Any]], field: str
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0]["message"][field] = "\ud800"
    with client(endpoint) as provider, pytest.raises(VlmError, match="invalid_upstream_response"):
        provider.generate(request())


def test_incomplete_http_body_is_rejected_even_if_prefix_is_valid_json(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["length_delta"] = 20
    with client(endpoint) as provider, pytest.raises(VlmError, match="invalid_upstream_response"):
        provider.generate(request())


def test_complete_close_delimited_connection_with_content_length_succeeds(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["length_delta"] = 0
    with client(endpoint) as provider:
        assert provider.generate(request()).text == "A red square."


def test_legacy_function_call_cannot_become_successful_empty_text(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0]["message"].update(
        content=None, function_call={"name": "unsupported", "arguments": "{}"}
    )
    with client(endpoint) as provider, pytest.raises(VlmError, match="invalid_upstream_response"):
        provider.generate(request())


def test_null_reasoning_alias_does_not_hide_actual_reasoning(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0]["message"].update(reasoning_content=None, reasoning="analysis")
    with client(endpoint) as provider:
        assert provider.generate(request()).reasoning == "analysis"


@pytest.fixture
def tls_server(tmp_path: Path) -> Iterator[tuple[int, Path]]:
    executable = shutil.which("openssl")
    if executable is None:
        pytest.skip("local TLS qualification requires openssl")
    certificate, key = tmp_path / "certificate.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            executable,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["content-length"]))
            payload = json.dumps(
                {
                    "model": "example/vision-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "TLS verified"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    worker.start()
    try:
        yield server.server_port, certificate
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize(
    "trusted,host,success",
    [(False, "localhost", False), (True, "localhost", True), (True, "127.0.0.1", False)],
)
def test_tls_validates_trust_and_hostname(
    tls_server: tuple[int, Path],
    monkeypatch: pytest.MonkeyPatch,
    trusted: bool,
    host: str,
    success: bool,
) -> None:
    port, certificate = tls_server
    if trusted:
        monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    with client(f"https://{host}:{port}/v1", timeout_seconds=2) as provider:
        if success:
            assert provider.generate(request()).text == "TLS verified"
        else:
            with pytest.raises(VlmError, match="upstream_unavailable"):
                provider.generate(request())


def test_stalled_tls_handshake_respects_total_deadline() -> None:
    release = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(1)

    def stall() -> None:
        try:
            peer, _ = listener.accept()
            with peer:
                release.wait(2)
        except OSError:
            pass

    worker = threading.Thread(target=stall, daemon=True)
    worker.start()
    try:
        with client(
            f"https://127.0.0.1:{listener.getsockname()[1]}/v1", timeout_seconds=0.08
        ) as provider:
            start = time.monotonic()
            with pytest.raises(VlmError, match="upstream_timeout"):
                provider.generate(request())
            assert time.monotonic() - start < 0.4
    finally:
        release.set()
        listener.close()
        worker.join(timeout=2)


def test_already_expired_transport_setup_is_a_sanitized_timeout() -> None:
    with client("http://127.0.0.1:1/v1", timeout_seconds=1e-12) as provider:
        with pytest.raises(VlmError, match="upstream_timeout") as error:
            provider.generate(request())
        assert error.value.status == 504


def test_tls_context_setup_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed() -> Any:
        raise OSError("private certificate configuration")

    monkeypatch.setattr(ssl, "create_default_context", failed)
    with client("https://localhost:1/v1") as provider:
        with pytest.raises(VlmError, match="upstream_unavailable") as error:
            provider.generate(request())
        assert "private" not in str(error.value)
