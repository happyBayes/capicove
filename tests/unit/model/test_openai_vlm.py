"""Real loopback transport tests without external services or model SDKs."""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from capicove.model import HttpModelProfile, ModelConfiguration
from capicove.model.providers.openai_vlm import OpenAICompatibleVlm, VlmError
from capicove.model.vlm import VlmMedia, VlmMessage, VlmRequest, VlmText
from tests.unit.model.test_vlm import configuration


@pytest.fixture
def upstream() -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {
        "status": 200,
        "delay": 0,
        "body": {
            "model": "example/vision-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "A red square."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            state["request_count"] = state.get("request_count", 0) + 1
            state["path"] = self.path
            state["auth"] = self.headers.get("authorization")
            state["request"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            time.sleep(state["delay"])
            try:
                payload = state.get("payload", json.dumps(state["body"]).encode())
                self.send_response(state["status"])
                self.send_header("Content-Type", "application/json")
                self.send_header("Location", "/never-follow")
                if "length_delta" in state:
                    self.send_header("Content-Length", str(len(payload) + state["length_delta"]))
                self.end_headers()
                self.wfile.flush()
                time.sleep(state.get("body_delay", 0))
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def client(endpoint: str, **changes: Any) -> OpenAICompatibleVlm:
    raw = configuration()
    raw["profiles"]["vision"].update({"endpoint": endpoint, **changes})
    profile = ModelConfiguration.from_mapping(raw).profiles["vision"]
    assert isinstance(profile, HttpModelProfile)
    return OpenAICompatibleVlm(profile)


def request() -> VlmRequest:
    return VlmRequest((VlmMessage("user", (VlmText("What is shown?"),)),))


def test_http_wire_auth_template_usage_and_lineage(
    upstream: tuple[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint, state = upstream
    monkeypatch.setenv("CAPICOVE_TEST_VLM_KEY", "synthetic-key")
    # Environment variable name for a synthetic credential, not a key value.
    env_name = "CAPICOVE_TEST_VLM_KEY"
    with client(endpoint, api_key_env=env_name) as provider:
        result = provider.generate(request())
    assert state["path"] == "/v1/chat/completions"
    assert state["auth"] == "Bearer synthetic-key"
    assert state["request"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.text == "A red square."
    assert result.usage is not None and result.usage.prompt_tokens == 12
    assert len(result.request_sha256) == 64
    assert result.to_openai()["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }


@pytest.mark.parametrize(
    "status,code",
    [
        (302, "upstream_unavailable"),
        (401, "upstream_unavailable"),
        (400, "upstream_rejected"),
        (429, "upstream_busy"),
        (503, "upstream_unavailable"),
    ],
)
def test_upstream_errors_are_sanitized_and_no_redirects(
    upstream: tuple[str, dict[str, Any]], status: int, code: str
) -> None:
    endpoint, state = upstream
    state.update(status=status, payload=b"secret upstream diagnostic")
    with client(endpoint) as provider:
        with pytest.raises(VlmError) as error:
            provider.generate(request())
    assert error.value.code == code
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "wrong-model"},
        {"choices": []},
        {"usage": {"prompt_tokens": True, "completion_tokens": 2}},
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 99}},
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": 42},
                    "finish_reason": "stop",
                }
            ]
        },
    ],
)
def test_invalid_upstream_response_is_rejected(
    upstream: tuple[str, dict[str, Any]], change: dict[str, Any]
) -> None:
    endpoint, state = upstream
    state["body"].update(change)
    with client(endpoint) as provider:
        with pytest.raises(VlmError, match="invalid_upstream_response"):
            provider.generate(request())


def test_deadline_releases_local_slot_and_subsequent_request_works(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["delay"] = 0.2
    with client(endpoint, timeout_seconds=0.05) as provider:
        start = time.monotonic()
        with pytest.raises(VlmError, match="upstream_timeout"):
            provider.generate(request())
        assert time.monotonic() - start < 0.5
        state["delay"] = 0
        assert provider.generate(request()).text


def test_limits_are_checked_before_network_and_close_is_final() -> None:
    provider = client("http://127.0.0.1:1/v1", modalities=["image"], max_images=1)
    image = VlmMedia(b"\x89PNG\r\n\x1a\nfixture", "image/png")
    with pytest.raises(ValueError, match="media limit"):
        provider.generate(VlmRequest((VlmMessage("user", (image, image)),)))
    with pytest.raises(ValueError, match="output token"):
        provider.generate(VlmRequest((VlmMessage("user", (VlmText("hello"),)),), max_tokens=8192))
    provider.close()
    provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.generate(request())


def test_absolute_deadline_covers_close_delimited_body(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state.update(delay=0.15, body_delay=0.2)
    with client(endpoint, timeout_seconds=0.25) as provider:
        start = time.monotonic()
        with pytest.raises(VlmError, match="upstream_timeout"):
            provider.generate(request())
        assert time.monotonic() - start < 0.32


@pytest.mark.parametrize(
    "payload", [b"{", b'{"model":"a","model":"b"}', b"x" * (2 * 1024 * 1024 + 1)]
)
def test_invalid_or_oversized_upstream_body_is_bounded(
    upstream: tuple[str, dict[str, Any]], payload: bytes
) -> None:
    endpoint, state = upstream
    state["payload"] = payload
    with client(endpoint) as provider, pytest.raises(VlmError, match="invalid_upstream_response"):
        provider.generate(request())


@pytest.mark.parametrize(
    "json_text,valid", [("not JSON", False), ("[]", False), ('{"color":"red"}', True)]
)
def test_json_mode_validates_completed_response(
    upstream: tuple[str, dict[str, Any]], json_text: str, valid: bool
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0]["message"]["content"] = json_text
    from dataclasses import replace

    with client(endpoint) as provider:
        if valid:
            assert provider.generate(replace(request(), json_object=True)).text == json_text
        else:
            with pytest.raises(VlmError, match="invalid_upstream_response"):
                provider.generate(replace(request(), json_object=True))


def test_content_filter_and_reasoning_only_truncation_are_explicit(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    choice = state["body"]["choices"][0]
    choice.update(message={"role": "assistant", "content": None}, finish_reason="content_filter")
    with client(endpoint) as provider:
        assert provider.generate(request()).finish_reason == "content_filter"
        choice.update(
            message={"role": "assistant", "content": None, "reasoning": "Analysis in progress."},
            finish_reason="length",
        )
        result = provider.generate(request())
        assert result.text == "" and result.reasoning == "Analysis in progress."
        assert result.finish_reason == "length"


def test_unknown_usage_stays_unknown(upstream: tuple[str, dict[str, Any]]) -> None:
    endpoint, state = upstream
    state["body"].pop("usage")
    with client(endpoint) as provider:
        assert provider.generate(request()).usage is None
