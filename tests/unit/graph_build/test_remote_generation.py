"""Exercise the compatible transport against a real authored HTTP server."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from capicove.graph_build.extraction import extraction_request
from capicove.graph_build.inference import DispatchControl, RateLimitError
from capicove.graph_build.profiles import StructuredRequest
from capicove.graph_build.remote_generation import (
    OpenAICompatibleGenerator,
    OpenAICompatibleProfile,
    OpenAICompatibleTokenizer,
)
from tests.unit.graph_build.test_extraction import build_profile, unit
from tests.unit.graph_build.test_local_generation import Runtime
from tests.unit.model.test_local_assets import assets_at


@contextmanager
def server(
    status: int = 200, *, change: dict[str, Any] | None = None, delay: float = 0
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    captured: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append(body)
            time.sleep(delay)
            self.send_response(status)
            self.send_header("Location", "http://127.0.0.1:1/leak")
            self.end_headers()
            response = {
                "model": "operator-selected-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"entities":[],"relationships":[]}'},
                    }
                ],
                "usage": {
                    "prompt_tokens": captured[-1].pop("_expected", 1),
                    "completion_tokens": 10,
                },
            }
            response.update(change or {})
            try:
                self.wfile.write(json.dumps(response).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", captured
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def profile_at(
    tmp_path: Path, endpoint: str = "http://127.0.0.1:1/v1", **options: Any
) -> OpenAICompatibleProfile:
    assets = assets_at(
        tmp_path / "tokenizer",
        {
            "tokenizer_config.json": json.dumps(
                {"chat_template": "<{{ messages }}:assistant>"}
            ).encode(),
        },
    )
    return OpenAICompatibleProfile(
        endpoint=endpoint,
        model="operator-selected-model",
        model_revision="operator-revision-1",
        tokenizer_assets=assets,
        context_limit=32768,
        network_enabled=True,
        **options,
    )


def request_for(profile: OpenAICompatibleProfile) -> StructuredRequest:
    policy = replace(
        build_profile().inference,
        provider_id=profile.provider_id,
        model_id=profile.model,
        model_revision=profile.model_revision,
        tokenizer_id="sha256:" + profile.tokenizer_assets.manifest_sha256,
        tokenizer_revision="sha256:" + profile.tokenizer_assets.manifest_sha256,
        context_window_tokens=32768,
        max_output_tokens=128,
    )
    return extraction_request(unit(), replace(build_profile(), inference=policy))


def control(seconds: float = 10, cancelled: Callable[[], bool] = lambda: False) -> DispatchControl:
    return DispatchControl(time.monotonic() + seconds, time.monotonic, cancelled)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/v1",
        "https://u:p@example.com/v1",
        "https://example.com/v1?key=secret",
        "file:///tmp/v1",
    ],
)
def test_reject_unsafe_endpoint(tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(ValueError):
        profile_at(tmp_path, endpoint)


def test_network_and_credentials_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = profile_at(tmp_path)
    with pytest.raises(ValueError, match="network"):
        replace(profile, network_enabled=False)
    monkeypatch.delenv("CAPICOVE_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="credential"):
        OpenAICompatibleGenerator(profile).generate(request_for(profile), control=control())


def test_complete_request_uses_verified_local_chat_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    request = request_for(profile)
    tokenizer = OpenAICompatibleTokenizer(profile)
    count = tokenizer.count_request(request)
    assert count > len(request.canonical_payload)
    assert runtime.tokenizer.messages[1]["content"] == request.canonical_payload.decode()
    assert all(call[2]["local_files_only"] for call in runtime.loads)


def test_tokenizer_accepts_transformers_batch_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    monkeypatch.setattr(
        runtime.tokenizer,
        "apply_chat_template",
        lambda *args, **kwargs: {
            "input_ids": [10, 20, 30],
            "attention_mask": [1, 1, 1],
        },
    )
    profile = profile_at(tmp_path)
    assert OpenAICompatibleTokenizer(profile).count_request(request_for(profile)) == 3


@pytest.mark.parametrize(
    "status,exception",
    [(401, ValueError), (403, ValueError), (429, RateLimitError), (302, ValueError)],
)
def test_transport_errors_are_sanitized_and_redirects_not_followed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    exception: type[Exception],
) -> None:
    monkeypatch.setenv("CAPICOVE_LLM_API_KEY", "private-test-value")
    monkeypatch.setattr(OpenAICompatibleTokenizer, "count_request", lambda self, request: 1)
    with server(status) as (endpoint, captured):
        profile = profile_at(tmp_path, endpoint)
        with pytest.raises(exception) as error:
            OpenAICompatibleGenerator(profile).generate(request_for(profile), control=control())
        assert "private-test-value" not in str(error.value)
        assert len(captured) == 1


def test_real_transport_preserves_model_and_schema_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPICOVE_LLM_API_KEY", "private-test-value")
    monkeypatch.setattr(OpenAICompatibleTokenizer, "count_request", lambda self, request: 1)
    with server() as (endpoint, captured):
        profile = profile_at(tmp_path, endpoint)
        result = OpenAICompatibleGenerator(profile).generate(
            request_for(profile), control=control()
        )
        assert json.loads(result.payload) == {"entities": [], "relationships": []}
        assert result.input_tokens == 1
        assert captured[0]["model"] == profile.model
        assert "tools" not in captured[0]


@pytest.mark.parametrize(
    "change",
    [
        {"model": "wrong-model"},
        {"usage": {"prompt_tokens": 2, "completion_tokens": 10}},
        {"choices": []},
        {"usage": {"prompt_tokens": True, "completion_tokens": 10}},
    ],
)
def test_server_identity_and_template_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    monkeypatch.setenv("CAPICOVE_LLM_API_KEY", "private-test-value")
    monkeypatch.setattr(OpenAICompatibleTokenizer, "count_request", lambda self, request: 1)
    with server(change=change) as (endpoint, _):
        profile = profile_at(tmp_path, endpoint)
        with pytest.raises(ValueError):
            OpenAICompatibleGenerator(profile).generate(request_for(profile), control=control())


def test_inflight_deadline_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPICOVE_LLM_API_KEY", "private-test-value")
    monkeypatch.setattr(OpenAICompatibleTokenizer, "count_request", lambda self, request: 1)
    with server(delay=2) as (endpoint, _):
        profile = profile_at(tmp_path, endpoint)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            OpenAICompatibleGenerator(profile).generate(request_for(profile), control=control(0.15))
        assert time.monotonic() - started < 1
        with pytest.raises(ValueError, match="cancelled"):
            OpenAICompatibleGenerator(profile).generate(
                request_for(profile), control=control(cancelled=lambda: True)
            )


def test_remote_tokenizer_fingerprint_is_usable_by_graph_chunking(tmp_path: Path) -> None:
    import re

    tokenizer = OpenAICompatibleTokenizer(profile_at(tmp_path))
    assert re.fullmatch(r"[0-9a-f]{64}", tokenizer.fingerprint)


def test_remote_generator_exposes_selected_model_identity(tmp_path: Path) -> None:
    profile = profile_at(tmp_path)
    generator = OpenAICompatibleGenerator(profile)
    assert generator.provider_id == profile.provider_id
    assert generator.model_id == profile.model
    assert generator.model_revision == profile.model_revision


def test_transport_does_not_run_host_policy_on_worker_thread(tmp_path: Path) -> None:
    from capicove.graph_build.remote_generation import _post

    owner = threading.get_ident()

    def policy() -> bool:
        if threading.get_ident() != owner:
            raise RuntimeError("host policy is thread-affine")
        return False

    with server() as (endpoint, _captured):
        response = _post(
            profile_at(tmp_path, endpoint),
            b"{}",
            "authored-test-key",
            8192,
            control(cancelled=policy),
        )
    assert json.loads(response)["model"] == "operator-selected-model"
