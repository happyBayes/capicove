"""Tests for the explicit client-observed video-search benchmark."""

from __future__ import annotations

import json
import socket
import sys
from collections.abc import Mapping
from importlib import util
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "benchmark_video_search.py"
_SPEC = util.spec_from_file_location("capicove_benchmark_video_search", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib invariant
    raise RuntimeError("unable to load benchmark_video_search.py")
_BENCHMARK = util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)

BenchmarkError = _BENCHMARK.BenchmarkError
BenchmarkQuery = _BENCHMARK.BenchmarkQuery
load_queries = _BENCHMARK.load_queries
measure_one_request = _BENCHMARK.measure_one_request
measure_requests = _BENCHMARK.measure_requests
percentile_ms = _BENCHMARK.percentile_ms
validate_endpoint = _BENCHMARK.validate_endpoint


class RecordingResponse:
    """Return one response while recording the full-body read."""

    def __init__(self, owner: RecordingHttpClient, status_code: int, body: bytes) -> None:
        self._owner = owner
        self.status_code = status_code
        self._body = body

    def read(self) -> bytes:
        self._owner.events.append("read_body")
        return self._body


class RecordingHttpClient:
    """Expose transport phases to prove the timer encloses all of them."""

    def __init__(
        self,
        response_body: bytes = b'{"status":"succeeded","result":{"query_id":"q-1","degraded":false,"hits":[]}}',
        *,
        status_code: int = 200,
        failure: Exception | None = None,
    ) -> None:
        self.response_body = response_body
        self.status_code = status_code
        self.failure = failure
        self.events: list[str] = []

    def serialize(self, payload: Mapping[str, object]) -> bytes:
        self.events.append("serialize")
        return json.dumps(dict(payload)).encode("utf-8")

    def send(self, body: bytes, *, timeout_ms: int) -> RecordingResponse:
        del body, timeout_ms
        self.events.append("send")
        if self.failure is not None:
            raise self.failure
        self.events.append("receive")
        return RecordingResponse(self, self.status_code, self.response_body)

    def parse(self, body: bytes) -> Mapping[str, object]:
        self.events.append("parse")
        value = json.loads(body)
        assert isinstance(value, dict)
        return value


def _query(expected_result_query_id: str | None = "q-1") -> Any:
    return BenchmarkQuery(
        query_id="fixture-query-1",
        payload={
            "query": {"type": "text", "text": "authored query"},
            "targets": ["keyframe", "video_segment"],
            "top_k": 20,
        },
        expected_result_query_id=expected_result_query_id,
    )


def test_client_timer_includes_serialization_transport_and_response_read() -> None:
    client = RecordingHttpClient()

    sample = measure_one_request(client, _query(), timeout_ms=1_000)

    assert client.events == ["serialize", "send", "receive", "read_body", "parse"]
    assert sample.elapsed_ns > 0
    assert sample.response_bytes == len(client.response_body)
    assert sample.succeeded is True


def test_measurement_baseline_is_captured_after_warmup() -> None:
    client = RecordingHttpClient()
    warmup_counts: list[int] = []
    run = measure_requests(
        client,
        (_query(),),
        warmup_requests=3,
        measured_requests=2,
        concurrency=1,
        timeout_ms=1000,
        after_warmup=lambda: warmup_counts.append(client.events.count("send")),
    )
    assert warmup_counts == [3]
    assert run.measured_requests == 2
    assert client.events.count("send") == 5


def test_nearest_rank_percentile_is_deterministic() -> None:
    assert percentile_ms((100.0, 200.0, 300.0, 400.0), 0.95) == 400.0
    assert percentile_ms((400.0, 100.0, 300.0, 200.0), 0.50) == 200.0


@pytest.mark.parametrize("quantile", [0.0, 1.1, float("nan")])
def test_percentile_rejects_invalid_quantile(quantile: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        percentile_ms((1.0,), quantile)


@pytest.mark.parametrize(
    ("client", "error_code"),
    [
        (RecordingHttpClient(status_code=503), "http_status_503"),
        (
            RecordingHttpClient(
                response_body=b'{"status":"failed","error":{"code":"unavailable"}}'
            ),
            "tool_unavailable",
        ),
        (RecordingHttpClient(response_body=b"not-json"), "malformed_response"),
        (
            RecordingHttpClient(
                response_body=b'{"status":"succeeded","result":{"query_id":"wrong","degraded":false,"hits":[]}}'
            ),
            "query_id_mismatch",
        ),
        (RecordingHttpClient(failure=TimeoutError()), "timeout"),
    ],
)
def test_failed_requests_are_measured_not_retried(
    client: Any,
    error_code: str,
) -> None:
    sample = measure_one_request(client, _query(), timeout_ms=1_000)

    assert sample.succeeded is False
    assert sample.error_code == error_code
    assert client.events.count("send") == 1


def test_degraded_response_is_counted() -> None:
    client = RecordingHttpClient(
        response_body=b'{"status":"succeeded","result":{"query_id":"q-1","degraded":true,"hits":[]}}'
    )

    sample = measure_one_request(client, _query(), timeout_ms=1_000)

    assert sample.succeeded is True
    assert sample.degraded is True


def test_measure_requests_excludes_warmup_and_preserves_exact_count() -> None:
    client = RecordingHttpClient()

    run = measure_requests(
        client,
        (_query(),),
        warmup_requests=3,
        measured_requests=7,
        concurrency=1,
        timeout_ms=1_000,
    )

    assert len(run.samples) == 7
    assert run.warmup_requests == 3
    assert run.measured_requests == 7
    assert run.failed_requests == 0
    assert client.events.count("send") == 10


def test_endpoint_defaults_to_loopback_and_rejects_credentials() -> None:
    assert validate_endpoint("http://127.0.0.1:8000/v1/search") == (
        "http://127.0.0.1:8000/v1/search"
    )
    with pytest.raises(BenchmarkError, match="credentials"):
        # Deliberately invalid, fictional credentials exercise URL rejection.
        validate_endpoint("http://user:secret@127.0.0.1:8000/v1/search")  # pragma: allowlist secret
    with pytest.raises(BenchmarkError, match="loopback"):
        validate_endpoint("http://192.168.1.10:8000/v1/search")


def test_endpoint_allows_private_literal_only_with_explicit_lan() -> None:
    assert (
        validate_endpoint("https://192.168.1.10:8000/v1/search", allow_lan=True)
        == "https://192.168.1.10:8000/v1/search"
    )
    with pytest.raises(BenchmarkError, match="HTTPS"):
        validate_endpoint("http://192.168.1.10:8000/v1/search", allow_lan=True)
    with pytest.raises(BenchmarkError, match="IP literal"):
        validate_endpoint("https://example.test:8000/v1/search", allow_lan=True)


def test_endpoint_rejects_non_private_and_ambiguous_urls() -> None:
    with pytest.raises(BenchmarkError, match="private"):
        validate_endpoint("https://8.8.8.8:443/v1/search", allow_lan=True)
    with pytest.raises(BenchmarkError, match="query or fragment"):
        validate_endpoint("http://127.0.0.1:8000/v1/search?token=secret")
    with pytest.raises(BenchmarkError, match="port"):
        validate_endpoint("http://127.0.0.1/v1/search")


def test_load_queries_requires_absolute_strict_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "fixture-query-1",
                "payload": {"query": "authored"},
                "expected_result_query_id": "q-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_queries(path)[0].query_id == "fixture-query-1"
    with pytest.raises(BenchmarkError, match="absolute"):
        load_queries(Path("queries.jsonl"))


def test_load_queries_rejects_unknown_fields_duplicates_and_secrets(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text('{"query_id":"q","payload":{"query":"x"},"extra":1}\n', encoding="utf-8")
    with pytest.raises(BenchmarkError, match="unknown fields"):
        load_queries(unknown)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"query_id":"q","payload":{"query":"x"}}\n{"query_id":"q","payload":{"query":"y"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="duplicate query_id"):
        load_queries(duplicate)

    secret = tmp_path / "secret.jsonl"
    secret.write_text(
        '{"query_id":"q","payload":{"authorization":"secret"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="credential"):
        load_queries(secret)


def test_real_network_resolution_is_not_used_for_literal_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolution(*args: object, **kwargs: object) -> object:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(socket, "getaddrinfo", fail_resolution)

    assert validate_endpoint("http://127.0.0.1:8000/v1/search").startswith("http://")
