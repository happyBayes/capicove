#!/usr/bin/env python3
"""Measure client-observed latency for an explicitly configured search endpoint."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, cast
from urllib.parse import urlsplit


_ALLOWED_QUERY_FIELDS = frozenset({"query_id", "payload", "expected_result_query_id"})
_CREDENTIAL_FIELD_PARTS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_REPORT_SCHEMA_VERSION = 1
_MAX_QUERY_FILE_BYTES = 16 * 1024**2
_MAX_QUERY_COUNT = 10_000
_MAX_QUERY_PAYLOAD_BYTES = 4 * 1024**2
_MAX_RESPONSE_BODY_BYTES = 1024**2


class BenchmarkError(RuntimeError):
    """Report invalid configuration or an unusable benchmark prerequisite."""


class BenchmarkResponse(Protocol):
    """Expose response status and a separately timed complete-body read."""

    status_code: int

    def read(self) -> bytes: ...


class BenchmarkClient(Protocol):
    """Expose each client phase included in end-to-end timing."""

    def serialize(self, payload: Mapping[str, object]) -> bytes: ...

    def send(self, body: bytes, *, timeout_ms: int) -> BenchmarkResponse: ...

    def parse(self, body: bytes) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """Bind one authored query identity to its canonical transport payload."""

    query_id: str
    payload: Mapping[str, object]
    expected_result_query_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("query_id", self.query_id)
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise BenchmarkError("payload must be a non-empty JSON object")
        payload = dict(self.payload)
        _reject_credentials(payload)
        encoded = _canonical_json(payload).encode("utf-8")
        if len(encoded) > _MAX_QUERY_PAYLOAD_BYTES:
            raise BenchmarkError("query payload exceeds the 4 MiB benchmark ceiling")
        object.__setattr__(self, "payload", MappingProxyType(payload))
        if self.expected_result_query_id is not None:
            _require_non_blank("expected_result_query_id", self.expected_result_query_id)


@dataclass(frozen=True, slots=True)
class RequestSample:
    """Record one attempt without raw query or response content."""

    query_id: str
    elapsed_ns: int
    response_bytes: int
    succeeded: bool
    degraded: bool = False
    error_code: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank("query_id", self.query_id)
        _require_positive_integer("elapsed_ns", self.elapsed_ns)
        _require_non_negative_integer("response_bytes", self.response_bytes)
        if not isinstance(self.succeeded, bool) or not isinstance(self.degraded, bool):
            raise TypeError("succeeded and degraded must be bool values")
        if self.succeeded and self.error_code is not None:
            raise ValueError("a successful sample cannot have an error_code")
        if not self.succeeded:
            _require_non_blank("error_code", self.error_code)
        if self.degraded and not self.succeeded:
            raise ValueError("a failed sample cannot be degraded")

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed client time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    def to_dict(self) -> dict[str, object]:
        """Return redacted JSON-compatible sample data."""
        return {
            "degraded": self.degraded,
            "elapsed_ns": self.elapsed_ns,
            "error_code": self.error_code,
            "query_id": self.query_id,
            "response_bytes": self.response_bytes,
            "succeeded": self.succeeded,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Aggregate one fixed-concurrency measured request run."""

    concurrency: int
    warmup_requests: int
    measured_requests: int
    samples: tuple[RequestSample, ...]

    def __post_init__(self) -> None:
        _require_positive_integer("concurrency", self.concurrency)
        _require_non_negative_integer("warmup_requests", self.warmup_requests)
        _require_positive_integer("measured_requests", self.measured_requests)
        if self.concurrency not in (1, 4):
            raise ValueError("concurrency must be 1 or 4")
        samples = tuple(self.samples)
        if any(not isinstance(sample, RequestSample) for sample in samples):
            raise TypeError("samples must contain RequestSample values")
        if len(samples) != self.measured_requests:
            raise ValueError("sample count must equal measured_requests")
        object.__setattr__(self, "samples", samples)

    @property
    def failed_requests(self) -> int:
        return sum(not sample.succeeded for sample in self.samples)

    @property
    def degraded_requests(self) -> int:
        return sum(sample.degraded for sample in self.samples)

    @property
    def p50_ms(self) -> float:
        return percentile_ms(tuple(sample.elapsed_ms for sample in self.samples), 0.50)

    @property
    def p95_ms(self) -> float:
        return percentile_ms(tuple(sample.elapsed_ms for sample in self.samples), 0.95)

    @property
    def p99_ms(self) -> float:
        return percentile_ms(tuple(sample.elapsed_ms for sample in self.samples), 0.99)

    @property
    def maximum_response_bytes(self) -> int:
        return max(sample.response_bytes for sample in self.samples)

    def to_dict(self) -> dict[str, object]:
        """Return the complete local raw timing report without request content."""
        error_counts = Counter(
            sample.error_code for sample in self.samples if sample.error_code is not None
        )
        return {
            "concurrency": self.concurrency,
            "degraded_requests": self.degraded_requests,
            "error_counts": dict(sorted(error_counts.items())),
            "failed_requests": self.failed_requests,
            "maximum_response_bytes": self.maximum_response_bytes,
            "measured_requests": self.measured_requests,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "samples": [sample.to_dict() for sample in self.samples],
            "warmup_requests": self.warmup_requests,
        }


class _ReadableResponse(Protocol):
    def read(self, amount: int | None = None) -> bytes: ...

    def close(self) -> None: ...


class _UrllibResponse:
    def __init__(self, response: _ReadableResponse, status_code: int) -> None:
        self._response = response
        self.status_code = status_code

    def read(self) -> bytes:
        try:
            value = self._response.read(_MAX_RESPONSE_BODY_BYTES + 1)
        finally:
            self._response.close()
        if not isinstance(value, bytes):
            raise TypeError("HTTP response body must be bytes")
        if len(value) > _MAX_RESPONSE_BODY_BYTES:
            raise BenchmarkError("response body exceeds the 1 MiB safety ceiling")
        return value


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        del request, file_pointer, code, message, headers, new_url
        return None


class UrllibHttpClient:
    """Use the standard library HTTP client without implicit retries."""

    def __init__(self, endpoint: str, *, bearer_token: str | None = None) -> None:
        self._endpoint = endpoint
        self._bearer_token = bearer_token
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def serialize(self, payload: Mapping[str, object]) -> bytes:
        return _canonical_json(dict(payload)).encode("utf-8")

    def send(self, body: bytes, *, timeout_ms: int) -> BenchmarkResponse:
        headers = {"Content-Type": "application/json"}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            response = self._opener.open(request, timeout=timeout_ms / 1_000)
        except urllib.error.HTTPError as error:
            return _UrllibResponse(cast(_ReadableResponse, error), error.code)
        return _UrllibResponse(cast(_ReadableResponse, response), int(response.status))

    def parse(self, body: bytes) -> Mapping[str, object]:
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("response must be a JSON object")
        return cast(dict[str, object], value)


def percentile_ms(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic nearest-rank percentile in milliseconds."""
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(quantile)
        or not 0 < quantile <= 1
    ):
        raise ValueError("quantile must be finite and within (0, 1]")
    samples = tuple(values)
    if not samples:
        raise ValueError("percentile values must not be empty")
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("percentile values must be finite non-negative numbers")
        if not math.isfinite(value) or value < 0:
            raise ValueError("percentile values must be finite non-negative numbers")
    ordered = sorted(float(value) for value in samples)
    rank = math.ceil(float(quantile) * len(ordered))
    return ordered[rank - 1]


def measure_one_request(
    client: BenchmarkClient,
    query: BenchmarkQuery,
    *,
    timeout_ms: int,
) -> RequestSample:
    """Time serialization through complete read and parse without a retry."""
    _require_positive_integer("timeout_ms", timeout_ms)
    started_ns = time.perf_counter_ns()
    response_bytes = 0
    try:
        request_body = client.serialize(query.payload)
        response = client.send(request_body, timeout_ms=timeout_ms)
        body = response.read()
        response_bytes = len(body)
        if not 200 <= response.status_code < 300:
            return _failed_sample(
                query,
                started_ns,
                response_bytes,
                f"http_status_{response.status_code}",
            )
        payload = client.parse(body)
        result, error_code = _decode_tool_result(payload)
        if error_code is not None:
            return _failed_sample(query, started_ns, response_bytes, error_code)
        if result is None:
            return _failed_sample(query, started_ns, response_bytes, "malformed_response")
        result_query_id = result.get("query_id")
        if not isinstance(result_query_id, str) or not result_query_id.strip():
            return _failed_sample(query, started_ns, response_bytes, "malformed_response")
        if (
            query.expected_result_query_id is not None
            and result_query_id != query.expected_result_query_id
        ):
            return _failed_sample(query, started_ns, response_bytes, "query_id_mismatch")
        degraded = result.get("degraded", False)
        if not isinstance(degraded, bool):
            return _failed_sample(query, started_ns, response_bytes, "malformed_response")
        elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
        return RequestSample(
            query_id=query.query_id,
            elapsed_ns=elapsed_ns,
            response_bytes=response_bytes,
            succeeded=True,
            degraded=degraded,
        )
    except TimeoutError:
        return _failed_sample(query, started_ns, response_bytes, "timeout")
    except urllib.error.URLError as error:
        code = "timeout" if isinstance(error.reason, TimeoutError) else "transport_error"
        return _failed_sample(query, started_ns, response_bytes, code)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        return _failed_sample(query, started_ns, response_bytes, "malformed_response")


def measure_requests(
    client: BenchmarkClient,
    queries: Sequence[BenchmarkQuery],
    *,
    warmup_requests: int,
    measured_requests: int,
    concurrency: int,
    timeout_ms: int,
    after_warmup: Callable[[], None] | None = None,
) -> BenchmarkRun:
    """Warm once, then measure a deterministic query cycle at fixed concurrency."""
    frozen_queries = tuple(queries)
    if not frozen_queries:
        raise BenchmarkError("queries must not be empty")
    if any(not isinstance(query, BenchmarkQuery) for query in frozen_queries):
        raise TypeError("queries must contain BenchmarkQuery values")
    _require_non_negative_integer("warmup_requests", warmup_requests)
    _require_positive_integer("measured_requests", measured_requests)
    _require_positive_integer("timeout_ms", timeout_ms)
    if concurrency not in (1, 4):
        raise BenchmarkError("concurrency must be 1 or 4")

    for index in range(warmup_requests):
        sample = measure_one_request(
            client,
            frozen_queries[index % len(frozen_queries)],
            timeout_ms=timeout_ms,
        )
        if not sample.succeeded:
            raise BenchmarkError(f"warm-up request failed with {sample.error_code}")

    if after_warmup is not None:
        after_warmup()

    measured_queries = tuple(
        frozen_queries[index % len(frozen_queries)] for index in range(measured_requests)
    )
    if concurrency == 1:
        samples = tuple(
            measure_one_request(client, query, timeout_ms=timeout_ms) for query in measured_queries
        )
    else:

        def measure(query: BenchmarkQuery) -> RequestSample:
            return measure_one_request(client, query, timeout_ms=timeout_ms)

        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="video-search-benchmark",
        ) as executor:
            samples = tuple(executor.map(measure, measured_queries))
    return BenchmarkRun(
        concurrency=concurrency,
        warmup_requests=warmup_requests,
        measured_requests=measured_requests,
        samples=samples,
    )


def load_queries(path: Path) -> tuple[BenchmarkQuery, ...]:
    """Load a strict absolute authored JSONL query set."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise BenchmarkError("queries path must be absolute")
    if path.is_symlink():
        raise BenchmarkError("queries path must not be a symlink")
    try:
        if path.stat().st_size > _MAX_QUERY_FILE_BYTES:
            raise BenchmarkError("queries file exceeds the 16 MiB safety ceiling")
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkError("unable to read queries path") from error
    if len(lines) > _MAX_QUERY_COUNT:
        raise BenchmarkError("queries file exceeds the 10000-row safety ceiling")
    queries: list[BenchmarkQuery] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BenchmarkError(f"blank JSONL row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkError(f"invalid JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise BenchmarkError(f"query row {line_number} must be an object")
        unknown = set(row).difference(_ALLOWED_QUERY_FIELDS)
        if unknown:
            raise BenchmarkError(f"query row {line_number} contains unknown fields")
        if set(row).difference({"expected_result_query_id"}) != {"query_id", "payload"}:
            raise BenchmarkError(f"query row {line_number} has invalid required fields")
        query_id = row["query_id"]
        payload = row["payload"]
        expected = row.get("expected_result_query_id")
        if not isinstance(query_id, str) or not isinstance(payload, dict):
            raise BenchmarkError(f"query row {line_number} has invalid field types")
        if expected is not None and not isinstance(expected, str):
            raise BenchmarkError(f"query row {line_number} has invalid field types")
        if query_id in seen:
            raise BenchmarkError(f"duplicate query_id {query_id!r}")
        seen.add(query_id)
        queries.append(
            BenchmarkQuery(
                query_id=query_id,
                payload=cast(dict[str, object], payload),
                expected_result_query_id=expected,
            )
        )
    if not queries:
        raise BenchmarkError("queries file must not be empty")
    return tuple(queries)


def validate_endpoint(endpoint: str, *, allow_lan: bool = False) -> str:
    """Allow only explicit loopback or encrypted private-IP search endpoints."""
    _require_non_blank("endpoint", endpoint)
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise BenchmarkError("endpoint scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise BenchmarkError("endpoint URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise BenchmarkError("endpoint URL must not contain a query or fragment")
    if parsed.path != "/v1/search":
        raise BenchmarkError("endpoint path must be /v1/search")
    try:
        port = parsed.port
    except ValueError as error:
        raise BenchmarkError("endpoint port is invalid") from error
    if port is None:
        raise BenchmarkError("endpoint must include an explicit port")
    hostname = parsed.hostname
    if hostname is None:
        raise BenchmarkError("endpoint must include an IP literal")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as error:
        raise BenchmarkError("endpoint host must be an IP literal") from error
    if address.is_loopback:
        return endpoint
    if not allow_lan:
        raise BenchmarkError("non-loopback endpoint requires --allow-lan")
    if parsed.scheme != "https":
        raise BenchmarkError("LAN qualification requires HTTPS")
    if (
        not address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
    ):
        raise BenchmarkError("LAN endpoint must use a private address")
    return endpoint


def measure_tcp_rtt_ms(endpoint: str, *, timeout_ms: int) -> float:
    """Measure one TCP-connect RTT-like probe to the validated endpoint."""
    _require_positive_integer("timeout_ms", timeout_ms)
    parsed = urlsplit(endpoint)
    if parsed.hostname is None or parsed.port is None:
        raise BenchmarkError("validated endpoint is missing host or port")
    started_ns = time.perf_counter_ns()
    try:
        with socket.create_connection(
            (parsed.hostname, parsed.port),
            timeout=timeout_ms / 1_000,
        ):
            pass
    except OSError as error:
        raise BenchmarkError("network RTT probe failed") from error
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def _decode_tool_result(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, str | None]:
    status = payload.get("status")
    if status != "succeeded":
        error = payload.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("code"), str):
            return None, f"tool_{error['code']}"
        return None, "tool_failed"
    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None, "malformed_response"
    nested_status = result.get("status")
    if nested_status is not None and nested_status != "succeeded":
        return None, "search_failed"
    return cast(Mapping[str, object], result), None


def _failed_sample(
    query: BenchmarkQuery,
    started_ns: int,
    response_bytes: int,
    error_code: str,
) -> RequestSample:
    return RequestSample(
        query_id=query.query_id,
        elapsed_ns=max(1, time.perf_counter_ns() - started_ns),
        response_bytes=response_bytes,
        succeeded=False,
        error_code=error_code,
    )


def _reject_credentials(value: object, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise BenchmarkError("payload object keys must be strings")
            lowered = key.casefold()
            if any(part in lowered for part in _CREDENTIAL_FIELD_PARTS):
                raise BenchmarkError(f"credential field is forbidden at {path}.{key}")
            _reject_credentials(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_credentials(nested, path=f"{path}[{index}]")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkError("payload must contain finite JSON values") from error


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field_name} must not be blank")


def _require_positive_integer(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkError(f"{field_name} must be a positive integer")


def _require_non_negative_integer(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{field_name} must be a non-negative integer")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure complete client-observed Capicove video-search latency."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), required=True)
    parser.add_argument("--timeout-ms", type=int, default=1_000)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--allow-lan", action="store_true")
    return parser


def _write_report(
    report_dir: Path,
    *,
    endpoint: str,
    queries_path: Path,
    network_rtt_ms: float,
    run: BenchmarkRun,
) -> Path:
    if not report_dir.is_absolute():
        raise BenchmarkError("report-dir must be absolute")
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if report_dir.is_symlink() or not report_dir.is_dir():
        raise BenchmarkError("report-dir must be a real directory")
    query_bytes = queries_path.read_bytes()
    endpoint_identity = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
    payload = {
        "endpoint_sha256": endpoint_identity,
        "network_rtt_ms": network_rtt_ms,
        "query_set_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "run": run.to_dict(),
        "schema_version": _REPORT_SCHEMA_VERSION,
    }
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    report_sha256 = hashlib.sha256(encoded).hexdigest()
    path = report_dir / f"video-search-latency-c{run.concurrency}-{report_sha256[:16]}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise BenchmarkError("identical benchmark report already exists") from error
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit fixed-concurrency benchmark and write a local report."""
    arguments = _build_parser().parse_args(argv)
    try:
        endpoint = validate_endpoint(arguments.endpoint, allow_lan=arguments.allow_lan)
        if arguments.warmup < 100:
            raise BenchmarkError("warmup must be at least 100")
        if arguments.requests < 1_000:
            raise BenchmarkError("requests must be at least 1000")
        _require_positive_integer("timeout-ms", arguments.timeout_ms)
        queries_path = arguments.queries
        queries = load_queries(queries_path)
        bearer_token = os.environ.get("CAPICOVE_VIDEO_SEARCH_BEARER_TOKEN")
        network_rtt_ms = measure_tcp_rtt_ms(endpoint, timeout_ms=arguments.timeout_ms)
        if network_rtt_ms > 20.0:
            raise BenchmarkError("network RTT exceeds the 20 ms qualification ceiling")
        run = measure_requests(
            UrllibHttpClient(endpoint, bearer_token=bearer_token),
            queries,
            warmup_requests=arguments.warmup,
            measured_requests=arguments.requests,
            concurrency=arguments.concurrency,
            timeout_ms=arguments.timeout_ms,
        )
        report_path = _write_report(
            arguments.report_dir,
            endpoint=endpoint,
            queries_path=queries_path,
            network_rtt_ms=network_rtt_ms,
            run=run,
        )
    except BenchmarkError as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 2
    print(report_path)
    return 0 if run.failed_requests == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
