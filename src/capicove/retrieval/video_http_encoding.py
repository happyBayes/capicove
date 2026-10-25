"""Reuse the local SigLIP2 API without loading PyTorch into the FAISS process.

The composition root must start the existing model API with a verified model
manifest and the declared preprocessing policy. This adapter accepts numeric
loopback endpoints only, disables proxies and redirects, and never downloads a
model. It does not treat local HTTP as a remotely authenticated model service.
"""

from __future__ import annotations

import ipaddress
import json
import math
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit
from uuid import uuid4

from capicove.contracts import EmbeddingSpaceRef, ImageQuery, TextQuery


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        raise ValueError("local model redirects are forbidden")


class LocalSiglip2QueryEncoder:
    """Encode through one explicitly started, isolated local model process."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        embedding_space: EmbeddingSpaceRef,
        resolve_image: Callable[[ImageQuery], bytes] | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or not ipaddress.ip_address(parsed.hostname).is_loopback
        ):
            raise ValueError("model endpoint must be a numeric loopback HTTP origin with a port")
        if not model_id.strip():
            raise ValueError("model identity is required")
        self._endpoint = endpoint.rstrip("/")
        self._model_id = model_id
        self._space = embedding_space
        self._resolve_image = resolve_image
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def encode(
        self,
        query: TextQuery | ImageQuery,
        *,
        embedding_space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[float, ...]:
        """Preserve the caller deadline and require the pinned response identity."""
        if embedding_space != self._space:
            raise ValueError("embedding space mismatch")
        self._remaining(deadline)
        if isinstance(query, TextQuery):
            if len(query.text) > 500:
                raise ValueError("local model text limit exceeded")
            return self._request(
                "/v1/embeddings/text",
                json.dumps({"texts": [query.text]}).encode(),
                "application/json",
                deadline,
            )
        if not isinstance(query, ImageQuery):
            raise ValueError("unsupported query modality")
        if self._resolve_image is None:
            raise PermissionError("query image requires an authorized resolver")
        return self._preview(self._resolve_image(query), deadline)

    def encode_preview_batch(self, previews: tuple[bytes, ...]) -> tuple[tuple[float, ...], ...]:
        """Reuse the same model image endpoint for bounded offline preview batches."""
        if not 1 <= len(previews) <= 16:
            raise ValueError("preview batch must contain between one and sixteen images")
        deadline = datetime.now(UTC) + timedelta(seconds=30)
        return tuple(self._preview(preview, deadline) for preview in previews)

    def _preview(self, payload: bytes, deadline: datetime) -> tuple[float, ...]:
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 8 * 1024**2:
            raise ValueError("preview must contain at most 8 MiB")
        boundary = f"capicove-{uuid4().hex}"
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="preview"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self._request(
            "/v1/embeddings/image", body, f"multipart/form-data; boundary={boundary}", deadline
        )

    def _request(
        self, path: str, body: bytes, content_type: str, deadline: datetime
    ) -> tuple[float, ...]:
        request = urllib.request.Request(
            self._endpoint + path, data=body, headers={"Content-Type": content_type}, method="POST"
        )
        # Only a validated numeric loopback origin and fixed endpoint path are used.
        with self._opener.open(request, timeout=self._remaining(deadline)) as response:
            payload = response.read(1024**2 + 1)
        self._remaining(deadline)
        if len(payload) > 1024**2:
            raise ValueError("local model response exceeds capacity")
        data = json.loads(payload)
        if not isinstance(data, dict) or (
            data.get("model_id") != self._model_id
            or data.get("model_revision") != self._space.model_revision
            or data.get("dimension") != self._space.dimension
        ):
            raise ValueError("local model response identity mismatch")
        rows = data.get("embeddings")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
            raise ValueError("local model embedding shape mismatch")
        vector = rows[0]
        if (
            len(vector) != self._space.dimension
            or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                for v in vector
            )
            or not math.isclose(sum(v * v for v in vector), 1.0, abs_tol=2e-4)
        ):
            raise ValueError("local model returned invalid normalized vector")
        return tuple(float(v) for v in vector)

    @staticmethod
    def _remaining(deadline: datetime) -> float:
        if deadline.tzinfo is None:
            raise ValueError("deadline must be timezone aware")
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("model deadline exceeded")
        return remaining


__all__ = ["LocalSiglip2QueryEncoder"]
