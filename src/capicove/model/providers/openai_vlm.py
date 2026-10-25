"""Generic OpenAI/vLLM multimodal HTTP adapter with explicit network authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import ssl
from threading import Lock
from typing import cast

from capicove.model.http_profile import HttpModelProfile
from capicove.model.vlm import (
    MAX_REQUEST_BYTES,
    VlmMedia,
    VlmRequest,
    VlmResult,
    VlmTokenCount,
    VlmUsage,
    parse_json,
)
from capicove.model.vlm import (
    VlmError as VlmError,
)
from capicove.model.vlm_tools import VlmToolCall, VlmToolChoice


MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class OpenAICompatibleVlm:
    """Call a configured served model without model SDKs, proxies, redirects or retries.

    One synchronous call owns the provider until its socket has closed. A local
    timeout cannot guarantee that an upstream GPU has cancelled generation.
    """

    def __init__(self, profile: HttpModelProfile) -> None:
        if not isinstance(profile, HttpModelProfile) or profile.provider != "openai-vlm":
            raise ValueError("openai-vlm HTTP profile required")
        self.profile = profile
        self.model_id = profile.model
        self._lock = Lock()
        self._closed = False

    def __enter__(self) -> OpenAICompatibleVlm:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Wait for active I/O to finish, then permanently close this provider."""
        with self._lock:
            self._closed = True

    def validate_request(self, request: VlmRequest) -> None:
        """Apply per-model modality and output limits before making any request."""
        if not isinstance(request, VlmRequest):
            raise ValueError("VlmRequest required")
        if request.max_tokens > self.profile.max_output_tokens:
            raise ValueError("output token limit exceeded")
        counts = {"image": 0, "video": 0}
        for message in request.messages:
            for part in message.content:
                if isinstance(part, VlmMedia):
                    if part.modality not in self.profile.modalities:
                        raise ValueError("unsupported model modality")
                    counts[part.modality] += 1
        if counts["image"] > self.profile.max_images or counts["video"] > self.profile.max_videos:
            raise ValueError("media limit exceeded")

    def generate(self, request: VlmRequest) -> VlmResult:
        """Return one choice and server usage, preserving truncation explicitly."""
        self.validate_request(request)
        if not self._lock.acquire(blocking=False):
            raise VlmError("provider_busy", 429)
        try:
            if self._closed:
                raise RuntimeError("VLM provider is closed")
            body = request.to_openai(self.model_id)
            if self.profile.chat_template_kwargs:
                body["chat_template_kwargs"] = dict(self.profile.chat_template_kwargs)
            payload = json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            if len(payload) > MAX_REQUEST_BYTES:
                raise ValueError("outbound request exceeds size limit")
            response = self._send(payload)
            try:
                return self._result(response, request, hashlib.sha256(payload).hexdigest())
            except (
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                UnicodeError,
                RecursionError,
            ) as error:
                raise VlmError("invalid_upstream_response") from error
        finally:
            self._lock.release()

    def count_tokens(self, request: VlmRequest) -> VlmTokenCount:
        """Count the template and expanded media through vLLM's tokenize API."""
        self.validate_request(request)
        if self.profile.token_counting != "vllm":  # nosec B105 - token-counting protocol selector
            raise VlmError("token_counting_not_configured", 501)
        if not self._lock.acquire(blocking=False):
            raise VlmError("provider_busy", 429)
        try:
            if self._closed:
                raise RuntimeError("VLM provider is closed")
            chat = request.to_openai(self.model_id)
            body = {key: chat[key] for key in ("model", "messages", "tools") if key in chat}
            body["add_generation_prompt"] = True
            if self.profile.chat_template_kwargs:
                body["chat_template_kwargs"] = dict(self.profile.chat_template_kwargs)
            payload = json.dumps(
                body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            if len(payload) > MAX_REQUEST_BYTES:
                raise ValueError("outbound request exceeds size limit")
            response = self._send(payload, path="/tokenize")
            try:
                raw = parse_json(response)
                if type(raw.get("count")) is not int or type(raw.get("max_model_len")) is not int:
                    raise ValueError("invalid token counts")
                count = VlmTokenCount(
                    self.model_id,
                    cast(int, raw["count"]),
                    cast(int, raw["max_model_len"]),
                    hashlib.sha256(payload).hexdigest(),
                    self.profile.identity,
                )
                tokens = raw.get("tokens")
                if (
                    not isinstance(tokens, list)
                    or len(tokens) != count.input_tokens
                    or any(type(t) is not int or t < 0 for t in tokens)
                ):
                    raise ValueError("inconsistent tokenizer result")
                return count
            except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
                raise VlmError("invalid_upstream_response") from None
        finally:
            self._lock.release()

    def _send(self, payload: bytes, *, path: str = "/v1/chat/completions") -> bytes:
        """Synchronous port; async callers should use their worker pool."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("run synchronous VLM calls in a worker thread")
        return asyncio.run(self._send_async(payload, path))

    async def _send_async(self, payload: bytes, path: str) -> bytes:
        # Optional transport dependencies never enter canonical contract imports.
        import aiohttp

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        if self.profile.api_key_env is not None:
            key = os.environ.get(self.profile.api_key_env, "")
            if not key or len(key) > 8192 or any(not 33 <= ord(c) <= 126 for c in key):
                raise ValueError("VLM API key environment value is missing or invalid")
            headers["Authorization"] = "Bearer " + key
        try:
            async with asyncio.timeout(self.profile.timeout_seconds):
                context = (
                    ssl.create_default_context()
                    if self.profile.endpoint.startswith("https:")
                    else True
                )
                # c-ares avoids uncancellable getaddrinfo executor threads.
                resolver = aiohttp.AsyncResolver()
                try:
                    connector = aiohttp.TCPConnector(
                        resolver=resolver, limit=1, force_close=True, ssl=context
                    )
                    async with aiohttp.ClientSession(
                        connector=connector,
                        trust_env=False,
                        auto_decompress=False,
                        cookie_jar=aiohttp.DummyCookieJar(),
                        timeout=aiohttp.ClientTimeout(
                            total=self.profile.timeout_seconds, ceil_threshold=float("inf")
                        ),
                    ) as session:
                        url = self.profile.endpoint.rstrip("/")[:-3] + path
                        async with session.post(
                            url, data=payload, headers=headers, allow_redirects=False
                        ) as response:
                            if response.status != 200:
                                if response.status == 429:
                                    raise VlmError("upstream_busy", 429)
                                if response.status in (400, 413, 422):
                                    raise VlmError("upstream_rejected", 400)
                                if path == "/tokenize" and response.status in (404, 405, 501):
                                    raise VlmError("token_counting_unsupported", 501)
                                raise VlmError("upstream_unavailable")
                            if response.headers.get("Content-Encoding", "identity") != "identity":
                                raise VlmError("invalid_upstream_response")
                            if (
                                response.content_length is not None
                                and response.content_length > MAX_RESPONSE_BYTES
                            ):
                                raise VlmError("invalid_upstream_response")
                            chunks = bytearray()
                            async for chunk in response.content.iter_chunked(65536):
                                chunks.extend(chunk)
                                if len(chunks) > MAX_RESPONSE_BYTES:
                                    raise VlmError("invalid_upstream_response")
                            return bytes(chunks)
                finally:
                    await resolver.close()
        except TimeoutError:
            raise VlmError("upstream_timeout", 504) from None
        except aiohttp.ClientPayloadError:
            raise VlmError("invalid_upstream_response") from None
        except (OSError, aiohttp.ClientError):
            raise VlmError("upstream_unavailable") from None

    def _result(self, payload: bytes, request: VlmRequest, digest: str) -> VlmResult:
        raw = parse_json(payload)
        if raw.get("model") != self.model_id:
            raise ValueError("upstream model mismatch")
        choices = raw["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("expected one choice")
        choice = choices[0]
        if type(choice.get("index")) is not int or choice["index"] != 0:
            raise ValueError("invalid choice index")
        reason = choice.get("finish_reason")
        if reason not in ("stop", "length", "content_filter", "tool_calls"):
            raise ValueError("unsupported finish reason")
        message = choice["message"]
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("function_call")
        ):
            raise ValueError("invalid assistant message")
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list) or len(raw_calls) > 32:
            raise ValueError("invalid tool calls")
        calls = tuple(VlmToolCall.from_openai(c) for c in raw_calls)
        if len({c.id for c in calls}) != len(calls):
            raise ValueError("duplicate tool call ids")
        previous_ids = {call.id for message in request.messages for call in message.tool_calls}
        if any(call.id in previous_ids for call in calls):
            raise ValueError("reused historical tool call id")
        names = {tool.name for tool in request.tools}
        if calls and (request.tool_choice == "none" or any(c.name not in names for c in calls)):
            raise ValueError("unexpected tool call")
        if request.parallel_tool_calls is False and len(calls) > 1:
            raise ValueError("unexpected parallel calls")
        if isinstance(request.tool_choice, VlmToolChoice) and any(
            c.name != request.tool_choice.name for c in calls
        ):
            raise ValueError("unexpected named tool")
        upstream_reason = reason
        if calls and reason == "stop":
            reason = "tool_calls"
        if reason == "tool_calls":
            if not calls:
                raise ValueError("missing tool calls")
            for call in calls:
                parse_json(call.arguments.encode())
        elif calls and reason != "length":
            raise ValueError("inconsistent tool finish reason")
        if reason == "stop" and request.tool_choice not in (None, "auto", "none") and not calls:
            raise ValueError("required tool call missing")
        text = message.get("content")
        reasoning = message.get("reasoning_content")
        if reasoning is None:
            reasoning = message.get("reasoning")
        if text is None:
            text = ""
        if not isinstance(text, str) or (reasoning is not None and not isinstance(reasoning, str)):
            raise ValueError("invalid generated text")
        text.encode("utf-8")
        if reasoning is not None:
            reasoning.encode("utf-8")
        if request.json_object and reason == "stop" and not request.stop:
            parse_json(text.encode("utf-8"))
        usage = None
        if raw.get("usage") is not None:
            reported = raw["usage"]
            if not isinstance(reported, dict):
                raise ValueError("invalid upstream usage")
            usage = VlmUsage(reported["prompt_tokens"], reported["completion_tokens"])
            total = reported.get("total_tokens", usage.prompt_tokens + usage.completion_tokens)
            if type(total) is not int or total != usage.prompt_tokens + usage.completion_tokens:
                raise ValueError("inconsistent upstream usage")
        return VlmResult(
            self.model_id,
            text,
            reason,
            usage,
            self.profile.identity,
            digest,
            request.media_sha256,
            self.profile.model_revision,
            reasoning,
            calls,
            upstream_reason if upstream_reason != reason else None,
        )
