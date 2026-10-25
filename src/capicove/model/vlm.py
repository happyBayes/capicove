"""Dependency-free multimodal generation port and bounded chat wire format."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol

from capicove.model.vlm_tools import VlmTool, VlmToolCall, VlmToolChoice


MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_MEDIA_BYTES = 20 * 1024 * 1024


class VlmError(RuntimeError):
    """Sanitized transport failure; upstream payloads and credentials stay private."""

    def __init__(self, code: str, status: int = 502) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


def parse_json(payload: bytes) -> dict[str, object]:
    """Reject duplicate keys, nonfinite numbers and non-object JSON bodies."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value: str) -> object:
        raise ValueError("nonfinite JSON value")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON value")
        return number

    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("request exceeds size limit")
    value = json.loads(
        payload, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite_float
    )
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


@dataclass(frozen=True, slots=True)
class VlmText:
    """One immutable text part in a multimodal message."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be nonblank")
        if len(self.text.encode("utf-8")) > 1024 * 1024:
            raise ValueError("text part exceeds 1 MiB")


@dataclass(frozen=True, slots=True)
class VlmMedia:
    """Caller-supplied bytes; inference never fetches URLs or local paths."""

    data: bytes
    media_type: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not 1 <= len(self.data) <= MAX_MEDIA_BYTES:
            raise ValueError("media exceeds byte limit")
        signatures = {
            "image/png": self.data.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/jpeg": self.data.startswith(b"\xff\xd8\xff"),
            "image/webp": self.data.startswith(b"RIFF") and self.data[8:12] == b"WEBP",
            "image/gif": self.data[:6] in (b"GIF87a", b"GIF89a"),
            "video/mp4": self.data[4:8] == b"ftyp",
            "video/quicktime": self.data[4:8] == b"ftyp",
            "video/webm": self.data.startswith(b"\x1a\x45\xdf\xa3"),
        }
        if not signatures.get(self.media_type, False):
            raise ValueError("unsupported media type or mismatched signature")
        if self.detail is not None and (
            self.modality != "image" or self.detail not in ("auto", "low", "high")
        ):
            raise ValueError("invalid image detail")

    @property
    def modality(self) -> str:
        return self.media_type.split("/", 1)[0]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def data_uri(self) -> str:
        """Encode bytes for OpenAI image or vLLM video content parts."""
        return f"data:{self.media_type};base64," + base64.b64encode(self.data).decode("ascii")


@dataclass(frozen=True, slots=True)
class VlmMessage:
    """Ordered content; only user messages carry media in this port."""

    role: str
    content: tuple[VlmText | VlmMedia, ...]
    tool_calls: tuple[VlmToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError("unsupported message role")
        if not isinstance(self.content, tuple) or not 0 <= len(self.content) <= 64:
            raise ValueError("message requires 1 to 64 content parts")
        if not self.content and self.role not in {"assistant", "tool"}:
            raise ValueError("message content is required")
        if (
            not isinstance(self.tool_calls, tuple)
            or len(self.tool_calls) > 32
            or any(not isinstance(c, VlmToolCall) for c in self.tool_calls)
            or (self.tool_calls and self.role != "assistant")
        ):
            raise ValueError("invalid assistant tool calls")
        if self.reasoning is not None and (
            self.role != "assistant"
            or not isinstance(self.reasoning, str)
            or len(self.reasoning.encode()) > 1024 * 1024
        ):
            raise ValueError("invalid assistant reasoning")
        if self.role == "tool":
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
                raise ValueError("tool result requires tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id requires tool role")
        for part in self.content:
            if not isinstance(part, (VlmText, VlmMedia)):
                raise ValueError("invalid content part")
            if isinstance(part, VlmMedia) and self.role != "user":
                raise ValueError("only user messages may contain media")


@dataclass(frozen=True, slots=True)
class VlmRequest:
    """Bounded nonstreaming generation with explicit sampling settings."""

    messages: tuple[VlmMessage, ...]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    json_object: bool = False
    tools: tuple[VlmTool, ...] = ()
    tool_choice: str | VlmToolChoice | None = None
    parallel_tool_calls: bool | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.messages, tuple)
            or not 1 <= len(self.messages) <= 64
            or any(not isinstance(m, VlmMessage) for m in self.messages)
            or not any(m.role == "user" for m in self.messages)
        ):
            raise ValueError("request requires 1 to 64 messages and a user message")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 32768:
            raise ValueError("max_tokens must be an integer from 1 to 32768")
        for value, minimum, maximum in ((self.temperature, 0, 2), (self.top_p, 0, 1)):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not minimum <= value <= maximum
            ):
                raise ValueError("invalid sampling parameter")
        if self.top_p == 0 or (
            self.seed is not None
            and (type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1)
        ):
            raise ValueError("invalid sampling parameter")
        if (
            not isinstance(self.stop, tuple)
            or len(self.stop) > 4
            or any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in self.stop)
        ):
            raise ValueError("invalid stop sequences")
        if type(self.json_object) is not bool:
            raise ValueError("invalid response format")
        self._validate_tools()
        encoded_size = 0
        for message in self.messages:
            for part in message.content:
                encoded_size += (
                    ((len(part.data) + 2) // 3 * 4)
                    if isinstance(part, VlmMedia)
                    else len(part.text.encode("utf-8"))
                )
                if encoded_size > MAX_REQUEST_BYTES:
                    raise ValueError("request exceeds size limit")
        if (
            len(json.dumps(self.to_openai("model"), ensure_ascii=False).encode())
            > MAX_REQUEST_BYTES
        ):
            raise ValueError("request exceeds size limit")

    def _validate_tools(self) -> None:
        if (
            not isinstance(self.tools, tuple)
            or len(self.tools) > 32
            or any(not isinstance(t, VlmTool) for t in self.tools)
        ):
            raise ValueError("expected at most 32 function tools")
        names = {t.name for t in self.tools}
        if len(names) != len(self.tools):
            raise ValueError("duplicate tool names")
        if isinstance(self.tool_choice, VlmToolChoice):
            if self.tool_choice.name not in names:
                raise ValueError("unknown named tool choice")
        elif self.tool_choice is not None:
            if not isinstance(self.tool_choice, str) or self.tool_choice not in {
                "auto",
                "none",
                "required",
            }:
                raise ValueError("invalid tool choice")
            if not self.tools and self.tool_choice not in {"auto", "none"}:
                raise ValueError("tool choice requires tools")
        if self.parallel_tool_calls is not None and type(self.parallel_tool_calls) is not bool:
            raise ValueError("invalid parallel tool calls option")
        pending: set[str] = set()
        seen: set[str] = set()
        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("unmatched or duplicate tool result")
                pending.remove(message.tool_call_id)
            else:
                if pending:
                    raise ValueError("tool results must precede the next message")
                for call in message.tool_calls:
                    if call.id in seen:
                        raise ValueError("duplicate tool call id")
                    parse_json(call.arguments.encode())
                    seen.add(call.id)
                    pending.add(call.id)
        if pending:
            raise ValueError("missing tool results")

    @property
    def media_sha256(self) -> tuple[str, ...]:
        return tuple(p.sha256 for m in self.messages for p in m.content if isinstance(p, VlmMedia))

    def to_openai(self, model: str) -> dict[str, object]:
        """Preserve part order, including multiple images and video extensions."""
        messages: list[dict[str, object]] = []
        for message in self.messages:
            parts: list[dict[str, object]] = []
            for part in message.content:
                if isinstance(part, VlmText):
                    parts.append({"type": "text", "text": part.text})
                else:
                    kind = part.modality + "_url"
                    media_info = {"url": part.data_uri()}
                    if part.detail is not None:
                        media_info["detail"] = part.detail
                    parts.append({"type": kind, kind: media_info})
            content: object = parts[0]["text"] if len(parts) == 1 and "text" in parts[0] else parts
            if not parts:
                content = None if message.tool_calls else ""
            entry: dict[str, object] = {"role": message.role, "content": content}
            if message.tool_calls:
                entry["tool_calls"] = [c.to_openai() for c in message.tool_calls]
            if message.reasoning is not None:
                entry["reasoning_content"] = message.reasoning
            if message.tool_call_id is not None:
                entry["tool_call_id"] = message.tool_call_id
            messages.append(entry)
        body: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
            "n": 1,
        }
        if self.tools:
            body["tools"] = [tool.to_openai() for tool in self.tools]
        if self.tool_choice is not None and self.tools:
            body["tool_choice"] = (
                self.tool_choice.to_openai()
                if isinstance(self.tool_choice, VlmToolChoice)
                else self.tool_choice
            )
        if self.parallel_tool_calls is not None and self.tools:
            body["parallel_tool_calls"] = self.parallel_tool_calls
        if self.seed is not None:
            body["seed"] = self.seed
        if self.stop:
            body["stop"] = list(self.stop)
        if self.json_object:
            body["response_format"] = {"type": "json_object"}
        return body


def _part(raw: object) -> VlmText | VlmMedia:
    if not isinstance(raw, dict):
        raise ValueError("invalid content part")
    kind = raw.get("type")
    if kind == "text" and set(raw) == {"type", "text"}:
        return VlmText(raw["text"])
    if kind not in ("image_url", "video_url") or set(raw) != {"type", kind}:
        raise ValueError("unsupported content part")
    media = raw[kind]
    if (
        not isinstance(media, dict)
        or "url" not in media
        or set(media) - {"url", "detail"}
        or not isinstance(media["url"], str)
    ):
        raise ValueError("media requires inline data URL")
    header, separator, encoded = media["url"].partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("media requires inline base64 data")
    mime = header[5:-7]
    if (
        mime.split("/", 1)[0] != kind.split("_", 1)[0]
        or len(encoded) > (MAX_MEDIA_BYTES + 2) // 3 * 4
    ):
        raise ValueError("media type or size mismatch")
    try:
        return VlmMedia(base64.b64decode(encoded, validate=True), mime, media.get("detail"))
    except (binascii.Error, UnicodeError) as error:
        raise ValueError("invalid base64 media") from error


def parse_chat_request(raw: dict[str, object]) -> VlmRequest:
    """Parse a deliberately bounded subset of Chat Completions; reject extras."""
    fields = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "stream",
        "n",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
    if (
        set(raw) - fields
        or raw.get("stream", False) is not False
        or type(raw.get("n", 1)) is not int
        or raw.get("n", 1) != 1
    ):
        raise ValueError("unsupported chat options")
    if "max_tokens" in raw and "max_completion_tokens" in raw:
        raise ValueError("specify only one output token limit")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError("invalid messages")
    parsed: list[VlmMessage] = []
    for entry in messages:
        if (
            not isinstance(entry, dict)
            or set(entry) - {"role", "content", "tool_calls", "tool_call_id", "reasoning_content"}
            or "role" not in entry
        ):
            raise ValueError("invalid message fields")
        content = entry.get("content")
        if (content is None or content == "") and entry["role"] in {"assistant", "tool"}:
            parts: tuple[VlmText | VlmMedia, ...] = ()
        elif isinstance(content, str):
            parts = (VlmText(content),)
        elif isinstance(content, list) and 1 <= len(content) <= 64:
            parts = tuple(_part(p) for p in content)
        else:
            raise ValueError("invalid message content")
        calls = entry.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list) or len(calls) > 32:
            raise ValueError("invalid tool calls")
        parsed.append(
            VlmMessage(
                entry["role"],
                parts,
                tuple(VlmToolCall.from_openai(c) for c in calls),
                entry.get("tool_call_id"),
                entry.get("reasoning_content"),
            )
        )
    response_format = raw.get("response_format", {"type": "text"})
    if response_format not in ({"type": "text"}, {"type": "json_object"}):
        raise ValueError("unsupported response format")
    stop = raw.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list):
        raise ValueError("invalid stop sequences")
    # Runtime validation is centralized in the immutable request constructor.
    from typing import Any, cast

    tools = raw.get("tools", [])
    if not isinstance(tools, list) or len(tools) > 32:
        raise ValueError("invalid tools")
    choice = raw.get("tool_choice")
    if isinstance(choice, dict):
        if (
            set(choice) != {"type", "function"}
            or choice["type"] != "function"
            or not isinstance(choice["function"], dict)
            or set(choice["function"]) != {"name"}
        ):
            raise ValueError("invalid named tool choice")
        choice = VlmToolChoice(choice["function"]["name"])
    return VlmRequest(
        tuple(parsed),
        cast(Any, raw.get("max_tokens", raw.get("max_completion_tokens", 512))),
        cast(Any, raw.get("temperature", 0.0)),
        cast(Any, raw.get("top_p", 1.0)),
        cast(Any, raw.get("seed")),
        tuple(stop),
        response_format == {"type": "json_object"},
        tuple(VlmTool.from_openai(t) for t in tools),
        cast(Any, choice),
        cast(Any, raw.get("parallel_tool_calls")),
    )


@dataclass(frozen=True, slots=True)
class VlmUsage:
    """Server-reported multimodal token counts; no text-only approximation."""

    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (self.prompt_tokens, self.completion_tokens)):
            raise ValueError("invalid upstream usage")


@dataclass(frozen=True, slots=True)
class VlmResult:
    """Generated text and explicit lineage, including operator-declared revision."""

    model: str
    text: str
    finish_reason: str
    usage: VlmUsage | None
    profile_identity: str
    request_sha256: str
    media_sha256: tuple[str, ...]
    model_revision: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[VlmToolCall, ...] = ()
    upstream_finish_reason: str | None = None

    def to_openai(self) -> dict[str, object]:
        """Render one assistant choice and additive Capicove provenance."""
        import time
        import uuid

        message: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            message["tool_calls"] = [c.to_openai() for c in self.tool_calls]
            if not self.text:
                message["content"] = None
        if self.reasoning is not None:
            message["reasoning_content"] = self.reasoning
        usage: dict[str, int] | None = None
        if self.usage is not None:
            usage = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            }
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
            "usage": usage,
            "capicove": {
                "profile_identity": self.profile_identity,
                "request_sha256": self.request_sha256,
                "media_sha256": list(self.media_sha256),
                "declared_model_revision": self.model_revision,
                "upstream_finish_reason": self.upstream_finish_reason,
            },
        }


@dataclass(frozen=True, slots=True)
class VlmTokenCount:
    """Upstream multimodal input count with the effective model context limit."""

    model: str
    input_tokens: int
    context_limit: int
    request_sha256: str
    profile_identity: str

    def __post_init__(self) -> None:
        if (
            type(self.input_tokens) is not int
            or self.input_tokens < 0
            or type(self.context_limit) is not int
            or self.context_limit < 1
        ):
            raise ValueError("invalid token counts")

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "context_limit": self.context_limit,
            "source": "vllm_tokenize",
            "request_sha256": self.request_sha256,
            "profile_identity": self.profile_identity,
        }


class VlmProvider(Protocol):
    """Public inference port for image/video understanding and text generation."""

    @property
    def model_id(self) -> str: ...

    def generate(self, request: VlmRequest) -> VlmResult: ...

    def count_tokens(self, request: VlmRequest) -> VlmTokenCount: ...

    def close(self) -> None: ...
