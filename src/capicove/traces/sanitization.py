"""ToolSpec-aware sanitization for persisted observable trace inputs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

from capicove.contracts import ToolSpec
from capicove.traces._json import freeze_json
from capicove.traces.content import MediaContent, MediaReference, TextContent
from capicove.traces.ports import TraceMediaResolver


_SECRET_KEYS = frozenset({"authorization", "api_key", "token", "password", "secret", "cookie"})
_BEARER_PATTERN = re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{8,}")
_OPENAI_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")
_AWS_KEY_PATTERN = re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])")
_DATA_URI_PATTERN = re.compile(r"(?i)\bdata:[^,\s]*;base64,[A-Za-z0-9+/=]+")
_FILE_URI_PATTERN = re.compile(r"(?i)\bfile:/+[^\s]*")
_REPEATED_POSIX_PATH_PATTERN = re.compile(
    r"(?i)(?<!http:)(?<!https:)(?<![A-Za-z0-9/])/{2,}(?:[^/\s]+(?:/[^/\s]*)*)?"
)
_POSIX_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9/])/(?!/)(?:[^/\s]+(?:/[^/\s]*)*)?")
_WINDOWS_PATH_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:\\(?:[^\\\s]+(?:\\[^\\\s]*)*)?")


class TraceSanitizationError(ValueError):
    """Report rejected trace content without including the sensitive value."""


class TracePrivacyError(ValueError):
    """Report one stable privacy-boundary violation without echoing content."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SanitizedToolArguments:
    """Return deeply frozen safe arguments and first-seen media identities."""

    arguments: Mapping[str, object]
    media: tuple[MediaReference, ...]


class TraceSanitizer:
    """Sanitize Tool arguments and visible text before trace persistence."""

    def __init__(
        self,
        media_resolver: TraceMediaResolver | None = None,
        *,
        max_text_bytes: int = 1_048_576,
        max_inline_media_bytes: int = 67_108_864,
    ) -> None:
        _require_positive_limit("max_text_bytes", max_text_bytes)
        _require_positive_limit("max_inline_media_bytes", max_inline_media_bytes)
        self._media_resolver = media_resolver
        self._max_text_bytes = max_text_bytes
        self._max_inline_media_bytes = max_inline_media_bytes

    def sanitize(
        self,
        spec: ToolSpec,
        arguments: Mapping[str, object],
        /,
    ) -> SanitizedToolArguments:
        """Return a safe immutable copy of Tool arguments."""
        if not isinstance(spec, ToolSpec):
            raise TraceSanitizationError("spec must be a ToolSpec")
        if not isinstance(arguments, Mapping):
            raise TraceSanitizationError("arguments must be a JSON object")
        patterns = tuple(self._parse_pattern(pattern) for pattern in spec.redacted_fields)
        active: set[int] = set()
        media_by_id: dict[str, MediaReference] = {}

        def add_media(reference: MediaReference) -> None:
            existing = media_by_id.get(reference.media_id)
            if existing is not None and existing != reference:
                raise TraceSanitizationError("one media ID resolved to conflicting identities")
            media_by_id.setdefault(reference.media_id, reference)

        def sanitize_value(value: object, *, path: tuple[str, ...], depth: int) -> object:
            if self._matches(path, patterns):
                return self._sanitize_declared(spec, path, value, add_media)
            if value is None or isinstance(value, (bool, int, float)):
                try:
                    return freeze_json(value, field_name="arguments")
                except ValueError as error:
                    raise TraceSanitizationError("arguments contain a non-finite value") from error
            if isinstance(value, str):
                return self._sanitize_text(value)
            if not isinstance(value, (Mapping, list, tuple)):
                raise TraceSanitizationError("arguments contain a non-JSON value")
            if depth >= 32:
                raise TraceSanitizationError("arguments exceed the maximum JSON depth")
            identity = id(value)
            if identity in active:
                raise TraceSanitizationError("arguments must not contain a cycle")
            active.add(identity)
            try:
                if isinstance(value, Mapping):
                    sanitized_object: dict[str, object] = {}
                    for key, item in value.items():
                        if not isinstance(key, str):
                            raise TraceSanitizationError(
                                "arguments must contain only string object keys"
                            )
                        item_path = (*path, key)
                        if key.casefold() in _SECRET_KEYS:
                            sanitized_object[key] = "[REDACTED]"
                        else:
                            sanitized_object[key] = sanitize_value(
                                item, path=item_path, depth=depth + 1
                            )
                    return sanitized_object
                return [
                    sanitize_value(item, path=(*path, str(index)), depth=depth + 1)
                    for index, item in enumerate(value)
                ]
            finally:
                active.remove(identity)

        sanitized = sanitize_value(arguments, path=(), depth=0)
        try:
            frozen = freeze_json(sanitized, field_name="arguments")
        except ValueError as error:
            raise TraceSanitizationError("sanitized arguments are not finite JSON") from error
        if not isinstance(frozen, Mapping):
            raise TraceSanitizationError("arguments must be a JSON object")
        return SanitizedToolArguments(frozen, tuple(media_by_id.values()))

    def sanitize_content(
        self,
        content: tuple[TextContent | MediaContent, ...],
        /,
    ) -> tuple[TextContent | MediaContent, ...]:
        """Sanitize visible text while retaining ordered media blocks."""
        sanitized: list[TextContent | MediaContent] = []
        for block in content:
            if isinstance(block, TextContent):
                sanitized.append(TextContent(self._sanitize_text(block.text)))
            elif isinstance(block, MediaContent):
                sanitized.append(block)
            else:
                raise TraceSanitizationError("content contains an unsupported block")
        return tuple(sanitized)

    def sanitize_json(
        self,
        value: Mapping[str, object],
        /,
        *,
        field_name: str,
    ) -> Mapping[str, object]:
        """Return a privacy-safe immutable copy of arbitrary JSON output."""
        if not isinstance(value, Mapping):
            raise TraceSanitizationError(f"{field_name} must be a JSON object")
        active: set[int] = set()

        def sanitize_value(
            item: object,
            *,
            key: str | None,
            depth: int,
        ) -> object:
            if key is not None and key.casefold() in _SECRET_KEYS:
                return "[REDACTED]"
            if key is not None and key.casefold() == "base64":
                return self._redact_inline_data(item)
            if item is None or isinstance(item, (bool, int, float)):
                try:
                    return freeze_json(item, field_name=field_name)
                except ValueError as error:
                    raise TraceSanitizationError(
                        f"{field_name} contains a non-finite value"
                    ) from error
            if isinstance(item, str):
                return self._sanitize_text(item)
            if not isinstance(item, (Mapping, list, tuple)):
                raise TraceSanitizationError(f"{field_name} contains a non-JSON value")
            if depth >= 32:
                raise TraceSanitizationError(f"{field_name} exceeds the maximum JSON depth")
            identity = id(item)
            if identity in active:
                raise TraceSanitizationError(f"{field_name} must not contain a cycle")
            active.add(identity)
            try:
                if isinstance(item, Mapping):
                    sanitized: dict[str, object] = {}
                    for child_key, child in item.items():
                        if not isinstance(child_key, str):
                            raise TraceSanitizationError(
                                f"{field_name} must contain only string object keys"
                            )
                        sanitized[child_key] = sanitize_value(
                            child,
                            key=child_key,
                            depth=depth + 1,
                        )
                    return sanitized
                return [sanitize_value(child, key=None, depth=depth + 1) for child in item]
            finally:
                active.remove(identity)

        sanitized = sanitize_value(value, key=None, depth=0)
        try:
            frozen = freeze_json(sanitized, field_name=field_name)
        except ValueError as error:
            raise TraceSanitizationError(f"sanitized {field_name} is not finite JSON") from error
        if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
            raise TraceSanitizationError(f"{field_name} must be a JSON object")
        return frozen

    def sanitize_text(self, value: str, /) -> str:
        """Return one privacy-safe bounded text value."""
        return self._sanitize_text(value)

    def _sanitize_declared(
        self,
        spec: ToolSpec,
        path: tuple[str, ...],
        value: object,
        add_media: object,
    ) -> object:
        field_path = ".".join(path)
        decoded: bytes | None = None
        if path and path[-1].casefold() == "base64":
            if not isinstance(value, str):
                raise TraceSanitizationError("inline media must be valid base64 text")
            try:
                decoded = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as error:
                raise TraceSanitizationError("inline media must be valid base64 text") from error
            if len(decoded) > self._max_inline_media_bytes:
                raise TraceSanitizationError("inline media exceeds the configured byte limit")
        reference = None
        if self._media_resolver is not None:
            try:
                reference = self._media_resolver.resolve(spec, field_path, value)
            except Exception as error:
                raise TraceSanitizationError("media resolution failed") from error
        if reference is not None:
            if not isinstance(reference, MediaReference):
                raise TraceSanitizationError("media resolver returned an invalid reference")
            add_media(reference)  # type: ignore[operator]
            return {"type": "media_reference", "media_id": reference.media_id}
        if decoded is not None:
            return {
                "type": "redacted_media",
                "size_bytes": len(decoded),
                "sha256": hashlib.sha256(decoded).hexdigest(),
            }
        return "[REDACTED]"

    def _redact_inline_data(self, value: object) -> Mapping[str, object]:
        if not isinstance(value, str):
            raise TraceSanitizationError("inline media must be valid base64 text")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise TraceSanitizationError("inline media must be valid base64 text") from error
        if len(decoded) > self._max_inline_media_bytes:
            raise TraceSanitizationError("inline media exceeds the configured byte limit")
        return {
            "type": "redacted_media",
            "size_bytes": len(decoded),
            "sha256": hashlib.sha256(decoded).hexdigest(),
        }

    def _sanitize_text(self, value: str) -> str:
        if len(value.encode("utf-8")) > self._max_text_bytes:
            raise TraceSanitizationError("text exceeds the configured byte limit")
        sanitized = _BEARER_PATTERN.sub("[REDACTED_BEARER_TOKEN]", value)
        sanitized = _OPENAI_KEY_PATTERN.sub("[REDACTED_SECRET_KEY]", sanitized)
        sanitized = _AWS_KEY_PATTERN.sub("[REDACTED_AWS_ACCESS_KEY]", sanitized)
        sanitized = _DATA_URI_PATTERN.sub("[REDACTED_INLINE_MEDIA]", sanitized)
        sanitized = _FILE_URI_PATTERN.sub("[REDACTED_ABSOLUTE_PATH]", sanitized)
        sanitized = _WINDOWS_PATH_PATTERN.sub("[REDACTED_ABSOLUTE_PATH]", sanitized)
        sanitized = _REPEATED_POSIX_PATH_PATTERN.sub("[REDACTED_ABSOLUTE_PATH]", sanitized)
        return _POSIX_PATH_PATTERN.sub("[REDACTED_ABSOLUTE_PATH]", sanitized)

    @staticmethod
    def _parse_pattern(value: str) -> tuple[str, ...]:
        parts = tuple(value.split("."))
        if not parts or any(not part for part in parts):
            raise TraceSanitizationError("redacted field pattern is invalid")
        return parts

    @staticmethod
    def _matches(path: tuple[str, ...], patterns: tuple[tuple[str, ...], ...]) -> bool:
        for pattern in patterns:
            if len(path) != len(pattern):
                continue
            if all(
                expected == "*" or expected == actual
                for expected, actual in zip(pattern, path, strict=True)
            ):
                return True
        return False


def _require_positive_limit(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def validate_trace_privacy(value: object, /) -> None:
    """Reject credentials, absolute paths, and inline media in trace JSON."""
    active: set[int] = set()

    def validate(item: object, *, key: str | None, depth: int) -> None:
        if key is not None and key.casefold() in _SECRET_KEYS:
            if item != "[REDACTED]":
                raise TracePrivacyError(
                    "PRIVACY_SECRET_KEY",
                    "trace JSON contains an unredacted secret-named field",
                )
            return
        if key is not None and key.casefold() == "base64":
            if isinstance(item, str):
                raise TracePrivacyError(
                    "PRIVACY_INLINE_BASE64",
                    "trace JSON contains inline base64 data",
                )
        if isinstance(item, str):
            if _DATA_URI_PATTERN.search(item):
                raise TracePrivacyError(
                    "PRIVACY_INLINE_BASE64",
                    "trace JSON contains inline base64 data",
                )
            if (
                _BEARER_PATTERN.search(item)
                or _OPENAI_KEY_PATTERN.search(item)
                or _AWS_KEY_PATTERN.search(item)
            ):
                raise TracePrivacyError(
                    "PRIVACY_CREDENTIAL",
                    "trace JSON contains an unredacted credential",
                )
            if (
                _FILE_URI_PATTERN.search(item)
                or _WINDOWS_PATH_PATTERN.search(item)
                or _REPEATED_POSIX_PATH_PATTERN.search(item)
                or _POSIX_PATH_PATTERN.search(item)
            ):
                raise TracePrivacyError(
                    "PRIVACY_ABSOLUTE_PATH",
                    "trace JSON contains an unredacted absolute path",
                )
            return
        if item is None or isinstance(item, (bool, int, float)):
            return
        if not isinstance(item, (Mapping, list, tuple)):
            raise TracePrivacyError(
                "PRIVACY_INVALID_JSON",
                "trace privacy validation requires finite JSON values",
            )
        if depth >= 64:
            raise TracePrivacyError(
                "PRIVACY_INVALID_JSON",
                "trace JSON exceeds the privacy validation depth limit",
            )
        identity = id(item)
        if identity in active:
            raise TracePrivacyError(
                "PRIVACY_INVALID_JSON",
                "trace JSON contains a cycle",
            )
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                for child_key, child in item.items():
                    if not isinstance(child_key, str):
                        raise TracePrivacyError(
                            "PRIVACY_INVALID_JSON",
                            "trace JSON contains a non-string object key",
                        )
                    validate(child, key=child_key, depth=depth + 1)
            else:
                for child in item:
                    validate(child, key=None, depth=depth + 1)
        finally:
            active.remove(identity)

    validate(value, key=None, depth=0)
