"""Tests for ToolSpec-aware trace sanitization and query contracts."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest

from capicove.contracts import ToolStatus
from capicove.tools import AUDIO_TRANSCRIBE_SPEC, VIDEO_FILTER_SPEC, VIDEO_SPLIT_SPEC
from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
from capicove.traces.content import MediaReference, TextContent, TraceModality
from capicove.traces.ports import TraceQuery
from capicove.traces.sanitization import (
    TracePrivacyError,
    TraceSanitizationError,
    TraceSanitizer,
    validate_trace_privacy,
)


SHA256 = "a" * 64


class FakeResolver:
    """Resolve configured values to one immutable media reference."""

    def __init__(self, values: set[object]) -> None:
        self.values = values
        self.paths: list[str] = []

    def resolve(self, spec: object, field_path: str, value: object, /) -> MediaReference | None:
        del spec
        self.paths.append(field_path)
        if value not in self.values:
            return None
        return MediaReference(
            media_id="media-1",
            modality=TraceModality.AUDIO,
            media_type="audio/wav",
            sha256=SHA256,
            size_bytes=5,
        )


def test_path_redaction_resolves_media_or_uses_fixed_marker() -> None:
    resolver = FakeResolver({"relative/audio.wav"})
    sanitizer = TraceSanitizer(resolver)

    resolved = sanitizer.sanitize(
        AUDIO_TRANSCRIBE_SPEC, {"source": {"type": "path", "path": "relative/audio.wav"}}
    )
    unresolved = sanitizer.sanitize(
        AUDIO_TRANSCRIBE_SPEC, {"source": {"type": "path", "path": "missing.wav"}}
    )

    assert resolved.arguments["source"] == {
        "type": "path",
        "path": {"type": "media_reference", "media_id": "media-1"},
    }
    assert resolved.media[0].media_id == "media-1"
    assert unresolved.arguments["source"] == {"type": "path", "path": "[REDACTED]"}
    assert resolver.paths == ["source.path", "source.path"]


def test_base64_and_wildcard_fields_never_retain_encoded_bytes() -> None:
    sanitizer = TraceSanitizer()
    video = sanitizer.sanitize(
        VIDEO_SPLIT_SPEC,
        {"source": {"type": "base64", "base64": "aGVsbG8="}},
    )
    filtered = sanitizer.sanitize(
        VIDEO_FILTER_SPEC,
        {
            "sources": [
                {"source": {"type": "base64", "base64": "aGVsbG8="}},
                {"source": {"type": "base64", "base64": "aGVsbG8="}},
            ]
        },
    )

    descriptor = {
        "type": "redacted_media",
        "size_bytes": 5,
        "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",  # pragma: allowlist secret
    }
    assert video.arguments["source"] == {"type": "base64", "base64": descriptor}
    assert filtered.arguments["sources"][0]["source"]["base64"] == descriptor  # type: ignore[index]
    assert "aGVsbG8=" not in repr(video.arguments)
    assert "aGVsbG8=" not in repr(filtered.arguments)


def test_declared_text_and_secret_named_fields_are_redacted_at_every_depth() -> None:
    image_spec = IMAGE_TOOL_SPECS_BY_NAME["image_remove_background_mapper"]
    arguments = {
        "text": "declared private text",
        "nested": {
            "Authorization": "Bearer abcdefghijklmnop",
            "api_key": "sk-abcdefghijklmnop",  # pragma: allowlist secret
            "password": "open sesame",  # pragma: allowlist secret
            "cookie": "session",
        },
    }

    result = TraceSanitizer().sanitize(image_spec, arguments)

    assert result.arguments["text"] == "[REDACTED]"
    assert result.arguments["nested"] == {
        "Authorization": "[REDACTED]",
        "api_key": "[REDACTED]",
        "password": "[REDACTED]",
        "cookie": "[REDACTED]",
    }


def test_visible_text_preserves_unicode_and_redacts_high_confidence_secrets_and_paths() -> None:
    text = (
        "保留中文 Bearer abcdefghijklmnop sk-abcdefghijklmnop "
        "AKIAABCDEFGHIJKLMNOP /Users/alice/private/file.txt C:\\Users\\alice\\file.txt end"  # pragma: allowlist secret
    )

    content = TraceSanitizer().sanitize_content((TextContent(text),))

    sanitized = content[0]
    assert isinstance(sanitized, TextContent)
    assert sanitized.text.startswith("保留中文 ")
    assert "Bearer abcdefghijklmnop" not in sanitized.text
    assert "sk-abcdefghijklmnop" not in sanitized.text
    assert "AKIAABCDEFGHIJKLMNOP" not in sanitized.text  # pragma: allowlist secret
    assert "/Users/alice/private/file.txt" not in sanitized.text
    assert "C:\\Users\\alice\\file.txt" not in sanitized.text
    assert sanitized.text.endswith(" end")


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/secret",
        "/secret/",
        "//secret",
        "///Users/alice/secret",
        "path:/secret",
        "path:////secret",
        "file:///Users/alice/secret",
        "C:\\",
        "C:\\secret",
        "C:\\secret\\",
    ],
)
def test_single_component_absolute_paths_are_redacted_and_rejected(path: str) -> None:
    sanitizer = TraceSanitizer()

    assert path not in sanitizer.sanitize_text(f"private location: {path}")
    with pytest.raises(TracePrivacyError, match="absolute path"):
        validate_trace_privacy({"value": path})


def test_absolute_path_redaction_does_not_modify_web_urls() -> None:
    value = "Read https://example.com/private and http://localhost/status"

    assert TraceSanitizer().sanitize_text(value) == value
    validate_trace_privacy({"value": value})


def test_media_type_less_data_uri_is_redacted_and_rejected() -> None:
    value = "data:;base64,aGVsbG8="

    assert value not in TraceSanitizer().sanitize_text(value)
    with pytest.raises(TracePrivacyError, match="inline base64"):
        validate_trace_privacy({"value": value})


def test_sanitizer_does_not_mutate_inputs_and_deduplicates_media_in_first_seen_order() -> None:
    arguments = {
        "sources": [
            {"source": {"type": "base64", "base64": "aGVsbG8="}},
            {"source": {"type": "base64", "base64": "aGVsbG8="}},
        ]
    }
    original = copy.deepcopy(arguments)
    resolver = FakeResolver({"aGVsbG8="})

    result = TraceSanitizer(resolver).sanitize(VIDEO_FILTER_SPEC, arguments)

    assert arguments == original
    assert tuple(reference.media_id for reference in result.media) == ("media-1",)


@pytest.mark.parametrize("bad", [b"not-json", {"set"}])
def test_sanitizer_rejects_non_json_values(bad: object) -> None:
    with pytest.raises(TraceSanitizationError):
        TraceSanitizer().sanitize(AUDIO_TRANSCRIBE_SPEC, {"value": bad})


def test_sanitizer_rejects_cycles_depth_and_configured_size_limits() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(TraceSanitizationError, match="cycle"):
        TraceSanitizer().sanitize(AUDIO_TRANSCRIBE_SPEC, cyclic)

    nested: object = "leaf"
    for _ in range(34):
        nested = [nested]
    with pytest.raises(TraceSanitizationError, match="depth"):
        TraceSanitizer().sanitize(AUDIO_TRANSCRIBE_SPEC, {"nested": nested})

    with pytest.raises(TraceSanitizationError, match="text"):
        TraceSanitizer(max_text_bytes=4).sanitize_content((TextContent("12345"),))
    with pytest.raises(TraceSanitizationError, match="inline media"):
        TraceSanitizer(max_inline_media_bytes=4).sanitize(
            VIDEO_SPLIT_SPEC,
            {"source": {"type": "base64", "base64": "aGVsbG8="}},
        )


def test_trace_query_validates_bounds_and_non_empty_filters() -> None:
    later = datetime(2026, 8, 19, tzinfo=UTC)
    earlier = datetime(2026, 8, 18, tzinfo=UTC)
    with pytest.raises(ValueError, match="bounds"):
        TraceQuery(started_after=later, completed_before=earlier)
    with pytest.raises(ValueError, match="tool_ids"):
        TraceQuery(tool_ids=frozenset())

    query = TraceQuery(
        modalities=frozenset({TraceModality.IMAGE}),
        tool_statuses=frozenset({ToolStatus.SUCCEEDED}),
    )
    assert query.modalities == frozenset({TraceModality.IMAGE})
