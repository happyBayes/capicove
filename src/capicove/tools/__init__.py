"""Transport-neutral Tool Registry for Capicove."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from capicove.tools.audio_clip import (
    AUDIO_CLIP_SPEC,
    AudioClipToolHandler,
    register_audio_clip_tool,
)
from capicove.tools.audio_transcribe import (
    AUDIO_TRANSCRIBE_SPEC,
    AudioTranscribeToolHandler,
    audio_transcribe_capabilities,
    register_audio_transcribe_tool,
    sanitize_audio_transcribe_arguments,
)
from capicove.tools.book import BOOK_TOOL_SPECS, BookToolHandler, register_book_tools
from capicove.tools.book_chunks import (
    CHUNK_TOOL_SPECS,
    BookChunkToolHandler,
    register_book_chunk_tools,
)
from capicove.tools.errors import DuplicateToolError, UnknownToolError
from capicove.tools.registry import RegisteredTool, ToolRegistry
from capicove.tools.tracing import TracedToolHandler, TraceProtocolError, tool_catalog_sha256
from capicove.tools.video_filter import (
    VIDEO_FILTER_SPEC,
    VideoFilterToolHandler,
    build_video_filter_tool_handler,
    register_video_filter_tool,
    sanitize_video_filter_arguments,
)
from capicove.tools.video_split import (
    VIDEO_SPLIT_SPEC,
    VideoSplitToolHandler,
    build_video_split_tool_handler,
    register_video_split_tool,
    sanitize_video_split_arguments,
)


if TYPE_CHECKING:
    from capicove.tools.image import (
        IMAGE_TOOL_SPECS_BY_NAME,
        ImageToolHandler,
        build_image_tool_handler,
        register_image_tools,
    )
    from capicove.tools.image_search import SEARCH_SPEC, register_search_tool
    from capicove.tools.image_search import SearchToolHandler as ImageSearchToolHandler
    from capicove.tools.search import CAPICOVE_SEARCH_SPEC, SearchToolHandler


_LAZY_IMAGE_EXPORTS = frozenset(
    {
        "IMAGE_TOOL_SPECS_BY_NAME",
        "ImageToolHandler",
        "build_image_tool_handler",
        "register_image_tools",
    }
)
_LAZY_SEARCH_EXPORTS = frozenset({"CAPICOVE_SEARCH_SPEC", "SearchToolHandler"})
_LAZY_IMAGE_SEARCH_EXPORTS = frozenset({"SEARCH_SPEC", "register_search_tool"})


def __getattr__(name: str) -> Any:
    """Load image Tool code only when an image export is requested."""
    if name in _LAZY_IMAGE_EXPORTS:
        from capicove.tools import image

        value = getattr(image, name)
    elif name in _LAZY_SEARCH_EXPORTS:
        from capicove.tools import search

        value = getattr(search, name)
    elif name in _LAZY_IMAGE_SEARCH_EXPORTS or name == "ImageSearchToolHandler":
        from capicove.tools import image_search

        value = getattr(
            image_search, "SearchToolHandler" if name == "ImageSearchToolHandler" else name
        )
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "AUDIO_CLIP_SPEC",
    "AUDIO_TRANSCRIBE_SPEC",
    "BOOK_TOOL_SPECS",
    "CAPICOVE_SEARCH_SPEC",
    "CHUNK_TOOL_SPECS",
    "IMAGE_TOOL_SPECS_BY_NAME",
    "SEARCH_SPEC",
    "VIDEO_FILTER_SPEC",
    "VIDEO_SPLIT_SPEC",
    "AudioClipToolHandler",
    "AudioTranscribeToolHandler",
    "BookChunkToolHandler",
    "BookToolHandler",
    "DuplicateToolError",
    "ImageSearchToolHandler",
    "ImageToolHandler",
    "RegisteredTool",
    "SearchToolHandler",
    "ToolRegistry",
    "TraceProtocolError",
    "TracedToolHandler",
    "UnknownToolError",
    "VideoFilterToolHandler",
    "VideoSplitToolHandler",
    "audio_transcribe_capabilities",
    "build_image_tool_handler",
    "build_video_filter_tool_handler",
    "build_video_split_tool_handler",
    "register_audio_clip_tool",
    "register_audio_transcribe_tool",
    "register_book_chunk_tools",
    "register_book_tools",
    "register_image_tools",
    "register_search_tool",
    "register_video_filter_tool",
    "register_video_split_tool",
    "sanitize_audio_transcribe_arguments",
    "sanitize_video_filter_arguments",
    "sanitize_video_split_arguments",
    "tool_catalog_sha256",
]
