"""Optional Model Context Protocol boundary for Capicove."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from capicove.mcp.audio_transcribe import (
    MCP_AUDIO_TRANSCRIBE_NAME,
    MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA,
    create_audio_transcribe_mcp_server,
    decode_audio_transcribe_mcp_arguments,
    encode_audio_transcribe_mcp_result,
    register_audio_transcribe_mcp,
)
from capicove.mcp.book import BOOK_MCP_NAMES, encode_book_tool_result, register_book_mcp
from capicove.mcp.book_chunks import create_book_chunk_mcp_server
from capicove.mcp.video_filter import (
    MAX_INLINE_VIDEO_BYTES as VIDEO_FILTER_MAX_INLINE_VIDEO_BYTES,
)
from capicove.mcp.video_filter import (
    MCP_VIDEO_FILTER_NAME,
    MCP_VIDEO_FILTER_OUTPUT_SCHEMA,
    create_video_filter_mcp_server,
    register_video_filter_mcp,
)
from capicove.mcp.video_filter import (
    decode_mcp_arguments as decode_video_filter_mcp_arguments,
)
from capicove.mcp.video_filter import (
    encode_mcp_result as encode_video_filter_mcp_result,
)
from capicove.mcp.video_split import (
    MAX_INLINE_VIDEO_BYTES,
    MCP_VIDEO_SPLIT_NAME,
    MCP_VIDEO_SPLIT_OUTPUT_SCHEMA,
    McpUnavailableError,
    create_mcp_server,
    decode_mcp_arguments,
    encode_mcp_result,
    register_video_split_mcp,
)


if TYPE_CHECKING:
    from capicove.mcp.bindings import McpToolBinding
    from capicove.mcp.search import (
        MCP_SEARCH_NAME,
        decode_search_mcp_arguments,
        encode_search_mcp_result,
        register_search_mcp,
    )
    from capicove.mcp.server import create_bound_mcp_server


_LAZY_SEARCH_EXPORTS = frozenset(
    {
        "MCP_SEARCH_NAME",
        "decode_search_mcp_arguments",
        "encode_search_mcp_result",
        "register_search_mcp",
    }
)


def __getattr__(name: str) -> Any:
    """Load text-search MCP code only when explicitly requested."""
    modules = {"McpToolBinding": "bindings", "create_bound_mcp_server": "server"}
    if name in modules:
        value = getattr(importlib.import_module(f"capicove.mcp.{modules[name]}"), name)
        globals()[name] = value
        return value
    if name not in _LAZY_SEARCH_EXPORTS:
        raise AttributeError(name)
    from capicove.mcp import search

    value = getattr(search, name)
    globals()[name] = value
    return value


__all__ = [
    "BOOK_MCP_NAMES",
    "MAX_INLINE_VIDEO_BYTES",
    "MCP_AUDIO_TRANSCRIBE_NAME",
    "MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA",
    "MCP_SEARCH_NAME",
    "MCP_VIDEO_FILTER_NAME",
    "MCP_VIDEO_FILTER_OUTPUT_SCHEMA",
    "MCP_VIDEO_SPLIT_NAME",
    "MCP_VIDEO_SPLIT_OUTPUT_SCHEMA",
    "VIDEO_FILTER_MAX_INLINE_VIDEO_BYTES",
    "McpToolBinding",
    "McpUnavailableError",
    "create_audio_transcribe_mcp_server",
    "create_book_chunk_mcp_server",
    "create_bound_mcp_server",
    "create_mcp_server",
    "create_video_filter_mcp_server",
    "decode_audio_transcribe_mcp_arguments",
    "decode_mcp_arguments",
    "decode_search_mcp_arguments",
    "decode_video_filter_mcp_arguments",
    "encode_audio_transcribe_mcp_result",
    "encode_book_tool_result",
    "encode_mcp_result",
    "encode_search_mcp_result",
    "encode_video_filter_mcp_result",
    "register_audio_transcribe_mcp",
    "register_book_mcp",
    "register_search_mcp",
    "register_video_filter_mcp",
    "register_video_split_mcp",
]
