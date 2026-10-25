"""Canonical preview snippets for retrieval hits over exact source text."""

from __future__ import annotations

import re


MAX_SNIPPET_CHARACTERS = 2_048
_HIGHLIGHT = re.compile(r"</?em>", re.IGNORECASE)


def preview_snippet(text: str, limit: int = MAX_SNIPPET_CHARACTERS) -> str:
    """Return a bounded preview whose literal ``<em>`` tokens cannot pose as highlights.

    Canonical hits reserve ``<em>`` for provider highlighting, so literal source tags
    are HTML-escaped in the preview only; evidence always returns the exact text.
    """
    escaped = _HIGHLIGHT.sub(
        lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"), text
    )
    return escaped[:limit]


__all__ = ["MAX_SNIPPET_CHARACTERS", "preview_snippet"]
