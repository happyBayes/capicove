"""Validate bounded declarative skill documents without executing their contents."""

from __future__ import annotations

import hashlib
import re


_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_FORBIDDEN = re.compile(
    r"(?i)(?:-----BEGIN .*PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
    r"|(?:ignore|bypass|disable)\s+(?:all\s+)?(?:security|permissions?|approval|safety)"
    r"|(?:grant|enable|allow)\s+(?:unrestricted|network|subprocess|credential)"
    r"|```\s*(?:python|bash|sh|javascript|powershell)\b)"
)


def validate_skill(name: str, content: str) -> str:
    """Return canonical text after rejecting non-declarative metadata and hazards.

    This is a bounded syntax and known-hazard check, not a semantic security proof.
    The runtime never grants permissions from document prose.
    """
    if not _SLUG.fullmatch(name) or len(name) > 80:
        raise ValueError("skill name must be a lowercase slug")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if len(content.encode("utf-8")) > 65536 or "\x00" in content:
        raise ValueError("skill document exceeds bounds")
    lines = content.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("skill requires frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("skill frontmatter must terminate") from exc
    fields: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator or key not in {"name", "description"} or key in fields:
            raise ValueError("unknown or duplicate skill frontmatter")
        fields[key] = value.strip().strip('"').strip("'")
    if fields.get("name") != name or not fields.get("description", "").strip():
        raise ValueError("skill name and description are required")
    if not "\n".join(lines[end + 1 :]).strip():
        raise ValueError("skill instruction body is required")
    if _FORBIDDEN.search(content):
        raise ValueError("skill contains executable code, secrets, or permission expansion")
    return content.rstrip() + "\n"


def content_digest(content: str) -> str:
    """Hash exact UTF-8 content bytes."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
