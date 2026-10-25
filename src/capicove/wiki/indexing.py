"""Deterministic page projection helpers for local Wiki search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import quote

from capicove.wiki.filesystem import page_paths, read_markdown


@dataclass(frozen=True, slots=True)
class WikiDocument:
    """Represent one validated page ready for indexing."""

    page_id: str
    page_path: str
    title: str
    kind: str
    body: str
    source_ids: tuple[str, ...]


def load_documents(wiki: Path) -> tuple[WikiDocument, ...]:
    """Load validated pages in stable human index order."""
    documents: list[WikiDocument] = []
    for path in page_paths(wiki):
        record = read_markdown(path)
        metadata = record.metadata
        documents.append(
            WikiDocument(
                page_id=str(metadata["id"]),
                page_path=path.relative_to(wiki).as_posix(),
                title=str(metadata["title"]),
                kind=str(metadata["kind"]),
                body=record.body,
                source_ids=tuple(str(item) for item in metadata.get("sources", [])),
            )
        )
    return tuple(sorted(documents, key=lambda item: (item.title.casefold(), item.page_path)))


def render_index(documents: tuple[WikiDocument, ...], *, built_on: date) -> str:
    """Render a deterministic Markdown catalog."""
    lines = [
        "# Wiki Index",
        "",
        f"> Last built: {built_on.isoformat()} | Total pages: {len(documents)}",
    ]
    if documents:
        lines.append("")
        for document in documents:
            title = " ".join(document.title.split())
            title = title.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            link_target = quote(document.page_path, safe="/")
            lines.append(f"- [{title}]({link_target}) — {document.kind} — `{document.page_id}`")
    return "\n".join(lines) + "\n"


def bounded_snippet(body: str, query: str, *, max_chars: int = 240) -> str:
    """Create a bounded plain-text snippet centered on the query when possible."""
    plain = " ".join(body.split())
    if len(plain) <= max_chars:
        return plain
    index = plain.casefold().find(query.casefold())
    if index < 0:
        first_token = query.split()[0]
        index = plain.casefold().find(first_token.casefold())
    if index < 0:
        index = 0
    start = max(0, index - max_chars // 3)
    end = min(len(plain), start + max_chars)
    start = max(0, end - max_chars)
    return plain[start:end]


__all__ = ["WikiDocument", "bounded_snippet", "load_documents", "render_index"]
