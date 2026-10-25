"""Deterministic cited Markdown built only from validated structured claims."""

import json
import posixpath
import re
from collections.abc import Mapping
from datetime import date
from urllib.parse import quote

from capicove.contracts.wiki_maintenance import PageRevision


def plain(text: str, *, heading: bool = False) -> str:
    """Reject active content and escape remaining Markdown punctuation."""
    if (
        re.search(r"[<>]|```|!?\[[^\]]*\]\(|(?:file|javascript|data):", text, re.I)
        or any(ord(c) < 32 and c not in "\n\t" for c in text)
        or (heading and ("\n" in text or "\r" in text))
    ):
        raise ValueError("invalid_submission: active markup or unsafe heading")
    return re.sub(r"([\\`*_{}\[\]()#+!|])", r"\\\1", text).replace("\n", " ")


def render_page(
    page: PageRevision,
    created_on: str,
    *,
    updated_on: str | None = None,
    page_paths: Mapping[str, str] | None = None,
) -> str:
    """Render a pinned version with host-built original-source links."""
    date.fromisoformat(created_on)
    updated_on = updated_on or created_on
    date.fromisoformat(updated_on)
    lines = [
        "---",
        f"id: {page.page_id}",
        "title: " + json.dumps(page.title, ensure_ascii=False),
        "kind: " + page.kind,
        "created: " + created_on,
        "updated: " + updated_on,
        "sources: " + json.dumps(page.source_ids),
        "aliases: " + json.dumps(page.aliases, ensure_ascii=False),
        "---",
        "",
        "# " + plain(page.title, heading=True),
    ]
    claims = {claim.claim_id: claim for claim in page.claims}
    for section in page.sections:
        lines.extend(["", "## " + plain(section.heading, heading=True), ""])
        for identity in section.claim_ids:
            claim = claims[identity]
            evidence = []
            for span in claim.evidence:
                suffix = "md" if span.kind == "markdown_span" else "pdf"
                locator = f"chars {span.start_char}:{span.end_char}; wiki-markdown-v1"
                if span.kind == "pdf_block_span":
                    locator = f"page {span.page_index}; block {span.block_id}; chars {span.start_char}:{span.end_char}"
                evidence.append(f"[evidence](../../raw/{span.source_sha256}.{suffix}) `{locator}`")
            lines.append(
                f"- {plain(claim.text)} **[{claim.status}; {claim.knowledge}]** "
                + " ".join(evidence)
            )
            if claim.derivation:
                lines.append("  Inference: " + plain(claim.derivation))
    if page.relations:
        lines.extend(["", "## Relations", ""])
        for relation in page.relations:
            target = f"`{relation.target_id}`"
            if relation.target_id.startswith("page:"):
                destination = (page_paths or {}).get(relation.target_id)
                if destination is None or not destination.startswith("pages/"):
                    raise ValueError("invalid_submission: unresolved page navigation")
                relative = posixpath.relpath(destination, posixpath.dirname(page.page_path))
                target = f"[{relation.target_id}]({quote(relative, safe='/')})"
            lines.append(f"- {relation.kind}: {target} — {plain(relation.reason)}")
    return "\n".join(lines) + "\n"
