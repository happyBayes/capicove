"""Bounded bilingual lexical discovery; inventory coverage is not semantic recall."""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal


POLICY = "wiki-lexical-v1-rrf60"


@dataclass(frozen=True, slots=True)
class DiscoveryPage:
    """Host-loaded page bytes and mechanically resolved navigation."""

    page_id: str
    path: str
    title: str
    body: str
    revision_sha256: str
    aliases: tuple[str, ...]
    links: tuple[str, ...]
    sources: tuple[str, ...]
    claim_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryCandidate:
    """Ranked page with auditable mechanical lane ranks."""

    page_id: str
    revision_sha256: str
    lane_ranks: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Bounded candidates with explicit truncation and unknown semantic recall."""

    candidates: tuple[DiscoveryCandidate, ...]
    scanned_pages: int
    omitted_candidates: int
    coverage: Literal["complete_inventory", "partial"]
    policy: str = POLICY
    semantic_recall: str = "unknown"


def normalize(text: str) -> str:
    """Normalize only search text, never original evidence locators."""
    return unicodedata.normalize("NFKC", text).casefold()


def han_terms(text: str) -> tuple[str, ...]:
    """Tokenize Han runs into unigrams and bigrams without crossing boundaries."""
    terms: set[str] = set()
    for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]+", normalize(text)):
        terms.update(run)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(sorted(terms))


def inventory_snapshot(pages: tuple[DiscoveryPage, ...]) -> bytes:
    """Bind every indexed field and lexical policy, with bounded inventory."""
    if (
        len(pages) > 10000
        or sum(len(p.body.encode()) + len(p.title.encode()) for p in pages) > 256 * 1024 * 1024
    ):
        raise ValueError("capacity_exceeded: inventory")
    if len({p.page_id for p in pages}) != len(pages):
        raise ValueError("duplicate page identity")
    return json.dumps(
        {"policy": POLICY, "pages": [asdict(p) for p in sorted(pages, key=lambda p: p.page_id)]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def _rows(pages: tuple[DiscoveryPage, ...]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            p.page_id,
            normalize(p.title + "\n" + p.body),
            " ".join(han_terms(p.title + "\n" + p.body + "\n" + " ".join(p.aliases))),
        )
        for p in pages
    )


def build_discovery(path: Path, pages: tuple[DiscoveryPage, ...]) -> None:
    """Build an explicitly requested disposable projection."""
    from capicove.wiki._adapters.maintenance_index import build_index

    build_index(path, inventory_snapshot(pages), _rows(pages))


def discover(
    path: Path,
    pages: tuple[DiscoveryPage, ...],
    query: str,
    *,
    source_ids: tuple[str, ...] = (),
    page_ids: tuple[str, ...] = (),
    limit: int = 20,
) -> DiscoveryResult:
    """Fuse FTS, Han, title/alias, backlink and shared-source ranks."""
    from capicove.wiki._adapters.maintenance_index import ranked_lanes

    if (
        type(limit) is not int
        or not 1 <= limit <= 20
        or not query.strip()
        or len(query.encode()) > 4096
    ):
        raise ValueError("capacity_exceeded: query")
    by_id = {p.page_id: p for p in pages}
    if not set(page_ids) <= by_id.keys():
        raise ValueError("citation_out_of_scope: page expansion")
    text = normalize(query)
    terms = tuple(dict.fromkeys(re.findall(r"[^\W_]+", text)))
    expression = " OR ".join('"' + term + '"' for term in terms)
    han = " OR ".join('"' + term + '"' for term in han_terms(text))
    lanes = ranked_lanes(path, inventory_snapshot(pages), _rows(pages), expression, han)
    lanes["alias"] = tuple(
        sorted(
            p.page_id
            for p in pages
            if any(text in normalize(value) for value in (p.title, *p.aliases))
        )
    )
    lanes["links"] = tuple(
        sorted(p.page_id for p in pages if p.page_id in page_ids or set(p.links) & set(page_ids))
    )
    lanes["source"] = tuple(sorted(p.page_id for p in pages if set(p.sources) & set(source_ids)))
    ranks: dict[str, list[tuple[str, int]]] = {}
    for lane, identities in lanes.items():
        for rank, identity in enumerate(identities, 1):
            ranks.setdefault(identity, []).append((lane, rank))
    ordered = sorted(
        ranks,
        key=lambda identity: (
            -sum((Fraction(1, 60 + rank) for _, rank in ranks[identity]), Fraction()),
            identity,
        ),
    )
    selected = tuple(
        DiscoveryCandidate(identity, by_id[identity].revision_sha256, tuple(ranks[identity]))
        for identity in ordered[:limit]
    )
    omitted = max(0, len(ordered) - limit)
    return DiscoveryResult(
        selected, len(pages), omitted, "partial" if omitted else "complete_inventory"
    )
