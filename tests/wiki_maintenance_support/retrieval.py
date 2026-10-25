"""Frozen synthetic bilingual retrieval labels, independent of ranking output."""

import hashlib
import json
from dataclasses import asdict, dataclass

from capicove.wiki.maintenance.discovery import DiscoveryPage


@dataclass(frozen=True)
class RetrievalCase:
    """One independently specified query and expected bounded candidates."""

    language: str
    category: str
    query: str
    relevant: tuple[str, ...]
    source_ids: tuple[str, ...] = ()
    page_ids: tuple[str, ...] = ()


def fixture() -> tuple[tuple[DiscoveryPage, ...], tuple[RetrievalCase, ...]]:
    """Build exactly 50 Chinese, 50 English and 20 mixed queries."""
    pages = []
    cases = []
    for index in range(120):
        language = "zh" if index < 50 else "en" if index < 100 else "mixed"
        category = ("alias", "substring", "negative", "dispute", "affected")[index % 5]
        identity = f"page:RETR{index:08d}"
        han = chr(0x4E00 + index * 2) + chr(0x4E01 + index * 2)
        token = han if language == "zh" else f"specimen{index:04d}"
        if language == "mixed":
            token += " " + han
        title = "record " + token
        body = f"Observation {token} has opposing source measurements."
        alias = "alias " + token
        source = "source:sha256:" + hashlib.sha256(identity.encode()).hexdigest()
        pages.append(
            DiscoveryPage(
                identity,
                f"pages/{index}.md",
                title,
                body,
                hashlib.sha256(body.encode()).hexdigest(),
                (alias,),
                (),
                (source,),
                (),
            )
        )
        query = alias if category == "alias" else token
        if category == "negative":
            query = "absentzz" + str(index)
        if category == "affected":
            query = "unrelatedzz"
        cases.append(
            RetrievalCase(
                language,
                category,
                query,
                () if category == "negative" else (identity,),
                (source,) if category == "affected" else (),
            )
        )
    return tuple(pages), tuple(cases)


def fixture_digest() -> str:
    pages, cases = fixture()
    return hashlib.sha256(
        json.dumps(
            {"pages": [asdict(p) for p in pages], "cases": [asdict(c) for c in cases]},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
