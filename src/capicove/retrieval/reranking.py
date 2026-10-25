"""Dependency-light text reranking shared by Python, CLI and HTTP callers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol, cast


MAX_DOCUMENTS = 128
MAX_REQUEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RerankItem:
    """Reference an unchanged input document by its zero-based position."""

    index: int
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Retain model identity, raw scores and actual unpadded pair token usage."""

    model: str
    model_revision: str
    results: tuple[RerankItem, ...]
    input_tokens: int


class TextReranker(Protocol):
    """Score query/document pairs; higher scores indicate greater relevance."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def rerank(
        self, query: str, documents: tuple[str, ...], *, top_n: int | None = None
    ) -> RerankResult: ...


def validate_rerank_request(query: str, documents: tuple[str, ...], top_n: int | None) -> int:
    """Bound candidate count and encoded text before tokenization or inference."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or not isinstance(documents, tuple)
        or not 1 <= len(documents) <= MAX_DOCUMENTS
        or any(not isinstance(doc, str) or not doc.strip() for doc in documents)
        or (top_n is not None and (type(top_n) is not int or not 1 <= top_n <= len(documents)))
    ):
        raise ValueError("invalid_rerank_request")
    try:
        if sum(len(text.encode("utf-8")) for text in (query, *documents)) > MAX_REQUEST_BYTES:
            raise ValueError("invalid_rerank_request")
    except UnicodeError:
        raise ValueError("invalid_rerank_request") from None
    return len(documents) if top_n is None else top_n


def rank_scores(scores: tuple[float, ...], *, top_n: int) -> tuple[RerankItem, ...]:
    """Return stable descending raw scores, keeping input order for equal scores."""
    if (
        not scores
        or type(top_n) is not int
        or not 1 <= top_n <= len(scores)
        or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)
    ):
        raise ValueError("invalid_reranker_output")
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    return tuple(RerankItem(index, float(scores[index])) for index in order[:top_n])


def parse_rerank_json(payload: bytes) -> dict[str, object]:
    """Reject duplicate fields, nonstandard numbers and non-object JSON."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("invalid_rerank_request")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError("invalid_rerank_request")

    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("invalid_rerank_request")
    value = json.loads(payload, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("invalid_rerank_request")
    return value


def parse_rerank_request(value: dict[str, object]) -> tuple[str, tuple[str, ...], int, bool]:
    """Parse the shared CLI and HTTP request fields without framework coercion."""
    if set(value) - {"model", "query", "documents", "top_n", "return_documents"}:
        raise ValueError("invalid_rerank_request")
    query, documents = value.get("query"), value.get("documents")
    top_n, include = value.get("top_n"), value.get("return_documents", False)
    if not isinstance(query, str) or not isinstance(documents, list) or type(include) is not bool:
        raise ValueError("invalid_rerank_request")
    if any(not isinstance(doc, str) for doc in documents):
        raise ValueError("invalid_rerank_request")
    docs = tuple(cast(list[str], documents))
    if top_n is not None and type(top_n) is not int:
        raise ValueError("invalid_rerank_request")
    return query, docs, validate_rerank_request(query, docs, top_n), include


def render_rerank_result(
    result: RerankResult, documents: tuple[str, ...], *, return_documents: bool = False
) -> dict[str, object]:
    """Use the same JSON representation at CLI and HTTP boundaries."""
    rows: list[dict[str, object]] = []
    for item in result.results:
        row: dict[str, object] = {"index": item.index, "relevance_score": item.relevance_score}
        if return_documents:
            row["document"] = {"text": documents[item.index]}
        rows.append(row)
    return {
        "model": result.model,
        "model_revision": result.model_revision,
        "score_type": "raw_logit",
        "results": rows,
        "usage": {"total_tokens": result.input_tokens},
    }
