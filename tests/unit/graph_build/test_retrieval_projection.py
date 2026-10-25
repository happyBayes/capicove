"""Exact bounded text projection without changing source graph evidence."""

import importlib
import importlib.util
from typing import Any

import pytest


def module() -> Any:
    assert importlib.util.find_spec("capicove.graph_build.retrieval") is not None, (
        "Task 21 retrieval projection is not implemented"
    )
    return importlib.import_module("capicove.graph_build.retrieval")


class CharacterTokenizer:
    fingerprint = "a" * 64

    def count(self, text: str) -> int:
        return len(text)


def test_embedding_subunits_do_not_truncate_graph_text() -> None:
    text = "abcdefg"
    spans = module().embedding_ranges(text, CharacterTokenizer(), 3)
    assert spans == ((0, 3), (3, 6), (6, 7))
    assert "".join(text[a:b] for a, b in spans) == text


@pytest.mark.parametrize("text", ["中🙂文" * 200, "a" * 1025])
def test_long_unicode_text_is_preserved(text: str) -> None:
    spans = module().embedding_ranges(text, CharacterTokenizer(), 512)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(0 < b - a <= 512 for a, b in spans)


@pytest.mark.parametrize("limit", [True, 0, -1, 2.5])
def test_invalid_limits_reject(limit: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        module().embedding_ranges("abc", CharacterTokenizer(), limit)


@pytest.mark.parametrize("text", [None, b"abc", "\ud800"])
def test_invalid_text_rejects(text: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        module().embedding_ranges(text, CharacterTokenizer(), 3)


@pytest.mark.parametrize("count", [True, -1, 0, 1.5, float("nan")])
def test_invalid_tokenizer_counts_reject(count: Any) -> None:
    class Invalid(CharacterTokenizer):
        def count(self, text: str) -> Any:
            return count

    with pytest.raises(ValueError):
        module().embedding_ranges("abc", Invalid(), 3)


def test_oversized_projection_rejects_before_tokenizer_calls() -> None:
    class NeverCount(CharacterTokenizer):
        def count(self, text: str) -> int:
            raise AssertionError("oversized input must fail before counting")

    with pytest.raises(ValueError, match="bounded"):
        module().embedding_ranges("a" * 65_537, NeverCount(), 512)


def test_nonmonotonic_counts_never_create_overlimit_ranges() -> None:
    class Nonmonotonic(CharacterTokenizer):
        def count(self, text: str) -> int:
            return {1: 1, 2: 8, 3: 1}.get(len(text), 8)

    text = "abcdefg"
    spans = module().embedding_ranges(text, Nonmonotonic(), 2)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(Nonmonotonic().count(text[a:b]) <= 2 for a, b in spans)


def test_single_codepoint_overflow_fails_explicitly() -> None:
    class Overflow(CharacterTokenizer):
        def count(self, text: str) -> int:
            return 5 if "🙂" in text else len(text)

    with pytest.raises(ValueError, match="codepoint"):
        module().embedding_ranges("a🙂b", Overflow(), 3)


def test_runtime_exposes_exact_document_wire_count_and_rejects_mismatch() -> None:
    from capicove.retrieval import TextEmbeddingService
    from tests.unit.retrieval.test_text_embedding import Encoder, text_space

    m = module()
    assert hasattr(m, "GraphEmbeddingRuntime"), "Missing explicit document embedding runtime"

    class DocumentTokenizer:
        fingerprint = "a" * 64
        space = text_space()
        verified_max_tokens = 5

        def count_document(self, text: str) -> int:
            return len(text) + 2

    runtime = m.GraphEmbeddingRuntime(
        DocumentTokenizer(), TextEmbeddingService(Encoder(), space=text_space()), "b" * 64, 2
    )
    assert m.embedding_ranges("abcdefg", runtime, runtime.max_tokens) == ((0, 3), (3, 6), (6, 7))
    assert runtime.space == text_space()
    with pytest.raises(ValueError):
        m.GraphEmbeddingRuntime(DocumentTokenizer(), runtime.service, "not-a-digest", 2)
