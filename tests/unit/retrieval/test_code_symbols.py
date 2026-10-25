"""S1 code chunking, identifiers, definitions, classifier and intake policy (RFC-0033)."""

from __future__ import annotations

import json

import pytest

from capicove.retrieval.code_symbols import (
    CHUNK_TARGET_CHARACTERS,
    MAX_CHUNK_JSON_BYTES,
    IntakeRejection,
    QueryKind,
    chunk_code,
    classify_query,
    detect_language,
    extract_symbols,
    identifier_components,
    normalize_identifier,
    path_terms,
    screen_source,
)


def _lines(text: str, start: int, end: int) -> tuple[int, int]:
    return text.count("\n", 0, start) + 1, text.count("\n", 0, max(start, end - 1)) + 1


def test_chunks_cover_source_exactly_with_one_based_lines() -> None:
    text = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(200))
    chunks = chunk_code(text)
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(text)
    for chunk in chunks:
        assert (chunk.start_line, chunk.end_line) == _lines(text, chunk.start_char, chunk.end_char)
        body = text[chunk.start_char : chunk.end_char]
        assert len(body) <= CHUNK_TARGET_CHARACTERS
        assert len(json.dumps(body).encode()) <= MAX_CHUNK_JSON_BYTES
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_char, chunk.end_char))
    assert covered == set(range(len(text)))


def test_crlf_unicode_and_blank_lines_are_preserved_exactly() -> None:
    text = "# 标题\r\n\r\nvalue = '小宁'\r\n\r\n\r\ndef ok():\r\n    pass\r\n"
    chunks = chunk_code(text)
    assert "".join(text[c.start_char : c.end_char] for c in chunks) == text
    assert chunks[-1].end_line == 7


def test_overlong_single_line_splits_into_exact_subranges_without_loss() -> None:
    text = "x = '" + "a" * 9000 + "'\nshort()\n"
    chunks = chunk_code(text)
    assert len(chunks) > 2
    assert "".join(text[c.start_char : c.end_char] for c in chunks) == text
    assert all(c.start_line == 1 for c in chunks[:-1])


def test_split_blocks_overlap_by_two_lines() -> None:
    text = "".join(f"    line_{i} = {i}  # {'x' * 40}\n" for i in range(120))
    chunks = chunk_code(text)
    assert len(chunks) >= 2
    first, second = chunks[0], chunks[1]
    assert second.start_line == first.end_line - 1


def test_identifiers_keep_raw_qualified_normalized_and_components_separately() -> None:
    symbols = extract_symbols(
        "python",
        "class HttpClient:\n    def fetch_user_profile(self):\n        return self.cache.getUserID()\n",
    )
    raw = {item.raw for item in symbols.identifiers}
    assert {"HttpClient", "fetch_user_profile", "self.cache.getUserID", "getUserID"} <= raw
    assert normalize_identifier("fetch_user_profile") == "fetch_user_profile"
    assert normalize_identifier("FetchUserProfile") == "fetchuserprofile"
    assert identifier_components("fetch_user_profile") == ("fetch", "user", "profile")
    assert identifier_components("getUserID") == ("get", "user", "id")
    assert identifier_components("HTTPServer2") == ("http", "server2")


@pytest.mark.parametrize(
    ("language", "source", "name", "kind"),
    [
        ("python", "async def load_index(path):\n    pass\n", "load_index", "function"),
        ("python", "class Planner(Base):\n    pass\n", "Planner", "class"),
        ("javascript", "export function renderTree(node) {}\n", "renderTree", "function"),
        ("javascript", "const parseArgs = (argv) => argv;\n", "parseArgs", "variable"),
        (
            "typescript",
            "export interface RetryPolicy { max: number }\n",
            "RetryPolicy",
            "interface",
        ),
        ("typescript", "type Handler = () => void;\n", "Handler", "type"),
        ("go", "func (s *Server) ServeHTTP(w http.ResponseWriter) {}\n", "ServeHTTP", "function"),
        ("rust", "pub fn build_graph(input: &str) -> Graph {}\n", "build_graph", "function"),
        ("rust", "pub struct Graph { nodes: Vec<u8> }\n", "Graph", "struct"),
    ],
)
def test_heuristic_definitions_record_line_kind_and_uncertainty(
    language: str, source: str, name: str, kind: str
) -> None:
    definitions = extract_symbols(language, source).definitions
    assert [(item.name, item.kind, item.line) for item in definitions] == [(name, kind, 1)]
    assert all(item.method == "lexical-heuristic-v1" for item in definitions)


def test_uncertain_definitions_are_omitted_rather_than_invented() -> None:
    assert extract_symbols("python", "x = call(def_value)\n").definitions == ()
    assert extract_symbols("markdown", "def looks_like_code():\n").definitions == ()


def test_query_classifier_separates_identifiers_paths_and_language() -> None:
    assert classify_query("fetch_user_profile") is QueryKind.IDENTIFIER
    assert classify_query("HttpClient.fetch") is QueryKind.IDENTIFIER
    assert classify_query("src/capicove/retrieval/fusion.py") is QueryKind.PATH
    assert classify_query("fusion.py") is QueryKind.PATH
    assert classify_query("how are search results fused") is QueryKind.NATURAL_LANGUAGE
    assert classify_query("如何融合检索结果") is QueryKind.NATURAL_LANGUAGE


def test_path_terms_use_relative_segments_and_components() -> None:
    terms = path_terms("src/retrieval/textFusion.py")
    assert {"src", "retrieval", "textfusion.py", "text", "fusion", "py"} <= set(terms)
    assert all(not term.startswith("/") for term in terms)


@pytest.mark.parametrize(
    ("path", "payload", "reason"),
    [
        ("data/model.bin", b"\x00\x01\x02", "unsupported_type"),
        ("notebook.ipynb", b"{}", "notebook_unsupported"),
        ("web/app.min.js", b"var a=1;", "generated_or_minified"),
        ("package-lock.json", b"{}", "lockfile"),
        ("uv.lock", b"version = 1", "lockfile"),
        ("node_modules/x/index.js", b"module.exports = 1;", "vendored"),
        (".github/workflow.py", b"x = 1", "hidden"),
        ("src/gen.py", b"# @generated by tool\nx = 1\n", "generated_or_minified"),
        # Synthetic credential markers exercise intake rejection, not real credentials.
        (
            "src/keys.py",
            b"KEY = '''-----BEGIN RSA PRIVATE KEY-----\nabc'''\n",  # pragma: allowlist secret
            "secret_detected",
        ),
        (
            "src/aws.py",
            b"KEY = 'AKIAABCDEFGHIJKLMNOP'\n",  # pragma: allowlist secret
            "secret_detected",
        ),
        ("src/latin1.py", "café = 1\n".encode("latin-1"), "invalid_encoding"),
        ("src/binary.py", b"x = 1\x00\n", "binary"),
        ("src/minified.js", (b"a" * 5000) + b"\n", "generated_or_minified"),
    ],
)
def test_screen_rejects_denied_generated_binary_and_secret_sources(
    path: str, payload: bytes, reason: str
) -> None:
    with pytest.raises(IntakeRejection) as error:
        screen_source(path, payload)
    assert error.value.reason == reason


def test_screen_accepts_code_and_docs_and_detects_language() -> None:
    assert screen_source("src/app.py", b"def main():\n    pass\n").language == "python"
    assert screen_source("README.md", b"# Title\n").language == "markdown"
    assert detect_language("web/view.tsx") == "typescript"
    assert detect_language("cmd/main.go") == "go"


def test_python_methods_get_qualified_definition_names_from_class_scope() -> None:
    source = (
        "class ProcessingFingerprint:\n"
        "    sha256: str\n"
        "\n"
        "    @classmethod\n"
        "    def from_manifest(cls, manifest):\n"
        "        def helper():\n"
        "            return 1\n"
        "        return helper()\n"
        "\n"
        "def top_level():\n"
        "    pass\n"
    )
    names = [
        (d.name, d.qualified_name, d.line) for d in extract_symbols("python", source).definitions
    ]
    assert names == [
        ("ProcessingFingerprint", "ProcessingFingerprint", 1),
        ("from_manifest", "ProcessingFingerprint.from_manifest", 5),
        ("helper", "ProcessingFingerprint.from_manifest.helper", 6),
        ("top_level", "top_level", 10),
    ]
