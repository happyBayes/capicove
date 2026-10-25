"""Private rebuildable SQLite FTS5 index for local Wiki pages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path

from capicove.wiki.filesystem import WikiIndexUnavailableError
from capicove.wiki.indexing import WikiDocument


_INDEX_SCHEMA_VERSION = "1"
_FTS_TOKENIZER = "unicode61"


def rebuild_index(database_path: Path, documents: tuple[WikiDocument, ...]) -> None:
    """Build a complete temporary FTS5 database and replace the old projection."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.", dir=database_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.execute(
                f"""
                CREATE VIRTUAL TABLE pages_fts USING fts5(
                    page_id UNINDEXED,
                    page_path UNINDEXED,
                    title,
                    body,
                    kind UNINDEXED,
                    source_ids UNINDEXED,
                    tokenize = '{_FTS_TOKENIZER}'
                )
                """
            )
            connection.execute(
                "CREATE TABLE wiki_index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.executemany(
                """
                INSERT INTO pages_fts(page_id, page_path, title, body, kind, source_ids)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.page_id,
                        item.page_path,
                        item.title,
                        item.body,
                        item.kind,
                        json.dumps(item.source_ids, ensure_ascii=False, separators=(",", ":")),
                    )
                    for item in documents
                ],
            )
            connection.executemany(
                "INSERT INTO wiki_index_metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", _INDEX_SCHEMA_VERSION),
                    ("content_fingerprint", _content_fingerprint(documents)),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, database_path)
    except (OSError, sqlite3.Error, UnicodeError) as error:
        raise WikiIndexUnavailableError("Wiki search index build failed") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def tokenize_query(query: str) -> tuple[str, ...]:
    """Return query tokens produced by the index's SQLite FTS5 tokenizer."""
    try:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                f"""
                CREATE VIRTUAL TABLE query_fts USING fts5(
                    text,
                    tokenize = '{_FTS_TOKENIZER}'
                )
                """
            )
            connection.execute(
                "CREATE VIRTUAL TABLE query_vocab USING fts5vocab(query_fts, instance)"
            )
            connection.execute("INSERT INTO query_fts(text) VALUES (?)", (query,))
            rows = connection.execute(
                "SELECT term FROM query_vocab ORDER BY doc, col, offset"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise WikiIndexUnavailableError("Wiki search tokenizer is unavailable") from error
    return tuple(str(row[0]) for row in rows)


def search_index(
    database_path: Path,
    query: str,
    *,
    tokens: tuple[str, ...],
    top_k: int,
) -> tuple[WikiDocument, ...]:
    """Return FTS candidates in stable provider order."""
    if not database_path.is_file() or database_path.is_symlink():
        raise WikiIndexUnavailableError("Wiki search index is unavailable; run wiki build")
    expression = _query_expression(query, tokens)
    try:
        connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT page_id, page_path, title, body, kind, source_ids
                FROM pages_fts
                WHERE pages_fts MATCH ?
                ORDER BY bm25(pages_fts), page_path
                LIMIT ?
                """,
                (expression, top_k),
            ).fetchall()
            documents = _decode_rows(rows)
        finally:
            connection.close()
    except WikiIndexUnavailableError:
        raise
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError) as error:
        raise WikiIndexUnavailableError(
            "Wiki search index is unreadable; run wiki build"
        ) from error
    return tuple(documents)


def check_index(database_path: Path, documents: tuple[WikiDocument, ...]) -> None:
    """Verify the derived index schema, integrity, and source fingerprint."""
    if not database_path.is_file() or database_path.is_symlink():
        raise WikiIndexUnavailableError("Wiki search index is unavailable; run wiki build")
    try:
        connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
        connection.row_factory = sqlite3.Row
        try:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            metadata = dict(connection.execute("SELECT key, value FROM wiki_index_metadata"))
            rows = connection.execute(
                """
                SELECT page_id, page_path, title, body, kind, source_ids
                FROM pages_fts
                ORDER BY rowid
                """
            ).fetchall()
            indexed_documents = _decode_rows(rows)
        finally:
            connection.close()
    except WikiIndexUnavailableError:
        raise
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as error:
        raise WikiIndexUnavailableError(
            "Wiki search index is unreadable; run wiki build"
        ) from error
    if integrity is None or integrity[0] != "ok":
        raise WikiIndexUnavailableError("Wiki search index is corrupt; run wiki build")
    if metadata.get("schema_version") != _INDEX_SCHEMA_VERSION:
        raise WikiIndexUnavailableError("Wiki search index schema is incompatible; run wiki build")
    if metadata.get("content_fingerprint") != _content_fingerprint(documents):
        raise WikiIndexUnavailableError("Wiki search index is stale; run wiki build")
    if _content_fingerprint(tuple(indexed_documents)) != _content_fingerprint(documents):
        raise WikiIndexUnavailableError("Wiki search index is stale; run wiki build")


def _decode_rows(rows: list[sqlite3.Row]) -> list[WikiDocument]:
    documents: list[WikiDocument] = []
    for row in rows:
        source_ids = json.loads(str(row["source_ids"]))
        if not isinstance(source_ids, list) or any(
            not isinstance(item, str) for item in source_ids
        ):
            raise WikiIndexUnavailableError("Wiki search index contains invalid sources")
        documents.append(
            WikiDocument(
                page_id=str(row["page_id"]),
                page_path=str(row["page_path"]),
                title=str(row["title"]),
                body=str(row["body"]),
                kind=str(row["kind"]),
                source_ids=tuple(source_ids),
            )
        )
    return documents


def _content_fingerprint(documents: tuple[WikiDocument, ...]) -> str:
    payload = [
        {
            "body": document.body,
            "kind": document.kind,
            "page_id": document.page_id,
            "page_path": document.page_path,
            "source_ids": list(document.source_ids),
            "title": document.title,
        }
        for document in documents
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _read_only_uri(database_path: Path) -> str:
    return f"{database_path.resolve().as_uri()}?mode=ro"


def _query_expression(query: str, tokens: tuple[str, ...]) -> str:
    phrases = re.findall(r"\w+", query, flags=re.UNICODE)
    if not phrases or tokenize_query(" ".join(phrases)) != tokens:
        phrases = list(tokens)
    return " AND ".join(f'"{phrase.replace(chr(34), chr(34) * 2)}"' for phrase in phrases)


__all__ = ["check_index", "rebuild_index", "search_index", "tokenize_query"]
