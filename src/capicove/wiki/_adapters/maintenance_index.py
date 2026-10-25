"""Disposable FTS5 projection with versioned, exact snapshot verification."""

import sqlite3
from pathlib import Path

from capicove.wiki.filesystem import atomic_write
from capicove.wiki.maintenance.store import read_authorized


def build_index(path: Path, snapshot: bytes, rows: tuple[tuple[str, str, str], ...]) -> None:
    """Replace only a derived index, retaining exact source inventory metadata."""
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE metadata (snapshot BLOB NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?)", (snapshot,))
        connection.execute("CREATE VIRTUAL TABLE lexical USING fts5(id UNINDEXED, text, han)")
        connection.executemany("INSERT INTO lexical VALUES (?, ?, ?)", rows)
        connection.commit()
        data = connection.serialize()
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("unsafe discovery index")
    atomic_write(path, data)


def ranked_lanes(
    path: Path,
    snapshot: bytes,
    rows: tuple[tuple[str, str, str], ...],
    latin_query: str,
    han_query: str,
) -> dict[str, tuple[str, ...]]:
    """Reject missing, corrupt or stale indexes before returning any candidates."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("discovery index unavailable; rebuild required")
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(read_authorized(path, (path.parent,), 768 * 1024 * 1024))
        connection.execute("PRAGMA query_only=ON")
        metadata = connection.execute("SELECT snapshot FROM metadata").fetchall()
        stored = connection.execute("SELECT id, text, han FROM lexical ORDER BY id").fetchall()
        if metadata != [(snapshot,)] or tuple(stored) != tuple(sorted(rows)):
            raise ValueError("discovery index stale; rebuild required")
        result: dict[str, tuple[str, ...]] = {}
        for lane, query in (("fts", latin_query), ("han", han_query)):
            if query:
                column = "text" if lane == "fts" else "han"
                values = connection.execute(
                    f"SELECT id FROM lexical WHERE {column} MATCH ? ORDER BY bm25(lexical), id",
                    (query,),
                ).fetchall()
                result[lane] = tuple(str(row[0]) for row in values)
        return result
    except sqlite3.Error as error:
        raise ValueError("discovery index corrupt; rebuild required") from error
    finally:
        connection.close()
