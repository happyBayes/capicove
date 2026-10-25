"""v1 index replacement is transactional and preserves the entity search budget."""

import json
import sqlite3

import pytest

from capicove.knowledge._adapters.sqlite_schema import _SCHEMA, migrate


def v1_database() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.executescript(_SCHEMA)
    for i in range(101):
        identity = f"entity{i:03d}"
        name = f"member {i:03d}"
        db.execute(
            "INSERT INTO entities VALUES (?,?,?,?)",
            ("w", "s", identity, json.dumps({"id": identity, "name": name})),
        )
        db.execute(
            "INSERT INTO entity_fts(workspace,scope,identity,text) VALUES (?,?,?,?)",
            ("w", "s", identity, name),
        )
    return db


def test_v1_existing_entities_retain_one_index_entry_per_candidate() -> None:
    db = v1_database()
    try:
        migrate(db)
        for _ in range(2):
            rows = db.execute(
                "SELECT identity FROM entity_fts WHERE text MATCH 'member' ORDER BY identity LIMIT 100"
            ).fetchall()
            assert len(rows) == len(set(rows)) == 100
            assert db.execute("SELECT COUNT(*) FROM entity_fts").fetchone() == (101,)
            migrate(db)
    finally:
        db.close()


def test_v1_index_insertion_failure_rolls_back_delete_columns_and_version() -> None:
    db = v1_database()
    try:
        before = db.execute("SELECT * FROM entity_fts ORDER BY identity").fetchall()

        def deny_insert(
            action: int,
            table: str | None,
            column: str | None,
            database: str | None,
            trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_INSERT and table == "entity_fts":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        db.set_authorizer(deny_insert)
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            migrate(db)
        db.set_authorizer(None)
        assert not db.in_transaction
        assert db.execute("SELECT * FROM entity_fts ORDER BY identity").fetchall() == before
        assert "valid_from" not in {
            row[1] for row in db.execute("PRAGMA table_info(fact_versions)")
        }
        assert db.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]
        migrate(db)
        assert db.execute("SELECT COUNT(*) FROM entity_fts").fetchone() == (101,)
    finally:
        db.close()
