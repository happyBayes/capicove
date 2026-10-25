"""Sqlite queries operations for temporal memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from capicove.contracts.knowledge import micros
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge._adapters.memory_query_terms import question_terms


if TYPE_CHECKING:
    from capicove.knowledge._adapters.sqlite_memory import SQLiteMemoryRepository


def candidates(
    self: SQLiteMemoryRepository, query: MemoryQuery, *, lane: str, identities: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Filter in SQLite before taking a lane's candidate budget."""
    self._guard()
    valid, known = (
        micros(query.valid_at or datetime.now(UTC)),
        micros(query.known_at or datetime.now(UTC)),
    )
    if lane not in {"lexical", "recency", "graph", "ids", "utility"}:
        raise ValueError("unknown lane")
    limit = 50 if lane in {"recency", "utility"} else 100
    result: list[dict[str, Any]] = []
    with self._read_connection() as db:
        for target in query.types:
            fact = target == "semantic"
            table, identity, fts = (
                ("fact_versions", "fact", "fact_fts")
                if fact
                else ("episodes", "identity", "episode_fts")
            )
            from_clause = f"{table} t"
            if lane == "recency":
                index = "facts_recency_ordered" if fact else "episode_recency_ordered"
                from_clause += f" INDEXED BY {index}"
            elif lane == "utility" and not fact:
                from_clause += " INDEXED BY sqlite_autoindex_episodes_1"
            terms = ["t.workspace=?", "t.scope=?"]
            values: list[Any] = [query.workspace_ref, query.scope_ref]
            if fact:
                terms += [
                    "t.recorded_from<=?",
                    "(t.recorded_to IS NULL OR ?<t.recorded_to)",
                    "(t.valid_from IS NULL OR t.valid_from<=?)",
                    "(t.valid_to IS NULL OR ?<t.valid_to)",
                    "json_extract(t.body,'$.state') IN ("
                    + ",".join("?" for _ in query.states)
                    + ")",
                ]
                values += [known, known, valid, valid, *query.states]
                order = "t.recorded_from DESC,t.fact"
            else:
                terms += [
                    "json_extract(t.body,'$.recorded')<=?",
                    "json_extract(t.body,'$.event_time')<=?",
                ]
                values += [known, valid]
                order = "json_extract(t.body,'$.event_time') DESC,t.identity"
            # Admission timestamps are ledger-assigned and strictly increasing
            # between commits. Select the latest policy before checking its state:
            # never fall back to an older admission after expiry or archive.
            # Target and column names come from the closed MemoryQuery kinds.
            latest_policy = (
                "SELECT m.identity FROM memory_records m "  # nosec B608
                "WHERE m.workspace=t.workspace AND m.scope=t.scope "
                f"AND json_extract(m.body,'$.target_ref')=+t.{identity} "
                f"AND json_extract(m.body,'$.kind')='{target}' "
                "ORDER BY json_extract(m.body,'$.recorded') DESC,m.identity DESC LIMIT 1"
            )
            terms.append(
                "EXISTS (SELECT 1 FROM memory_records p "  # nosec B608
                "WHERE p.workspace=t.workspace AND p.scope=t.scope "
                f"AND p.identity=({latest_policy}) "
                "AND (json_extract(p.body,'$.retention_deadline') IS NULL "
                "OR ?<json_extract(p.body,'$.retention_deadline')) "
                "AND NOT EXISTS (SELECT 1 FROM memory_state s "
                "WHERE s.workspace=p.workspace AND s.scope=p.scope AND s.identity=p.identity))"
            )
            values.append(micros(datetime.now(UTC)))
            if lane == "lexical":
                planned = question_terms(query.text)
                tokens = list(planned) if planned is not None else query.text.split()
                if not tokens:
                    continue
                word_search = planned is not None or all(
                    token.isascii() and token.isalnum() for token in tokens
                )
                if word_search:
                    fts = "fact_words" if fact else "episode_words"
                index = "facts_history" if fact else "sqlite_autoindex_episodes_1"
                from_clause = f"{fts} f CROSS JOIN {table} t INDEXED BY {index} ON t.workspace=f.workspace AND t.scope=f.scope AND t.{identity}=f.identity"
                if word_search:
                    terms.append("f.text MATCH ?")
                    values.append(
                        (" OR " if planned is not None else " AND ").join(
                            '"' + token.replace('"', '""') + '"' for token in tokens
                        )
                    )
                    order = "f.rank ASC," + order
                else:
                    for token in tokens:
                        terms.append("f.text LIKE ? ESCAPE '\\'")
                        values.append(
                            "%"
                            + token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                            + "%"
                        )
            elif lane == "ids":
                if not identities:
                    continue
                terms.append(f"t.{identity} IN (" + ",".join("?" for _ in identities) + ")")
                values += list(identities)
            elif lane == "graph":
                if not fact or not identities:
                    continue
                marks = ",".join("?" for _ in identities)
                filtered = " AND ".join(terms).replace("t.", "g.")
                outgoing = f"SELECT g.version FROM fact_versions g INDEXED BY facts_subject WHERE {filtered} AND json_extract(g.body,'$.subject') IN ({marks}) ORDER BY g.recorded_from DESC LIMIT 100"  # nosec B608
                incoming = f"SELECT g.version FROM fact_versions g INDEXED BY facts_object WHERE {filtered} AND json_extract(g.body,'$.object.kind')='entity' AND json_extract(g.body,'$.object.value') IN ({marks}) ORDER BY g.recorded_from DESC LIMIT 100"  # nosec B608
                from_clause = f"(SELECT version FROM ({outgoing}) UNION SELECT version FROM ({incoming})) neighbors CROSS JOIN fact_versions t ON t.version=neighbors.version"  # nosec B608
                values = [*values, *identities, *values, *identities, *values]
            elif lane == "utility":
                # Seed by admitted targets so an empty policy set avoids scanning
                # the lifetime corpus. The common gate still checks the latest one.
                terms.append(
                    f"t.{identity} IN (SELECT json_extract(body,'$.target_ref') "  # nosec B608
                    "FROM memory_records WHERE workspace=? AND scope=?)"
                )
                values += [query.workspace_ref, query.scope_ref]
                order = (
                    "(SELECT json_extract(p.body,'$.importance') FROM memory_records p "
                    "WHERE p.workspace=t.workspace AND p.scope=t.scope "
                    f"AND p.identity=({latest_policy})) DESC," + order  # nosec B608
                )
            sql = (
                "SELECT t.body"
                + (",t.recorded_to" if fact else "")
                + f" FROM {from_clause} WHERE "
                + " AND ".join(terms)
                + f" ORDER BY {order} LIMIT ?"
            )
            values.append(limit)
            for lane_rank, row in enumerate(db.execute(sql, values), 1):
                body = json.loads(row[0])
                if fact:
                    body["recorded_to"] = row[1]
                body["target_kind"] = target
                body["lane_rank"] = lane_rank
                result.append(body)
    result.sort(key=lambda b: (b["lane_rank"], b["id"]))
    return result[:limit]


def entity_assertions_for_merge(
    self: SQLiteMemoryRepository, workspace: str, scope: str, entity_id: str
) -> list[dict[str, Any]]:
    self._guard()
    with self._connection() as db:
        rows = db.execute(
            "SELECT body FROM fact_versions WHERE workspace=? AND scope=? AND recorded_to IS NULL AND COALESCE(json_extract(body,'$.retracted'),0)=0 AND (json_extract(body,'$.subject')=? OR (json_extract(body,'$.object.kind')='entity' AND json_extract(body,'$.object.value')=?))",
            (workspace, scope, entity_id, entity_id),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def memory_policy_for(
    self: SQLiteMemoryRepository, workspace: str, scope: str, target: str
) -> dict[str, Any] | None:
    self._guard()
    with self._connection() as db:
        row = db.execute(
            "SELECT body FROM memory_records WHERE workspace=? AND scope=? AND json_extract(body,'$.target_ref')=? ORDER BY json_extract(body,'$.recorded') DESC LIMIT 1",
            (workspace, scope, target),
        ).fetchone()
    return None if row is None else json.loads(row[0])
