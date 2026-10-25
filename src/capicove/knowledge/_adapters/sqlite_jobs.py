"""Sqlite jobs operations for temporal memory."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from capicove.contracts.knowledge import micros


if TYPE_CHECKING:
    from capicove.knowledge._adapters.sqlite_memory import SQLiteMemoryRepository


def claim_job(
    self: SQLiteMemoryRepository,
    workspace: str,
    scope: str,
    identity: str,
    *,
    now: datetime,
    lease_seconds: int,
    force: bool = False,
) -> dict[str, Any] | None:
    self._guard()
    if type(lease_seconds) is not int or not 1 <= lease_seconds <= 300:
        raise ValueError("invalid lease")
    with self._connection() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute(
                "SELECT * FROM outbox_jobs WHERE workspace=? AND scope=? AND identity=?",
                (workspace, scope, identity),
            ).fetchone()
            if (
                row is None
                or row["state"] in {"completed", "partially_completed", "dead_letter"}
                or (row["state"] == "leased" and row["lease_until"] > micros(now))
                or (not force and row["next_attempt"] > micros(now))
            ):
                db.rollback()
                return None
            token = uuid4().hex
            attempts = row["attempts"] + 1
            db.execute(
                "UPDATE outbox_jobs SET state='leased',token=?,lease_until=?,attempts=? WHERE workspace=? AND scope=? AND identity=?",
                (
                    token,
                    micros(now) + lease_seconds * 1000000,
                    attempts,
                    workspace,
                    scope,
                    identity,
                ),
            )
            db.commit()
            return {"identity": identity, "token": token, "attempts": attempts}
        except BaseException:
            db.rollback()
            raise
