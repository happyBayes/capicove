"""Sqlite diagnostics operations for temporal memory."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from capicove.knowledge._adapters.sqlite_memory import SQLiteMemoryRepository


def _verify(self: SQLiteMemoryRepository) -> bool:
    # Stream in ledger order and retain only operation metadata. No projection
    # rebuild, payload decoding, or optional provider is involved in recovery.
    encode = json.JSONEncoder(
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode
    sha256 = hashlib.sha256
    with self._connection() as db:
        erased = set(
            db.execute(
                "SELECT DISTINCT workspace,scope FROM memory_events WHERE event_type='MemoryErased' AND payload IS NOT NULL"
            ).fetchall()
        )
        erased_scopes = {(row[0], row[1]) for row in erased}
        for head in db.execute("SELECT * FROM workspace_heads"):
            workspace = head["workspace"]
            previous, revision = "", 0
            previous_operation = None
            operation_count = 0
            operation_last = 0
            recorded_last = 0
            # Operations are fetched once per batch while events are streamed.
            cursor = db.cursor()
            cursor.row_factory = None
            events = cursor.execute(
                "SELECT revision,scope,event_id,event_type,recorded,payload,payload_hash,previous_hash,hash,operation_id,schema_version,aggregate_type,aggregate_id FROM memory_events WHERE workspace=? ORDER BY revision",
                (workspace,),
            )
            request_hash = receipt_hash = ""
            for event in events:
                (
                    number,
                    scope,
                    event_id,
                    event_type,
                    recorded,
                    payload,
                    payload_hash,
                    previous_hash,
                    event_hash,
                    operation_id,
                    schema_version,
                    aggregate_type,
                    aggregate_id,
                ) = event
                revision += 1
                if operation_id != previous_operation:
                    operation_count += 1
                    operation = db.execute(
                        "SELECT scope,request_hash,receipt,receipt_hash,first_revision,last_revision FROM memory_operations WHERE workspace=? AND operation_id=?",
                        (workspace, operation_id),
                    ).fetchone()
                    if (
                        operation is None
                        or operation["first_revision"] != revision
                        or (previous_operation is not None and operation_last != revision - 1)
                        or recorded <= recorded_last
                    ):
                        self._corrupt = True
                        return False
                    if (workspace, operation["scope"]) not in erased_scopes and sha256(
                        operation["receipt"].encode()
                    ).hexdigest() != operation["receipt_hash"]:
                        self._corrupt = True
                        return False
                    previous_operation = operation_id
                    operation_last = operation["last_revision"]
                    request_hash, receipt_hash = (
                        operation["request_hash"],
                        operation["receipt_hash"],
                    )
                recorded_last = recorded
                expected = sha256(
                    encode(
                        [
                            workspace,
                            scope,
                            revision,
                            event_id,
                            event_type,
                            recorded,
                            payload_hash,
                            previous,
                            operation_id,
                            schema_version,
                            aggregate_type,
                            aggregate_id,
                            request_hash,
                            receipt_hash,
                        ]
                    ).encode()
                ).hexdigest()
                if (
                    number != revision
                    or previous_hash != previous
                    or event_hash != expected
                    or (payload is None and (workspace, scope) not in erased_scopes)
                    or (
                        payload is not None and sha256(payload.encode()).hexdigest() != payload_hash
                    )
                ):
                    self._corrupt = True
                    return False
                previous = expected
            if (
                operation_count
                != db.execute(
                    "SELECT COUNT(*) FROM memory_operations WHERE workspace=?", (workspace,)
                ).fetchone()[0]
                or head["revision"] != revision
                or head["hash"] != previous
                or head["recorded"] != recorded_last
                or operation_last != revision
            ):
                self._corrupt = True
                return False
    self._corrupt = False
    return True
