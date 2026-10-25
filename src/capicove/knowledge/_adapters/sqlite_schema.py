"""Private SQLite schema and projection inventory."""

import sqlite3


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_resolution_candidates(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity));
CREATE TABLE IF NOT EXISTS workspace_heads(workspace TEXT PRIMARY KEY, revision INTEGER NOT NULL, hash TEXT NOT NULL, recorded INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS memory_operations(workspace TEXT, key TEXT, scope TEXT NOT NULL, request_hash TEXT NOT NULL, receipt TEXT NOT NULL, receipt_hash TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE, first_revision INTEGER NOT NULL, last_revision INTEGER NOT NULL, PRIMARY KEY(workspace,key), UNIQUE(workspace,operation_id), CHECK(first_revision>0 AND last_revision>=first_revision), FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS memory_events(workspace TEXT, revision INTEGER, scope TEXT NOT NULL, event_id TEXT UNIQUE NOT NULL, event_type TEXT NOT NULL, recorded INTEGER NOT NULL, payload TEXT, payload_hash TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL, operation_id TEXT NOT NULL, schema_version INTEGER NOT NULL CHECK(schema_version=1), aggregate_type TEXT NOT NULL, aggregate_id TEXT NOT NULL, PRIMARY KEY(workspace,revision), FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace), FOREIGN KEY(workspace,operation_id) REFERENCES memory_operations(workspace,operation_id) DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX IF NOT EXISTS events_operation ON memory_events(workspace,operation_id);
CREATE INDEX IF NOT EXISTS events_erased ON memory_events(workspace,scope) WHERE event_type='MemoryErased' AND payload IS NOT NULL;
CREATE TABLE IF NOT EXISTS episodes(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS episode_processing_state(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace,scope,identity) REFERENCES episodes(workspace,scope,identity) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS entities(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS entity_aliases(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS entity_redirects(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS memory_records(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS memory_state(workspace TEXT,scope TEXT,identity TEXT,body TEXT NOT NULL,PRIMARY KEY(workspace,scope,identity),FOREIGN KEY(workspace,scope,identity) REFERENCES memory_records(workspace,scope,identity) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS memory_target ON memory_records(workspace,scope,json_extract(body,'$.target_ref'));
CREATE INDEX IF NOT EXISTS episode_recency_ordered ON episodes(workspace,scope,json_extract(body,'$.event_time') DESC,identity);
CREATE INDEX IF NOT EXISTS episode_recorded ON episodes(workspace,scope,json_extract(body,'$.recorded'));
CREATE TABLE IF NOT EXISTS projection_offsets(workspace TEXT PRIMARY KEY,revision INTEGER NOT NULL,FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE TABLE IF NOT EXISTS fact_versions(workspace TEXT, scope TEXT, version TEXT, fact TEXT, slot TEXT, assertion TEXT, recorded_from INTEGER, recorded_to INTEGER, body TEXT NOT NULL, PRIMARY KEY(workspace,scope,version), CHECK(recorded_to IS NULL OR recorded_from < recorded_to), FOREIGN KEY(workspace) REFERENCES workspace_heads(workspace));
CREATE INDEX IF NOT EXISTS facts_subject ON fact_versions(workspace,scope,json_extract(body,'$.subject'),recorded_from,recorded_to);
CREATE INDEX IF NOT EXISTS facts_object ON fact_versions(workspace,scope,json_extract(body,'$.object.value'),recorded_from,recorded_to);
CREATE INDEX IF NOT EXISTS facts_slot ON fact_versions(workspace,scope,slot,recorded_to);
CREATE INDEX IF NOT EXISTS facts_recency_ordered ON fact_versions(workspace,scope,recorded_from DESC,fact);
CREATE INDEX IF NOT EXISTS facts_parent ON fact_versions(workspace,scope,json_extract(body,'$.parent_version_id'));
CREATE INDEX IF NOT EXISTS facts_history ON fact_versions(workspace,scope,fact,recorded_from);
CREATE INDEX IF NOT EXISTS facts_assertion ON fact_versions(workspace,scope,assertion,recorded_to);
CREATE UNIQUE INDEX IF NOT EXISTS facts_current ON fact_versions(workspace,scope,assertion) WHERE recorded_to IS NULL;
CREATE TABLE IF NOT EXISTS fact_evidence(workspace TEXT,scope TEXT,version TEXT,digest TEXT,start INTEGER,end INTEGER,PRIMARY KEY(workspace,scope,version,digest,start,end),CHECK(start>=0 AND end>start),FOREIGN KEY(workspace,scope,version) REFERENCES fact_versions(workspace,scope,version) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS fact_conflicts(workspace TEXT,scope TEXT,version TEXT,conflict_key TEXT,state TEXT CHECK(state IN ('accepted','disputed','rejected')),PRIMARY KEY(workspace,scope,version),FOREIGN KEY(workspace,scope,version) REFERENCES fact_versions(workspace,scope,version) ON DELETE CASCADE);
CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(workspace UNINDEXED,scope UNINDEXED,identity UNINDEXED,text,tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(workspace UNINDEXED, scope UNINDEXED, identity UNINDEXED, text, tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(workspace UNINDEXED, scope UNINDEXED, identity UNINDEXED, text, tokenize='trigram');
CREATE VIRTUAL TABLE IF NOT EXISTS episode_words USING fts5(workspace UNINDEXED,scope UNINDEXED,identity UNINDEXED,text);
CREATE VIRTUAL TABLE IF NOT EXISTS fact_words USING fts5(workspace UNINDEXED,scope UNINDEXED,identity UNINDEXED,text);
CREATE TABLE IF NOT EXISTS outbox_jobs(workspace TEXT, scope TEXT, identity TEXT, state TEXT NOT NULL, token TEXT, lease_until INTEGER, attempts INTEGER NOT NULL, body TEXT NOT NULL, next_attempt INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(workspace,scope,identity));
CREATE TABLE IF NOT EXISTS fts_entries(workspace TEXT,scope TEXT,kind TEXT,identity TEXT,docid INTEGER NOT NULL,PRIMARY KEY(workspace,scope,kind,identity),UNIQUE(kind,docid));
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_migrations VALUES(1);
"""


_RECORD_TABLES = {
    "resolution": "entity_resolution_candidates",
    "episode": "episodes",
    "entity": "entities",
    "alias": "entity_aliases",
    "redirect": "entity_redirects",
    "memory": "memory_records",
    "archive": "memory_state",
    "extraction": "episode_processing_state",
}
_PROJECTION_TABLES = (
    "entity_resolution_candidates",
    "fts_entries",
    "fact_evidence",
    "fact_conflicts",
    "episode_processing_state",
    "entity_aliases",
    "entity_redirects",
    "memory_state",
    "memory_records",
    "fact_versions",
    "entities",
    "episodes",
    "outbox_jobs",
    "projection_offsets",
    "episode_fts",
    "fact_fts",
    "entity_fts",
    "episode_words",
    "fact_words",
)


# Additive v1 -> v2 migration. Events and receipts are never rewritten.
def migrate(db: sqlite3.Connection) -> None:
    columns = {row[1] for row in db.execute("PRAGMA table_info(fact_versions)")}
    if "valid_from" not in columns:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("ALTER TABLE fact_versions ADD COLUMN valid_from INTEGER")
            db.execute(
                "ALTER TABLE fact_versions ADD COLUMN valid_to INTEGER CHECK(valid_to IS NULL OR valid_from IS NULL OR valid_from < valid_to)"
            )
            db.execute(
                "UPDATE fact_versions SET valid_from=json_extract(body,'$.validity[0]'), valid_to=json_extract(body,'$.validity[1]')"
            )
            db.execute("INSERT OR IGNORE INTO schema_migrations VALUES(2)")
            # v1 projections may already have populated this derived index.
            # Rebuild it in the migration transaction rather than append duplicates.
            db.execute("DELETE FROM entity_fts")
            db.execute(
                "INSERT INTO entity_fts(workspace,scope,identity,text) SELECT workspace,scope,identity,json_extract(body,'$.name') FROM entities"
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise
    db.execute(
        "CREATE INDEX IF NOT EXISTS facts_bitemporal ON fact_versions(workspace,scope,valid_from,valid_to,recorded_from,recorded_to)"
    )
