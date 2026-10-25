# Wiki maintenance (experimental v3)

Wiki maintenance lets an Agent or person propose source-cited concept pages while
the host keeps every published byte under review. Sources are captured unchanged,
an Agent receives a bounded task with exact evidence, its proposal is validated
and previewed, a reviewer with `wiki:review` accepts or rejects it, and only then
can it be applied. Answers cite the original source spans.

!!! warning "Experimental, not release-qualified"
    Schema-v3 workspaces are not readable by older Capicove versions, and v3
    migration has no command-line entry point yet: it stays an explicit operator
    action through `capicove.wiki.maintenance.migration.migrate_v3` until the W3
    review and qualification gates pass. Do not migrate a workspace you cannot
    restore. Semantic quality has not been qualified with a real Agent or
    independent reviewers. The acceptance record is
    [known limitations](../reference/known-limitations.md).

## Prepare a disposable v3 workspace

```bash
pip install "capicove[wiki]"
capicove wiki init /srv/wiki-demo --date 2026-10-23
python -c "from pathlib import Path; from capicove.wiki.maintenance.migration import migrate_v3; print(migrate_v3(Path('/srv/wiki-demo/wiki'), preview=True))"
python -c "from pathlib import Path; from capicove.wiki.maintenance.migration import migrate_v3; migrate_v3(Path('/srv/wiki-demo/wiki'), preview=False)"
```

The preview lists every created and updated path and changes nothing. Applying
it keeps all pages and raw sources byte-identical, records a migration revision,
and builds the disposable discovery index used by tasks.

## Operate with `capicove wiki maintain`

Every operation takes the workspace path and, where it needs input, a closed JSON
request file (`--request FILE`; unknown or duplicate fields are rejected). Sources
must lie below the workspace or a directory named with `--source-root`. Output is
the canonical Tool result as JSON; exit status 2 means rejected or failed, with no
change to the Wiki.

```bash
capicove wiki maintain capture /srv/wiki-demo --source-root /srv/sources --request capture.json
capicove wiki maintain task /srv/wiki-demo --request task.json
capicove wiki maintain submit /srv/wiki-demo --request submit.json
capicove wiki maintain preview /srv/wiki-demo --request preview.json
capicove wiki maintain review /srv/wiki-demo --request review.json
capicove wiki maintain apply /srv/wiki-demo --request apply.json
capicove wiki maintain status /srv/wiki-demo
capicove wiki ask /srv/wiki-demo "Which claims are disputed?"
```

| Operation | Request fields | Permission |
| --- | --- | --- |
| `capture` | `source` | `wiki:write` |
| `task`, `context` | `source_ids`, `query`; optional `page_ids`, `intervals` (`context` adds `task_id`) | `wiki:read`, `wiki:write` |
| `submit` | `submission` (the encoded proposal record) | `wiki:write` |
| `preview` | `proposal_id` | `wiki:read` |
| `review` | `proposal_id`, `decisions` (one per review subject), `accept` | `wiki:review` |
| `apply` | `proposal_id`, `review_id` | `wiki:write`, `wiki:publish` |
| `follow-up` | `intent_id`, `page_ids` | `wiki:read`, `wiki:write` |
| `rollback` | `page_id`, `publication_id`, `idempotency_key` | `wiki:read`, `wiki:write` |
| `status` | none | `wiki:read` |
| `recover` | `abort` | `wiki:publish` |
| `withdraw` | `source_id`, `reason`, `idempotency_key` | `wiki:publish` |
| `answer-task`, `ask` | `question`, `include_history` | `wiki:read` |
| `answer-submit` | `task`, `submission` | `wiki:read` |

The command-line operator holds every permission locally; the same handlers are
the canonical Tools, and the default local MCP catalog exposes only the read
operations.

## What the host enforces

- **Review is separate from writing.** A review is recorded only for a caller with
  `wiki:review`, with a supported decision for every claim, disposition and
  affected page. A reviewer identity inside submitted JSON is rejected.
- **Pinned tasks.** A task pins the Wiki revision, source policy and evidence. Any
  later publication, including legacy `ingest` or `build`, makes it stale.
- **Dirty and managed pages.** A legacy writer cannot revise a managed page, and
  apply refuses a target edited by hand; the edit is preserved.
- **Withdrawal versus erasure.** `withdraw` removes a source from tasks and
  answers immediately and is recorded; history and raw bytes are retained.
- **Recovery.** While a publication is pending, every reader and writer refuses.
  `recover` resumes from recorded hashes, or `{"abort": true}` restores the prior
  view. Process-kill recovery is tested; power-loss durability is not claimed.
- **Rollback** is a new reviewed compensating revision, never history deletion.
