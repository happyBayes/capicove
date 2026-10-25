# Memory commands

The `memory` command maps local JSON request files to canonical memory Tools.
Every invocation requires a trusted host configuration and a bounded request:

```text
capicove memory --config CONFIG COMMAND --request REQUEST
```

| Command | Operation | Main request fields |
| --- | --- | --- |
| `remember` | `remember_episode` | `scope`, `content`, `event_time`, `idempotency_key` |
| `assert-fact` | `assert_fact` | `scope`, subject, predicate, object, validity, evidence, key |
| `recall` | `recall` | `scope`, query, time filters, result and context budgets |
| `get-episode` | `get_episode` | `scope`, `episode_id` |
| `history` | `get_fact_history` | `scope`, `fact_id` |
| `retry` | `retry_extraction` | `scope`, `episode_id`, `idempotency_key` |
| `work` | `work_extractions` | `scope`, optional `limit` from 1 to 100 |
| `resolve-entity` | `resolve_entity` | `scope`, name, entity type, idempotency key |
| `archive` | `archive_memory` | `scope`, `memory_id`, `idempotency_key` |
| `erase` | `erase_scope` | `scope`, `idempotency_key` |
| `doctor` | `doctor` | `scope` |

The configuration file is limited to 64 KiB and the request file to 2 MiB. Both
must contain JSON objects. Successful Tool results exit with status `0`; failed
Tool results and configuration or request failures exit with status `1`.

Configuration and validation failures emit JSON with a stable code and safe field
or hint. Errors do not echo request values, source content, or provider details.
The CLI acts as the authenticated local owner. Hosted MCP integrations must
provide their own authenticated context and scope authorization.

See the [local temporal memory guide](../guides/temporal-memory.md) for complete
examples, privacy constraints, backup guidance, and recovery behavior.
