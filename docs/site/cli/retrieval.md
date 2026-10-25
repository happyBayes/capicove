# Offline search commands

These commands use a version-2 workspace host configuration with
`capabilities.retrieval`. See the [offline workspace search guide](../guides/offline-retrieval.md)
for the configuration and the full journey.

```text
capicove source import --runtime-config CONFIG --root ALIAS --relative-path PATH
capicove source forget --runtime-config CONFIG --root ALIAS --relative-path PATH
capicove index build --runtime-config CONFIG
capicove search QUERY --runtime-config CONFIG [filters]
capicove get-evidence EVIDENCE_ID --runtime-config CONFIG [--section SECTION ...]
```

| Command | Profile | Effect |
| --- | --- | --- |
| `source import` | `write` or `admin` | Records one verified source; never modifies it. |
| `source forget` | `write` or `admin` | Retires one source from the next build's inventory; the file need not exist. |
| `index build` | `write` or `admin` | Verifies every import, then builds and activates one generation. |
| `search` | any; needs `capicove.search` in `enabled_tools` | Reads the active generation. |
| `get-evidence` | any; needs `capicove.get_evidence` in `enabled_tools` | Reads one generation-bound evidence record. |

`source import`, `source forget`, and `index build` print one JSON object and exit
with status `0`. They exit with status `2` and print `Import failed: …`,
`Forget failed: …`, or `Build failed: …` on stderr when the configuration, path, media type, or source identity is rejected.

`search` and `get-evidence` print the canonical Tool result envelope. They exit
with status `0` on success, `1` for a failed Tool result such as an unavailable
or stale evidence ID, and `2` when the runtime configuration cannot start.
Without `--runtime-config`, these two commands still require a Python host to
inject their handlers.
