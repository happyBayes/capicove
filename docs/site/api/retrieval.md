# Retrieval API

The public retrieval contracts describe text, image, and video-clip queries,
searchable units, filters, execution profiles, compact or full hits, temporal
locators, and result status. The service depends on injected index, embedding, and
source-resolution ports rather than a database or model SDK.

::: capicove.retrieval

The optional SigLIP2 composition API is local-only and keeps model selection
explicit.

::: capicove.api.siglip2


## Explicit modality entry points

`capicove.search` exposes independent `search_text`, `search_image`, and
`search_video` functions. They do not infer a modality or silently fall back to
another search implementation.

| Function | Tool identity | Status |
| --- | --- | --- |
| `search_text` | `capicove.search` | Existing text schema and permissions preserved |
| `search_image` | `capicove.search.image` | Personal preview; explicit authorization factory required |
| `search_video` | `capicove.search.video` | Rejected as unavailable; no backend or MCP registration |

Text transports continue importing `capicove.tools.search` and
`capicove.mcp.search`. Image transports import `capicove.tools.image_search` and
`capicove.mcp.image_search`; the image MCP name is `capicove_search_image`.
These names allow text and image registration without duplicate Tool keys or
MCP names. Previously drafted image clients must update both names and imports.
This does not grant image permissions to a text caller or enable remote providers.

The image MCP server admits at most four synchronous searches. Excess requests
receive a retry-later MCP error without starting work. Cancelling a caller stops
waiting but does not terminate the worker; its admission slot and generation
leases remain held until completion under the existing deadline. Event-loop
shutdown must drain the default executor before closing shared runtime resources.

The text generation catalog remains in `retrieval.generations`; personal image
generation values are isolated in `retrieval.image_generations`. Their SQLite
adapters and formats remain separate. No database migration is performed.

::: capicove.search

## Reverse-image runtime

::: capicove.runtime.reverse_image

The runtime factory composes only local CPU providers. It does not build, publish,
qualify, or activate a generation as a side effect.
