# Image commands

`capicove image search` queries an activated reverse-image index, and
`capicove image serve` starts its local MCP stdio server. See the
[executable search setup](image-search.md) for configuration, requests, exit codes,
and a synthetic end-to-end reproduction. Neither command builds or activates an index.

## `capicove image run`

Invoke one canonical image Tool from the command line. Multiple `--input` values
are converted to the canonical `sources` array. For complex or model-backed tools,
pass the complete canonical request using `--arguments`.

```text
capicove image run --tool TOOL [--input PATH ...]
  [--arguments FILE]
  [--output {human,json}]
```

The default CLI composition supports the local rule-based tools:
`image_aspect_ratio_filter`, `image_shape_filter`, `image_size_filter`, and
`image_subplot_filter`, plus deterministic `image_phash` and
`image_lsh_search`. Other tools require an explicitly configured local provider
through Python or MCP; selecting one here exits with status 2 and a message on
stderr, without running the Tool.

Example:

```bash
capicove image run \
  --tool image_aspect_ratio_filter \
  --input ./images/a.png \
  --input ./images/b.png \
  --output json
```

Batch pHash arguments use stable records and optional LSH acceleration:

```json
{
  "records": [
    {"record_id": "first", "images": [{"path": "./images/first.png"}]},
    {"record_id": "second", "images": [{"path": "./images/second.jpg"}]}
  ],
  "max_hamming_distance": 2,
  "enable_lsh": true
}
```

```bash
capicove image run --tool image_phash --arguments phash.json --output json
```

Stateless LSH search consumes fixed-width hexadecimal hashes:

```json
{
  "query": {"hashes": ["0123456789abcdef"]},
  "candidates": [
    {"record_id": "first", "hashes": ["0123456789abcdee"]}
  ],
  "hash_version": "phash-64-v1",
  "max_hamming_distance": 2,
  "top_k": 100
}
```

```bash
capicove image run --tool image_lsh_search --arguments lsh-search.json --output json
```
