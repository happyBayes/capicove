# CLI reference

Capicove uses capability-first nested commands:

```text
capicove doctor
capicove kb --help
capicove evolution --help
capicove setup init --help
capicove setup doctor --help
capicove setup check-llm --help
capicove image run
capicove image search
capicove image serve
capicove video split
capicove video filter
capicove video deduplicate
capicove audio transcribe
capicove audio import --help
capicove audio capabilities
capicove batch run
capicove jobs --help
capicove traces export
capicove wiki init
capicove mcp catalog
capicove mcp serve
capicove http serve --help
capicove models --help
capicove source import --help
capicove source forget --help
capicove index build --help
capicove search --help
capicove get-evidence --help
capicove retrieval --help
capicove code --help
capicove book --help
capicove memory --help
```

The [native knowledge guide](../guides/native-knowledge.md) covers unified import,
update, graph construction and evidence-backed questions through `capicove kb`.
The [Skill evolution guide](../guides/skill-evolution.md) covers local versioning,
evaluation and promotion through `capicove evolution` on POSIX platforms.
The [native audio guide](../guides/native-audio-search.md) covers `capicove retrieval`
(shared-space audio, image, video and note search), and the
[code search guide](../guides/code-search.md) covers `capicove code`.

Explicit image-model acquisition uses the separate
`capicove-download-image-models` executable.

See the [capability matrix](../getting-started/capabilities.md) for prerequisites.
Model installation, book parsing, search and evidence access require the
configuration described in their guides. `--help` does not download models.
Use the [workspace setup reference](setup.md) for local startup checks and the
explicit compatible LLM readiness probe.

Run any command with `--help` for the parser's current option list. Output formats
are capability-specific; use `--output json` where the command exposes that option.
Image search and Wiki commands emit JSON directly. Image serve and MCP serve
reserve stdout for the MCP stdio protocol.

Capabilities with canonical Tool contracts use the same handlers across CLI and
other transports. The standalone Wiki command is a direct adapter over
`WikiService`; its local path checks, structured results, and failure categories
remain capability-owned.
