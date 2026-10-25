# Local code search (preview)

Search one or more local Git repositories by identifier, qualified name, relative
path or plain words, and get exact line-cited evidence. The S1 profile
`local-code-lexical-v1` uses the policy `code-symbols-v1`. It needs no model
and never runs repository code, hooks or Git filters.

!!! note "What S1 does and does not claim"
    Definitions come from versioned lexical heuristics (`lexical-heuristic-v1`)
    and are labelled as such: no AST parsing, type resolution or call graph. Queries
    in a different natural language from the code (for example Chinese questions
    about English code) need the hybrid stage below. The repository acceptance
    restrictions are listed in [known limitations](../reference/known-limitations.md).

## Configure

Use a version-2 workspace host file with `capabilities.retrieval` schema 2:

```json
"capabilities": {
  "retrieval": {
    "schema_version": 2,
    "source_roots": {"repo": "/src/my-project"},
    "index_root": "/srv/workspace/code-index",
    "corpus_ref": "corpus:my-project",
    "retrieval_profile_ref": "local-code-lexical-v1",
    "max_source_bytes": 1048576,
    "model_config_path": null,
    "model_config_sha256": null,
    "model_binding": null,
    "max_units": 10000,
    "max_vector_bytes": 0,
    "code_policy_ref": "code-symbols-v1"
  }
}
```

Each source root must be a Git repository root. Enable `capicove.search` and
`capicove.get_evidence`; import and build need a `write` or `admin` profile.
Schema-1 retrieval configurations keep their existing Markdown behavior.

## Import, build, search

```bash
capicove code import --runtime-config host.json --root repo --git /usr/bin/git
capicove index build --runtime-config host.json
capicove code search fetch_user_profile --runtime-config host.json
capicove code search HttpClient.fetch_user_profile --runtime-config host.json
capicove code search src/app/views.ts --runtime-config host.json
capicove get-evidence --runtime-config host.json cev-...
```

- **Discovery.** Git only enumerates candidates: `ls-files` with hooks and
  fsmonitor disabled, a scrubbed environment and bounded output, plus
  `check-ignore --no-index`, so ignored tracked files stay out. Hidden paths,
  symlinks, submodules, vendored and build directories, lockfiles, notebooks,
  binary, generated or minified files, non-UTF-8 files and files with private keys
  or common access tokens are excluded and counted by reason. Each import call is
  capped at 10,000 candidates and 64 MiB.
- **Single files.** `capicove source import --root repo --relative-path PATH`
  applies exactly the same policy.
- **Builds** verify every live file against its imported hash, fail if a file
  changed without re-import, seal the generation and activate it atomically.
  Rebuilding unchanged inputs returns the same `cg-` generation ID.
- **Ranking.** For identifier queries the order is exact definitions, then exact
  uses, then normalized matches, then other lexical matches. Path queries rank
  exact relative-path and file-name matches first. Ties use weighted reciprocal
  rank fusion and then unit ID.
- **Evidence** returns the exact chunk text, character range, one-based lines,
  language, source hash and every heuristic definition in the chunk, read from the
  sealed generation rather than the live file.

The same Tools serve MCP clients through `capicove mcp serve --config host.json`.

## Optional structure-aware chunking (S2)

Install the optional extra (`pip install "capicove[code-ast]"`) and set
`"code_policy_ref": "code-ast-v1"`. Builds then parse Python, JavaScript/JSX and
TypeScript/TSX with Tree-sitter. Chunks follow whole functions and classes, and
evidence reports each definition's qualified scope, signature and `tree-sitter-v1`
method. Other languages, Markdown and files with syntax errors keep exact S1 line
chunks, and evidence says so (`parse_status`, `parse_reason`). `capicove index build`
prints `parse_outcomes`. Switching policy produces a new generation.

## Optional hybrid search with a local model (S3)

The hybrid profile adds a dense lane from the locally verified EmbeddingGemma 2
model on llama.cpp, so natural-language and cross-language questions (for example
Chinese questions about English code) can find code. Nothing is downloaded: first
prepare the pinned assets and write a code model configuration with the setup
script, then point the retrieval configuration at it.

Run the `receipt` command from the [native audio guide](native-audio-search.md)
with one extra option, `--code-model-config-output /srv/capicove/code-models.json`;
it writes the code (text-only) model configuration next to the multimodal one.

In `capabilities.retrieval` set `"retrieval_profile_ref": "local-code-hybrid-v1"`,
`"code_policy_ref": "code-hybrid-v1"`, `"model_binding": "code"`,
`"model_config_path"` to that file, `"model_config_sha256"` to its SHA-256 and a
positive `"max_vector_bytes"` (at most 268435456). Builds embed every chunk on the
CPU, which is slow (tens of vectors per minute on a 4-core host); rebuilding an
unchanged chunk reuses its cached vector. Searches warn
`text_dense_embedding_unavailable` and return lexical results only if the encoder
cannot run; corrupt vectors or a changed model pin fail the search instead.

## Optional relation expansion (S4)

With `"code_policy_ref": "code-relations-v1"` on the hybrid profile, builds also
record containment, internal imports and unambiguous direct calls (Python). A
search keeps its first five results, then adds up to 20 directly called or
contained definitions before the remaining results, within the same `top_k` and a
16 KiB snippet budget. Evidence lists the relations that justified a unit.
Ambiguous, dynamic and external calls are recorded as unresolved and never expand.
This stage is not the default; its benefit has not been measured against a
predeclared threshold.
