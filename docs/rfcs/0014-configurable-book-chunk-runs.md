# RFC-0014: Configurable Book Chunk Runs

- Status: Accepted for the configuration and chunk-projection scope authorized in
  conversation on 2026-09-03.
- Design: Configurable book chunking (archived development reference).
- Decision: [ADR-0019](../adr/0019-independent-configurable-book-chunks.md).

Introduce version-1 immutable chunk runs, source-mapped core/context spans,
configuration/preview/read Tools, and reference-only caller task graphs. All
fields, defaults, precedence, validation, identity, and failure rules follow the
linked design. The built-in byte counter is labeled estimated; exact local asset
tokenizers are opt-in and cannot download files.

Existing document evidence, SummaryTask v1, submissions, and Wiki schemas retain
their semantics. Chunk task export is a distinct contract and does not implicitly
change the old submission/publication workflow. Creating or reconfiguring a chunk
run never overwrites source evidence, existing summaries, or published Wiki pages.

Chunk configuration and runtime read budgets have separate fingerprints. New
chunk runs bind source/structure/tokenizer identity and regenerate dependent task
IDs. Read budgets cannot disable immutable-source validation, strict coverage,
source-span bounds, authorization boundaries, or acyclic task dependencies.

No network access, inference provider selection, or graph-build dependency is
introduced. Metadata and source text remain confidential and locally stored.

## Accepted front-matter clarification (2026-09-03)

The user authorized retaining TOC navigation while excluding cover/copyright
pages from new chunk extraction inputs, without removing original evidence.
Guidance, prefaces and uncertain pages remain content. Implement this as the
versioned `front_matter_policy`: `navigation-v1` by default, or `preserve` for
legacy behavior. A missing persisted field means `preserve`; its canonical
configuration omits the field to preserve existing run identities and bytes.

The version-1 run envelope is unchanged. Navigation chunks use reserved parents
`front-matter:toc:<first-page-index>`. Reads expose derived `content_role` and
`extraction_eligible` fields. Navigation is available through paginated chunk
reads, but is not a knowledge-summary task or a body-context source. Preview
reports expose navigation/content counts and exclusion counts by reason. Exact
block/reason pairs remain in the immutable run. Coverage validation recomputes
authorized exclusions and navigation boundaries from source, never trusts a
caller-supplied exclusion list, and rejects context crossing those boundaries.

Detection is offline and conservative: inspect at most the first 16 physical
pages of evidence beginning at page zero, close cover/copyright detection when
substantive content begins, and preserve ambiguous content. This is not a general
page classifier or a change to legacy SummaryTask submission/publication.
