# PDF book fast-reading workflow

Capicove can turn a local structured textbook, technical book, or nonfiction PDF
into a resumable, page-aware Wiki. Capicove performs local parsing and deterministic
validation; the caller's Agent supplies structure judgments and summaries. Parsing
uses explicitly configured local models. The
[model factory](model-factory.md) can select the parser with
`book prepare --model-config /absolute/models.toml --model-binding book.parser`.

## Scope

!!! warning "Check support before installing models"

    Audience: users with a compatible local parser environment and an Agent to
    complete review tasks. Profile v2 rejects pages containing raster images and
    pages without a valid nonblank text layer, including many covers and scans.
    The legacy workflow can publish a Wiki; chunk reading/task references are
    not accepted by legacy `book submit` or `book publish`. Decide which workflow
    you need before parsing. Model installation requires explicit network access.

The command examples below require replacing local paths and preparing the
listed models/configuration. They are integration recipes, not standalone demos.
For an offline first task with no model setup, use the
[Quickstart](../getting-started/quickstart.md).

The first profile targets Chinese and English books with an explicit chapter
structure. It prioritizes embedded PDF bookmarks, then printed table-of-contents
alignment, PP-StructureV3 title blocks, and deterministic numbering rules. A
globally or locally consistent high-confidence section passes directly. Only
low-confidence local regions become structure-review tasks.

Profile version 2 supports Chinese, English, and Japanese layout OCR. It inspects a
valid embedded Unicode text layer only to detect a dominant script outside that
scope, then fails explicitly instead of promoting unverified or invisible PDF text
to source evidence. Missing, blank, or invalid text mappings are rejected.
Unsupported OCR language hints fail before model initialization and are recorded
in the processing identity, but they do not prove the source script. A page without
a valid, nonblank text layer is rejected because this profile cannot independently
verify its source language. Profile version 2 also rejects every page containing a
raster image, even when the page has a valid text layer, because it cannot yet
independently verify the script inside that image. Covers, logos, charts, figures,
photos, scan-only pages, and hybrid native/OCR pages therefore remain outside the
qualified capability. Supporting them or additional scripts requires a new reviewed
profile with an independent source-image script gate.

Printed TOC rows may be multiline text blocks, and chapter headings may be
classified as images. Plain section numbers restart beneath the current chapter.
Contradictory numbering stays low-confidence. An explicit chapter heading missing
from the detected TOC is retained for local review; it neither advances the trusted
chapter number nor suppresses a later TOC-matched heading. Unlisted local callouts
remain section content.

Semantic topic segmentation for unstructured prose, remote PDF acquisition,
provider credential management, and automatic model invocation are outside this
profile.

## Install the optional parser

First install a PaddlePaddle runtime compatible with the target CPU or accelerator
by following PaddlePaddle's platform instructions. Confirm it in the target
environment:

```bash
python -c "import paddle; print(paddle.__version__)"
```

Then install Capicove and the separate parser plugin:

```bash
pip install "capicove[wiki]"
pip install capicove-processor-paddleocr
```

From a repository checkout, register the local plugin in the same environment as
the Capicove CLI before installing models:

```bash
uv pip install -e ./plugins/processor-paddleocr
```

The plugin uses PaddleOCR's `paddleocr[doc-parser]` extra and PP-StructureV3. It
does not import PaddleOCR during ordinary Capicove startup and never downloads a
model during parsing.

Preview the exact pinned repositories, revisions, expected size, hosts, and local
destination before authorizing the only default network operation:

```bash
capicove models install paddleocr \
  --profile book-balanced \
  --dry-run

capicove models install paddleocr \
  --profile book-balanced \
  --yes
```

Set `CAPICOVE_MODEL_HOME` to change the model storage root, or pass
`--destination`. The `book-balanced` manifest pins PP-DocLayout-L,
PP-OCRv5-server-det, and PP-OCRv5-server-rec to exact Git revisions. Parsing binds
all three local directories and disables table, formula, chart, seal, translation,
and information-extraction modules. The manifest also records the supported
languages, compatible PaddleOCR, PaddlePaddle, PaddleX, pypdf, and PDFium ranges,
the fixed pipeline configuration, and every installed file's size and SHA-256. The
dry-run prints those identities before confirmation.

At parse time the plugin validates and records the exact installed PaddleOCR,
PaddlePaddle CPU or GPU, PaddleX, pypdf, and PDFium runtime versions. Those exact
versions are part of the configuration and processing fingerprints; missing,
ambiguous, or incompatible runtimes fail before model or source access.

The SDK model names are explicitly bound to `PP-DocLayout-L`,
`PP-OCRv5_server_det`, and `PP-OCRv5_server_rec`; a local model directory alone
must not allow the SDK to select a different model architecture. Every parse checks
the schema 2 installation receipt and rehashes the closed 18-file model tree before
opening the PDF. Re-run the same `models install ... --yes` command to upgrade an
older receipt; already present files are verified locally and are not downloaded
again when they match the packaged manifest.

## Prepare a book

```bash
capicove book prepare ./reading-workspace ./books/systems.pdf \
  --parser paddleocr \
  --profile book-balanced \
  --profile-version 2 \
  --language-hint zh \
  --language-hint en
```

The command stores the PDF by SHA-256, parses page evidence, preserves PDF
bookmarks and explicitly embedded page labels, builds the chapter tree, and writes
immutable Agent tasks. The parser copies and hashes a temporary source snapshot;
metadata extraction and page inference read that same snapshot. The returned
`book_id` identifies all later operations. Source paths are not included in Agent
tasks.

Test a bounded range before committing to a long parse:

```bash
capicove book prepare ./range-workspace ./books/systems.pdf \
  --start-page-index 14 \
  --end-page-index 16
```

Both indices are required together. The range is zero-based and half-open: this
example parses physical PDF pages 15 and 16, retaining original indices 14 and 15
in evidence and citations. The complete source PDF is preserved, but only the
requested pages enter inference. Out-of-bounds ranges and encrypted PDFs are
rejected before model initialization. In-range child bookmarks survive even when
their parent starts before the range. Range preparations have a different
`book_id` from full-book preparations. A range excluding the TOC has less hierarchy
evidence and may produce more local-review tasks; it is not whole-book acceptance.

For accepted pages, the parser processes one temporary single-page PDF at a time,
with a maximum text-detector side of 736 pixels, recognition batch size 1, and 2 CPU
threads. These settings bound inference inputs, not total RAM. Native model memory
is still substantial, and OCR can still misread small or stylized text. There is no
hard memory limit, execution timeout, or per-page OCR checkpoint yet. Progress
counts and SDK diagnostics go to stderr, leaving the CLI's JSON result on stdout.

The default model directory is
`$CAPICOVE_MODEL_HOME/paddleocr/book-balanced`, or
`~/.local/share/capicove/models/paddleocr/book-balanced` when the environment
variable is absent. Override it with `--model-root`.

## Run tasks with any Agent model

Export dependency-ordered JSONL:

```bash
capicove book tasks ./reading-workspace BOOK_ID --format jsonl > tasks.jsonl
```

Each line has `task_type: structure_review` or `task_type: summary`.

A structure-review submission uses only the task's candidates, context blocks,
allowed parent positions, and permitted operations:

```json
{
  "task_id": "structure-review:...",
  "operation": "accept",
  "selected_block_ids": [],
  "target_section_id": null
}
```

The closed operations are `accept`, `reject`, `merge`, `split`, and `reparent`.
Merge targets must be the direct parent or an adjacent sibling with a touching or
overlapping page range. Remote sections cannot be used as arbitrary merge targets.
After every local structure task is valid, Capicove creates the bottom-up summary
graph.

Each summary task currently allows at most 128 evidence blocks. A larger section
raises an explicit error before task persistence; evidence is never silently
truncated. Parent content preceding a child heading on the same page is retained.
Automatic chunk-and-reduce execution is not implemented. For long sections, use
the independent configurable chunk workflow below. Its reference task graph does
not use the legacy summary submission contract.

A summary submission is provider-neutral:

```json
{
  "task_id": "summary:...",
  "idempotency_key": "...",
  "abstract": "A concise section abstract.",
  "key_points": ["One important point."],
  "claims": [
    {
      "claim_id": "claim:stable-local-id",
      "text": "One factual statement.",
      "citation_ids": ["p0012:b000042"],
      "quotation": null
    }
  ],
  "provider": "caller-owned-provider",
  "model": "caller-owned-model"
}
```

Every factual claim requires at least one citation ID allowed by its task.
Dependencies must be complete and the idempotency key must still match. Submit one
result at a time:

```bash
capicove book submit ./reading-workspace BOOK_ID --input result.json
capicove book status ./reading-workspace BOOK_ID
```

Accepted summary resubmissions append a new version linked to the previous one;
they never overwrite evidence or an older result.

## Publish the Wiki

```bash
capicove book publish ./reading-workspace BOOK_ID
capicove wiki build ./reading-workspace
capicove wiki query ./reading-workspace "distributed systems"
```

For a new workspace, publication creates a Wiki v2 projection. An existing Wiki
v1 must be migrated explicitly:

```bash
capicove wiki migrate-v2 ./reading-workspace
capicove wiki migrate-v2 ./reading-workspace --apply
```

The first command is a no-write preview. Wiki v2 retains the content-addressed PDF,
the parser representation, one Markdown page per section, and a citation JSONL
ledger. Citation links display the parsed printed page label when available and
always retain the zero-based PDF page index.

If the source has no `/PageLabels` dictionary, `printed_page_label` remains null.
pypdf's synthesized physical numbering is not treated as a printed book label.
OCR-derived printed-page mapping is not implemented; PDF page indices remain the
authoritative navigation anchors.

Strict publication runs deterministic validation:

- L0 resolves claim, page, source, representation, and block identities.
- L1 checks page lineage and exact-evidence SHA-256 digests.
- L2 checks quotations after NFC and ASCII-whitespace normalization.

L0-L2 errors block publication. Optional semantic support checking is an L3
caller-Agent judgment and is never treated as deterministic proof.

## Tool and MCP operations

Agents can use the same workflow without shell parsing through these canonical
Tools:

- `capicove.book.prepare`
- `capicove.book.next-task`
- `capicove.book.submit`
- `capicove.book.status`
- `capicove.book.publish`

The MCP names are `capicove_book_prepare`, `capicove_book_next_task`,
`capicove_book_submit`, `capicove_book_status`, and `capicove_book_publish`.
Inputs and results remain bounded; PDF bytes and whole-book text are never inlined
into MCP calls.

## Configure hybrid book chunks

Chunk runs are independent immutable projections over prepared OCR evidence and
the finalized chapter tree. Tuning chunks never reruns OCR or overwrites old
summaries or Wiki pages. The implementation combines structure-aware grouping,
visible content-type cues, recursive splitting, bounded packing and Parent reads.

For a new long book, prepare evidence without generating legacy summary tasks:

```bash
capicove book prepare ./reading-workspace ./books/systems.pdf --evidence-only
```

This explicit option has a distinct `book_id`; it is not a migration of an older
preparation. It still creates necessary structure-review tasks. Export and submit
those reviews using `book tasks` and `book submit` as above. Review finalization
does not create legacy summaries for an evidence-only preparation. Existing
prepared books can be chunked directly once their structure is finalized.

### Preview, save and compare

```bash
capicove book chunk ./reading-workspace BOOK_ID \
  --chunk-preset textbook-balanced --dry-run

capicove book chunk ./reading-workspace BOOK_ID \
  --chunk-preset textbook-balanced

capicove book chunk ./reading-workspace BOOK_ID \
  --target-core-tokens 300 --max-chunk-tokens 600 \
  --context-before-tokens 50 --compare-run RUN_ID --dry-run
```

Save returns `run_id` and `read_policy_id`. Repeating identical configuration is
idempotent; changing chunk-affecting parameters creates another run. Dry runs do
not write files or change active state. There is no implicit mutable latest run.
Comparisons name the baseline explicitly and report changed fields.

The preview includes core/rendered min, median, p95 and max counts, number of
chunks, added-input fraction, warning counts, exclusions by reason, complete
core coverage and tokenizer identity. Count sums include serialization separators
and can change after repacking; source-span coverage is the losslessness check.

### Front matter: navigation versus extraction content

New runs default to `front_matter_policy = "navigation-v1"`:

- Confidently recognized cover/title and copyright pages are excluded from core,
  overlap and original Parent reads. The PDF and parsed blocks are not deleted.
  Immutable `run.json` records each excluded block with its reason; previews
  summarize `front_matter:cover` and `front_matter:copyright` counts.
- TOC text remains source-mapped, separately packed navigation. `book chunks`
  returns `content_role: navigation` and `extraction_eligible: false`; these chunks
  are available for chapter navigation/structure inspection, not knowledge
  summary tasks or body context. Paginate all chunks to discover navigation;
  `--tasks` deliberately omits it.
- Reader guidance, prefaces and ambiguous pages remain extraction content.
  Detection uses conservative Chinese/English signals within the first 16
  physical pages when evidence starts at page zero. It stops excluding when
  substantive content begins. Later partial ranges do not assume a new cover.
  This is not a general classifier; unusual layouts or poor OCR may be retained.

```bash
capicove book chunk ./reading-workspace BOOK_ID \
  --front-matter-policy navigation-v1 --compare-run OLD_RUN_ID --dry-run
capicove book chunk ./reading-workspace BOOK_ID \
  --front-matter-policy navigation-v1
capicove book chunks ./reading-workspace BOOK_ID NEW_RUN_ID --limit 50
```

Use `--front-matter-policy preserve` to reproduce the previous behavior. Existing
runs without the field keep their original semantics and remain readable; new
rules require creating a new run, never overwriting an old one. CLI, TOML and MCP
use the same policy. This policy applies to independent chunk runs, not the
legacy `book tasks` / `submit` / `publish` summary workflow.

### Public parameters

Configuration precedence is built-in defaults, selected preset, TOML values,
then explicit CLI or canonical Tool overrides. These are initial tuning values,
not measured optimal settings for every textbook or model.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `preset` / `--chunk-preset` | `textbook-balanced` | Versioned content-protection defaults. |
| `target_core_tokens` | 600 | Soft packing target; an intact block or protected group may exceed it. |
| `max_chunk_tokens` | 1200 | Hard limit for rendered heading path + before/core/after text. |
| `context_before_tokens` | 100 | Optional preceding evidence within the same Parent. |
| `context_after_tokens` | 0 | Optional following evidence within the same Parent. |
| `boundary_level` | `deepest` | Group by deepest section, level-2 section, or level-1 chapter (`section`, `chapter`). |
| `tokenizer` | `utf8-bytes-v1` | Explicit estimated byte counter, or `local:/absolute/path/tokenizer.json`. |
| `protected_types` | Preset-specific | Visible content types whose compatible adjacent groups stay together when budgets permit. |
| `oversize_policy` | `split` | Recursively split oversized blocks; `error` fails instead. Formulas fail when unsplittable under either policy. |
| `page_boundary` | `allow_continuations` | Cross pages only for same-type visible continuations under the same Parent; `strict` forbids it. |
| `merge_small_chunks` | `true` | Pack ordinary adjacent blocks; disabling it does not disable protected-group packing. |
| `front_matter_policy` | `navigation-v1` | Keep TOC as navigation and exclude recognized cover/copyright; `preserve` reproduces legacy behavior. |
| `max_context_read_tokens` | 2000 | Separate saved ceiling for on-demand original Parent reads, not a chunk fingerprint input. |

CLI field flags use hyphens, for example `--context-after-tokens 50`. Use
`--no-merge-small-chunks` to disable ordinary packing. `--protected-types` accepts
space-separated names; passing no names clears the list. Valid types are
`definition`, `principle`, `proof`, `procedure`, `example`, `comparison`, `code`,
`table` and `formula`.

All count budgets are integers, not booleans. Core target, chunk maximum and
Parent-read budget must be 1–32768; optional context may be zero. The core target
cannot exceed the maximum, and before/after budgets must sum to less than it.
Unknown fields/enums and invalid combinations fail without writing a run.

| Preset | Protected types | Page policy |
| --- | --- | --- |
| `textbook-balanced` | definition, example, procedure, formula | allow_continuations |
| `technical-code` | code, procedure, table | strict |
| `math-heavy` | definition, principle, proof, formula, example | allow_continuations |

Numerical defaults are identical across these first presets. For code-heavy
books, compare `technical-code` against the baseline and inspect split warnings.
For long proofs, increase the hard budget only within the Agent's available
input capacity. Smaller targets can improve locality but increase repeated
title/context overhead; measure both rather than minimizing chunk size alone.

An explicitly selected TOML file may contain only this table:

```toml
[chunking]
preset = "math-heavy"
target_core_tokens = 700
max_chunk_tokens = 1400
context_before_tokens = 80
context_after_tokens = 0
boundary_level = "deepest"
oversize_policy = "split"
page_boundary = "allow_continuations"
merge_small_chunks = true
max_context_read_tokens = 2200
```

```bash
capicove book chunk ./reading-workspace BOOK_ID \
  --chunk-config ./math-chunks.toml --target-core-tokens 800 --dry-run
```

### Token accounting and evidence ownership

`utf8-bytes-v1` counts UTF-8 bytes and reports `token_count_mode: estimated`.
In particular, a Chinese character usually consumes three count units. This is
not an arbitrary model's token budget guarantee. For exact tokenizer-input
counts, install the optional `tokenizers` package in the same environment as
Capicove, supply an existing local asset, and choose it explicitly:

```bash
capicove book chunk ./reading-workspace BOOK_ID \
  --tokenizer local:/absolute/path/tokenizer.json --dry-run
```

No tokenizer asset is downloaded. The local JSON hash and runtime version bind
the run identity. Asset padding and truncation are disabled before counting.
Exact counts cover the configured tokenizer, without special tokens, chat
templates, prompts or output reservations; the caller must budget those extras.
Context reads reject a changed tokenizer asset or runtime fingerprint.

Each chunk has separate `core_spans`, `before_spans` and `after_spans`. A span
contains a block ID, zero-based PDF page index, half-open Unicode-code-point
offsets and the exact slice SHA-256. The core covers every retained source block
exactly once. Header/footer/page-number and whitespace-only exclusions are
explicit; readers revalidate the excluded block set against original evidence.
Extract entities and relations only for core evidence, not for repeated context.

Read bounded pages and original Parent context:

```bash
capicove book chunks ./reading-workspace BOOK_ID RUN_ID --limit 10
capicove book chunks ./reading-workspace BOOK_ID RUN_ID --offset 10 --limit 10
capicove book context ./reading-workspace BOOK_ID RUN_ID CHUNK_ID \
  --read-policy-id READ_POLICY_ID --max-context-read-tokens 1000
```

Follow `next_offset`; page size is at most 50 and may be shortened by the byte
limit. Chunk text is capped at 128 KiB and 128 total evidence spans independently
of tokenizer counts. Parent text is anchored on the requested core, then expands
to nearby same-parent cores within those ceilings. A small budget can return a
partial core, explicitly marked `core_complete: false`. Context read budgets
measure the returned original text, not its surrounding JSON metadata.

The requested read limit is clamped to the selected saved policy. Recreating a
run with a different `max_context_read_tokens` adds a policy without rechunking or
changing task IDs. Pass the returned `read_policy_id` to select it; if several
policies exist and none is selected, the most restrictive applies. Reads remain
read-only and return the effective policy and budget.

Content typing is deterministic cue matching, not semantic adjudication. Prefer
paragraph/sentence boundaries for prose, line/row boundaries for code/tables,
then safe Unicode prefixes with explicit warnings. Protected groups may be split
at hard constraints, with `protected_group_split:*` warnings. The current OCR
profile disables specialized formula/table recognition: chunk tuning cannot
recover missing mathematical structure, guarantee correct table headers, or
prove that each chunk represents exactly one complete knowledge point.

### Caller-owned task export and MCP

```bash
capicove book chunks ./reading-workspace BOOK_ID RUN_ID --tasks --limit 50
capicove book chunk-mcp /absolute/path/reading-workspace
```

The task export is a reference-only dependency DAG: chunk summaries, bounded
reduction groups (at most 32 inputs), section summaries, then a book summary.
Tasks contain run/chunk references rather than unbounded book text. The Agent
uses chunk/context reads and its own model. `submission_contract` is null:
**these new tasks cannot yet be submitted through legacy `book submit` or
published through legacy `book publish`**. Automatic task execution, persistent
new-style summary results and Wiki publication from these runs remain separate
workflow integration. The existing v1 summary/Wiki path remains usable.

The optional MCP SDK is loaded only when starting the stdio server. Its four
Tools are `capicove_book_chunk`, `capicove_book_chunk_preview`,
`capicove_book_chunks` and `capicove_book_context`. The corresponding canonical
Tool IDs are `capicove.book.chunk`, `capicove.book.chunk-preview`,
`capicove.book.chunks` and `capicove.book.context`; Python hosts can register them
with `capicove.tools.register_book_chunk_tools`.

MCP create/preview accepts structured configuration, not arbitrary TOML paths:

```json
{
  "workspace": "/absolute/path/reading-workspace",
  "book_id": "book:0123456789abcdef01234567",
  "chunking": {
    "preset": "technical-code",
    "target_core_tokens": 700,
    "max_chunk_tokens": 1400
  }
}
```

The host binds the MCP server to one authorized workspace. External workspaces
and local tokenizer assets outside that workspace are rejected, including assets
referenced by an existing run. Preview/read Tools require only read permissions;
creation requires write permissions. Filesystem paths are local and no inference
or network calls are made.

## Privacy and recovery

PDFs, OCR text, tasks, summaries, and citations are confidential local data. Model
installation is explicit network access; parsing, task exchange, validation, Wiki
rendering, and search are offline. A missing model, failed page, unresolved
structure task, incomplete dependency, stale submission, or invalid citation stays
visible and blocks strict publication. Re-running `book prepare` with the same
source, parser, profile, and page range reuses a completed preparation. It does
not rerun OCR or upgrade old structure results after a code change; use a fresh
workspace when validating a repaired parser. An interrupted initial parse must
restart parsing the requested range.

Workspace schema 2 binds every prepared artifact to its manifest. Schema 1
workspaces are rejected rather than automatically trusted or rewritten. To retain
a legacy source, create a fresh workspace and run `book prepare` again from the
original PDF with the intended parser and profile. Read-only commands never migrate
or modify a legacy workspace.

Structure decisions are validated before they enter the append-only history.
Invalid submissions leave the task pending. After all decisions are valid, the
reviewed structure and regenerated task files become visible only as a complete
pair. If writing that pair is interrupted, reopen the workspace, export tasks,
and resubmit a valid structure decision through the same `book submit` command.
Capicove reconstructs the projection from durable decisions without duplicating an
identical decision version. This is review-finalization recovery, not OCR resume.
