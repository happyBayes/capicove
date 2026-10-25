# RFC draft: Bounded qualification for hybrid PDF routing

- Status: Draft; local experiment implemented and qualified; production acceptance pending
- Date: 2026-09-14
- Repository: happyBayes/capicove
- Related work: PR #24; RFC-0011 and RFC-0012
- Proposed implementation owner: document parser plugin maintainers
- Proposed reviewer: @happyBayes
- Decision requested: accept the bounded prototype and its stop rules below
- Authority: `RFC_ACCEPTANCE=PENDING`; `PROTOTYPE_AUTHORITY=USER_AUTHORIZED`;
  `LOCAL_IMPLEMENTATION_AUTHORITY=USER_AUTHORIZED`; `MERGE_AUTHORITY=NONE`

## Current disposition (2026-09-15)

Carl explicitly authorized continued local development and the two local model
acquisitions, and reported verbal reviewer agreement. That is not a recorded
GitHub approval. The experiment now lives under
`plugins/processor-paddleocr/experiments/hybrid`; it uses existing public parse
contracts and a distinct, unregistered experimental profile. The historical
proposal and stop decisions below are retained as context, not current results.
Repeated user instructions to continue superseded the original experiment stop
limit. No merge, production registration, or release is authorized by these tests.

The final verification report records current qualification, limitations, and
exact identities. Earlier conclusions that fraction decomposition was infeasible
were too broad: 4-neighbor components, loss of source resolution, fixed component
counts, and an unsuitable isolated-digit reader were concrete implementation
limitations. Their failed results remain part of the evidence history.

## Historical problem and decision

The local PR #24 qualification candidate passes engineering checks but only two
of six fixed page cases. One successful case extracts Chinese financial facts;
the other correctly rejects an unsupported script. Image-bearing and scan-only
pages are rejected before useful OCR can run. This is not general PDF reading.

Do not extend the blanket image rejection into the next profile. An image can be
a logo, a text-bearing scan, or a figure; its presence alone cannot choose a page
route. Conversely, a nonblank Unicode text layer does not establish that its text
matches the visible page. Script classification also cannot validate numbers.

The next deliverable is one offline prototype that demonstrates routing and
evidence fidelity on the fixed pages before production contracts are changed.
No independent image-script detector is currently qualified by this RFC. Its
availability and effectiveness are explicit feasibility questions, not assumed
solved dependencies.

## MVP and exclusions

The prototype accepts one authorized local PDF page at a time and outputs an
external diagnostic record plus an existing canonical parse result where valid.
The initial OCR language scope remains Chinese, English, and Japanese. Greek and
Russian page cases continue to require the frozen unsupported-script rejection.

No new download, cloud service, model training, whole-book processing, automatic
summary publication, or workspace migration is part of this experiment. Work
uses a new external output directory and the existing pinned local assets.
Existing v2 caches and profile semantics remain unchanged. Any future production
route uses a distinct, explicitly selected profile identity.

## Page routing proposal

Render the authorized page and collect native text spans with geometry. Record
rotation, page dimensions, clipping, and text-bearing regions in the same page
coordinate system. Unicode validity alone is insufficient for native acceptance.

| Observed page | Proposed route | Required evidence before acceptance |
| --- | --- | --- |
| Visible native text with consistent mapping | Preserve native spans | Native spans correspond to visible text, with no unresolved text-bearing region |
| Native text plus a text-bearing image region | Preserve valid spans and OCR that region | Region evidence covers the visible image text; overlap is resolved deterministically |
| Missing or unusable text layer | OCR the rendered page | Supported script evidence, usable text, and coverage of visible text-bearing regions |
| Unsupported dominant script | Explicit unsupported error | Script evidence comes from a usable layer or independently assessed image evidence |
| Disagreement or unknown route | Withhold successful result | Bounded diagnostic naming the unresolved region and reason |

For the prototype, a reviewer labels region visibility, script, and native-text
alignment from rendered pages. These labels are assessment ground truth only;
they must never be supplied to the parser or used as source-specific routing
exceptions. A script detector or visibility check is accepted for automatic use
only after comparison with those independent labels. If no suitable already-local
component exists, the experiment stops with that dependency documented.

Do not silently omit an unrecognized image as decorative. It can be retained as a
non-text image block only when the assessment establishes that no required text
was lost. Charts may contain text without their meaning being captured by OCR;
the MVP makes no chart interpretation claim.

## Evidence and numerical fidelity

Preserve extracted text exactly and keep normalized search text separate. Never
correct a digit, unit, or punctuation mark using a language model or fixture
answer. Retain each selected block's route, page/region coordinates, source hash,
runtime/model identity, and evidence used to accept or withhold it in the external
diagnostic record. Retain disagreements instead of merging them into a plausible
sentence.

Use layout confidence only for geometry and ordering. OCR confidence measures a
provider score, not truth. Two OCR passes agreeing is useful evidence but is not
independent proof: both may repeat the same error. Native/OCR disagreement over
digits or units blocks automatic acceptance for the affected region.

The fixed acceptance cases compare visible numbers directly against independently
recorded facts, including 100 sequences, 110,000 notes, and the disk size. This
tests that sample only; it does not certify all numbers on unseen documents.
If the already-local OCR stack cannot preserve those facts, report the remaining
recognition limitation instead of adding case-specific substitutions.

## Contracts, compatibility, and ownership

Routing, rendering, and OCR adapters belong in the optional parser plugin.
Canonical contracts remain independent of PDF and model SDKs. Book preparation
must continue to consume the public parser contract.

RFC-0011 offers succeeded/failed pages, not a human-review-pending page state.
Use an explicit failed result or exception for unresolved pages in this MVP.
Do not overload low heading confidence to imply text verification. Persisting
per-block route lineage or adding text-review states requires a separate contract
decision and an ADR before production implementation.

The prototype uses external diagnostics to avoid creating an unreviewed workspace
format. The previous schema 2 changes remain an unmerged compatibility proposal;
this RFC does not retroactively approve them. Schema 1 sources can be reparsed
into a fresh workspace; no read operation may rewrite old caches.

## Frozen assessment

Retain the original six sources, hashes, page indices, language hints, expected
outcomes, and twelve anchors from the v2 protocol. Keep that protocol immutable.
A separately versioned protocol is necessary for a new profile identity; its
diff must change only the profile identity and explicitly reviewed receipt fields,
not cases, anchors, numerical alternatives, or error expectations.

Required outcomes are:

| Case | Required outcome |
| --- | --- |
| Z.AI page 28 | Complete; both self-training and technical-vision anchors |
| Laekna page 2 | Complete; all three financial anchors |
| Greek page 1 | Unsupported-script error, not a generic image rejection |
| Russian page 9 | Unsupported-script error |
| Sparse pdf.js page | Exactly the visible pdf.js text |
| Linn image-only page | Complete; all six product/numerical anchors |

Pass requires 6/6 case outcomes and 12/12 anchors, with the existing identity,
closed-output, and input-unchanged checks. These are development regression pages,
not a blind benchmark. Successful scan-only processing cannot be replaced with a
human-review outcome and still count as meeting this gate.

Before integration, freeze a small independent set of six unseen pages: two
native, two mixed, and two scan-only, including dense numbers and unsupported
scripts. Select and label them before running the candidate. Record every omission,
wrong digit, and unsupported-script acceptance; any such critical error blocks an
automatic-reading claim. This small set still cannot establish general accuracy.

## Execution chain and stop rules

1. Feasibility inventory: identify already-local rendering, span geometry, and
   image-script options; inspect their actual outputs on the four blocked page
   cases. Deliver a route/evidence matrix. Stop if a required capability is absent.
2. One prototype: implement only the route chosen from that evidence. Keep
   source identity and geometry regression tests; test empty output, unresolved
   regions, and numerical disagreement. No source-name or source-hash branches.
3. One fixed qualification: freeze candidate, assets, protocol, runner, and
   evaluator, then run all six pages. Limit each page to 300 seconds and the six
   cases to 30 minutes, with no automatic retries.
4. At most one corrective patch for an observed implementation defect, followed
   by one full six-page rerun. If recognition or script feasibility still fails,
   stop and report the precise missing capability and viable alternatives.
5. Only after regression success, run the frozen unseen set once and prepare the
   integration decision. Any new dependency, network service, or model acquisition
   is a scope decision, not an implicit fallback.

This caps fixed qualification at two batches. Broader security audits and unrelated
cleanup are excluded from this work item. Passing engineering tests alone never
substitutes for the page-fidelity gate.

## Draft validation and current disposition

This change is STANDARD: a non-executable RFC draft only. Validate its local links,
protocol mapping, and diff. No production code or qualification run is changed by
this document. Previous test results remain historical evidence for the earlier
candidate and do not certify this proposal.

Correctness review: page routes, numerical fidelity, module ownership, and six
expected outcomes are explicit. Failure-mode review: missing detectors, invisible
text, lost regions, correlated OCR errors, stale caches, and bounded retries have
explicit disposition. Architecture acceptance, prototype feasibility, and hybrid
qualification remain pending. The combined PR #24 candidate remains DO_NOT_COMMIT.

## References

- [Document parser contracts](0011-document-parser-contracts.md)
- [Book task contracts](0012-book-agent-task-contracts.md)
- [Module boundaries](../architecture/module-boundaries.md)
- [Contribution workflow](../../CONTRIBUTING.md)
