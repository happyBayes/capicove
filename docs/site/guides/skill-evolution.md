# Supervised Skill evolution

Capicove can turn observable task feedback into an immutable Skill candidate, compare
it with the active version, obtain an explicit local operator decision, and use the
approved version in subsequent recorded tasks. New feedback can produce the next
version. It never trains model weights or grants permissions from generated text.

## What is connected

The `capicove evolution` CLI and `capicove.runtime.evolution_loop.SkillLearningLoop`
connect trace snapshots, learning, versioned content, paired evaluation, review,
activation, actual host consumption, feedback and supervised rollback. The default
extractor learns from structured corrections. An explicit compatible-model adapter
can infer lessons from bounded, filtered visible observations. Synthesis creates or
patches declarative `SKILL.md` content; weak, stale or non-skill evidence yields
`no_change`. Runtime integration supplies exact approved bytes to a host callback;
it does not silently change `kb ask`, install skills into another agent, or expand
that host's tool permissions.

The first implementation supports supervised activation. It does not implement
semantic skill merging, time-based autonomous archival, or automatic canary promotion.
Existing automatic promotion enum values do not enable those features.

## Run the local closed loop

Evolution storage currently requires POSIX ownership and permission checks (Linux
or macOS). Windows is rejected before creating the registry directory; Windows ACL
support is not implemented.

Use dedicated private directories outside the source repository. All commands emit
JSON. Replace the IDs below with IDs returned by the preceding command.

```console
capicove evolution init --root /absolute/private/evolution
capicove evolution run "hello world" --operation uppercase \
  --episode-id example-1 --store /absolute/private/traces/episodes.sqlite3 \
  --root /absolute/private/evolution
```

No skill is selected, so the fixture host echoes the text and records failure against
its uppercase oracle. Add an explicit reusable correction:

```console
capicove evolution feedback example-1 --feedback-id correction-1 \
  --task-signature text-formatting --lesson "Convert text to uppercase." \
  --applicability "When formatting text" --exclusion "Unrelated tasks" \
  --attribution skill_gap --confidence 1 \
  --store /absolute/private/traces/episodes.sqlite3 --root /absolute/private/evolution
capicove evolution distill --store /absolute/private/traces/episodes.sqlite3 \
  --root /absolute/private/evolution
capicove evolution generate BATCH_ID --name text-formatting --root /absolute/private/evolution
```

An explicit confident correction can support a proposal. Ordinary inferred lessons
require agreement across at least three distinct Episodes. Corrections classified
as `agent_misuse`, `environment_failure`, or `uncertain` do not rewrite instructions.
Existing-skill patches require evidence attributed to that exact active version.

Create an independently authored campaign JSON file:

```json
{
  "replay": [
    {"episode_id":"replay-1","input_text":"alpha beta","expected_output":"ALPHA BETA","operation":"uppercase"},
    {"episode_id":"replay-2","input_text":"gamma delta","expected_output":"GAMMA DELTA","operation":"uppercase"}
  ],
  "held_out": [
    {"episode_id":"held-1","input_text":"epsilon zeta","expected_output":"EPSILON ZETA","operation":"uppercase"},
    {"episode_id":"held-2","input_text":"theta lambda","expected_output":"THETA LAMBDA","operation":"uppercase"}
  ]
}
```

Evaluation IDs must be unique and disjoint from candidate synthesis sources. Keep
held-out goals and expected outputs out of synthesis input. This CLI accepts explicit
curated splits; it does not derive a random split or prove that an operator never
looked at a held-out case. Reports bind the campaign digest, content hash, baseline,
evaluator identity and each paired result.

```console
capicove evolution evaluate CANDIDATE_ID --campaign /absolute/private/campaign.json \
  --root /absolute/private/evolution
capicove evolution review show CANDIDATE_ID --root /absolute/private/evolution
capicove evolution review approve CANDIDATE_ID --reviewer local-owner \
  --reason "Reviewed evidence and paired outcomes" \
  --expect-state awaiting_approval --expect-active none --root /absolute/private/evolution
capicove evolution run "another example" --operation uppercase --skill text-formatting \
  --episode-id example-2 --store /absolute/private/traces/episodes.sqlite3 \
  --root /absolute/private/evolution
```

The final run records the actual candidate/version/hash and uses its instructions.
Append feedback for `example-2`, then repeat distill/generate to produce a child
version. Repeated unchanged batch writes and candidate creation are idempotent.
A candidate is never an active fallback. Explicitly selecting a missing or unapproved
skill fails before host execution.

For a later version, `--expect-active` must name the exact baseline candidate.
Concurrent approval, stale state, changed content and absent/passing-invalid reports
fail closed. Rejection and rollback are explicit:

```console
capicove evolution review reject CANDIDATE_ID --reviewer local-owner \
  --reason "Insufficient generality" --expect-active ACTIVE_ID --root /absolute/private/evolution
capicove evolution rollback text-formatting --to PREVIOUSLY_APPROVED_ID \
  --expect-active ACTIVE_ID --reviewer local-owner --reason "Observed regression" \
  --root /absolute/private/evolution
capicove evolution status --root /absolute/private/evolution
```

Rollback switches the active pointer to a retained previously approved version of the
same skill. It does not erase candidates, reports, reviews or trace evidence.

## Evaluation meaning

Starting a valid evaluation revokes any prior approval readiness before executing
tasks. An exception or process interruption leaves the candidate unapprovable;
retry evaluation to obtain a fresh passing report. Completed pairs are retained
on caught execution failures, without persisting raw provider error messages.
If evaluations overlap, only the latest attempt may finalize its result. An older
attempt fails with `evaluation attempt was superseded` rather than replacing a
newer failed or incomplete report.

The built-in fixture executor recognizes `uppercase`, `lowercase`, `strip`,
`sort-lines`, and `echo`. It executes both instruction sets against an independent
expected output. It qualifies only those text transformations, not arbitrary agent
quality. A new skill must produce at least one observed improvement. Existing
perfect-baseline confirmation may pass; otherwise the mean score must improve by
at least 0.02 in each split. Critical failures, leakage flags, increasing errors,
policy regressions and excessive major regressions prevent approval.

At least two replay and two held-out cases are required. The implementation uses
curated campaigns and bounded exact-text leakage detection; it does not claim
statistical significance or semantic leakage detection. Scores cannot independently
authorize activation. Programmatic report registration is a trusted local host API,
not a remote authentication boundary; the CLI executes evaluators and has no command
to import a claimed passing verdict.

## Explicit model-backed execution

`distill`, `evaluate`, and `run` accept `--generation-profile PROFILE.json` and
`--allow-network`. Both this flag and the existing compatible generation profile's
`network_enabled: true` are required. Use the existing verified local tokenizer
manifest and an operator-selected endpoint/model revision as described in
[deployment](deployment.md). Credentials remain in the profile's named environment
variable, never in Skill documents or command arguments. No model is downloaded.

- `distill` uses `CompatibleExperienceExtractor` with sanitized bounded visible
  examples, not raw tool arguments or storage access. Filtering is conservative;
  it cannot prove that arbitrary natural-language text contains no private detail.
- `evaluate` uses `CompatibleSkillExecutor`; expected answers stay in the local
  verifier and are never supplied to the executor. Use operation `text` for general
  text tasks. Exact-output scoring is intentionally strict; hosts needing another
  verifier implement the `ReplayEnvironment` port.
- `run` requires `--expected-output` for model-backed tasks so completion is not
  confused with task success. Host integrations may provide their own verifier.

Only text is generated. Generated commands and scripts are never executed. Provider
identity binds endpoint, configured model/revision, verified tokenizer and settings;
a remote server's actual weights remain an operator assertion. The repository tests
use injected transport responses and do not qualify a real remote deployment.

## Opt-in background integration

```python
from pathlib import Path

from capicove.runtime.evolution_loop import SkillLearningLoop
from capicove.traces import EpisodeFeedback

with SkillLearningLoop(
    Path("/absolute/private/evolution"),
    Path("/absolute/private/traces/episodes.sqlite3"),
    enabled=True,
    skill_names={"text-formatting": "text-formatting"},
) as loop:
    # executor receives goal and the exact immutable approved SkillVersion tuple.
    run = loop.run(
        names=("text-formatting",),
        goal="Task input",
        episode_id="host-run-1",
        executor=host_executor,
    )
    loop.feedback(
        run,
        "correction-1",
        EpisodeFeedback(
            "correction",
            {
                "task_signature": "text-formatting",
                "lesson": "Preserve paragraph boundaries.",
                "applicability": "When formatting text",
                "exclusion": "Other tasks",
                "attribution": "skill_gap",
                "confidence": 1.0,
            },
            "user",
        ),
    )
    loop.wait(timeout=5)
```

`host_executor` is your existing explicitly authorized execution callback returning
`ExecutionResult(output, succeeded)`. The loop is disabled unless explicitly enabled.
It opens a separate registry connection per background worker and only creates drafts.
Before a new live task it requests cooperative cancellation; waiting is bounded.
Python threads cannot forcibly terminate an uncooperative provider. Close requests
cancellation, so a short-lived host should wait for desired review work before closing.

## Persistence and limits

Registry content and evidence are local private SQLite data, with schema checks,
immutable content and append-only history. Approval and pointer changes share one
transaction. Unknown schema, unsafe filesystem links, content hash mismatch and
conflicting retry keys are rejected. Raw Episode records remain owned by traces.
The local operator controls the filesystem: reviewer labels are audit identities,
not multi-user ACL authentication.

Distillation rejects confidential/restricted or broken Episodes, unresolved skill
versions and oversized input. Defaults cap one scan at 10,000 records and 1,000
Episodes; provider batches are bounded further. Evidence records include source
hashes, extraction policy and extractor identity. Narrow queries or separate stores
are required when a scan exceeds those limits. No source content is uploaded without
the explicit model opt-ins described above.
