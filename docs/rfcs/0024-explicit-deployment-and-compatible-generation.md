# RFC-0024: Explicit deployment and compatible generation

- Status: Accepted for operator-requested implementation
- Date: 2026-10-19
- Authorization: The project operator requested completion of unqualified runtime
  paths, use of their self-hosted OpenAI-compatible LLM without a model binding,
  and the deployment/delivery workflow. This records implementation scope, not
  independent maintainer review, merge approval, or permission to publish.
- Decision: [ADR-0028](../adr/0028-explicit-deployment-and-compatible-generation.md).

## Scope

Reuse existing capability ports, model inventories, release wheels, private
workspace initialization and inference receipts. Avoid a universal inference API
or a second model registry. Local model factory configuration version 1 remains
offline; remote generation has a separate, closed operator configuration.

`capicove.graph_build.remote_generation` implements the existing
`StructuredGenerator`, `RequestTokenizer` and text-tokenizer interfaces. An
operator explicitly chooses an HTTPS endpoint (loopback HTTP is allowed), model
identifier, model revision assertion, locally verified tokenizer inventory,
context limit, credential environment reference and optional chat-template
settings. Configuration requires `network_enabled: true`. No provider/model
fallback, implicit download, inherited proxy, redirect or model-requested tools
are allowed. Hosts remain responsible for evidence-egress authorization.

The provider identity binds endpoint, selected model/revision, tokenizer digest,
template settings and wire serialization. Complete input tokens are counted with
the verified local tokenizer and compared with server usage; mismatches fail
closed. Output is bounded and passes existing graph validators. Server model
revisions are operator attestations: the compatible API cannot prove the identity
of remote weight bytes. Credentials are read at invocation and never serialized.

Transport deadlines include slow headers and responses. Cancellation rejects
results immediately. At most two transport workers remain in flight; a cancelled
worker retains its slot until its bounded socket operation finishes. DNS/native
loading cannot be forcibly interrupted. Configuration parsing never sends data.

## Delivery

Delivery uses selected package profiles and a closed wheel inventory with hashes.
Installation occurs in a dedicated environment, with an explicit network step to
prepare dependencies and an offline installation path for the resulting bundle.
Runtime assets and secrets stay outside package archives. Existing model receipts
and native configuration formats remain authoritative. A deployment report must
distinguish installed dependencies, verified assets, configured services and
successful real inference; imports alone never establish runtime readiness.

Missing weights, credentials or platform evidence are reported as unresolved.
No automatic Git operation or publication is part of deployment. Model acquisition
still requires the repository's explicit source/transfer authorization.

## Validation

Cover authentication, redirect refusal, malformed responses, exact token budgets,
server identity/template drift, cancellation and replay authorization. Qualify the
real user-selected server using authored evidence only. Repeat clean installation,
quality, regression, build and dependency checks against the final artifacts.
