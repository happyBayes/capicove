# RFC-0030: Unified model HTTP serving and registered graph generation

- Status: Accepted for operator-requested implementation
- Date: 2026-10-21
- Authorization: The operator approved the proposed architecture and requested
  implementation, tests, and adversarial acceptance.
- Decision: [ADR-0034](../adr/0034-unified-model-serving.md).

Add the `audio_transcription` capability and `whisper-cpp` provider. Local model
profiles pin the Whisper and Silero file inventory with SHA-256, explicit native
executable paths, CPU execution, and a decoded-duration limit. Runtime never
acquires models. HTTP accepts multipart `file`, `model`, optional `language`, and
`response_format=json|text|verbose_json` at `/v1/audio/transcriptions`. Uploads,
decoded PCM duration, inference time and concurrency are bounded. Temporary
uploads are removed after native execution, including cancelled requests.

Use `models list --config` and `models serve --config --model NAME|--all` for all
registered capabilities. Only explicitly bound profiles may be served. Selecting
more profiles than `max_instances` fails before construction. The listener owns
selected instances until shutdown; each model has independent admission. Existing
capability applications retain their validation and route formats. `/v1/models`
advertises names, capabilities, identities and paths. Raw PDF/image routes use a
`model` query parameter when multiple instances share their capability. Incoming
credentials are environment references; non-loopback CLI listeners require them.

Add generation configuration schema 2 with `models` (absolute configuration
path), `generation_model` (bound profile name), and `context_limit`. Knowledge
build and answer paths use the registered VLM through the existing structured
adapter interface. Exact upstream token counting and a declared revision are
required. Graph validation, source evidence, receipts and host network authority
remain in their existing owners. Model/configuration changes alter receipt
identity. Schema 1 local-tokenizer configurations remain supported explicitly;
there is no silent model migration or approximate token-count fallback.

Remote-counting adapters additionally implement the optional
`ControlledRequestTokenizer` port. The runner passes its remaining call/build
budget and reauthorizes evidence before each preflight, including retries. Local
legacy tokenizers retain their existing interface. Token-counting traffic has the
same evidence authority as generation traffic. Remote counting runs after receipt
claiming and inside the provider admission slot. Counting and generation share a
call deadline and attempt budget: a counting timeout or rate limit consumes an
attempt and uses bounded backoff; other counting failures produce a failed
receipt. Cancellation and build deadline expiry produce terminal receipts.

VLM requests and model lookup also accept an unambiguous upstream model name.
Registered profile names take precedence; shared upstream names require an
explicit profile name. A completed local provider construction is disposed even
when startup is cancelled before ownership transfers to the listener.
