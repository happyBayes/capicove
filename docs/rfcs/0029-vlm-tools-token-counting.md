# RFC-0029: VLM tool messages and explicit token counting

- Status: Accepted for operator-requested implementation
- Date: 2026-10-21
- Authorization: The operator requested the complete VLM provider improvements
  after reviewing the transport, deployment, token-counting and tool-use gaps.
- Decision: [ADR-0033](../adr/0033-vlm-tools-token-counting.md).

Extend the existing VLM port with immutable function definitions, assistant tool
calls, tool-result messages, named/automatic/required selection, reasoning replay,
and explicit upstream token counting. Preserve the single Chat Completions route
for text, images and video. Capicove does not execute model-proposed functions.
The host owns execution, schema validation and authorization. Conversation
validation rejects missing, duplicate and unmatched call/result IDs.

HTTP profiles optionally declare `token_counting = "vllm"`; the provider sends
chat messages, tools and template settings to the same origin's `/tokenize`.
Expose the count through `POST /v1/count_tokens`, CLI `count-tokens`, and
`VlmProvider.count_tokens`. This is a Capicove extension, not an OpenAI standard
endpoint. Missing support returns 501 without text-only estimation. Count results
record input count, context limit, profile identity and exact request hash.

Replace custom socket/DNS/TLS code with lazy aiohttp and c-ares dependencies in
`capicove[vlm]`. Preserve total deadlines, response byte limits, certificate checks,
explicit network opt-in, no redirects, no automatic retries and no inherited
proxies. The synchronous port remains worker-owned in HTTP composition.

Provide explicit pinned acquisition, hash receipts and offline vLLM startup as
operator tooling. Keep weights and credentials outside the source repository.
Model-specific parser settings belong to deployment recipes. Streaming and
multipart uploads remain outside this change; JSON supports embedded media.
