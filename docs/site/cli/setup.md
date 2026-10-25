# Workspace setup and runtime checks

## Initialize a workspace

```bash
capicove setup init /absolute/private/cove \
  --source-root notes=/absolute/authorized/notes
capicove setup doctor --config /absolute/private/cove/host.json
```

`setup init` creates a version-2 host configuration with only the explicitly
selected local capabilities. It does not read source files, build an index,
download a model or make a network request. `setup doctor` checks the configured
workspace and selected Tools locally; it does not probe remote model services.
See [offline workspace search](../guides/offline-retrieval.md) and
[workspace media](../guides/workspace-media.md) to configure those capabilities.

## Check a compatible chat service

Install the graph-generation dependencies if they are not already present:

```bash
python -m pip install 'capicove[graph-remote]'
```

Create a secret-free profile as described in
[deployment and remote generation](../guides/deployment.md). Set its referenced
credential in the process environment, then explicitly call:

```bash
capicove setup check-llm --config /absolute/private/generation.json
```

This sends an authored readiness request to the configured OpenAI-compatible
`/chat/completions` endpoint. It verifies the served model identifier, response
shape and local/server input-token accounting. The configured model revision is
operator-attested; the chat API does not prove the remote weight bytes. This
command does not submit a document, build a graph, configure a workspace Tool,
or grant general network/evidence access. Normal `setup init`, `doctor` and MCP
startup remain offline.

## Read the result

On success the command prints JSON containing `ready`, model and provider identity,
tokenizer digest, token counts and elapsed time. On failure it exits nonzero with
a bounded diagnostic. Missing credentials, rejected authentication, model/template
drift and service timeouts fail closed. Review the selected host's network policy
and permission gate before composing this provider into an application.
