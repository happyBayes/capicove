# `capicove http serve`

Serve the Tools selected by a version-2 workspace configuration over HTTP on
`127.0.0.1`, behind a per-start bearer token
([RFC-0020](https://github.com/happyBayes/capicove/blob/main/docs/rfcs/0020-workspace-loopback-http.md)).

```text
capicove http serve --config PATH [--port PORT] [--token-file PATH]
```

Install `capicove[api]` first. `--port 0` (the default) picks a free port. After
startup the server prints one JSON line on stderr:

```json
{"event": "ready", "address": "127.0.0.1", "port": 49152, "token_file": "/home/me/cove/http.token"}
```

Read the token from that file (mode `0600`) and send it as
`Authorization: Bearer <token>`. The token is never printed.

| Route | Purpose |
| --- | --- |
| `GET /v1/health` | `{"status": "ok"}`; no token needed |
| `GET /v1/tools` | The same catalog as `capicove mcp catalog` |
| `POST /v1/tools/{tool_id}` | Invoke one selected Tool with its canonical arguments |
| `POST /v1/search`, `POST /v1/evidence` | Aliases, present only when those Tools are selected |

```bash
token=$(cat /home/me/cove/http.token)
curl -s -H "Authorization: Bearer $token" -H "Content-Type: application/json" \
  -d '{"query": "Paris"}' http://127.0.0.1:49152/v1/search
```

Completed invocations return `200` with the canonical result envelope, even when
the Tool reports a failure. Other statuses: `400` invalid JSON or arguments,
`401` missing or wrong token, `404` unknown or unselected Tool, `413` body over
2 MiB, `429` another invocation is running, `503` shutting down, `504` the Tool
missed its deadline.

The server always binds loopback, even when the host config enables network
access. One server runs per workspace; a second start exits with status `2`
and leaves the first server's token alone. `--token-file` must stay inside
`workspace_root`. On SIGTERM or Ctrl-C the server stops admitting calls, stops
in-flight work including FFmpeg or Whisper child processes, removes its token,
and exits within 10 seconds. Workers always use the configuration read at
startup; edit the file and restart the server to change authority. HTTP serve is
Linux-only in this release.
