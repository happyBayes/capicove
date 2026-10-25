# Quickstart: inspect a local image

**Goal:** complete one real image filter in about five minutes after installation.
This task uses four synthetic pixels, base dependencies, and no models or network
calls. The input is never modified.

## Prepare the environment

Follow [Installation](installation.md). Open a terminal in the repository root
(the directory containing `pyproject.toml`):

```bash
uv sync --locked --group dev
uv run --no-sync capicove doctor
```

Installation may download dependencies. Subsequent commands use `--no-sync` to
avoid implicit dependency installation.

## Run the filter

The repository includes a text-format image at `docs/site/assets/quickstart.ppm`.
Without a checkout, download the [synthetic image](../assets/quickstart.ppm), use
its local path and replace `uv run --no-sync capicove` with `capicove`.

<!-- docs-smoke: image -->
```bash
uv run --no-sync capicove image run \
  --tool image_aspect_ratio_filter \
  --input docs/site/assets/quickstart.ppm \
  --output json
```

## Verify the result

The command exits successfully and prints JSON describing the filter result.
Check these fields (other metadata is omitted here):

```json
{
  "status": "succeeded",
  "error": null,
  "result": {
    "keep": true,
    "metrics": [{"name": "aspect_ratio", "values": [1.0]}]
  }
}
```

The square image has width and height of 2 pixels and an aspect ratio of 1, so it
passes the default bounds. Source metadata includes a hash; absolute paths may
differ across machines. This is a shape check, not OCR or semantic understanding.

The operation reads the fixture and prints to the terminal. It creates no image
or persistent database, so there are no generated artifacts to clean up.

## Recover from an error

| Symptom | Next action |
| --- | --- |
| `uv` unavailable | Complete the prerequisite on the Installation page. |
| `capicove` unavailable | Prepare the environment above from the repository root. |
| Input cannot be read | Check the working directory and fixture path; use an absolute path for a downloaded fixture. |
| Another Tool needs a provider | Use the exact Tool above; check the [Image CLI](../cli/image.md) for default support. |

Do not broaden permissions or download a model to repair this example. For
[support](../reference/support.md), include the version, doctor output, and
sanitized error.

## Give this task to an Agent

> From the Capicove repository root, use the installed environment to run the
> Quickstart image-aspect-ratio command above against only
> `docs/site/assets/quickstart.ppm`. Do not modify inputs or other repository files.
> Do not install packages, download models, or contact external services without
> asking me. Show the exit status and relevant JSON result, explain whether the
> square image passes, and report warnings. If an input or dependency is missing,
> stop and name it; do not guess replacement configuration.

## Next task

Choose a workflow in the [capability matrix](capabilities.md), or continue to the
[Image CLI](../cli/image.md) and [Image API](../api/image.md).
