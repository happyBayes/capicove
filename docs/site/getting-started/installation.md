# Installation

Capicove requires Python 3.11 or later. Version 0.1.0 is prepared locally;
publication to the package index is a separate maintainer action. The commands
using `==0.1.0` apply after that version is available in your approved index.

## Install a release

```bash
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
. .venv/bin/activate
python -m pip install "capicove[wiki,mcp,api]==0.1.0"
capicove --version
capicove doctor
```

For only contracts, image primitives and dependency-light processing, install
`capicove==0.1.0`. No model downloads or external APIs are enabled by installation.
For an unpublished candidate, install the reviewed local wheel instead:

```bash
python -m pip install "/path/to/capicove-0.1.0-py3-none-any.whl[wiki,mcp,api]"
```

Use [Quickstart](quickstart.md) for a local task, and
[optional dependencies](optional-dependencies.md) to choose execution paths.

## Runtime requirements

| Capability | Additional requirements |
| --- | --- |
| Local Wiki, HTTP API, MCP | Corresponding `wiki`, `api`, `mcp` extras |
| Temporal Memory | Python linked to SQLite 3.42 or later with FTS5; private local storage as described in [Temporal Memory](../guides/temporal-memory.md) |
| Model-backed image, video and graph operations | Compatible optional runtime; verified local model files; explicit model selection |
| Video and audio command execution | FFmpeg/ffprobe with the protocols required by the selected adapter; Whisper.cpp and VAD models for transcription |
| Audio clipping | FFmpeg/ffprobe with working seekable `fd:` protocols; verify with `capicove audio capabilities --component clip --output json` |
| Spark | Matching `capicove-execution-spark` plugin, PySpark 4.1–4.2, Java 17 |
| PaddleOCR book parsing | Matching `capicove-processor-paddleocr` plugin, supported PaddlePaddle runtime and verified `book-balanced` model receipt |
| Elasticsearch/OpenSearch | Matching retriever plugin and an explicitly configured compatible service |
| FAISS | Matching `capicove-retriever-faiss` plugin and native FAISS dependencies |
| Remote graph generation | `graph-remote` extra, verified local tokenizer, compatible OpenAI-style chat endpoint, explicit credential environment reference and network opt-in |

Installing every extra is not a universal deployment profile: CPU/GPU libraries,
CUDA and system libraries must agree. Use a fresh environment for the selected
profile. GPU qualification on one host does not establish portability to another
GPU driver or operating system. `capicove doctor` reports discovery; exercising the
selected operation is the runtime acceptance test.

## Development checkout

```bash
git clone https://github.com/happyBayes/capicove.git
cd capicove
uv sync --locked --group dev
uv run --no-sync capicove doctor
```

Install uv using your approved installation method. Dependency installation needs
network access or a populated cache. Until the repository is public, cloning needs
repository access. Preserve local changes before updating an existing checkout.
Open the documentation with `make docs-serve` at `http://127.0.0.1:8000/`.

See [Releasing](../development/releasing.md) for isolated install validation and
release gates. Model files and optional services are configured explicitly; the
package does not bundle private assets or activate network fallbacks.

To use the locally prepared 0.1.0 archives before they appear in an approved
package index, follow the hash-verified offline workflow in
[Deployment and delivery](../guides/deployment.md). The core wheel alone does not
include every plugin or optional dependency profile.
