"""Isolated bounded entry point: request metadata only, never expected answers."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
configuration = json.loads(Path(os.environ["CAPICOVE_HYBRID_CONFIGURATION"]).read_text())
repo = Path(configuration["repository"])
sys.path[:0] = [
    str(ROOT),
    str(repo / "src"),
    str(repo / "plugins/processor-paddleocr/src"),
]
os.environ.update(
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    PADDLE_PDX_CACHE_HOME=configuration["cache_directory"],
)


def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network is forbidden in the local qualification worker")


sys.addaudithook(no_network)
from backend import (  # noqa: E402 - explicit isolated worker bootstrap
    ExperimentalHybridBackend,
    digest,
)

from capicove.contracts import (  # noqa: E402 - explicit isolated worker bootstrap
    DocumentPageRange,
    DocumentParseRequest,
    LocalPathSource,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--hints", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False, parents=True)
    backend = ExperimentalHybridBackend(configuration, output / "artifacts")
    request = DocumentParseRequest(
        LocalPathSource(args.source),
        args.sha256,
        "application/pdf",
        backend.profile_id,
        backend.profile_version,
        tuple(json.loads(args.hints)),
        DocumentPageRange(args.page, args.page + 1),
    )
    started = time.monotonic()
    record = {
        "source_sha256": args.sha256,
        "page_index": args.page,
        "language_hints": list(request.language_hints),
    }
    try:
        value = backend.parse(request)
        record.update(status="completed", result=asdict(value))
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error=str(error))
    finally:
        backend.close()
    record.update(
        elapsed_seconds=time.monotonic() - started,
        source_unchanged=digest(args.source) == args.sha256,
    )
    (output / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "result"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
