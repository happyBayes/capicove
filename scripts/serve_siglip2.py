#!/usr/bin/env python3
"""Run the optional local SigLIP2 retrieval API."""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse deployment configuration, load the model, and serve HTTP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise SystemExit("install capicove[model-serving] before serving SigLIP2") from error

    from capicove.api.siglip2 import build_service, create_app

    service = build_service(
        args.model_directory,
        expected_model_manifest_sha256=args.model_manifest_sha256,
        device=args.device,
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
