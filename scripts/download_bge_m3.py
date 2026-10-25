#!/usr/bin/env python3
"""Provision pinned official BGE-M3 dense weights as verified safetensors."""

import argparse
import json
from pathlib import Path

from capicove.model.bge_m3 import load_download_lock, provision_bge_m3


def main() -> int:
    """Print the pinned plan or explicitly acquire and convert model assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--confirm-download", action="store_true")
    args = parser.parse_args()
    if not args.confirm_download:
        print(json.dumps(load_download_lock(), indent=2))
        return 0
    receipt = provision_bge_m3(args.destination, confirmed=True)
    print(
        json.dumps(
            {
                "root": str(args.destination / "bge-m3"),
                "manifest_sha256": receipt["asset_manifest_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
