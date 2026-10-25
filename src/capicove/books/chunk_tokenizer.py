"""Offline counters with explicit estimation and local tokenizer provenance."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Protocol

from capicove.books.chunk_config import ChunkingConfig


class TokenCounter(Protocol):
    """Count rendered input without choosing or calling a generation model."""

    fingerprint: str
    mode: str

    def count(self, text: str) -> int:
        """Return the configured counter's input size."""
        ...


class Utf8ByteCounter:
    """Provide a deterministic byte estimate, not a model-token guarantee."""

    fingerprint = "utf8-bytes-v1"
    mode = "estimated"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


def load_token_counter(config: ChunkingConfig) -> TokenCounter:
    """Load only an explicitly authorized local tokenizer; never fetch assets."""
    if config.tokenizer == "utf8-bytes-v1":
        return Utf8ByteCounter()
    path = Path(config.tokenizer[6:])
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("local tokenizer must be a regular file at most 64 MiB")
    raw = path.read_bytes()
    try:
        runtime = importlib.import_module("tokenizers")
    except ImportError as error:
        raise ValueError("local tokenizer requires the optional tokenizers package") from error
    tokenizer = runtime.Tokenizer.from_str(raw.decode("utf-8"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    identity = f"tokenizers:{runtime.__version__}:{hashlib.sha256(raw).hexdigest()}"

    class LocalCounter:
        fingerprint = identity
        mode = "exact"

        def count(self, text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return LocalCounter()
