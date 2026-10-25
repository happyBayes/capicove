"""Verify explicitly provisioned local assets without acquisition."""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_SUFFIXES = {".json", ".safetensors", ".txt", ".model", ".tiktoken", ".jinja"}
_MAX_JSON_BYTES = 64 * 1024 * 1024


def _walk_error(error: OSError) -> NoReturn:
    """Reject incomplete enumeration without exposing filesystem diagnostics."""
    raise ValueError("unsafe_model_assets") from None


def _relative_file(name: str) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and "\\" not in name
        and "\x00" not in name
        and not PurePosixPath(name).is_absolute()
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


def require_local_model_directory(root: Path) -> Path:
    """Return an explicit existing absolute directory, never a Hub identifier."""
    if not isinstance(root, Path) or not root.is_absolute() or not root.is_dir():
        raise ValueError("model_unavailable")
    if root.resolve() != root or any(part == ".." for part in root.parts):
        raise ValueError("unsafe_asset_path")
    return root


@dataclass(frozen=True, slots=True)
class LocalModelAssets:
    """A closed inventory of local regular files and their immutable hashes.

    The manifest digest is SHA-256 of UTF-8 JSON encoding of the sorted file/hash
    pairs, with ensure_ascii=False and separators=(",", ":"). Roots and all
    descendants must be unlinked local paths. Provision these directories with
    host permissions preventing concurrent writes while a runtime is loading.
    """

    root: Path
    files: tuple[tuple[str, str], ...]
    manifest_sha256: str

    def verify(self) -> None:
        """Verify the complete inventory and reject executable or unsafe assets."""
        root = require_local_model_directory(self.root)
        if (
            not isinstance(self.files, tuple)
            or not 0 < len(self.files) <= 4096
            or not isinstance(self.manifest_sha256, str)
            or _SHA256.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError("invalid_asset_manifest")
        for pair in self.files:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not _relative_file(pair[0])
                or not isinstance(pair[1], str)
                or _SHA256.fullmatch(pair[1]) is None
            ):
                raise ValueError("invalid_asset_manifest")
        if self.files != tuple(sorted(self.files)) or len(dict(self.files)) != len(self.files):
            raise ValueError("invalid_asset_manifest")
        encoded = json.dumps(self.files, ensure_ascii=False, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != self.manifest_sha256:
            raise ValueError("manifest_digest_mismatch")
        try:
            actual: set[str] = set()
            for directory, dirs, files in os.walk(root, followlinks=False, onerror=_walk_error):
                for name in (*dirs, *files):
                    path = Path(directory) / name
                    info = path.lstat()
                    if stat.S_ISDIR(info.st_mode):
                        # os.walk suppresses DirEntry.is_dir errors without onerror;
                        # a directory in its file list would never be traversed.
                        if name not in dirs:
                            raise ValueError("unsafe_model_assets") from None
                        continue
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("unsafe_asset_path")
                    actual.add(path.relative_to(root).as_posix())
                    if len(actual) > 4096:
                        raise ValueError("asset_inventory_mismatch")
            if actual != set(dict(self.files)):
                raise ValueError("asset_inventory_mismatch")
            for name, expected in self.files:
                path = root / name
                if (
                    path.name == "adapter_config.json"
                    or path.suffix not in _SAFE_SUFFIXES
                    or (
                        path.suffix == ".model"
                        and path.name not in {"tokenizer.model", "spiece.model"}
                    )
                ):
                    raise ValueError("unsafe_model_assets")
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as source:
                    before = os.fstat(source.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        raise ValueError("unsafe_asset_path")
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                    if digest != expected:
                        raise ValueError("asset_digest_mismatch")
                    if path.suffix == ".json":
                        if before.st_size > _MAX_JSON_BYTES:
                            raise ValueError("unsafe_model_assets")
                        source.seek(0)
                        data = json.load(source)
                        # Vocabulary keys are arbitrary token text, not loader options.
                        if path.name not in {"tokenizer.json", "vocab.json", "added_tokens.json"}:
                            _verify_json(data, actual)
                    after = os.fstat(source.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        raise ValueError("asset_changed_during_verification")
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            raise ValueError("unsafe_model_assets") from None


def _verify_json(value: Any, files: set[str]) -> None:
    if isinstance(value, list):
        for item in value:
            _verify_json(item, files)
    elif isinstance(value, dict):
        for name, item in value.items():
            if (
                name in {"auto_map", "custom_pipelines", "trust_remote_code", "code_revision"}
                and item
            ):
                raise ValueError("unsafe_model_assets")
            if (name.endswith("_file") or name.endswith("_files")) and item is not None:
                targets = item if isinstance(item, list) else [item]
                if any(not _relative_file(target) or target not in files for target in targets):
                    raise ValueError("unsafe_model_assets")
            if name == "weight_map":
                if not isinstance(item, dict) or not item:
                    raise ValueError("unsafe_model_assets")
                for target in item.values():
                    if (
                        not _relative_file(target)
                        or not target.endswith(".safetensors")
                        or target not in files
                    ):
                        raise ValueError("unsafe_model_assets")
            _verify_json(item, files)
