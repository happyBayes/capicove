"""Offline immutable chunk-run storage, previews and bounded source readers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import statistics
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from capicove.books.chunk_codec import decode_run, encode_run
from capicove.books.chunk_config import ChunkingConfig, integer_limit, persisted_chunk_config
from capicove.books.chunk_front_matter import source_roles
from capicove.books.chunk_models import ChunkRun, chunk_run_id, digest, render_chunk, spans_text
from capicove.books.chunk_tasks import reference_tasks
from capicove.books.chunk_tokenizer import load_token_counter
from capicove.books.chunking import bounded_context, build_chunk_run, validate_core_coverage
from capicove.books.codec import book_structure_to_json
from capicove.books.models import BookStructure, StructureConfidence
from capicove.books.source_security import source_sha256
from capicove.books.workspace import LocalBookWorkspace
from capicove.contracts import DocumentParseResult


class BookChunkService:
    """Manage independent chunk versions without replacing existing book results."""

    def __init__(
        self, workspace: LocalBookWorkspace, *, tokenizer_root: Path | None = None
    ) -> None:
        self.workspace = workspace
        self.tokenizer_root = tokenizer_root.resolve() if tokenizer_root is not None else None

    def _check_tokenizer(self, config: ChunkingConfig) -> None:
        if (
            self.tokenizer_root is not None
            and config.tokenizer.startswith("local:")
            and not Path(config.tokenizer[6:]).resolve().is_relative_to(self.tokenizer_root)
        ):
            raise ValueError("MCP local tokenizer must be inside the authorized workspace")

    def _directory(self, book_id: str, run_id: str | None = None) -> Path:
        if re.fullmatch(r"book:[0-9a-f]{24}", book_id) is None:
            raise ValueError("invalid book ID")
        root = self.workspace.root
        components = ["books", book_id.replace(":", "-"), "chunk-runs"]
        if run_id is not None:
            if re.fullmatch(r"chunk-run:[0-9a-f]{24}", run_id) is None:
                raise ValueError("invalid chunk run ID")
            components.append(run_id.replace(":", "-"))
        for component in components:
            root /= component
            if root.is_symlink():
                raise ValueError("chunk managed path must not be a symlink")
        return root

    def _evidence(self, book_id: str) -> tuple[DocumentParseResult, BookStructure]:
        self._directory(book_id)
        directory = self.workspace.root / "books" / book_id.replace(":", "-")
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError("book evidence must not use symlinks")
        result = self.workspace.read_result(book_id)
        structure = self.workspace.read_structure(book_id)
        if any(
            section.confidence is not StructureConfidence.HIGH for section in structure.sections
        ):
            raise ValueError("finalize local structure review before chunking")
        source = self.workspace.root / "raw" / f"{result.source_sha256}.pdf"
        if source_sha256(source) != result.source_sha256:
            raise ValueError("source_changed")
        return result, structure

    def create(
        self,
        book_id: str,
        config: ChunkingConfig,
        *,
        dry_run: bool = False,
        compare_run: str | None = None,
    ) -> dict[str, Any]:
        """Preview without writes or persist one complete immutable projection."""
        self._check_tokenizer(config)
        result, structure = self._evidence(book_id)
        baseline = self._load(book_id, compare_run)[0] if compare_run else None
        run = build_chunk_run(result, structure, config)
        directory = self._directory(book_id, run.run_id)
        created = not directory.exists() and not dry_run
        policy = {"max_context_read_tokens": config.max_context_read_tokens}
        if not dry_run:
            payload = encode_run(run)
            if len(payload) > 64 * 1024 * 1024:
                raise ValueError("chunk run exceeds the 64 MiB storage limit")
            if directory.exists():
                if self._read_payload(directory) != payload:
                    raise ValueError("immutable chunk run conflicts with stored content")
            else:
                directory.parent.mkdir(exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=".chunk-", dir=directory.parent))
                try:
                    (staging / "run.json").write_bytes(payload)
                    (staging / "sha256").write_text(
                        hashlib.sha256(payload).hexdigest(), encoding="ascii"
                    )
                    staging.replace(directory)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            policy_path = directory / f"read-policy-{digest(policy)}.json"
            if policy_path.is_symlink():
                raise ValueError("read policy must not be a symlink")
            if not policy_path.exists():
                with policy_path.open("x", encoding="utf-8") as stream:
                    json.dump(policy, stream, sort_keys=True)
        report = preview_report(run)
        report.update(
            {
                "book_id": book_id,
                "created": created,
                "dry_run": dry_run,
                "resolved_config": asdict(config),
                "task_count": len(reference_tasks(run, structure)),
                "comparison": None,
                "task_workflow": "reference_only; not accepted by legacy book submit/publish",
                "read_policy_id": "read-policy:" + digest(policy),
            }
        )
        if baseline is not None:
            report["comparison"] = {
                "baseline_run_id": baseline.run_id,
                "chunk_count_delta": len(run.chunks) - len(baseline.chunks),
                "changed_fields": [
                    key
                    for key, value in json.loads(run.config_json).items()
                    if value != json.loads(baseline.config_json).get(key)
                ],
                "baseline": preview_report(baseline),
            }
        return report

    @staticmethod
    def _read_payload(directory: Path) -> bytes:
        path = directory / "run.json"
        checksum = directory / "sha256"
        if (
            path.is_symlink()
            or checksum.is_symlink()
            or not path.is_file()
            or not checksum.is_file()
            or path.stat().st_size > 64 * 1024 * 1024
        ):
            raise ValueError("invalid chunk run files")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum.read_text(encoding="ascii"):
            raise ValueError("chunk run hash mismatch")
        return raw

    def _load(
        self, book_id: str, run_id: str
    ) -> tuple[ChunkRun, DocumentParseResult, BookStructure]:
        run = decode_run(self._read_payload(self._directory(book_id, run_id)))
        result, structure = self._evidence(book_id)
        if (
            run.run_id != run_id
            or run.source_sha256 != result.source_sha256
            or run.representation_sha256 != result.representation_sha256
            or run.structure_sha256 != digest(book_structure_to_json(structure))
        ):
            raise ValueError("chunk run source or structure identity mismatch")
        blocks = {block.block_id: block for page in result.pages for block in page.blocks}
        config = persisted_chunk_config(json.loads(run.config_json))
        if run.run_id != chunk_run_id(
            run.source_sha256,
            run.representation_sha256,
            run.structure_sha256,
            config,
            run.tokenizer_fingerprint,
        ):
            raise ValueError("chunk run policy identity mismatch")
        page_indices = frozenset(page.page_index for page in result.pages)
        allowed, _ = source_roles(blocks, config.front_matter_policy, page_indices=page_indices)
        if dict(run.exclusions) != allowed:
            raise ValueError("chunk exclusion reasons disagree with source policy")
        validate_core_coverage(
            run.chunks,
            blocks,
            frozenset(allowed),
            policy=config.front_matter_policy,
            page_indices=page_indices,
        )
        for chunk in run.chunks:
            for span in (*chunk.before_spans, *chunk.after_spans):
                if span.block_id not in blocks:
                    raise ValueError("context references unavailable source")
                span.extract(blocks[span.block_id])
        return run, result, structure

    def read_chunks(
        self, book_id: str, run_id: str, *, offset: int = 0, limit: int = 20, tasks: bool = False
    ) -> dict[str, Any]:
        """Page through source-mapped chunks or the separate reference-only task DAG."""
        integer_limit("offset", offset, minimum=0, maximum=1000000)
        integer_limit("limit", limit, minimum=1, maximum=50)
        run, result, structure = self._load(book_id, run_id)
        rows: list[dict[str, Any]] = []
        blocks = {block.block_id: block for page in result.pages for block in page.blocks}
        task_rows = reference_tasks(run, structure) if tasks else ()
        total = len(task_rows) if tasks else len(run.chunks)
        used = 0
        for index in range(offset, min(offset + limit, total)):
            if tasks:
                row = task_rows[index]
            else:
                chunk = run.chunks[index]
                row = asdict(chunk)
                core = spans_text(chunk.core_spans, blocks)
                before = spans_text(chunk.before_spans, blocks)
                after = spans_text(chunk.after_spans, blocks)
                row.update(
                    {
                        "core_text": core,
                        "before_text": before,
                        "after_text": after,
                        "rendered_text": render_chunk(chunk.heading_path, core, before, after),
                        "content_role": chunk.content_role,
                        "extraction_eligible": chunk.content_role == "content",
                    }
                )
            size = len(json.dumps(row, ensure_ascii=False).encode())
            if used + size > 1024 * 1024:
                if not rows:
                    raise ValueError("one chunk exceeds the response byte limit")
                break
            rows.append(row)
            used += size
        next_offset = offset + len(rows)
        return {
            "run_id": run_id,
            "items": rows,
            "total": total,
            "next_offset": next_offset if next_offset < total else None,
            "token_count_mode": run.token_count_mode,
        }

    def read_context(
        self,
        book_id: str,
        run_id: str,
        chunk_id: str,
        *,
        max_tokens: int | None = None,
        read_policy_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a bounded original-parent view anchored at the selected chunk."""
        if max_tokens is not None:
            integer_limit("max_context_read_tokens", max_tokens, minimum=1, maximum=32768)
        run, result, _ = self._load(book_id, run_id)
        policy_id, policy_limit = self._read_policy(
            self._directory(book_id, run_id), read_policy_id
        )
        max_tokens = min(max_tokens, policy_limit) if max_tokens is not None else policy_limit
        config = persisted_chunk_config(json.loads(run.config_json))
        self._check_tokenizer(config)
        counter = load_token_counter(config)
        if counter.fingerprint != run.tokenizer_fingerprint:
            raise ValueError("local tokenizer asset changed since chunk creation")
        indices = [index for index, chunk in enumerate(run.chunks) if chunk.chunk_id == chunk_id]
        if len(indices) != 1:
            raise ValueError("unknown chunk ID")
        index = indices[0]
        chunk = run.chunks[index]
        blocks = {block.block_id: block for page in result.pages for block in page.blocks}
        core = bounded_context(chunk.core_spans, max_tokens, blocks, counter)
        chosen = core
        # Expand locally in source order; never load an unbounded parent into the response.
        candidates = [
            (abs(position - index), position, other)
            for position, other in enumerate(run.chunks)
            if other.parent_id == chunk.parent_id and position != index
        ]
        for _, position, other in sorted(candidates) if core == chunk.core_spans else ():
            trial = (
                (*other.core_spans, *chosen) if position < index else (*chosen, *other.core_spans)
            )
            if (
                len(trial) > 128
                or len(spans_text(trial, blocks).encode()) > 131072
                or counter.count(spans_text(trial, blocks)) > max_tokens
            ):
                break
            chosen = trial
        text = spans_text(chosen, blocks)
        return {
            "run_id": run_id,
            "chunk_id": chunk_id,
            "parent_id": chunk.parent_id,
            "content_role": chunk.content_role,
            "extraction_eligible": chunk.content_role == "content",
            "heading_path": chunk.heading_path,
            "text": text,
            "spans": [asdict(span) for span in chosen],
            "tokens": counter.count(text),
            "max_tokens": max_tokens,
            "read_policy_id": policy_id,
            "policy_max_tokens": policy_limit,
            "core_complete": core == chunk.core_spans,
            "token_count_mode": counter.mode,
            "tokenizer_fingerprint": counter.fingerprint,
        }

    @staticmethod
    def _read_policy(directory: Path, policy_id: str | None) -> tuple[str, int]:
        if policy_id is not None and re.fullmatch(r"read-policy:[0-9a-f]{64}", policy_id) is None:
            raise ValueError("invalid read policy ID")
        paths = (
            [directory / (policy_id.replace(":", "-") + ".json")]
            if policy_id
            else list(directory.glob("read-policy-*.json"))
        )
        if not paths or len(paths) > 1000:
            raise ValueError("missing or excessive read policies")
        policies: list[tuple[str, int]] = []
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024:
                raise ValueError("invalid read policy file")
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(value, dict)
                or set(value) != {"max_context_read_tokens"}
                or path.name != f"read-policy-{digest(value)}.json"
            ):
                raise ValueError("read policy identity mismatch")
            limit = integer_limit(
                "max_context_read_tokens",
                value["max_context_read_tokens"],
                minimum=1,
                maximum=32768,
            )
            policies.append(("read-policy:" + digest(value), limit))
        # No mutable latest pointer: ambiguous reads use the most restrictive policy.
        return min(policies, key=lambda policy: policy[1])


def preview_report(run: ChunkRun) -> dict[str, Any]:
    def distribution(values: list[int]) -> dict[str, int | float]:
        ordered = sorted(values)
        if not ordered:
            return {"min": 0, "median": 0, "p95": 0, "max": 0}
        return {
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p95": ordered[min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1)],
            "max": ordered[-1],
        }

    warnings = Counter(warning for chunk in run.chunks for warning in chunk.warnings)
    core = sum(chunk.core_tokens for chunk in run.chunks)
    rendered = sum(chunk.rendered_tokens for chunk in run.chunks)
    return {
        "run_id": run.run_id,
        "chunk_count": len(run.chunks),
        "core_tokens": distribution([chunk.core_tokens for chunk in run.chunks]),
        "rendered_tokens": distribution([chunk.rendered_tokens for chunk in run.chunks]),
        "core_token_sum": core,
        "rendered_token_sum": rendered,
        "added_input_fraction": max(0, rendered - core) / rendered if rendered else 0.0,
        "warning_chunk_counts": dict(warnings),
        "excluded_block_count": len(run.exclusions),
        "exclusion_reason_counts": dict(Counter(reason for _, reason in run.exclusions)),
        "navigation_chunk_count": sum(chunk.content_role == "navigation" for chunk in run.chunks),
        "extraction_chunk_count": sum(chunk.content_role == "content" for chunk in run.chunks),
        "coverage_valid": run.coverage_valid,
        "tokenizer_fingerprint": run.tokenizer_fingerprint,
        "token_count_mode": run.token_count_mode,
    }
