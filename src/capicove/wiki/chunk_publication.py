"""Recoverable Wiki-v2 publication of verified Book chunk-result claims."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

from capicove.books import AtomicClaim
from capicove.documents import document_parse_result_to_json
from capicove.wiki._adapters.sqlite_index import check_index, rebuild_index
from capicove.wiki.citations import validate_book_citations
from capicove.wiki.filesystem import (
    WikiIndexUnavailableError,
    WikiUnsafePathError,
    WikiValidationError,
    atomic_write,
    jsonl_records,
    jsonl_records_from_bytes,
    load_yaml_mapping,
    scan,
    wiki_write_lock,
)
from capicove.wiki.indexing import WikiDocument, load_documents, render_index
from capicove.wiki.models import ChunkPublicationPacket, ClaimCitation
from capicove.wiki.validation import lint


@dataclass(frozen=True, slots=True)
class _Prepared:
    source_bytes: bytes
    representation_bytes: bytes
    page_bytes: dict[str, bytes]
    citations: tuple[dict[str, Any], ...]
    claim_ids: tuple[str, ...]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _plain(value: str) -> str:
    normalized = html.escape(" ".join(value.split()), quote=False)
    return normalized.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _source_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _publication_date(value: object) -> date:
    """Decode persisted publication dates without exposing malformed content."""
    if type(value) is not str:
        raise ValueError("publication_conflict")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError("publication_conflict") from None


def _read_source_bytes(path: Path) -> bytes:
    """Read one regular PDF through a pinned, no-follow POSIX file handle."""
    try:
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        for parent in path.parents:
            parent_info = parent.lstat()
            if stat.S_ISLNK(parent_info.st_mode) or (
                getattr(parent_info, "st_file_attributes", 0) & reparse_flag
            ):
                raise ValueError("source_changed")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source_changed")
        if os.name == "posix":
            if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
                raise ValueError("source_changed")
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY
            directory_fd = os.open(path.anchor, directory_flags)
            try:
                for component in path.parts[1:-1]:
                    next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                    os.close(directory_fd)
                    directory_fd = next_fd
                file_fd = os.open(path.name, file_flags, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
            with os.fdopen(file_fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or _source_identity(opened) != _source_identity(
                    before
                ):
                    raise ValueError("source_changed")
                content = stream.read()
                after = os.fstat(stream.fileno())
        else:
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or _source_identity(opened) != _source_identity(
                    before
                ):
                    raise ValueError("source_changed")
                content = stream.read()
                after = os.fstat(stream.fileno())
        if _source_identity(after) != _source_identity(before) or _source_identity(
            path.lstat()
        ) != _source_identity(before):
            raise ValueError("source_changed")
        return content
    except OSError as error:
        raise ValueError("source_changed") from error


def _prepare(packet: ChunkPublicationPacket, published_on: date) -> _Prepared:
    manifest = {
        "book_id": packet.book_id,
        "run_id": packet.run_id,
        "result_ids": list(packet.result_ids),
        "source_sha256": packet.source_sha256,
        "representation_sha256": packet.representation_sha256,
        "wiki_schema_version": packet.wiki_schema_version,
        "render_policy_version": packet.render_policy_version,
    }
    if (
        packet.publication_id != "chunk-publication:" + _digest(manifest)[:24]
        or packet.wiki_schema_version != 2
        or packet.render_policy_version != 1
        or not packet.result_ids
        or not packet.pages
        or packet.parsed.source_sha256 != packet.source_sha256
        or packet.parsed.representation_sha256 != packet.representation_sha256
    ):
        raise ValueError("publication_incomplete")
    source_bytes = _read_source_bytes(packet.source_pdf)
    if hashlib.sha256(source_bytes).hexdigest() != packet.source_sha256:
        raise ValueError("source_changed")
    representation = _canonical(document_parse_result_to_json(packet.parsed)) + b"\n"
    blocks = {
        (page.page_index, block.block_id): block
        for page in packet.parsed.pages
        for block in page.blocks
    }
    labels = {page.page_index: page.printed_page_label for page in packet.parsed.pages}
    prefix = (
        f"pages/books/{packet.book_id.replace(':', '-')}/{packet.publication_id.replace(':', '-')}/"
    )
    paths = {
        page.section_id: prefix + f"section-{_digest(page.section_id)[:24]}.md"
        for page in packet.pages
    }
    if len(paths) != len(packet.pages):
        raise ValueError("publication_conflict")
    page_bytes: dict[str, bytes] = {}
    citations: list[ClaimCitation] = []
    claims: list[AtomicClaim] = []
    claim_ids: list[str] = []
    for page in packet.pages:
        page_id = "page:" + _digest([packet.publication_id, page.section_id])[:26].upper()
        lines = [
            "---",
            f"id: {page_id}",
            f"title: {json.dumps(page.title, ensure_ascii=False)}",
            "kind: summary",
            f"created: {published_on.isoformat()}",
            f"updated: {published_on.isoformat()}",
            f'sources: ["source:sha256:{packet.source_sha256}"]',
            "---",
            "",
            f"# {_plain(page.title)}",
            "",
        ]
        if page.section_id == "book":
            lines.extend(("## Sections", ""))
            for other in packet.pages:
                if other.section_id != "book":
                    lines.append(f"- [{_plain(other.title)}]({Path(paths[other.section_id]).name})")
            lines.append("")
        lines.extend(("## Grounded claims", ""))
        for claim_number, claim in enumerate(page.claims):
            if not claim.text.strip() or not 1 <= len(claim.citations) <= 16:
                raise ValueError("citation_invalid")
            claim_id = (
                "claim:"
                + _digest([packet.publication_id, page.section_id, claim.result_id, claim_number])[
                    :24
                ]
            )
            claim_ids.append(claim_id)
            block_ids: list[str] = []
            links: list[str] = []
            seen_blocks: set[tuple[int, str]] = set()
            for citation in claim.citations:
                block = blocks.get((citation.page_index, citation.block_id))
                if (
                    block is None
                    or type(citation.start_char) is not int
                    or type(citation.end_char) is not int
                    or not 0 <= citation.start_char < citation.end_char <= len(block.exact_text)
                    or hashlib.sha256(
                        block.exact_text[citation.start_char : citation.end_char].encode()
                    ).hexdigest()
                    != citation.text_sha256
                ):
                    raise ValueError("citation_invalid")
                key = (citation.page_index, citation.block_id)
                if key in seen_blocks:
                    continue
                seen_blocks.add(key)
                block_ids.append(citation.block_id)
                citation_id = "citation:" + _digest([claim_id, *key])[:24]
                citations.append(
                    ClaimCitation(
                        citation_id,
                        claim_id,
                        page_id,
                        packet.book_id,
                        packet.source_sha256,
                        packet.representation_sha256,
                        citation.page_index,
                        labels[citation.page_index],
                        (citation.block_id,),
                        hashlib.sha256(block.exact_text.encode()).hexdigest(),
                        None,
                        claim.version,
                    )
                )
                links.append(
                    "[PDF page "
                    + str(citation.page_index + 1)
                    + "](../../../../raw/"
                    + packet.source_sha256
                    + ".pdf#page="
                    + str(citation.page_index + 1)
                    + ")"
                )
            claims.append(AtomicClaim(claim_id, claim.text, tuple(dict.fromkeys(block_ids))))
            lines.append(f"- {_plain(claim.text)} {' '.join(links)}")
        page_bytes[paths[page.section_id]] = ("\n".join(lines) + "\n").encode()
    report = validate_book_citations(tuple(citations), tuple(claims), packet.parsed)
    if not report.valid:
        raise ValueError("citation_invalid")
    return _Prepared(
        source_bytes,
        representation,
        page_bytes,
        tuple(asdict(citation) for citation in citations),
        tuple(claim_ids),
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("publication_conflict")
        value[key] = item
    return value


def _read_json(path: Path, *, wiki: Path) -> dict[str, Any]:
    if os.name != "posix" and (
        path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024
    ):
        raise ValueError("publication_conflict")
    try:
        raw = (
            _read_wiki_bytes_posix(wiki, path.relative_to(wiki), max_bytes=1024 * 1024)
            if os.name == "posix"
            else path.read_bytes()
        )
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("publication_conflict") from error
    if not isinstance(value, dict):
        raise ValueError("publication_conflict")
    return value


def _ensure_content(path: Path, content: bytes, *, wiki: Path) -> None:
    if os.name == "posix":
        _ensure_content_posix(wiki, path.relative_to(wiki), content)
        return
    if path.is_symlink() or (
        path.exists() and (not path.is_file() or path.read_bytes() != content)
    ):
        raise ValueError("publication_conflict")
    if not path.exists():
        atomic_write(path, content)


def _read_wiki_bytes_posix(wiki: Path, relative: Path, *, max_bytes: int | None = None) -> bytes:
    """Read one Wiki file through no-follow directory and file handles."""
    with _pinned_parent_posix(wiki, relative, create_parents=False) as (directory_fd, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise WikiUnsafePathError("publication path is unsafe")
            if max_bytes is not None and metadata.st_size > max_bytes:
                raise ValueError("publication_conflict")
            content = source.read(max_bytes + 1 if max_bytes is not None else -1)
            if max_bytes is not None and len(content) > max_bytes:
                raise ValueError("publication_conflict")
            return content


def _check_existing_content_posix(wiki: Path, relative: Path, content: bytes) -> None:
    """Reject an occupied publication path before writing a pending marker."""
    try:
        existing = _read_wiki_bytes_posix(wiki, relative)
    except FileNotFoundError:
        return
    except WikiUnsafePathError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return
        raise
    except OSError as error:
        raise WikiUnsafePathError("publication path is unsafe") from error
    if existing != content:
        raise ValueError("publication_conflict")


@contextmanager
def _pinned_parent_posix(
    wiki: Path, relative: Path, *, create_parents: bool
) -> Iterator[tuple[int, str]]:
    """Pin a Wiki-owned parent without following any directory symlink."""
    if (
        not wiki.is_absolute()
        or relative.is_absolute()
        or not relative.parts
        or any(component in {".", ".."} for component in relative.parts)
    ):
        raise WikiUnsafePathError("publication path is unsafe")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(wiki.anchor, flags)
    try:
        for component in wiki.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        for component in relative.parts[:-1]:
            if create_parents:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=directory_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        yield directory_fd, relative.name
    except OSError as error:
        raise WikiUnsafePathError("publication path is unsafe") from error
    finally:
        os.close(directory_fd)


def _ensure_content_posix(wiki: Path, relative: Path, content: bytes) -> None:
    """Create or verify a publication file beneath pinned no-follow directories."""
    with _pinned_parent_posix(wiki, relative, create_parents=True) as (directory_fd, name):
        try:
            file_fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
        except FileNotFoundError:
            file_fd = None
        if file_fd is not None:
            with os.fdopen(file_fd, "rb") as existing:
                if (
                    not stat.S_ISREG(os.fstat(existing.fileno()).st_mode)
                    or existing.read() != content
                ):
                    raise ValueError("publication_conflict")
            return
        temporary = ".chunk-publication-" + uuid4().hex
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(temporary_fd, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise ValueError("publication_conflict") from error
            os.fsync(directory_fd)
        finally:
            os.unlink(temporary, dir_fd=directory_fd)


def _atomic_write_posix(wiki: Path, relative: Path, content: bytes) -> None:
    """Replace one publication projection relative to a pinned parent."""
    with _pinned_parent_posix(wiki, relative, create_parents=True) as (directory_fd, name):
        temporary = ".chunk-publication-" + uuid4().hex
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(temporary_fd, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _rebuild_index_posix(wiki: Path, documents: tuple[WikiDocument, ...]) -> None:
    """Build privately, then install the index beneath a pinned Wiki parent."""
    with tempfile.TemporaryDirectory(prefix="capicove-wiki-index-") as temporary_root:
        source = Path(temporary_root) / "search.sqlite3"
        rebuild_index(source, documents)
        with _pinned_parent_posix(wiki, Path("_derived/search.sqlite3"), create_parents=True) as (
            directory_fd,
            name,
        ):
            temporary = ".chunk-publication-" + uuid4().hex
            temporary_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                with source.open("rb") as index, os.fdopen(temporary_fd, "wb") as output:
                    shutil.copyfileobj(index, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                os.fsync(directory_fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass


def _unlink_marker(wiki: Path, marker: Path) -> None:
    """Remove a publication marker without following a changed POSIX parent."""
    if os.name == "posix":
        with _pinned_parent_posix(wiki, marker.relative_to(wiki), create_parents=False) as (
            directory_fd,
            name,
        ):
            os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
    else:
        marker.unlink()


def publish_chunk_book(
    wiki: Path, packet: ChunkPublicationPacket, *, published_on: date
) -> dict[str, Any]:
    """Publish one verified result set with a recoverable visibility marker."""
    if (
        re.fullmatch(r"chunk-publication:[0-9a-f]{24}", packet.publication_id) is None
        or re.fullmatch(r"book:[0-9a-f]{24}", packet.book_id) is None
        or re.fullmatch(r"chunk-run:[0-9a-f]{24}", packet.run_id) is None
        or any(
            re.fullmatch(r"chunk-result:[0-9a-f]{24}", result_id) is None
            for result_id in packet.result_ids
        )
    ):
        raise ValueError("publication_conflict")
    with wiki_write_lock(wiki):
        if scan(wiki).schema_version != 2:
            raise ValueError("Wiki v2 is required for chunk publication")
        ontology_path = wiki / "graph" / "ontology.yaml"
        if os.name == "posix":
            load_yaml_mapping(
                ontology_path,
                content=_read_wiki_bytes_posix(wiki, ontology_path.relative_to(wiki)),
            )
        else:
            load_yaml_mapping(ontology_path)
        marker = wiki / "_migration" / "chunk-publication.pending"
        receipt = (
            wiki
            / "_migration"
            / "chunk-publications"
            / (packet.publication_id.replace(":", "-") + ".json")
        )
        if marker.exists() or marker.is_symlink():
            pending = _read_json(marker, wiki=wiki)
            if set(pending) != {"schema_version", "publication_id", "published_on"} or (
                type(pending["schema_version"]) is not int
                or pending["schema_version"] != 1
                or pending["publication_id"] != packet.publication_id
            ):
                raise ValueError("publication_conflict")
            published_on = _publication_date(pending["published_on"])
        elif receipt.exists() or receipt.is_symlink():
            previous = _read_json(receipt, wiki=wiki)
            if previous.get("publication_id") != packet.publication_id:
                raise ValueError("publication_conflict")
            published_on = _publication_date(previous.get("published_on"))
        prepared = _prepare(packet, published_on)
        response = {
            "publication_id": packet.publication_id,
            "wiki_schema_version": 2,
            "page_paths": list(prepared.page_bytes),
            "claim_ids": list(prepared.claim_ids),
            "book_id": packet.book_id,
            "run_id": packet.run_id,
            "source_sha256": packet.source_sha256,
            "representation_sha256": packet.representation_sha256,
            "published_on": published_on.isoformat(),
        }
        receipt_bytes = _canonical(response)
        if receipt.exists() or receipt.is_symlink():
            if receipt.is_symlink():
                raise ValueError("publication_conflict")
            existing_receipt = (
                _read_wiki_bytes_posix(wiki, receipt.relative_to(wiki), max_bytes=1024 * 1024)
                if os.name == "posix"
                else receipt.read_bytes()
            )
            if existing_receipt != receipt_bytes:
                raise ValueError("publication_conflict")
        citation_path = wiki / "graph" / "citations.jsonl"
        existing = (
            jsonl_records_from_bytes(
                _read_wiki_bytes_posix(wiki, citation_path.relative_to(wiki)),
                name=citation_path.name,
            )
            if os.name == "posix"
            else jsonl_records(citation_path)
        )
        by_id = {str(item.get("citation_id")): item for item in existing}
        for item in prepared.citations:
            old = by_id.get(item["citation_id"])
            if old is not None and _canonical(old) != _canonical(item):
                raise ValueError("publication_conflict")
            by_id[item["citation_id"]] = item
        merged = tuple(sorted(by_id.values(), key=lambda item: str(item["citation_id"])))
        citation_bytes = b"".join(_canonical(item) + b"\n" for item in merged)
        if receipt.exists():
            for path, payload in (
                (wiki / "raw" / f"{packet.source_sha256}.pdf", prepared.source_bytes),
                (
                    wiki / "representations" / f"{packet.representation_sha256}.json",
                    prepared.representation_bytes,
                ),
                *((wiki / relative, content) for relative, content in prepared.page_bytes.items()),
            ):
                if path.is_symlink():
                    raise ValueError("publication_conflict")
                if os.name == "posix":
                    try:
                        existing_bytes = _read_wiki_bytes_posix(wiki, path.relative_to(wiki))
                    except FileNotFoundError as error:
                        raise ValueError("publication_conflict") from error
                elif path.is_file():
                    existing_bytes = path.read_bytes()
                else:
                    raise ValueError("publication_conflict")
                if existing_bytes != payload:
                    raise ValueError("publication_conflict")
            if (
                _read_wiki_bytes_posix(wiki, citation_path.relative_to(wiki))
                if os.name == "posix"
                else citation_path.read_bytes()
            ) != citation_bytes:
                raise ValueError("publication_conflict")
            report = lint(wiki)
            if not report.valid:
                raise WikiValidationError(report)
            documents = load_documents(wiki)
            try:
                check_index(wiki / "_derived" / "search.sqlite3", documents)
            except WikiIndexUnavailableError:
                if os.name == "posix":
                    _rebuild_index_posix(wiki, documents)
                else:
                    rebuild_index(wiki / "_derived" / "search.sqlite3", documents)
            if marker.exists():
                _unlink_marker(wiki, marker)
            return {key: value for key, value in response.items() if key != "published_on"}
        for path, payload in (
            (wiki / "raw" / f"{packet.source_sha256}.pdf", prepared.source_bytes),
            (
                wiki / "representations" / f"{packet.representation_sha256}.json",
                prepared.representation_bytes,
            ),
            *((wiki / relative, content) for relative, content in prepared.page_bytes.items()),
        ):
            if os.name == "posix":
                _check_existing_content_posix(wiki, path.relative_to(wiki), payload)
            elif path.is_symlink() or (path.exists() and path.read_bytes() != payload):
                raise ValueError("publication_conflict")
        if not marker.exists():
            marker_bytes = _canonical(
                {
                    "schema_version": 1,
                    "publication_id": packet.publication_id,
                    "published_on": published_on.isoformat(),
                }
            )
            if os.name == "posix":
                _atomic_write_posix(wiki, marker.relative_to(wiki), marker_bytes)
            else:
                atomic_write(marker, marker_bytes)
        _ensure_content(
            wiki / "raw" / f"{packet.source_sha256}.pdf", prepared.source_bytes, wiki=wiki
        )
        _ensure_content(
            wiki / "representations" / f"{packet.representation_sha256}.json",
            prepared.representation_bytes,
            wiki=wiki,
        )
        for relative, payload in prepared.page_bytes.items():
            _ensure_content(wiki / relative, payload, wiki=wiki)
        if (
            _read_wiki_bytes_posix(wiki, citation_path.relative_to(wiki))
            if os.name == "posix"
            else citation_path.read_bytes()
        ) != citation_bytes:
            if os.name == "posix":
                _atomic_write_posix(wiki, citation_path.relative_to(wiki), citation_bytes)
            else:
                atomic_write(citation_path, citation_bytes)
        report = lint(wiki)
        if not report.valid:
            raise WikiValidationError(report)
        documents = load_documents(wiki)
        if os.name == "posix":
            _rebuild_index_posix(wiki, documents)
        else:
            rebuild_index(wiki / "_derived" / "search.sqlite3", documents)
        index_bytes = render_index(documents, built_on=published_on).encode("utf-8")
        if os.name == "posix":
            _atomic_write_posix(wiki, Path("index.md"), index_bytes)
        else:
            atomic_write(wiki / "index.md", index_bytes)
        if os.name == "posix":
            _atomic_write_posix(wiki, receipt.relative_to(wiki), receipt_bytes)
        else:
            atomic_write(receipt, receipt_bytes)
        _unlink_marker(wiki, marker)
        return {key: value for key, value in response.items() if key != "published_on"}
