"""Resolve Markdown and committed Book snapshots without caller hash authority."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from capicove.contracts import DocumentParseResult
from capicove.contracts.wiki_maintenance import EvidenceSpan, PublicationRecord, SourceReference
from capicove.wiki.maintenance.codec import decode_record
from capicove.wiki.maintenance.evidence import markdown_span, resolve_markdown, resolve_pdf
from capicove.wiki.maintenance.representations import MarkdownRepresentation
from capicove.wiki.maintenance.store import read_authorized, record_path


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService

SourceInterval = tuple[str, int, int] | tuple[str, int, str, int, int]


@dataclass(frozen=True, slots=True)
class SourceMaterial:
    """Host-loaded evidence and its verified representation commitment."""

    reference: SourceReference
    raw: bytes
    markdown: MarkdownRepresentation | None = None
    pdf: DocumentParseResult | None = None
    snapshot_sha256: str | None = None


def committed_bytes(service: "MaintenanceService", relative: str) -> bytes:
    """Read an exact current artifact only if committed by active history."""
    from capicove.wiki.maintenance.tasks import state

    if not relative.startswith(("maintenance/book-snapshots/", "maintenance/follow-ups/")):
        raise ValueError("invalid committed artifact")
    raw = read_authorized(service.wiki / relative, (service.wiki,), 1048576)
    current = state(service)
    identity = current["publication_id"]
    expected_revision = current["wiki_revision"]
    seen: set[str] = set()
    while identity is not None:
        if identity in seen or len(seen) >= 10000:
            raise ValueError("invalid committed lineage")
        seen.add(identity)
        receipt = decode_record(
            read_authorized(
                record_path(service.wiki, "publications", identity, "publication"),
                (service.wiki,),
                1048576,
            )
        ).payload
        if (
            not isinstance(receipt, PublicationRecord)
            or receipt.publication_id != identity
            or receipt.wiki_revision != expected_revision
            or receipt.status == "aborted"
        ):
            raise ValueError("invalid committed lineage")
        for change in receipt.changes:
            if change.relative_path == relative:
                if hashlib.sha256(raw).hexdigest() != change.after_sha256:
                    raise ValueError("evidence_digest_mismatch: committed snapshot")
                return raw
        identity = receipt.parent_id
        expected_revision -= 1
    raise ValueError("snapshot is not committed")


def load_source(
    service: "MaintenanceService",
    source_id: str,
    withdrawn: tuple[str, ...],
    *,
    representation_sha256: str | None = None,
) -> SourceMaterial:
    """Load a source using host-owned representation identity, never Agent input."""
    if re.fullmatch(r"source:sha256:[0-9a-f]{64}", source_id) is None or source_id in withdrawn:
        raise ValueError("evidence_unavailable: invalid or withdrawn source")
    digest = source_id.removeprefix("source:sha256:")
    if (service.wiki / "raw" / (digest + ".md")).exists():
        value = service._read_markdown_unlocked(source_id, withdrawn)
        return SourceMaterial(
            SourceReference(
                source_id=source_id,
                source_sha256=digest,
                representation_sha256=value.representation_sha256,
            ),
            read_authorized(service.wiki / "raw" / (digest + ".md"), (service.wiki,), 16777216),
            markdown=value,
        )
    from capicove.documents import document_parse_result_from_json, document_parse_result_to_json

    raw = read_authorized(service.wiki / "raw" / (digest + ".pdf"), (service.wiki,), 16777216)
    if (
        representation_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", representation_sha256) is None
    ):
        raise ValueError("invalid representation identity")
    suffix = "-" + representation_sha256 if representation_sha256 is not None else ""
    commitment = json.loads(
        committed_bytes(service, "maintenance/book-snapshots/" + digest + suffix + ".json")
    )
    if (
        not isinstance(commitment, dict)
        or set(commitment)
        != {"schema_version", "source_sha256", "representation_sha256", "snapshot_sha256"}
        or type(commitment["schema_version"]) is not int
        or commitment["schema_version"] != 1
        or commitment["source_sha256"] != digest
        or not isinstance(commitment["representation_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", commitment["representation_sha256"]) is None
    ):
        raise ValueError("invalid committed PDF snapshot")
    encoded = read_authorized(
        service.wiki / "representations" / (commitment["representation_sha256"] + ".json"),
        (service.wiki,),
        16777216,
    )
    parsed = document_parse_result_from_json(json.loads(encoded))
    canonical = json.dumps(
        document_parse_result_to_json(parsed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    if (
        hashlib.sha256(raw).hexdigest() != digest
        or parsed.source_sha256 != digest
        or parsed.representation_sha256 != commitment["representation_sha256"]
        or hashlib.sha256(canonical).hexdigest() != commitment["snapshot_sha256"]
    ):
        raise ValueError("evidence_digest_mismatch: PDF snapshot")
    return SourceMaterial(
        SourceReference(
            source_id=source_id,
            source_sha256=digest,
            representation_sha256=parsed.representation_sha256,
        ),
        raw,
        pdf=parsed,
        snapshot_sha256=commitment["snapshot_sha256"],
    )


def resolve(material: SourceMaterial, span: EvidenceSpan) -> str:
    """Re-resolve one exact locator against loaded original evidence."""
    if material.markdown is not None:
        return resolve_markdown(span, material.raw)
    if material.pdf is None or material.snapshot_sha256 is None:
        raise ValueError("evidence_unavailable")
    return resolve_pdf(span, material.pdf, material.raw, snapshot_sha256=material.snapshot_sha256)


def slices(
    material: SourceMaterial, intervals: tuple[SourceInterval, ...] | None
) -> tuple[tuple[EvidenceSpan, str], ...]:
    """Select bounded half-open codepoint intervals, preserving PDF block identity."""
    from capicove.contracts import DocumentPageStatus

    source = material.reference
    selections = (
        tuple(item for item in intervals if item[0] == source.source_id)
        if intervals is not None
        else None
    )
    result = []
    if material.markdown is not None:
        text = material.markdown.text
        if selections is not None and any(len(item) != 3 for item in selections):
            raise ValueError("invalid Markdown interval")
        ranges = (
            [(start, min(start + 4096, len(text))) for start in range(0, len(text), 4096)]
            if selections is None
            else [(item[1], item[2]) for item in selections]
        )
        for start, end in ranges:
            if type(start) is not int or type(end) is not int:
                raise ValueError("invalid Markdown interval")
            span = markdown_span(material.markdown, start, end)
            result.append((span, text[start:end]))
    else:
        if material.pdf is None:
            raise ValueError("evidence_unavailable")
        if selections is not None and any(len(item) != 5 for item in selections):
            raise ValueError("PDF intervals require source, page, block, start and end")
        for page in material.pdf.pages:
            if page.status != DocumentPageStatus.SUCCEEDED:
                continue
            for block in page.blocks:
                text = block.exact_text
                ranges = (
                    [(start, min(start + 4096, len(text))) for start in range(0, len(text), 4096)]
                    if selections is None
                    else [
                        (item[3], item[4])
                        for item in selections
                        if len(item) == 5
                        and item[1] == page.page_index
                        and item[2] == block.block_id
                    ]
                )
                for start, end in ranges:
                    if (
                        type(start) is not int
                        or type(end) is not int
                        or not 0 <= start < end <= len(text)
                    ):
                        raise ValueError("invalid PDF interval")
                    span = EvidenceSpan(
                        kind="pdf_block_span",
                        source_id=source.source_id,
                        source_sha256=source.source_sha256,
                        representation_sha256=source.representation_sha256,
                        start_char=start,
                        end_char=end,
                        exact_sha256=hashlib.sha256(text[start:end].encode()).hexdigest(),
                        page_index=page.page_index,
                        block_id=block.block_id,
                    )
                    result.append((span, text[start:end]))
                    if len(result) > 128:
                        raise ValueError("capacity_exceeded: PDF context")
        if selections is not None and len(result) != len(selections):
            raise ValueError("citation_out_of_scope: PDF selection")
    if not result or len(result) > 128 or sum(len(text.encode()) for _, text in result) > 262144:
        raise ValueError("capacity_exceeded: source context")
    return tuple(result)
