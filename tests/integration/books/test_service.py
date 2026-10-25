from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from capicove.books import (
    AtomicClaim,
    BookService,
    LocalBookWorkspace,
    StructureReviewSubmission,
    SummarySubmission,
)
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseRequest,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)


class _Parser:
    parser_id = "fake"

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        pages = []
        for page_index, title in enumerate(("Chapter 1 Foundations", "Chapter 2 Systems")):
            block = DocumentBlock(
                f"p{page_index:04d}:b000000",
                page_index,
                DocumentBlockType.DOCUMENT_TITLE,
                title,
                title,
                NormalizedBox(0.1, 0.1, 0.9, 0.2),
                0,
                0.99,
            )
            pages.append(
                DocumentPage(
                    page_index,
                    str(page_index + 1),
                    100,
                    200,
                    DocumentPageStatus.SUCCEEDED,
                    (block,),
                )
            )
        receipt = DocumentParserReceipt(
            "fake.plugin",
            "1",
            request.profile_id,
            request.profile_version,
            "c" * 64,
            "d" * 64,
            "e" * 64,
        )
        return DocumentParseResult(
            request.source_sha256,
            "b" * 64,
            tuple(pages),
            receipt,
        )


class _LowConfidenceParser(_Parser):
    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        parsed = super().parse(request)
        pages = []
        for page in parsed.pages:
            block = page.blocks[0]
            title = "Preface" if page.page_index == 0 else "Background"
            pages.append(
                DocumentPage(
                    page.page_index,
                    page.printed_page_label,
                    page.width,
                    page.height,
                    page.status,
                    (
                        DocumentBlock(
                            block.block_id,
                            block.page_index,
                            block.block_type,
                            title,
                            title,
                            block.box,
                            block.reading_order,
                            0.7,
                        ),
                    ),
                )
            )
        return DocumentParseResult(
            parsed.source_sha256,
            parsed.representation_sha256,
            tuple(pages),
            parsed.receipt,
        )


def test_service_prepare_submit_resume_and_append_versions(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)

    first = service.prepare(source, _Parser())
    second = service.prepare(source, _Parser())

    assert first.book_id == second.book_id
    assert first.created is True
    assert second.created is False
    assert workspace.read_structure(first.book_id).profile_version == 2
    tasks = service.tasks(first.book_id)
    assert len(tasks) == 2
    task = service.next_task(first.book_id)
    assert task is not None
    submission = SummarySubmission(
        task.task_id,
        task.idempotency_key,
        "An abstract",
        ("A point",),
        (AtomicClaim("claim:first", "A fact", (task.allowed_citation_ids[0],)),),
        "caller",
        "caller-model",
    )

    first_version = service.submit(first.book_id, submission)
    second_version = service.submit(first.book_id, submission)

    assert first_version.version == 1
    assert first_version.previous_version is None
    assert second_version.version == 2
    assert second_version.previous_version == 1
    reopened = BookService(LocalBookWorkspace.open(workspace.root))
    assert reopened.status(first.book_id).completed_tasks == 1
    assert reopened.next_task(first.book_id) is not None


def test_service_passes_language_hints_and_isolates_their_cache_identity(
    tmp_path: Path,
) -> None:
    requests: list[DocumentParseRequest] = []

    class CapturingParser(_Parser):
        def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
            requests.append(request)
            return super().parse(request)

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    service = BookService(LocalBookWorkspace.create(tmp_path / "workspace"))

    default = service.prepare(source, CapturingParser())
    hinted = service.prepare(source, CapturingParser(), language_hints=("ru",))
    resumed = service.prepare(source, CapturingParser(), language_hints=("ru",))

    assert default.book_id != hinted.book_id
    assert hinted.book_id == resumed.book_id
    assert requests[-1].language_hints == ("ru",)
    assert resumed.created is False


def test_service_rejects_parser_receipt_for_a_different_profile(tmp_path: Path) -> None:
    class WrongProfileParser(_Parser):
        def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
            parsed = super().parse(request)
            return replace(
                parsed,
                receipt=replace(parsed.receipt, profile_version=request.profile_version + 1),
            )

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    service = BookService(LocalBookWorkspace.create(tmp_path / "workspace"))

    with pytest.raises(ValueError, match="profile"):
        service.prepare(source, WrongProfileParser())


def test_service_revalidates_cached_parser_profile_identity(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    representation_path = (
        workspace.root / "books" / prepared.book_id.replace(":", "-") / "representation.json"
    )
    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    representation["receipt"]["profile_version"] = 1
    representation_path.write_text(
        json.dumps(representation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest"):
        service.prepare(source, _Parser())


def test_service_revalidates_cached_structure_identity(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    structure_path = (
        workspace.root / "books" / prepared.book_id.replace(":", "-") / "structure.json"
    )
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    structure["source_sha256"] = "f" * 64
    structure_path.write_text(
        json.dumps(structure, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="structure"):
        service.prepare(source, _Parser())


def test_service_rejects_cached_task_text_tampering_even_with_forged_inner_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    tasks_path = workspace.root / "books" / prepared.book_id.replace(":", "-") / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    tasks[0]["evidence"][0]["exact_text"] = "forged task evidence"
    tasks[0]["evidence"][0]["exact_text_sha256"] = hashlib.sha256(
        b"forged task evidence"
    ).hexdigest()
    tasks_path.write_text(
        "".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in tasks),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"tasks\.jsonl"):
        service.prepare(source, _Parser())


def test_schema_v1_workspace_is_rejected_without_mutation(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    book = workspace.root / "books" / prepared.book_id.replace(":", "-")
    manifest_path = book / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy = {
        key: current[key]
        for key in (
            "schema_version",
            "book_id",
            "source_sha256",
            "representation_sha256",
            "review_task_count",
        )
    }
    legacy["schema_version"] = 1
    manifest_path.write_text(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (workspace.root / "workspace.json").write_text('{"schema_version":1}\n', encoding="utf-8")

    legacy_book = manifest_path.read_bytes()
    legacy_workspace = (workspace.root / "workspace.json").read_bytes()

    with pytest.raises(ValueError, match="workspace manifest"):
        workspace.read_result(prepared.book_id)
    with pytest.raises(ValueError, match="workspace manifest"):
        service.status(prepared.book_id)
    with pytest.raises(ValueError, match="workspace manifest"):
        LocalBookWorkspace.open(workspace.root)

    assert manifest_path.read_bytes() == legacy_book
    assert (workspace.root / "workspace.json").read_bytes() == legacy_workspace


def test_schema_v1_workspace_cannot_self_sign_forged_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    book = workspace.root / "books" / prepared.book_id.replace(":", "-")
    manifest_path = book / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy = {
        key: current[key]
        for key in (
            "schema_version",
            "book_id",
            "source_sha256",
            "representation_sha256",
            "review_task_count",
        )
    }
    legacy["schema_version"] = 1
    manifest_path.write_text(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tasks_path = book / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    tasks[0]["evidence"][0]["exact_text"] = "FORGED MIGRATION EVIDENCE"
    tasks[0]["evidence"][0]["exact_text_sha256"] = hashlib.sha256(
        b"FORGED MIGRATION EVIDENCE"
    ).hexdigest()
    tasks_path.write_text(
        "".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in tasks),
        encoding="utf-8",
    )
    (workspace.root / "workspace.json").write_text('{"schema_version":1}\n', encoding="utf-8")

    legacy_manifest = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="workspace manifest"):
        LocalBookWorkspace.open(workspace.root)
    assert manifest_path.read_bytes() == legacy_manifest


def test_manifest_downgrade_cannot_bypass_artifact_verification(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    book = workspace.root / "books" / prepared.book_id.replace(":", "-")
    manifest_path = book / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    current.pop("artifact_sha256")
    current["schema_version"] = 1
    manifest_path.write_text(
        json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tasks_path = book / "tasks.jsonl"
    tasks_path.write_bytes(tasks_path.read_bytes().replace(b"Foundations", b"FORGED DATA"))
    (workspace.root / "workspace.json").write_text('{"schema_version":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="workspace manifest"):
        service.tasks(prepared.book_id)
    with pytest.raises(ValueError, match="workspace manifest"):
        service.prepare(source, _Parser())


def test_schema_v1_profile_can_be_reparsed_into_a_new_workspace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nlegacy profile book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    prepared = BookService(workspace).prepare(source, _Parser(), profile_version=1)
    book = workspace.root / "books" / prepared.book_id.replace(":", "-")
    manifest_path = book / "manifest.json"
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    current.pop("artifact_sha256")
    current["schema_version"] = 1
    manifest_path.write_text(
        json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (workspace.root / "workspace.json").write_text('{"schema_version":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="workspace manifest"):
        LocalBookWorkspace.open(workspace.root)

    replacement = LocalBookWorkspace.create(tmp_path / "replacement")
    reparsed = BookService(replacement).prepare(source, _Parser(), profile_version=1)

    assert BookService(replacement).status(reparsed.book_id).total_tasks == 2
    assert replacement.read_manifest(reparsed.book_id)["schema_version"] == 2


def test_service_restores_missing_cached_source_and_rejects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    preserved = workspace.root / "raw" / f"{prepared.source_sha256}.pdf"
    preserved.unlink()

    resumed = service.prepare(source, _Parser())

    assert resumed.created is False
    assert preserved.read_bytes() == source.read_bytes()
    preserved.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content-addressed"):
        service.prepare(source, _Parser())


def test_service_revalidates_cached_evidence_only_policy(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nambiguous book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _LowConfidenceParser(), evidence_only=True)
    policy = (
        workspace.root / "books" / prepared.book_id.replace(":", "-") / "preparation-policy.json"
    )
    policy.unlink()

    with pytest.raises(ValueError, match="policy"):
        service.prepare(source, _LowConfidenceParser(), evidence_only=True)


def test_invalid_profile_does_not_preserve_source(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nstructured book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")

    with pytest.raises(ValueError, match="profile_version"):
        BookService(workspace).prepare(source, _Parser(), profile_version=0)

    assert tuple((workspace.root / "raw").iterdir()) == ()


def test_service_accepts_only_local_structure_reviews_then_creates_summary_tasks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nambiguous book")
    service = BookService(LocalBookWorkspace.create(tmp_path / "workspace"))

    prepared = service.prepare(source, _LowConfidenceParser())
    reviews = service.structure_tasks(prepared.book_id)

    assert reviews
    assert service.tasks(prepared.book_id) == ()
    for review in reviews:
        service.submit_structure(
            prepared.book_id,
            StructureReviewSubmission(review.task_id, "accept"),
        )
    assert service.structure_tasks(prepared.book_id) == ()
    assert len(service.tasks(prepared.book_id)) == 2
    assert service.status(prepared.book_id).structure_review_tasks == 0


def test_reviewed_projection_rejects_tampered_tasks(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nambiguous book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _LowConfidenceParser())
    for review in service.structure_tasks(prepared.book_id):
        service.submit_structure(
            prepared.book_id, StructureReviewSubmission(review.task_id, "accept")
        )
    reviewed_tasks = (
        workspace.root / "books" / prepared.book_id.replace(":", "-") / "reviewed-tasks.jsonl"
    )
    original = reviewed_tasks.read_bytes()
    tampered = original.replace(b"Preface", b"FORGED!")
    assert tampered != original
    reviewed_tasks.write_bytes(tampered)

    with pytest.raises(ValueError, match=r"reviewed-tasks\.jsonl"):
        service.tasks(prepared.book_id)
    with pytest.raises(ValueError, match=r"reviewed-tasks\.jsonl"):
        service.prepare(source, _LowConfidenceParser())


def test_legacy_reviewed_projection_is_verified_before_manifest_migration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nambiguous book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _LowConfidenceParser())
    for review in service.structure_tasks(prepared.book_id):
        service.submit_structure(
            prepared.book_id, StructureReviewSubmission(review.task_id, "accept")
        )
    book = workspace.root / "books" / prepared.book_id.replace(":", "-")
    reviewed_manifest = book / "reviewed-manifest.json"
    reviewed_manifest.unlink()

    with pytest.raises(ValueError, match="migration"):
        service.tasks(prepared.book_id)
    with pytest.raises(ValueError, match="migration"):
        service.status(prepared.book_id)
    resumed = service.prepare(source, _LowConfidenceParser())

    assert resumed.created is False
    assert reviewed_manifest.is_file()
    assert workspace.reviewed_projection_is_verified(prepared.book_id)


def test_invalid_review_is_not_persisted_and_can_be_corrected(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _LowConfidenceParser())
    first, last = service.structure_tasks(prepared.book_id)
    service.submit_structure(prepared.book_id, StructureReviewSubmission(first.task_id, "accept"))

    with pytest.raises(ValueError, match="outside local context"):
        service.submit_structure(
            prepared.book_id, StructureReviewSubmission(last.task_id, "split", ("p9999:b999999",))
        )

    assert last.task_id not in workspace.latest_structure_submissions(prepared.book_id)
    assert service.structure_tasks(prepared.book_id) == (last,)
    service.submit_structure(prepared.book_id, StructureReviewSubmission(last.task_id, "accept"))
    assert service.status(prepared.book_id).total_tasks == 2


@pytest.mark.parametrize("failed_name", ["reviewed-structure.json", "reviewed-tasks.jsonl"])
def test_review_finalization_failure_can_resume_after_restart(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    failed_name: str,
) -> None:
    import capicove.books.workspace as workspace_module

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _LowConfidenceParser())
    first, last = service.structure_tasks(prepared.book_id)
    service.submit_structure(prepared.book_id, StructureReviewSubmission(first.task_id, "accept"))

    original_write = workspace_module._atomic_write

    def fail_write(path: Path, payload: bytes) -> None:
        if path.name == failed_name:
            raise OSError("interrupted finalization")
        original_write(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(workspace_module, "_atomic_write", fail_write)
        with pytest.raises(OSError, match="interrupted"):
            service.submit_structure(
                prepared.book_id, StructureReviewSubmission(last.task_id, "accept")
            )

    assert not workspace.has_reviewed(prepared.book_id)
    assert workspace.read_structure(prepared.book_id) == workspace.read_structure(
        prepared.book_id, original=True
    )
    assert workspace.read_tasks(prepared.book_id) == ()
    reopened = BookService(LocalBookWorkspace.open(workspace.root))
    assert reopened.structure_tasks(prepared.book_id)
    reopened.submit_structure(prepared.book_id, StructureReviewSubmission(last.task_id, "accept"))
    assert reopened.status(prepared.book_id).total_tasks == 2
    assert workspace.has_reviewed(prepared.book_id)
    assert workspace.latest_structure_submissions(prepared.book_id)[last.task_id].version == 1


def test_evidence_only_prepare_bypasses_legacy_task_size_and_is_idempotent(tmp_path: Path) -> None:
    from dataclasses import replace

    class LongParser(_Parser):
        def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
            parsed = super().parse(request)
            page = parsed.pages[0]
            template = page.blocks[0]
            blocks = (
                template,
                *(
                    replace(
                        template,
                        block_id=f"p0000:b{index:06d}",
                        reading_order=index,
                        block_type=DocumentBlockType.TEXT,
                        exact_text=f"Source paragraph {index}.",
                        normalized_text=f"Source paragraph {index}.",
                    )
                    for index in range(1, 150)
                ),
            )
            return replace(parsed, pages=(replace(page, blocks=blocks),))

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-evidence")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    with pytest.raises(ValueError):
        service.prepare(source, LongParser())
    prepared = service.prepare(source, LongParser(), evidence_only=True)
    assert prepared.summary_tasks == 0
    assert len(workspace.read_result(prepared.book_id).pages[0].blocks) == 150
    assert service.prepare(source, LongParser(), evidence_only=True).created is False
    assert workspace.is_evidence_only(prepared.book_id)


def test_evidence_only_review_finalization_does_not_build_legacy_summaries(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-evidence")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _LowConfidenceParser(), evidence_only=True)
    for review in service.structure_tasks(prepared.book_id):
        service.submit_structure(
            prepared.book_id, StructureReviewSubmission(review.task_id, "accept")
        )
    assert workspace.has_reviewed(prepared.book_id)
    assert service.tasks(prepared.book_id) == ()
    assert not service.status(prepared.book_id).publishable


def test_failed_page_blocks_publication_after_all_summary_tasks_complete(tmp_path: Path) -> None:
    class PartialParser(_Parser):
        def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
            result = super().parse(request)
            failed = DocumentPage(2, "3", 100, 200, DocumentPageStatus.FAILED, (), "OCR failed")
            return replace(result, pages=(*result.pages, failed))

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-partial")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, PartialParser())
    for index, task in enumerate(service.tasks(prepared.book_id)):
        service.submit(
            prepared.book_id,
            SummarySubmission(
                task.task_id,
                task.idempotency_key,
                "Abstract",
                ("Point",),
                (AtomicClaim(f"claim:{index}", "Fact", (task.allowed_citation_ids[0],)),),
            ),
        )

    status = service.status(prepared.book_id)
    assert status.completed_tasks == status.total_tasks == 2
    assert not status.publishable
    with pytest.raises(ValueError, match="not publishable"):
        service.publish(prepared.book_id, published_on=date(2026, 9, 3))
    assert not (workspace.root / "wiki").exists()
