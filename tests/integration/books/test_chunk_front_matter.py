import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from capicove.books import BookService, LocalBookWorkspace
from capicove.books.chunk_config import ChunkingConfig
from capicove.books.chunk_models import SourceSpan
from capicove.books.chunk_service import BookChunkService
from capicove.contracts import DocumentBlockType as Kind
from capicove.contracts import DocumentOutlineEntry, DocumentParseRequest, DocumentParseResult

from .test_service import _Parser


class _FrontMatterParser(_Parser):
    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        template = super().parse(request)
        page = template.pages[0]
        block = page.blocks[0]
        contents = (
            ((Kind.DOCUMENT_TITLE, "物理"), (Kind.TEXT, "普通高中教科书\n必修第一册\n示例出版社")),
            (
                (Kind.TEXT, "责任编辑: 甲\n出版: 示例出版社"),
                (Kind.OTHER, "版权所有, 未经许可不得复制。ISBN 000-0"),
            ),
            ((Kind.TEXT, "目录\n第一章 测量 10\n第二章 运动 20"),),
            ((Kind.DOCUMENT_TITLE, "致同学们"), (Kind.TEXT, "请先观察现象, 再设计实验")),
            ((Kind.DOCUMENT_TITLE, "序言"), (Kind.TEXT, "本书介绍实验方法。")),
        )
        pages = tuple(
            replace(
                page,
                page_index=index,
                blocks=tuple(
                    replace(
                        block,
                        block_id=f"p{index:04d}:b{number:06d}",
                        page_index=index,
                        reading_order=number,
                        block_type=kind,
                        exact_text=text,
                        normalized_text=" ".join(text.split()),
                    )
                    for number, (kind, text) in enumerate(items)
                ),
            )
            for index, items in enumerate(contents)
        )
        return replace(
            template,
            pages=pages,
            outline=(DocumentOutlineEntry("序言", 4, 1),),
        )


def _prepared(tmp_path: Path) -> tuple[BookChunkService, str]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-synthetic-front-matter")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    prepared = BookService(workspace).prepare(source, _FrontMatterParser(), evidence_only=True)
    return BookChunkService(workspace), prepared.book_id


def test_front_matter_reads_preview_and_context_preserve_policy(tmp_path: Path) -> None:
    service, book_id = _prepared(tmp_path)
    created = service.create(book_id, ChunkingConfig())
    assert created["exclusion_reason_counts"] == {
        "front_matter:cover": 2,
        "front_matter:copyright": 2,
    }
    assert created["navigation_chunk_count"] == 1
    chunks = service.read_chunks(book_id, created["run_id"])["items"]
    assert any(chunk["content_role"] == "navigation" for chunk in chunks)
    for chunk in chunks:
        assert chunk["extraction_eligible"] == (chunk["content_role"] == "content")
        context = service.read_context(book_id, created["run_id"], chunk["chunk_id"])
        assert context["content_role"] == chunk["content_role"]
        assert all(span["page_index"] >= 2 for span in context["spans"])
        if chunk["content_role"] == "navigation":
            assert {span["page_index"] for span in context["spans"]} == {2}
        else:
            assert all(span["page_index"] != 2 for span in context["spans"])


def test_legacy_run_stays_readable_and_unchanged_after_policy_upgrade(tmp_path: Path) -> None:
    service, book_id = _prepared(tmp_path)
    old = service.create(book_id, ChunkingConfig(front_matter_policy="preserve"))
    saved = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    new = service.create(book_id, ChunkingConfig(), compare_run=old["run_id"])
    assert old["run_id"] != new["run_id"]
    assert any(
        span["page_index"] == 0
        for chunk in service.read_chunks(book_id, old["run_id"])["items"]
        for span in chunk["core_spans"]
    )
    assert all(path.read_bytes() == raw for path, raw in saved.items())


def test_new_exclusion_policy_cannot_be_rehashed_under_legacy_run_id(tmp_path: Path) -> None:
    service, book_id = _prepared(tmp_path)
    old = service.create(book_id, ChunkingConfig(front_matter_policy="preserve"))
    new = service.create(book_id, ChunkingConfig())
    paths = {path.parent.name: path for path in tmp_path.rglob("run.json")}
    old_path = paths[old["run_id"].replace(":", "-")]
    new_path = paths[new["run_id"].replace(":", "-")]
    value = json.loads(new_path.read_bytes())
    value["run"]["run_id"] = old["run_id"]
    raw = json.dumps(value).encode()
    old_path.write_bytes(raw)
    old_path.with_name("sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="identity"):
        service.read_chunks(book_id, old["run_id"])


@pytest.mark.parametrize(
    "tamper", ["reason", "context", "excluded-context", "navigation", "excluded-content"]
)
def test_rehashed_front_matter_run_cannot_bypass_source_policy(tmp_path: Path, tamper: str) -> None:
    service, book_id = _prepared(tmp_path)
    created = service.create(book_id, ChunkingConfig())
    path = next(tmp_path.rglob("run.json"))
    value = json.loads(path.read_bytes())
    chunks = value["run"]["chunks"]
    if tamper == "reason":
        value["run"]["exclusions"][0][1] = "front_matter:copyright"
    elif tamper == "context":
        chunks[-1]["before_spans"] = chunks[0]["core_spans"]
    elif tamper == "excluded-context":
        chunks[-1]["before_spans"] = [
            asdict(
                SourceSpan("p0000:b000000", 0, 0, 2, hashlib.sha256("物理".encode()).hexdigest())
            )
        ]
    elif tamper == "navigation":
        chunks[0]["parent_id"] = "front-matter"
    else:
        removed = chunks.pop()
        value["run"]["exclusions"].extend(
            [[span["block_id"], "front_matter:cover"] for span in removed["core_spans"]]
        )
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    path.with_name("sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match=r"exclusion|excluded|navigation"):
        service.read_chunks(book_id, created["run_id"])
