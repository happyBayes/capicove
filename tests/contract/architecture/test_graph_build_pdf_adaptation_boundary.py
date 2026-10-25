"""Keep document composition separate from the text-only graph core."""

import ast
from pathlib import Path


_DESIGN_PATH = Path("docs/architecture/graph-pdf-adaptation.md")
_REPRESENTATION_MODULES = (
    Path("src/capicove/graph_build/representations.py"),
    Path("src/capicove/graph_build/ports.py"),
)
_GRAPH_BUILD_PATH = Path("src/capicove/graph_build")


def test_pdf_adaptation_design_records_the_deferred_public_contract() -> None:
    """Fail if Task 4D stops making the unlanded-parser boundary explicit."""
    assert _DESIGN_PATH.is_file()
    design = _DESIGN_PATH.read_text(encoding="utf-8")

    assert "Status: Full source-aware locator adaptation remains deferred" in design
    assert "`DocumentParseResult` -> `TextRepresentation`" in design
    assert "Full PDF/OCR/ASR locator support is not implemented" in design
    assert "re-check authorization before reuse" in design
    assert 'adapter_mapping_version: str = "pdf-adaptation-v1"' in design
    assert "parser_result_ref: str" in design
    assert "parser_receipt_ref: str" in design
    assert "representation_sha256 = fingerprint(" in design
    assert '"adapter_mapping_version": self.adapter_mapping_version' in design
    assert "separator rows have `page_index=None`" in design
    assert "origin: TextOrigin" in design
    assert "pdf_representation(document_ref, result, scope, origin: TextOrigin)" in design
    assert "TextRepresentation.origin" in design
    assert "page_number=page_index + 1" in design
    assert "partial_page_indices" in design
    assert "failed_page_indices" in design
    assert "omitted_page_indices" in design
    assert "does not define, import," in design
    assert "or emulate them" in design


def test_text_representation_modules_do_not_import_book_or_document_capabilities() -> None:
    """Keep text-only graph representation code independent of parser ownership."""
    forbidden_prefixes = ("capicove.books", "capicove.documents")

    for path in _REPRESENTATION_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = [module for node in ast.walk(tree) for module in _imported_modules(node)]
        assert not any(module.startswith(forbidden_prefixes) for module in imported_modules), (
            path,
            imported_modules,
        )


def test_only_document_preparation_imports_the_public_parser_dto() -> None:
    """Use the landed owner contract only at the explicit preparation boundary."""
    for path in _GRAPH_BUILD_PATH.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "DocumentParseResult" not in defined_names, path
        if "DocumentParseResult" in imported_names:
            assert path == _GRAPH_BUILD_PATH / "document_preparation.py", path
            owners = [
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and any(alias.name == "DocumentParseResult" for alias in node.names)
            ]
            assert owners == ["capicove.contracts"], owners
        modules = [module for node in ast.walk(tree) for module in _imported_modules(node)]
        assert not any(module.startswith("capicove_processor_paddleocr") for module in modules), (
            path
        )


def _imported_modules(node: ast.AST) -> tuple[str, ...]:
    """Return directly imported module names for one AST node."""
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module,) if node.level == 0 and node.module is not None else ()
    return ()
