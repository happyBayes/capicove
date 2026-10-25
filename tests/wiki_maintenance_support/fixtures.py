"""Temporary byte-authored evidence and page fixtures, without private data."""

from dataclasses import dataclass
from pathlib import Path

from tests.wiki_maintenance_support.oracle import snapshot


@dataclass(frozen=True)
class WikiCase:
    """Original evidence and observable filesystem oracle."""

    root: Path
    source_path: Path
    source_bytes: bytes
    expected_text: str
    expected_spans: tuple[tuple[int, int, str], ...]
    manual_path: Path

    def snapshot(self) -> dict[str, bytes]:
        """Read complete fixture state."""
        return snapshot(self.root)

    def page_bytes(self) -> dict[str, bytes]:
        """Read pages independently of the application index."""
        return snapshot(self.root / "wiki" / "pages")


def make_case(root: Path) -> WikiCase:
    """Create a valid legacy Wiki plus three independently authored pages."""
    from datetime import date

    from capicove.wiki import WikiService

    WikiService(root).initialize(created_on=date(2026, 10, 22))
    source = root / "source.md"
    raw = b"# Sample\r\nAlpha evidence.\r\n"
    source.write_bytes(raw)
    pages = root / "wiki" / "pages"
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    (root / "wiki" / "raw" / (digest + ".md")).write_bytes(raw)
    for index, name in enumerate(("concept-one", "concept-two", "manual")):
        body = "Manual note.\n" if name == "manual" else "Synthetic fixture.\n"
        (pages / (name + ".md")).write_text(
            f"---\nid: page:FIXTURE000{index}\ntitle: {name}\nkind: concept\n"
            f"created: 2026-10-22\nupdated: 2026-10-22\nsources: [source:sha256:{digest}]\n---\n\n{body}"
        )
    manual = pages / "manual.md"
    return WikiCase(root, source, raw, "# Sample\nAlpha evidence.\n", ((9, 14, "Alpha"),), manual)
