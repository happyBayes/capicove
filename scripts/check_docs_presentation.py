"""Compare two MkDocs builds; permit presentation outside the existing shell only."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def check_presentation(before: Path, after: Path) -> int:
    """Require identical routes, document markup, navigation and search records."""
    pages = {path.relative_to(before) for path in before.rglob("*.html")}
    actual = {path.relative_to(after) for path in after.rglob("*.html")}
    if not pages or pages != actual:
        raise ValueError("The HTML page inventory changed or the baseline is empty")
    for path in sorted(pages):
        old = (before / path).read_text(encoding="utf-8")
        new = (after / path).read_text(encoding="utf-8")
        for tag in ("header", "main", "footer"):
            pattern = rf"<{tag}\b[^>]*>.*?</{tag}>"
            previous = re.findall(pattern, old, re.S)
            current = re.findall(pattern, new, re.S)
            if tag == "main" and not previous:
                raise ValueError(f"Missing baseline main content: {path}")
            if previous != current:
                raise ValueError(f"Document or navigation markup changed: {path} ({tag})")
    index = Path("search/search_index.json")
    if (before / index).read_bytes() != (after / index).read_bytes():
        raise ValueError("Search index changed")
    return len(pages)


def main() -> int:
    """Run the visual-only gate against a separately built baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        count = check_presentation(args.before, args.after)
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    print(f"PASS: {count} pages retain identical content, navigation and search index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
