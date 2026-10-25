"""Read real files without invoking production serialization or rendering."""

from pathlib import Path


def snapshot(root: Path) -> dict[str, bytes]:
    """Capture exact bytes, rejecting symlinks instead of hiding extra files."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssertionError("unexpected symlink")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def assert_unchanged(root: Path, before: dict[str, bytes]) -> None:
    """Check both file inventory and bytes independently of product hashes."""
    assert snapshot(root) == before
