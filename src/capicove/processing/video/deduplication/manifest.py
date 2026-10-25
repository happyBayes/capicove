"""Upload manifest projection for video deduplication results."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from capicove.processing.video.deduplication.models import (
    DuplicateActionResult,
    ManifestItem,
    VideoDeduplicationResult,
)


MANIFEST_FIELDS = (
    "path",
    "upload_mode",
    "duplicate_group_id",
    "keeper_path",
    "size_bytes",
    "moved_to",
)


def build_manifest_items(
    result: VideoDeduplicationResult,
    actions: DuplicateActionResult | None = None,
) -> tuple[ManifestItem, ...]:
    """Project every readable file into one stable upload decision."""
    group_by_path = {path: group for group in result.duplicate_groups for path in group.all_paths}
    eliminated = {path for group in result.duplicate_groups for path in group.eliminated}
    moved = {record.source: record.destination for record in (actions.moves if actions else ())}
    return tuple(
        ManifestItem(
            path=record.path,
            upload_mode="skip" if record.path in eliminated else "copy",
            duplicate_group_id=(
                group_by_path[record.path].group_id if record.path in group_by_path else None
            ),
            keeper_path=(
                group_by_path[record.path].keeper if record.path in group_by_path else None
            ),
            size_bytes=record.size_bytes,
            moved_to=moved.get(record.path),
        )
        for record in result.files
    )


def _item_dict(item: ManifestItem) -> dict[str, object]:
    values = asdict(item)
    for name in ("path", "keeper_path", "moved_to"):
        value = values[name]
        values[name] = str(value) if value is not None else ""
    values["duplicate_group_id"] = values["duplicate_group_id"] or ""
    return values


def write_deduplication_manifest(
    path: Path,
    result: VideoDeduplicationResult,
    actions: DuplicateActionResult | None = None,
) -> None:
    """Atomically write a JSON or CSV upload manifest selected by suffix."""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".csv"}:
        raise ValueError("manifest path must end in .json or .csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [_item_dict(item) for item in build_manifest_items(result, actions)]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            if suffix == ".json":
                json.dump(
                    {"schema_version": 1, "mode": result.mode.value, "items": items},
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                stream.write("\n")
            else:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(items)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
