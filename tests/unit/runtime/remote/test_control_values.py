"""Remote artifact, operation and approval handles reject malformed values."""

import pytest

from capicove.contracts.remote import ApprovalRef, ArtifactRef, OperationRef


def test_artifact_references_are_opaque_and_bounded() -> None:
    good = ArtifactRef("artifact:" + "a" * 32, "b" * 64, 1, "text/plain", 1, ())
    assert good.size == 1
    for changes in (
        {"artifact_id": "artifact:../x"},
        {"artifact_id": "upload:" + "a" * 32},
        {"sha256": "B" * 64},
        {"size": 0},
        {"size": True},
        {"version": 0},
        {"lineage": ["list"]},
        {"media_type": ""},
    ):
        values = {
            "artifact_id": good.artifact_id,
            "sha256": good.sha256,
            "size": good.size,
            "media_type": good.media_type,
            "version": good.version,
            "lineage": good.lineage,
            **changes,
        }
        with pytest.raises(ValueError):
            ArtifactRef(**values)  # type: ignore[arg-type]


def test_operation_and_approval_handles_validate_identity() -> None:
    OperationRef("operation:" + "1" * 32, "queued", "capicove.notes.append", 1, "c" * 64)
    with pytest.raises(ValueError):
        OperationRef("job:1", "queued", "capicove.notes.append", 1, "c" * 64)
    with pytest.raises(ValueError):
        OperationRef("operation:1", "queued", "capicove.notes.append", 0, "c" * 64)
    with pytest.raises(ValueError):
        OperationRef("operation:1", "queued", "capicove.notes.append", 1, "short")
    ApprovalRef("approval:" + "1" * 32, "pending", "d" * 64, None, None)
    with pytest.raises(ValueError):
        ApprovalRef("plan:1", "pending", "d" * 64, None, None)
    with pytest.raises(ValueError):
        ApprovalRef("approval:1", "pending", "short", None, None)
