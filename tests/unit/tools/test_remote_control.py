"""Remote control Tool contracts and typed rejections, independent of transport."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.contracts import InvocationContext, SideEffect, ToolErrorCode, ToolStatus
from capicove.contracts.remote import AuthorityKey, PrincipalKey
from capicove.runtime.remote_artifacts import ArtifactService
from capicove.runtime.remote_operations import OperationService
from capicove.tools.remote_control import (
    ARTIFACT_SPECS,
    OPERATION_SPECS,
    artifact_handlers,
    operation_handlers,
)


def _authority(subject: str = "alice") -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", "app", "human", subject),
        "workspace-1",
        "grant-1",
        "https://mcp.test/mcp",
    )


def _context(permissions: frozenset[str] = frozenset({"operation:submit"})) -> InvocationContext:
    return InvocationContext(
        "r",
        "t",
        "workspace-1",
        "remote",
        "production",
        datetime.now(UTC) + timedelta(seconds=30),
        permissions,
    )


def test_specs_are_closed_versioned_and_classified() -> None:
    for spec in (*ARTIFACT_SPECS.values(), *OPERATION_SPECS.values()):
        assert spec.contract_version == 1 and not spec.network_access
        assert spec.input_schema["additionalProperties"] is False
        assert len(spec.required_permissions) == 1
    writes = {
        spec.tool_id for spec in ARTIFACT_SPECS.values() if spec.side_effect is SideEffect.WRITE
    }
    assert writes == {
        "capicove.artifact.begin-upload",
        "capicove.artifact.write-chunk",
        "capicove.artifact.abort-upload",
        "capicove.artifact.commit-upload",
        "capicove.artifact.delete",
    }
    assert "capicove.operation.submit" not in {spec.tool_id for spec in OPERATION_SPECS.values()}


def test_handlers_return_typed_rejections_without_effects(tmp_path: Path) -> None:
    service = ArtifactService(tmp_path / "artifacts", application_quota_bytes=10)
    handlers = artifact_handlers(service, _authority())
    begin = handlers["capicove.artifact.begin-upload"]
    over = begin(
        _context(),
        {"size": 11, "sha256": "0" * 64, "media_type": "text/plain", "idempotency_key": "k"},
    )
    assert over.status is ToolStatus.REJECTED and over.error is not None
    assert over.error.code is ToolErrorCode.UNAVAILABLE and over.error.retryable
    data = b"hello"
    good = begin(
        _context(),
        {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "media_type": "text/plain",
            "idempotency_key": "k2",
        },
    )
    assert good.status is ToolStatus.SUCCEEDED and good.result is not None
    upload = good.result["upload_id"]
    bad = handlers["capicove.artifact.write-chunk"](
        _context(), {"upload_id": upload, "index": 0, "data_base64": "not base64!"}
    )
    assert bad.status is ToolStatus.REJECTED and bad.error is not None
    assert bad.error.code is ToolErrorCode.INVALID_INPUT
    written = handlers["capicove.artifact.write-chunk"](
        _context(),
        {"upload_id": upload, "index": 0, "data_base64": base64.b64encode(data).decode()},
    )
    assert written.status is ToolStatus.SUCCEEDED
    foreign = artifact_handlers(service, _authority("bob"))["capicove.artifact.upload-status"](
        _context(), {"upload_id": upload}
    )
    assert foreign.status is ToolStatus.REJECTED and "not_found" in foreign.error.message  # type: ignore[union-attr]
    operations = operation_handlers(OperationService(tmp_path / "ops.sqlite3", ()), _authority())
    missing = operations["capicove.operation.status"](
        _context(), {"operation_id": "operation:" + "0" * 32}
    )
    assert missing.status is ToolStatus.REJECTED
    denied = operations["capicove.operation.cancel"](
        _context(frozenset()), {"operation_id": "operation:" + "0" * 32}
    )
    assert denied.error is not None and denied.error.code is ToolErrorCode.PERMISSION_DENIED
