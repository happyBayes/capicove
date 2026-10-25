"""Real local publication and evidence expansion with authored model outputs."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from capicove.contracts import InvocationContext, ProcessingFingerprint
from capicove.processing.video.models import VideoSource
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationRequest
from capicove.retrieval.generations import TextIndexGenerationManifest, TextIndexValidationReport
from capicove.retrieval.ports import TextRetrieverBackend
from capicove.runtime.video_search import LocalVideoSearchRuntime
from tests.integration.processing.video.test_retrieval_preparation import authored_video
from tests.integration.retrieval.test_video_runtime import Encoder
from tests.unit.mcp.test_search import Server
from tests.unit.retrieval.test_video_models import visual_space


__all__ = ["authored_video"]


def _runtime(root: Path, authorize: Mock | None = None) -> LocalVideoSearchRuntime:
    backend = Mock()
    backend.provider_id = "test.lexical"
    backend.generation_lifecycle.mapping_sha256.return_value = "c" * 64
    backend.lexical_retriever.retrieve.return_value = ()

    def validate(manifest: TextIndexGenerationManifest) -> TextIndexValidationReport:
        return TextIndexValidationReport(
            manifest.generation_id,
            manifest.manifest_sha256,
            manifest.generation_id,
            manifest.mapping_sha256,
            manifest.expected_unit_count,
            manifest.expected_unit_count,
            (),
        )

    backend.generation_lifecycle.validate.side_effect = validate
    return LocalVideoSearchRuntime(
        workspace_root=root,
        workspace_ref="test",
        corpus_refs=("test",),
        embedding_space=visual_space(),
        encoder=Encoder(),
        text_backend=cast(TextRetrieverBackend, backend),
        text_index_format="test-v1",
        authorize_evidence=authorize,
    )


def _build(runtime: LocalVideoSearchRuntime, data: bytes, *, identity: str = "authored") -> None:
    fingerprint = ProcessingFingerprint("b" * 64)
    runtime.build(
        VideoRetrievalPreparationRequest(
            VideoSource(data=data),
            f"artifact:{identity}",
            f"evidence:{identity}",
            "representation:asr",
            "run:authored",
            "video-local-exact",
            fingerprint,
            fingerprint,
            fingerprint,
            datetime.now(UTC) + timedelta(seconds=30),
        )
    )


def test_runtime_search_evidence_and_reopen(
    tmp_path: Path, authored_video: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    authorize = Mock(
        side_effect=lambda ctx, evidence, resource: (
            ctx.workspace_id == "local"
            and ctx.actor_id == "capicove-cli"
            and ctx.environment_id == "local"
        )
    )
    with _runtime(tmp_path, authorize) as runtime:
        _build(runtime, authored_video)
        assert runtime.cli(["search", "red", "--target", "video_segment"]) == 0
        hit = json.loads(capsys.readouterr().out)["result"]["hits"][0]
        evidence = hit["evidence_id"]
        assert runtime.cli(["get-evidence", evidence]) == 0
        result = json.loads(capsys.readouterr().out)["result"]
        assert set(result) == {"evidence_id", "metadata"}
        assert result["metadata"][0]["unit_id"] == hit["id"]
        assert result["metadata"][0]["source_sha256"] == sha256(authored_video).hexdigest()
        assert runtime.cli(["get-evidence", evidence, "--section", "preview"]) == 0
        preview = json.loads(capsys.readouterr().out)["result"]["preview"]
        assert preview["resource_ref"].startswith("resource:preview:")
        assert str(tmp_path) not in json.dumps(result)

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; from unittest.mock import Mock; "
                "from tests.integration.retrieval.test_video_runtime_evidence import _runtime; "
                "\nwith _runtime(Path(sys.argv[1]), Mock(return_value=True)) as runtime: "
                "raise SystemExit(runtime.cli(['get-evidence', sys.argv[2]]))"
            ),
            str(tmp_path),
            evidence,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["result"] == result

    with _runtime(tmp_path, authorize) as reopened:
        assert reopened.cli(["get-evidence", evidence]) == 0
        assert json.loads(capsys.readouterr().out)["result"] == result
        authorize.side_effect = None
        authorize.return_value = False
        assert reopened.cli(["get-evidence", evidence]) == 1
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "permission_denied"


def test_missing_or_foreign_authority_denies_before_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_resource = Mock(side_effect=AssertionError("unauthorized content read"))
    monkeypatch.setattr(
        "capicove.runtime.video_search.read_local_preparation_resource", read_resource
    )
    context = InvocationContext(
        "request",
        "trace",
        "foreign",
        "actor",
        "environment",
        datetime.now(UTC) + timedelta(seconds=30),
        permissions=frozenset({"evidence:read"}),
    )
    for authorize in (None, Mock(return_value=False)):
        with _runtime(tmp_path, authorize) as runtime:
            result = runtime.evidence_handler(
                context, {"evidence_id": "resource:sha256:" + "a" * 64}
            )
            assert result.error and result.error.code.value == "permission_denied"
            assert not (tmp_path / "objects").exists()
            if authorize is not None:
                assert authorize.call_count == 1
    read_resource.assert_not_called()


def test_evidence_tampering_and_closed_runtime_fail(
    tmp_path: Path, authored_video: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    with _runtime(tmp_path, Mock(return_value=True)) as runtime:
        _build(runtime, authored_video)
        assert runtime.cli(["search", "red", "--target", "video_segment"]) == 0
        evidence = json.loads(capsys.readouterr().out)["result"]["hits"][0]["evidence_id"]
        (tmp_path / "objects" / evidence.rsplit(":", 1)[1]).write_bytes(b"tampered")
        assert runtime.cli(["get-evidence", evidence]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] is None and payload["error"] is not None
    with pytest.raises(RuntimeError, match="closed"):
        runtime.cli(["get-evidence", evidence])


def test_evidence_is_bound_to_active_generation_and_leases_are_released(
    tmp_path: Path, authored_video: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    with _runtime(tmp_path, Mock(return_value=True)) as runtime:
        _build(runtime, authored_video)
        old = runtime.components.generation_sets.resolve_active(
            workspace_ref="test", profile_ref="video-local-exact"
        )
        assert runtime.cli(["search", "red", "--target", "video_segment"]) == 0
        evidence = json.loads(capsys.readouterr().out)["result"]["hits"][0]["evidence_id"]
        _build(runtime, authored_video, identity="replacement")
        assert runtime.cli(["get-evidence", evidence]) == 1
        assert json.loads(capsys.readouterr().out)["result"] is None
        assert runtime.components.generation_sets.can_evict(old.set_id)


def test_evidence_http_and_mcp_require_explicit_context_and_match_cli(
    tmp_path: Path, authored_video: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    def context() -> InvocationContext:
        return InvocationContext(
            "request",
            "trace",
            "local",
            "capicove-cli",
            "local",
            datetime.now(UTC) + timedelta(seconds=30),
            permissions=frozenset({"evidence:read"}),
        )

    with _runtime(tmp_path, Mock(return_value=True)) as runtime:
        _build(runtime, authored_video)
        assert runtime.cli(["search", "red", "--target", "video_segment"]) == 0
        evidence = json.loads(capsys.readouterr().out)["result"]["hits"][0]["evidence_id"]
        assert runtime.cli(["get-evidence", evidence]) == 0
        expected = json.loads(capsys.readouterr().out)["result"]
        arguments = {"evidence_id": evidence}
        server = Server()
        invoke = runtime.register_evidence_mcp(server, context_factory=context)  # type: ignore[arg-type]
        assert invoke(arguments)["result"] == expected
        with TestClient(runtime.http_app()) as client:
            assert client.post("/v1/evidence", json=arguments).status_code == 404
        with TestClient(runtime.http_app(evidence_context_factory=context)) as client:
            assert client.post("/v1/evidence", json=arguments).json()["result"] == expected
        assert runtime.cli(["get-evidence", evidence, "--section", "scores"]) == 1
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "unavailable"
