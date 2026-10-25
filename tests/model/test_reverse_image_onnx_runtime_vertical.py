"""Real synthetic ONNX vertical for the personal reverse-image runtime."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


onnx = pytest.importorskip("onnx", reason="install the model-test dependency group")
pytest.importorskip("onnxruntime", reason="install capicove[retrieval-cpu]")

from capicove.contracts import (  # noqa: E402
    ImageQuery,
    InlineBytesSource,
    InvocationContext,
    PrivacyClass,
    SearchableUnit,
    SearchFailureCode,
    SearchRequest,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    ToolStatus,
)
from capicove.model.bundle import (  # noqa: E402
    VerifiedImageEncoderBundle,
    load_verified_image_encoder_bundle,
)
from capicove.model.manifest import ImageModelSpec  # noqa: E402
from capicove.model.receipt import (  # noqa: E402
    build_receipt,
    verify_model_installation,
    write_receipt,
)
from capicove.processing.image.models import ImageSource  # noqa: E402
from capicove.processing.image.representations import (  # noqa: E402
    ImageRepresentationSource,
)
from capicove.runtime import ReverseImageRuntime, build_reverse_image_runtime  # noqa: E402
from capicove.runtime.image_query import ResolvedImageArtifact  # noqa: E402
from capicove.runtime.retrieval import CpuExecutionBudget  # noqa: E402
from capicove.runtime.retrieval_build import PersonalImageBuildPlan  # noqa: E402
from capicove.runtime.retrieval_qualification import (  # noqa: E402
    LocalPreviewQualificationStore,
)
from capicove.runtime.retrieval_representations import ImageBuildSource  # noqa: E402
from capicove.tools.image_search import (  # noqa: E402
    SEARCH_SPEC,
    StaticSearchAuthorizationFactory,
)


pytestmark = [pytest.mark.model_onnx, pytest.mark.cpu, pytest.mark.e2e]

NOW = datetime.now(UTC)
FUTURE = NOW + timedelta(hours=1)
SOURCE_MANIFEST_SHA256 = "a" * 64


class _NoArtifactResolver:
    def resolve(
        self,
        artifact_ref: str,
        *,
        authorization: object,
        deadline: datetime,
    ) -> ResolvedImageArtifact:
        del artifact_ref, authorization, deadline
        raise AssertionError("the synthetic inline path must not resolve artifacts")


class _ContextFactory:
    def create(self, *, tool_id: str, deadline_cap: datetime) -> InvocationContext:
        assert tool_id == SEARCH_SPEC.tool_id
        return _context(deadline=min(deadline_cap, FUTURE))


class _RegistrationServer:
    callback: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[
        [Callable[[Mapping[str, object]], Mapping[str, object]]],
        Callable[[Mapping[str, object]], Mapping[str, object]],
    ]:
        assert name == "capicove_search_image"
        assert description
        assert input_schema

        def register(
            callback: Callable[[Mapping[str, object]], Mapping[str, object]],
        ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
            self.callback = callback
            return callback

        return register


def _context(*, deadline: datetime = FUTURE) -> InvocationContext:
    return InvocationContext(
        request_id="real-onnx-request",
        trace_id="real-onnx-trace",
        workspace_id="workspace-real-onnx",
        actor_id="actor-real-onnx",
        environment_id="synthetic-real-onnx",
        deadline=deadline,
        permissions=frozenset({"retrieval:search"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, "PNG")
    return buffer.getvalue()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _file_entry(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_bundle(tmp_path: Path, lane: str) -> VerifiedImageEncoderBundle:
    root = tmp_path / "bundles" / lane
    root.mkdir(parents=True)
    revision = ("1" if lane == "instance" else "2") * 40
    model_path = root / "vision.onnx"
    model_input = onnx.helper.make_tensor_value_info(
        "pixel_values",
        onnx.TensorProto.FLOAT,
        ["batch", 3, 2, 2],
    )
    model_output = onnx.helper.make_tensor_value_info(
        "image_embedding",
        onnx.TensorProto.FLOAT,
        ["batch", 3],
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ReduceMean",
                ["pixel_values"],
                ["image_embedding"],
                axes=[2, 3],
                keepdims=0,
            )
        ],
        f"synthetic_{lane}_rgb_spatial_mean",
        [model_input],
        [model_output],
    )
    model = onnx.helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx.save(model, model_path)
    model_payload = model_path.read_bytes()
    license_payload = b"Apache License 2.0\n"
    (root / "LICENSE").write_bytes(license_payload)
    checksums_payload = _canonical_json(
        {
            "files": [
                _file_entry("vision.onnx", model_payload),
                _file_entry("LICENSE", license_payload),
            ],
            "schema_version": 1,
        }
    )
    (root / "checksums.json").write_bytes(checksums_payload)
    manifest = {
        "artifacts": [
            {"role": "onnx-model", **_file_entry("vision.onnx", model_payload)},
            {"role": "checksums", **_file_entry("checksums.json", checksums_payload)},
            {"role": "license", **_file_entry("LICENSE", license_payload)},
        ],
        "bundle_id": f"synthetic-{lane}-rgb-v1",
        "embedding_space": {
            "dimension": 3,
            "distance_metric": "cosine",
            "model_revision": revision,
            "normalization": "l2",
            "preprocessing_version": "rgb-exact-2-v1",
            "provider_id": f"synthetic-{lane}-onnx",
            "query_modalities": ["image"],
            "schema_version": 1,
            "space_id": f"synthetic-{lane}-rgb-v1",
            "target_modalities": ["image"],
        },
        "input": {
            "dtype": "float32",
            "name": "pixel_values",
            "rank": 4,
            "shape": ["batch", 3, 2, 2],
        },
        "license": "apache-2.0",
        "network_behavior": "offline",
        "output": {
            "dimension": 3,
            "dtype": "float32",
            "l2_normalized": True,
            "metric": "cosine",
            "name": "image_embedding",
            "pooling": "none",
            "rank": 2,
            "selection": "only",
            "vector_schema": "image-vector-v1",
        },
        "preprocessing": {
            "apply_exif_orientation": True,
            "channel_mean": [0.0, 0.0, 0.0],
            "channel_std": [1.0, 1.0, 1.0],
            "color_mode": "RGB",
            "crop_size": None,
            "crop_strategy": "none",
            "interpolation": "nearest",
            "resize_size": [2, 2],
            "resize_strategy": "exact",
            "schema_version": 1,
            "tensor_dtype": "float32",
            "tensor_layout": "NCHW",
            "value_range": [0.0, 1.0],
        },
        "repository": f"owner/synthetic-{lane}",
        "revision": revision,
        "runtime": {
            "custom_operators_allowed": False,
            "execution_provider": "CPUExecutionProvider",
            "graph_optimization": "all",
            "provider_fallback_allowed": False,
            "version_specifier": ">=1.20,<2",
        },
        "schema_version": 1,
    }
    (root / "manifest.json").write_bytes(_canonical_json(manifest))
    model_spec = ImageModelSpec(
        key=f"synthetic-{lane}-encoder",
        repository=f"owner/synthetic-{lane}",
        revision=revision,
        license="apache-2.0",
        license_url="https://example.invalid/license",
        capability="image_embedding",
        adapter="onnx-image-embedding",
        architecture="SyntheticRgbMean",
        preprocess_revision="rgb-exact-2-v1",
        default=False,
        restricted=False,
    )
    receipt = build_receipt(
        model_spec,
        root,
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        hf_version="not-applicable",
    )
    write_receipt(root, receipt)
    installation = verify_model_installation(
        root,
        model_spec,
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )
    return load_verified_image_encoder_bundle(installation)


def _build_source(
    payloads: tuple[bytes, ...],
    *,
    runtime: ReverseImageRuntime,
    instance_bundle: VerifiedImageEncoderBundle,
    concept_bundle: VerifiedImageEncoderBundle,
) -> ImageBuildSource:
    return ImageBuildSource(
        tuple(
            ImageRepresentationSource(
                ordinal=ordinal,
                unit=SearchableUnit(
                    unit_id=f"real-onnx-unit-{ordinal}",
                    unit_type=TargetType.IMAGE,
                    source_hash=hashlib.sha256(payload).hexdigest(),
                    artifact_ref=f"real-onnx-artifact-{ordinal}",
                    evidence_ref=f"real-onnx-evidence-{ordinal}",
                    representation_refs=(f"real-onnx-source-{ordinal}",),
                    processing_run_ref="real-onnx-build",
                    processing_version="personal-reverse-image-v1",
                    processing_fingerprint=runtime.processing_fingerprint,
                    embedding_spaces=(
                        instance_bundle.manifest.embedding_space,
                        concept_bundle.manifest.embedding_space,
                    ),
                ),
                source=ImageSource(data=payload, media_type="image/png"),
            )
            for ordinal, payload in enumerate(payloads)
        )
    )


def _request(payload: bytes) -> SearchRequest:
    return SearchRequest(
        query=ImageQuery(InlineBytesSource(payload, "image/png")),
        targets=frozenset({TargetType.IMAGE}),
    )


def _arguments(payload: bytes) -> dict[str, object]:
    return {
        "query": {
            "type": "image",
            "source": {
                "type": "inline_bytes",
                "base64": base64.b64encode(payload).decode("ascii"),
                "media_type": "image/png",
            },
        },
        "targets": ["image"],
    }


def test_real_onnx_runtime_completes_reverse_image_vertical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove real ONNX build, explicit activation, Tool, and MCP parity offline."""
    from capicove.mcp.image_search import register_search_mcp

    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules

    instance_bundle = _write_bundle(tmp_path, "instance")
    concept_bundle = _write_bundle(tmp_path, "concept")
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    authorization_policy = StaticSearchAuthorizationFactory(
        authorization_scope_ref="scope:real-onnx-vertical",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset({"local:real-onnx"}),
    )
    qualification_store = LocalPreviewQualificationStore(
        (tmp_path / "qualifications").resolve(),
        hmac_key=b"real-onnx-preview-qualification-key",
    )

    def deny_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the real synthetic ONNX vertical must remain offline")

    monkeypatch.setattr(socket, "socket", deny_socket)
    runtime = build_reverse_image_runtime(
        workspace_id="workspace-real-onnx",
        generation_root=(tmp_path / "generations").resolve(),
        lifecycle_database_path=(tmp_path / "lifecycle.sqlite3").resolve(),
        instance_bundle=instance_bundle,
        concept_bundle=concept_bundle,
        artifact_resolver=_NoArtifactResolver(),
        local_roots_by_policy={"local-only": {"local:real-onnx": trusted_root}},
        preview_activation_verifier=qualification_store,
        cursor_signing_key=b"r" * 32,
        cpu_budget=CpuExecutionBudget(2, 2, 1),
        authorization_policy=authorization_policy,
        owner_id="real-onnx-worker",
        owner_session_epoch="real-onnx-session",
        clock=lambda: datetime.now(UTC),
    )
    assert runtime._bundle_provisioning_manifest_sha256 is None
    payloads = (
        _png_bytes((250, 10, 10)),
        _png_bytes((10, 250, 10)),
        _png_bytes((10, 10, 250)),
    )
    query_payload = payloads[0]
    context = _context()
    authorization = authorization_policy.create(context)
    try:
        unavailable = runtime.service.search(
            _request(query_payload),
            authorization=authorization,
            deadline=FUTURE,
        )
        assert unavailable.status is SearchStatus.FAILED
        assert unavailable.failure_code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE

        source = _build_source(
            payloads,
            runtime=runtime,
            instance_bundle=instance_bundle,
            concept_bundle=concept_bundle,
        )
        plan = PersonalImageBuildPlan(
            build_id="real-onnx-build",
            activation_set_id="real-onnx-set",
            workspace_id="workspace-real-onnx",
            corpus_ref="synthetic-authored-images",
            batch_size=2,
        )
        built = runtime.administrator.build(source, plan, deadline=FUTURE)
        validated = runtime.administrator.validate(built, deadline=FUTURE)
        published = runtime.administrator.publish(validated, deadline=FUTURE)
        registration = runtime.register_published(
            published,
            published_at=NOW,
            deadline=FUTURE,
        )
        assert runtime.resolve_active_snapshot("personal-lite", deadline=FUTURE) is None

        qualification_id = runtime.issue_preview_qualification(
            registration,
            report_sha256="b" * 64,
            policy_sha256="c" * 64,
            verification_receipt_sha256="d" * 64,
            verified_at=NOW,
            deadline=FUTURE,
        )
        assert runtime.resolve_active_snapshot("personal-lite", deadline=FUTURE) is None

        activated = runtime.activate_preview(
            registration,
            qualification_id=qualification_id,
            expected_previous_snapshot=None,
            activated_at=NOW,
            deadline=FUTURE,
        )
        assert activated.activation_set_id == published.manifest.activation_set_id
        assert activated.qualification_id == qualification_id
        assert activated.qualification_tier == "preview"
        assert runtime.resolve_active_snapshot("personal-lite", deadline=FUTURE) == activated

        direct = runtime.service.search(
            _request(query_payload),
            authorization=authorization,
            deadline=FUTURE,
        )
        tool = runtime.tool_handler(context, _arguments(query_payload))
        server = _RegistrationServer()
        callback = register_search_mcp(
            server,
            runtime.tool_handler,
            context_factory=_ContextFactory(),
            clock=lambda: datetime.now(UTC),
        )
        mcp = callback(_arguments(query_payload))

        assert direct.status is SearchStatus.SUCCEEDED
        assert direct.hits[0].id == "real-onnx-unit-0"
        assert SearchWarningCode.RETRIEVAL_PREVIEW in direct.warning_codes
        assert tool.status is ToolStatus.SUCCEEDED
        assert tool.result is not None
        tool_hits = tool.result["hits"]
        assert isinstance(tool_hits, list)
        assert [hit["id"] for hit in tool_hits] == [hit.id for hit in direct.hits]
        assert tool.result["warning_codes"] == [warning.value for warning in direct.warning_codes]
        assert mcp["status"] == "succeeded"
        mcp_result = mcp["result"]
        assert isinstance(mcp_result, dict)
        assert mcp_result == tool.result
        assert mcp_result["warning_codes"] == ["retrieval_preview"]
    finally:
        runtime.close()


def test_image_cli_and_real_stdio_reopen_activated_index(tmp_path: Path) -> None:
    """Exercise both installed entry points against real synthetic ONNX models."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from tests.e2e.test_reverse_image_runtime import _write_model_provisioning

    root = tmp_path.resolve()
    bundles = (_write_bundle(root, "instance"), _write_bundle(root, "concept"))
    provisioning = _write_model_provisioning(root, bundles)
    qualification_key = b"q" * 32
    sources = root / "sources"
    sources.mkdir()
    store = LocalPreviewQualificationStore(root / "qualifications", hmac_key=qualification_key)
    runtime = build_reverse_image_runtime(
        workspace_id="workspace-real-onnx",
        generation_root=root / "generations",
        lifecycle_database_path=root / "lifecycle.sqlite3",
        model_provisioning=provisioning,
        artifact_resolver=_NoArtifactResolver(),
        local_roots_by_policy={"local-only": {"photos": sources}},
        preview_activation_verifier=store,
        cursor_signing_key=b"r" * 32,
        cpu_budget=CpuExecutionBudget(2, 2, 1),
        authorization_policy=StaticSearchAuthorizationFactory(
            authorization_scope_ref="local-image:workspace-real-onnx",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset({"photos"}),
        ),
        owner_id="fixture",
        owner_session_epoch="fixture-session",
    )
    payload = _png_bytes((250, 10, 10))
    with runtime:
        source = _build_source(
            (payload,), runtime=runtime, instance_bundle=bundles[0], concept_bundle=bundles[1]
        )
        plan = PersonalImageBuildPlan(
            build_id="external-build",
            activation_set_id="external-set",
            workspace_id="workspace-real-onnx",
            corpus_ref="synthetic-authored-images",
            batch_size=1,
        )
        built = runtime.administrator.build(source, plan, deadline=FUTURE)
        validated = runtime.administrator.validate(built, deadline=FUTURE)
        published = runtime.administrator.publish(validated, deadline=FUTURE)
        registration = runtime.register_published(published, published_at=NOW, deadline=FUTURE)
        qualification_id = runtime.issue_preview_qualification(
            registration,
            report_sha256="b" * 64,
            policy_sha256="c" * 64,
            verification_receipt_sha256="d" * 64,
            verified_at=NOW,
            deadline=FUTURE,
        )
        runtime.activate_preview(
            registration,
            qualification_id=qualification_id,
            expected_previous_snapshot=None,
            activated_at=NOW,
            deadline=FUTURE,
        )
    for name, key in (("cursor.key", b"r" * 32), ("qualification.key", qualification_key)):
        (root / name).write_bytes(key)
        (root / name).chmod(0o600)
    config = {
        "schema_version": 1,
        "workspace_id": "workspace-real-onnx",
        "actor_id": "local-owner",
        "generation_root": str(root / "generations"),
        "lifecycle_database_path": str(root / "lifecycle.sqlite3"),
        "model_root": str(root),
        "provisioning_manifest": str(provisioning.manifest_relative_path),
        "provisioning_sha256": provisioning.manifest_sha256,
        "source_roots": {"photos": str(sources)},
        "qualification_root": str(root / "qualifications"),
        "qualification_key_file": str(root / "qualification.key"),
        "cursor_key_file": str(root / "cursor.key"),
        "permissions": ["retrieval:search"],
        "cpu_threads": 2,
    }
    config_path = root / "host.json"
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)
    query_path = root / "query.json"
    query_path.write_text(json.dumps(_arguments(payload)))
    command = [sys.executable, "-m", "capicove", "image"]
    completed = subprocess.run(
        [*command, "search", "--config", str(config_path), "--arguments", str(query_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == "succeeded"
    assert envelope["result"]["hits"][0]["id"] == "real-onnx-unit-0"

    local_image = sources / "query.png"
    local_image.write_bytes(payload)
    local_query = {
        "query": {"type": "image", "source": {"type": "local_path", "path": str(local_image)}},
        "targets": ["image"],
        "data_policy_ref": "local-only",
    }
    query_path.write_text(json.dumps(local_query))
    local_completed = subprocess.run(
        [*command, "search", "--config", str(config_path), "--arguments", str(query_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert local_completed.returncode == 0, local_completed.stderr
    assert json.loads(local_completed.stdout)["result"] == envelope["result"]
    outside = root / "private-outside.png"
    outside.write_bytes(payload)
    denied_query = {
        "query": {"type": "image", "source": {"type": "local_path", "path": str(outside)}},
        "targets": ["image"],
    }
    query_path.write_text(json.dumps(denied_query))
    denied = subprocess.run(
        [*command, "search", "--config", str(config_path), "--arguments", str(query_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert denied.returncode == 1
    assert json.loads(denied.stdout)["status"] != "succeeded"
    assert "private-outside" not in denied.stdout + denied.stderr

    async def invoke() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "capicove", "image", "serve", "--config", str(config_path)],
        )
        async with asyncio.timeout(30), stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert [tool.name for tool in listed.tools] == ["capicove_search_image"]
                result = await session.call_tool("capicove_search_image", _arguments(payload))
                assert not result.isError
                block = result.content[0]
                assert block.type == "text"
                document = json.loads(block.text)
                assert document["status"] == "succeeded"
                assert document["result"] == envelope["result"]
                rejected = await session.call_tool("capicove_search_image", denied_query)
                block = rejected.content[0]
                assert block.type == "text"
                assert json.loads(block.text)["status"] != "succeeded"
                assert "private-outside" not in block.text

    asyncio.run(invoke())
    config["provisioning_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    tampered = subprocess.run(
        [*command, "search", "--config", str(config_path), "--arguments", str(query_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert tampered.returncode == 2
    assert tampered.stdout == ""
