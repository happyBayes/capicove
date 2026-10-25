"""End-to-end authored-media acceptance for the personal reverse-image runtime."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from capicove.contracts import (
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
    ToolErrorCode,
    ToolStatus,
)
from capicove.model.bundle import (
    VerifiedImageEncoderBundle,
    load_verified_image_encoder_bundle,
)
from capicove.model.manifest import ImageModelSpec
from capicove.model.provisioning import load_reverse_image_model_provisioning
from capicove.model.receipt import (
    build_receipt,
    verify_model_installation,
    write_receipt,
)
from capicove.processing.image.models import ImageSource
from capicove.processing.image.representations import ImageRepresentationSource
from capicove.runtime import (
    ReverseImageRuntime,
    build_reverse_image_runtime,
)
from capicove.runtime.image_query import ResolvedImageArtifact
from capicove.runtime.retrieval import CpuExecutionBudget
from capicove.runtime.retrieval_build import PersonalImageBuildPlan
from capicove.runtime.retrieval_qualification import LocalPreviewQualificationStore
from capicove.runtime.retrieval_representations import ImageBuildSource
from capicove.runtime.reverse_image import ReverseImageModelProvisioning
from capicove.tools.image_search import SEARCH_SPEC, StaticSearchAuthorizationFactory


NOW = datetime.now(UTC)
FUTURE = NOW + timedelta(hours=1)
SOURCE_MANIFEST_SHA256 = "a" * 64


@dataclass(frozen=True, slots=True)
class _ValueInfo:
    name: str
    shape: tuple[int | str | None, ...]
    type: str = "tensor(float)"


class _SessionOptions:
    intra_op_num_threads = 0
    inter_op_num_threads = 0
    use_per_session_threads = False
    use_deterministic_compute = False
    execution_mode: object | None = None
    graph_optimization_level: object | None = None

    def add_session_config_entry(self, _name: str, _value: str) -> None:
        pass


class _FakeSession:
    def __init__(self, runtime: _FakeOnnxRuntime) -> None:
        self._runtime = runtime

    def get_inputs(self) -> Sequence[_ValueInfo]:
        return (_ValueInfo("pixel_values", ("batch", 3, 2, 2)),)

    def get_outputs(self) -> Sequence[_ValueInfo]:
        return (_ValueInfo("image_embedding", ("batch", 3)),)

    def get_providers(self) -> Sequence[str]:
        return ("CPUExecutionProvider",)

    def run(
        self,
        output_names: Sequence[str],
        feeds: Mapping[str, np.ndarray],
    ) -> Sequence[object]:
        assert tuple(output_names) == ("image_embedding",)
        tensor = np.asarray(feeds["pixel_values"], dtype=np.float32)
        self._runtime.run_count += 1
        return (np.asarray(tensor.mean(axis=(2, 3)), dtype=np.float32),)


class _ExecutionMode:
    ORT_SEQUENTIAL = object()


class _GraphOptimizationLevel:
    ORT_DISABLE_ALL = object()
    ORT_ENABLE_BASIC = object()
    ORT_ENABLE_EXTENDED = object()
    ORT_ENABLE_ALL = object()


class _FakeOnnxRuntime(ModuleType):
    __version__ = "1.20.0"
    ExecutionMode = _ExecutionMode
    GraphOptimizationLevel = _GraphOptimizationLevel
    SessionOptions = _SessionOptions

    def __init__(self) -> None:
        super().__init__("onnxruntime")
        self.session_count = 0
        self.run_count = 0

    def get_available_providers(self) -> Sequence[str]:
        return ("CPUExecutionProvider",)

    def InferenceSession(
        self,
        _model_path: str,
        *,
        sess_options: object,
        providers: Sequence[str],
    ) -> _FakeSession:
        assert isinstance(sess_options, _SessionOptions)
        assert tuple(providers) == ("CPUExecutionProvider",)
        self.session_count += 1
        return _FakeSession(self)


class _NoArtifactResolver:
    def resolve(
        self,
        artifact_ref: str,
        *,
        authorization: object,
        deadline: datetime,
    ) -> ResolvedImageArtifact:
        del artifact_ref, authorization, deadline
        raise AssertionError("the authored inline acceptance path must not resolve artifacts")


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
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="authored-e2e",
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
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
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
    model_payload = f"authored fake ONNX graph for {lane}\n".encode()
    license_payload = b"Apache License 2.0\n"
    (root / "vision.onnx").write_bytes(model_payload)
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
    model = ImageModelSpec(
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
        model,
        root,
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        hf_version="not-applicable",
    )
    write_receipt(root, receipt)
    installation = verify_model_installation(
        root,
        model,
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )
    return load_verified_image_encoder_bundle(installation)


def _write_model_provisioning(
    tmp_path: Path,
    bundles: tuple[VerifiedImageEncoderBundle, VerifiedImageEncoderBundle],
) -> ReverseImageModelProvisioning:
    root = tmp_path.resolve()
    lock_path = root / "locks" / "reverse-image-models.toml"
    lock_path.parent.mkdir()
    lock_rows = ["manifest_version = 1\n"]
    for bundle in bundles:
        model = bundle.installation.model
        lock_rows.append(
            "\n".join(
                (
                    "",
                    "[[models]]",
                    f'key = "{model.key}"',
                    f'repository = "{model.repository}"',
                    f'revision = "{model.revision}"',
                    f'license = "{model.license}"',
                    f'license_url = "{model.license_url}"',
                    f'capability = "{model.capability}"',
                    f'adapter = "{model.adapter}"',
                    f'architecture = "{model.architecture}"',
                    f'preprocess_revision = "{model.preprocess_revision}"',
                    "default = false",
                    "restricted = false",
                    "",
                )
            )
        )
    lock_path.write_text("".join(lock_rows), encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    entries = []
    for lane, bundle in zip(("instance", "concept"), bundles, strict=True):
        installation = bundle.installation.path
        receipt = build_receipt(
            bundle.installation.model,
            installation,
            manifest_sha256=lock_sha256,
            hf_version="not-applicable",
        )
        receipt_path = write_receipt(installation, receipt)
        receipt_payload = receipt_path.read_bytes()
        entries.append(
            {
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "installation_relative_path": f"bundles/{lane}",
                "lane": lane,
                "model_key": bundle.installation.model.key,
                "receipt_relative_path": (f"bundles/{lane}/capicove-model-receipt.json"),
                "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
            }
        )
    document = {
        "entries": entries,
        "model_lock_relative_path": "locks/reverse-image-models.toml",
        "model_lock_sha256": lock_sha256,
        "provisioning_id": "authored-personal-reverse-image-v1",
        "schema_version": 1,
    }
    relative_path = PurePosixPath("reverse-image-provisioning.json")
    payload = _canonical_json(document)
    root.joinpath(*relative_path.parts).write_bytes(payload)
    return ReverseImageModelProvisioning(
        root=root,
        manifest_relative_path=relative_path,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _build_source(
    payloads: tuple[bytes, ...],
    *,
    runtime: ReverseImageRuntime,
    instance_bundle: VerifiedImageEncoderBundle,
    concept_bundle: VerifiedImageEncoderBundle,
) -> ImageBuildSource:
    sources = []
    for ordinal, payload in enumerate(payloads):
        sources.append(
            ImageRepresentationSource(
                ordinal=ordinal,
                unit=SearchableUnit(
                    unit_id=f"unit-{ordinal}",
                    unit_type=TargetType.IMAGE,
                    source_hash=hashlib.sha256(payload).hexdigest(),
                    artifact_ref=f"artifact-{ordinal}",
                    evidence_ref=f"evidence-{ordinal}",
                    representation_refs=(f"source-{ordinal}",),
                    processing_run_ref="authored-e2e-build",
                    processing_version="personal-reverse-image-v1",
                    processing_fingerprint=runtime.processing_fingerprint,
                    embedding_spaces=(
                        instance_bundle.manifest.embedding_space,
                        concept_bundle.manifest.embedding_space,
                    ),
                ),
                source=ImageSource(data=payload, media_type="image/png"),
            )
        )
    return ImageBuildSource(tuple(sources))


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


def test_reverse_image_runtime_requires_explicit_preview_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise build, activation, Python, Tool, and MCP on authored PNG bytes."""
    from capicove.mcp.image_search import register_search_mcp
    from capicove.runtime.reverse_image import _processing_fingerprint

    fake_onnx = _FakeOnnxRuntime()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    unprovisioned_bundles = (
        _write_bundle(tmp_path, "instance"),
        _write_bundle(tmp_path, "concept"),
    )
    model_provisioning = _write_model_provisioning(tmp_path, unprovisioned_bundles)
    provisioned = load_reverse_image_model_provisioning(
        model_provisioning.root,
        model_provisioning.manifest_relative_path,
        expected_manifest_sha256=model_provisioning.manifest_sha256,
    )
    instance_bundle = provisioned.instance_bundle
    concept_bundle = provisioned.concept_bundle
    authorization_policy = StaticSearchAuthorizationFactory(
        authorization_scope_ref="scope:authored-e2e",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset({"local:authored-e2e"}),
    )
    qualification_store = LocalPreviewQualificationStore(
        (tmp_path / "qualifications").resolve(),
        hmac_key=b"preview-qualification-key-for-authored-e2e",
    )
    runtime = build_reverse_image_runtime(
        workspace_id="workspace-1",
        generation_root=(tmp_path / "generations").resolve(),
        lifecycle_database_path=(tmp_path / "lifecycle.sqlite3").resolve(),
        model_provisioning=model_provisioning,
        artifact_resolver=_NoArtifactResolver(),
        local_roots_by_policy={"local-only": {"local:authored-e2e": trusted_root}},
        preview_activation_verifier=qualification_store,
        cursor_signing_key=b"k" * 32,
        cpu_budget=CpuExecutionBudget(2, 2, 1),
        authorization_policy=authorization_policy,
        owner_id="worker-1",
        owner_session_epoch="session-1",
        clock=lambda: datetime.now(UTC),
    )
    assert (
        runtime._bundle_provisioning_manifest_sha256
        == provisioned.manifest.source_sha256
        == model_provisioning.manifest_sha256
    )
    assert runtime.processing_fingerprint == _processing_fingerprint(
        unprovisioned_bundles[0],
        unprovisioned_bundles[1],
    )
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
            build_id="authored-e2e-build",
            activation_set_id="authored-e2e-set",
            workspace_id="workspace-1",
            corpus_ref="authored-photos",
            batch_size=2,
        )
        built = runtime.administrator.build(source, plan, deadline=FUTURE)
        validated = runtime.administrator.validate(built, deadline=FUTURE)
        published = runtime.administrator.publish(validated, deadline=FUTURE)

        still_unavailable = runtime.service.search(
            _request(query_payload),
            authorization=authorization,
            deadline=FUTURE,
        )
        assert still_unavailable.status is SearchStatus.FAILED
        assert still_unavailable.failure_code is SearchFailureCode.INDEX_GENERATION_UNAVAILABLE

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
        forged_arguments = _arguments(query_payload)
        forged_arguments["data_policy_ref"] = "request-granted-policy"
        forged_policy = runtime.tool_handler(context, forged_arguments)
        server = _RegistrationServer()
        callback = register_search_mcp(
            server,
            runtime.tool_handler,
            context_factory=_ContextFactory(),
            clock=lambda: datetime.now(UTC),
        )
        mcp = callback(_arguments(query_payload))

        assert direct.status is SearchStatus.SUCCEEDED
        assert direct.execution_summary is not None
        assert direct.execution_summary.resolved_profile_ref == "personal-lite"
        assert direct.execution_summary.activation_tier == "preview"
        assert SearchWarningCode.RETRIEVAL_PREVIEW in direct.warning_codes
        assert direct.hits[0].id == "unit-0"
        assert tool.status is ToolStatus.SUCCEEDED
        assert tool.result is not None
        tool_hits = tool.result["hits"]
        assert isinstance(tool_hits, list)
        assert tool_hits[0]["id"] == direct.hits[0].id
        assert forged_policy.status is ToolStatus.REJECTED
        assert forged_policy.error is not None
        assert forged_policy.error.code is ToolErrorCode.PERMISSION_DENIED
        assert mcp["status"] == "succeeded"
        mcp_result = mcp["result"]
        assert isinstance(mcp_result, dict)
        assert mcp_result == tool.result
        assert mcp_result["warning_codes"] == ["retrieval_preview"]
        assert fake_onnx.session_count == 4
        assert fake_onnx.run_count >= 8
    finally:
        runtime.close()


def test_provisioned_receipt_drift_fails_build_before_runtime_state_or_media_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unprovisioned_bundles = (
        _write_bundle(tmp_path, "instance"),
        _write_bundle(tmp_path, "concept"),
    )
    model_provisioning = _write_model_provisioning(tmp_path, unprovisioned_bundles)
    provisioned = load_reverse_image_model_provisioning(
        model_provisioning.root,
        model_provisioning.manifest_relative_path,
        expected_manifest_sha256=model_provisioning.manifest_sha256,
    )
    receipt_path = tmp_path / "bundles" / "instance" / "capicove-model-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert isinstance(receipt, dict)
    receipt["hf_version"] = "tampered-after-provisioning"
    receipt_path.write_bytes(_canonical_json(receipt))
    decoder_calls = 0

    def forbidden_decoder(source: ImageSource) -> object:
        del source
        nonlocal decoder_calls
        decoder_calls += 1
        raise AssertionError("invalid model provisioning must fail before media decode")

    import capicove.runtime.reverse_image as reverse_image_runtime

    monkeypatch.setattr(reverse_image_runtime, "load_image", forbidden_decoder)
    monkeypatch.setattr(
        reverse_image_runtime,
        "load_reverse_image_model_provisioning",
        lambda *_args, **_kwargs: provisioned,
    )
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    authorization_policy = StaticSearchAuthorizationFactory(
        authorization_scope_ref="scope:authored-e2e",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset({"local:authored-e2e"}),
    )
    generation_root = (tmp_path / "generations").resolve()
    lifecycle_path = (tmp_path / "lifecycle.sqlite3").resolve()

    with pytest.raises(ValueError, match="receipt digest"):
        build_reverse_image_runtime(
            workspace_id="workspace-1",
            generation_root=generation_root,
            lifecycle_database_path=lifecycle_path,
            model_provisioning=model_provisioning,
            artifact_resolver=_NoArtifactResolver(),
            local_roots_by_policy={"local-only": {"local:authored-e2e": trusted_root}},
            preview_activation_verifier=LocalPreviewQualificationStore(
                (tmp_path / "qualifications").resolve(),
                hmac_key=b"preview-qualification-key-for-authored-e2e",
            ),
            cursor_signing_key=b"k" * 32,
            cpu_budget=CpuExecutionBudget(2, 2, 1),
            authorization_policy=authorization_policy,
            owner_id="worker-1",
            owner_session_epoch="session-1",
        )

    assert decoder_calls == 0
    assert not generation_root.exists()
    assert not lifecycle_path.exists()
