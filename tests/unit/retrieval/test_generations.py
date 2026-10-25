"""Unit tests for immutable personal-lite generation manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import PurePosixPath

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    IndexGeneration,
    IndexParametersRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.image_generations import (
    TARGET_REPRESENTATION_MANIFEST_ROLE,
    GenerationManifestError,
    GenerationSetManifest,
    LaneGeneration,
    PayloadFileRef,
    UnitCatalogManifest,
    decode_generation_set_manifest,
    encode_generation_set_manifest,
)
from capicove.retrieval.search_models import RetrievalLane


NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _payload(role: str) -> PayloadFileRef:
    content = role.encode()
    return PayloadFileRef(
        relative_path=PurePosixPath("payloads") / f"{role.replace(':', '-')}.bin",
        sha256=_sha(content),
        size_bytes=len(content),
        role=role,
    )


def _embedding_space(space_id: str, model_revision: str) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=space_id,
        provider_id="local-test-provider",
        model_revision=model_revision,
        preprocessing_version="rgb-224-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="opencv-phash-64-v1",
        provider_id="opencv-phash",
        provider_version="4.12.0",
        preprocessing_version="pillow-luma-opencv-area32-dct8-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def generation_set_manifest() -> GenerationSetManifest:
    """Return one complete personal-lite generation set."""
    payloads = tuple(
        _payload(role)
        for role in (
            TARGET_REPRESENTATION_MANIFEST_ROLE,
            "catalog",
            "validation:catalog",
            "perceptual-parameters",
            "perceptual-shard",
            "validation:perceptual",
            "instance-parameters",
            "instance-shard",
            "validation:instance",
            "concept-parameters",
            "concept-shard",
            "validation:concept",
        )
    )
    payload_by_role = {payload.role: payload for payload in payloads}
    fingerprint = ProcessingFingerprint(SHA_A)

    def generation(
        lane: RetrievalLane,
        *,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...] = (),
        hash_spaces: tuple[HashSpaceRef, ...] = (),
    ) -> LaneGeneration:
        parameter_role = f"{lane.value}-parameters"
        return LaneGeneration(
            lane=lane,
            generation=IndexGeneration(
                generation_id=f"generation-{lane.value}-1",
                activation_set_id="set-1",
                workspace_id="workspace-1",
                corpus_ref="personal-photos",
                target_type=TargetType.IMAGE,
                unit_manifest_sha256=SHA_B,
                embedding_spaces=embedding_spaces,
                hash_spaces=hash_spaces,
                retriever_id=f"capicove.{lane.value}-exact",
                index_format=f"{lane.value}-flat-v1",
                index_parameters_ref=IndexParametersRef(
                    parameter_schema=f"{lane.value}-flat-v1",
                    payload_ref=parameter_role,
                    sha256=payload_by_role[parameter_role].sha256,
                ),
                shard_refs=(f"{lane.value}-shard",),
                calibration_profile_ref=f"{lane.value}-personal-lite-v1",
                build_processing_fingerprint=fingerprint,
                created_at=NOW,
                validation_report_ref=f"validation:{lane.value}",
            ),
            target_bundle_manifest_sha256=(None if lane is RetrievalLane.PERCEPTUAL else SHA_C),
        )

    return GenerationSetManifest(
        activation_set_id="set-1",
        workspace_id="workspace-1",
        corpus_ref="personal-photos",
        retrieval_profile_ref="personal-lite",
        target_type=TargetType.IMAGE,
        unit_manifest_sha256=SHA_B,
        frozen_unit_manifest_proof_sha256=SHA_C,
        target_representation_manifest_sha256=(
            payload_by_role[TARGET_REPRESENTATION_MANIFEST_ROLE].sha256
        ),
        processing_fingerprint=fingerprint,
        unit_catalog=UnitCatalogManifest(
            catalog_id="catalog-1",
            unit_count=2,
            unit_manifest_sha256=SHA_B,
            catalog_format="jsonl-v1",
            payload_ref="catalog",
            validation_report_ref="validation:catalog",
        ),
        lanes=(
            generation(
                RetrievalLane.CONCEPT, embedding_spaces=(_embedding_space("concept", "c1"),)
            ),
            generation(RetrievalLane.PERCEPTUAL, hash_spaces=(_hash_space(),)),
            generation(
                RetrievalLane.INSTANCE, embedding_spaces=(_embedding_space("instance", "i1"),)
            ),
        ),
        payloads=payloads,
        created_at=NOW,
    )


def test_generation_manifest_round_trips_as_canonical_json() -> None:
    original = generation_set_manifest()

    encoded = encode_generation_set_manifest(original)
    decoded = decode_generation_set_manifest(encoded)

    assert decoded == original
    assert encoded == encode_generation_set_manifest(decoded)
    assert encoded.endswith(b"\n")
    assert (
        encoded
        == json.dumps(
            json.loads(encoded),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


@pytest.mark.parametrize(
    "path",
    (
        "/absolute/index.npy",
        "../escape.npy",
        "lane/../../escape",
        "C:/windows/index.npy",
        "C:\\windows\\index.npy",
        "//server/share/index.npy",
        "lane\\index.npy",
        "lane/CON",
        "lane/index.npy\x00suffix",
    ),
)
def test_payload_path_must_be_safe_and_relative(path: str) -> None:
    with pytest.raises(ValueError, match=r"relative|safe|reserved"):
        PayloadFileRef(
            relative_path=PurePosixPath(path),
            sha256=SHA_A,
            size_bytes=16,
            role="instance-vectors",
        )


@pytest.mark.parametrize("field", ("state", "active", "unknown"))
def test_decoder_rejects_unknown_and_mutable_lifecycle_fields(field: str) -> None:
    document = json.loads(encode_generation_set_manifest(generation_set_manifest()))
    document[field] = True

    with pytest.raises(GenerationManifestError, match="unknown field"):
        decode_generation_set_manifest(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )


def test_decoder_rejects_schema_version_and_noncanonical_encoding() -> None:
    document = json.loads(encode_generation_set_manifest(generation_set_manifest()))
    document["schema_version"] = 2
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    with pytest.raises(GenerationManifestError, match="schema_version"):
        decode_generation_set_manifest(encoded)

    noncanonical = json.dumps(
        json.loads(encode_generation_set_manifest(generation_set_manifest())),
        indent=2,
    ).encode()
    with pytest.raises(GenerationManifestError, match="canonical"):
        decode_generation_set_manifest(noncanonical)


def test_manifest_rejects_incomplete_or_mismatched_lane_identity() -> None:
    manifest = generation_set_manifest()
    instance = next(item for item in manifest.lanes if item.lane is RetrievalLane.INSTANCE)
    mismatched = replace(
        instance,
        generation=replace(instance.generation, workspace_id="another-workspace"),
    )

    with pytest.raises(ValueError, match="workspace"):
        replace(
            manifest,
            lanes=tuple(mismatched if item is instance else item for item in manifest.lanes),
        )

    with pytest.raises(ValueError, match=r"exactly.*perceptual.*instance.*concept"):
        replace(manifest, lanes=manifest.lanes[:-1])


def test_manifest_rejects_unreferenced_missing_and_wrong_parameter_payloads() -> None:
    manifest = generation_set_manifest()
    extra = _payload("unreferenced")
    with pytest.raises(ValueError, match="unreferenced"):
        replace(manifest, payloads=(*manifest.payloads, extra))

    without_representation_manifest = tuple(
        payload
        for payload in manifest.payloads
        if payload.role != TARGET_REPRESENTATION_MANIFEST_ROLE
    )
    with pytest.raises(ValueError, match="target representation manifest"):
        replace(manifest, payloads=without_representation_manifest)

    with pytest.raises(ValueError, match=r"target representation.*SHA-256"):
        replace(manifest, target_representation_manifest_sha256=SHA_A)

    without_catalog = tuple(payload for payload in manifest.payloads if payload.role != "catalog")
    with pytest.raises(ValueError, match="catalog"):
        replace(manifest, payloads=without_catalog)

    instance = next(item for item in manifest.lanes if item.lane is RetrievalLane.INSTANCE)
    wrong_parameter = replace(
        instance,
        generation=replace(
            instance.generation,
            index_parameters_ref=replace(instance.generation.index_parameters_ref, sha256=SHA_A),
        ),
    )
    with pytest.raises(ValueError, match=r"parameter.*SHA-256"):
        replace(
            manifest,
            lanes=tuple(wrong_parameter if item is instance else item for item in manifest.lanes),
        )


def test_personal_manifest_rejects_enterprise_profile_and_capacity() -> None:
    manifest = generation_set_manifest()

    with pytest.raises(ValueError, match="personal-lite"):
        replace(manifest, retrieval_profile_ref="enterprise")
    with pytest.raises(ValueError, match="10,000"):
        replace(manifest, unit_catalog=replace(manifest.unit_catalog, unit_count=10_001))
