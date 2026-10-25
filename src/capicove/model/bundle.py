"""Verified, offline-only contracts for local ONNX image-encoder bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.receipt import (
    VerifiedModelInstallation,
    read_receipt,
)


_DEFAULT_MAXIMUM_CONTROL_BYTES = 1_048_576
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_SPECIFIER_PATTERN = re.compile(
    r">=(?P<lower_major>\d+)\.(?P<lower_minor>\d+)(?:\.(?P<lower_patch>\d+))?,"
    r"<(?P<upper_major>\d+)(?:\.(?P<upper_minor>\d+))?(?:\.(?P<upper_patch>\d+))?\Z"
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "bundle_id",
        "repository",
        "revision",
        "license",
        "preprocessing",
        "input",
        "output",
        "embedding_space",
        "artifacts",
        "runtime",
        "network_behavior",
        "schema_version",
    }
)
_PREPROCESSING_FIELDS = frozenset(
    {
        "color_mode",
        "apply_exif_orientation",
        "resize_strategy",
        "resize_size",
        "crop_strategy",
        "crop_size",
        "interpolation",
        "value_range",
        "channel_mean",
        "channel_std",
        "tensor_layout",
        "tensor_dtype",
        "schema_version",
    }
)
_INPUT_FIELDS = frozenset({"name", "rank", "shape", "dtype"})
_OUTPUT_FIELDS = frozenset(
    {
        "name",
        "rank",
        "selection",
        "pooling",
        "dimension",
        "dtype",
        "l2_normalized",
        "metric",
        "vector_schema",
    }
)
_SPACE_FIELDS = frozenset(
    {
        "space_id",
        "provider_id",
        "model_revision",
        "preprocessing_version",
        "dimension",
        "normalization",
        "distance_metric",
        "query_modalities",
        "target_modalities",
        "schema_version",
    }
)
_ARTIFACT_FIELDS = frozenset({"role", "path", "size_bytes", "sha256"})
_RUNTIME_FIELDS = frozenset(
    {
        "version_specifier",
        "execution_provider",
        "graph_optimization",
        "custom_operators_allowed",
        "provider_fallback_allowed",
    }
)
_CHECKSUMS_FIELDS = frozenset({"schema_version", "files"})
_CHECKSUM_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_RECEIPT_FILE_FIELDS = frozenset({"path", "size", "sha256"})
_ARTIFACT_ROLES = frozenset({"onnx-model", "checksums", "license"})


@dataclass(frozen=True, slots=True)
class ImagePreprocessingContract:
    """Describe deterministic conversion from decoded RGB pixels to an ONNX tensor."""

    color_mode: str
    apply_exif_orientation: bool
    resize_strategy: str
    resize_size: tuple[int, int]
    crop_strategy: str
    crop_size: tuple[int, int] | None
    interpolation: str
    value_range: tuple[float, float]
    channel_mean: tuple[float, float, float]
    channel_std: tuple[float, float, float]
    tensor_layout: str
    tensor_dtype: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class OnnxTensorContract:
    """Describe the single image input expected by the pinned ONNX graph."""

    name: str
    rank: int
    shape: tuple[int | str, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class OnnxOutputContract:
    """Describe selection and vector semantics for the graph output."""

    name: str
    rank: int
    selection: str
    pooling: str
    dimension: int
    dtype: str
    l2_normalized: bool
    metric: str
    vector_schema: str


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    """Bind one closed bundle role to exact regular-file bytes."""

    role: str
    relative_path: PurePosixPath
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class OnnxRuntimeContract:
    """Constrain later runtime composition to a qualified local CPU execution path."""

    version_specifier: str
    execution_provider: str
    graph_optimization: str
    custom_operators_allowed: bool
    provider_fallback_allowed: bool


@dataclass(frozen=True, slots=True)
class ImageEncoderBundleManifest:
    """Contain the complete closed contract for one immutable image encoder."""

    bundle_id: str
    repository: str
    revision: str
    license: str
    preprocessing: ImagePreprocessingContract
    input: OnnxTensorContract
    output: OnnxOutputContract
    embedding_space: EmbeddingSpaceRef
    artifacts: tuple[BundleArtifact, ...]
    runtime: OnnxRuntimeContract
    network_behavior: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class VerifiedImageEncoderBundle:
    """Bind a parsed image-encoder contract to trusted installation evidence."""

    installation: VerifiedModelInstallation
    manifest: ImageEncoderBundleManifest
    manifest_sha256: str

    def verify(self) -> None:
        """Recheck trusted receipt, manifest bytes, and every declared asset."""
        manifest, manifest_sha256 = _load_and_verify_manifest(
            self.installation,
            maximum_manifest_bytes=_DEFAULT_MAXIMUM_CONTROL_BYTES,
            maximum_checksums_bytes=_DEFAULT_MAXIMUM_CONTROL_BYTES,
        )
        if manifest != self.manifest or manifest_sha256 != self.manifest_sha256:
            raise ValueError("image-encoder bundle manifest changed after verification")

    def artifact_path(self, role: str) -> Path:
        """Reverify and return the one role-owned regular file below the root."""
        self.verify()
        artifacts = tuple(artifact for artifact in self.manifest.artifacts if artifact.role == role)
        if len(artifacts) != 1:
            raise ValueError(f"unknown image-encoder bundle artifact role: {role}")
        return self.installation.path.joinpath(*artifacts[0].relative_path.parts)


def load_verified_image_encoder_bundle(
    installation: VerifiedModelInstallation,
    *,
    maximum_manifest_bytes: int = _DEFAULT_MAXIMUM_CONTROL_BYTES,
    maximum_checksums_bytes: int = _DEFAULT_MAXIMUM_CONTROL_BYTES,
) -> VerifiedImageEncoderBundle:
    """Load a closed offline image-encoder bundle from trusted evidence."""
    if not isinstance(installation, VerifiedModelInstallation):
        raise TypeError("installation must be a VerifiedModelInstallation")
    _require_positive_int("maximum_manifest_bytes", maximum_manifest_bytes)
    _require_positive_int("maximum_checksums_bytes", maximum_checksums_bytes)
    manifest, manifest_sha256 = _load_and_verify_manifest(
        installation,
        maximum_manifest_bytes=maximum_manifest_bytes,
        maximum_checksums_bytes=maximum_checksums_bytes,
    )
    return VerifiedImageEncoderBundle(
        installation=installation,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )


def _load_and_verify_manifest(
    installation: VerifiedModelInstallation,
    *,
    maximum_manifest_bytes: int,
    maximum_checksums_bytes: int,
) -> tuple[ImageEncoderBundleManifest, str]:
    installation.verify()
    if installation.model.adapter != "onnx-image-embedding":
        raise ValueError("trusted lock entry is not an ONNX image-embedding model")
    root = _verified_root(installation.path)
    manifest_payload = _read_regular_file(
        root,
        PurePosixPath("manifest.json"),
        maximum_bytes=maximum_manifest_bytes,
    )
    manifest = _parse_manifest(manifest_payload, installation)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    checksums_artifact = _artifact_for_role(manifest.artifacts, "checksums")
    checksums_payload = _read_regular_file(
        root,
        checksums_artifact.relative_path,
        maximum_bytes=maximum_checksums_bytes,
    )
    if _file_record(checksums_artifact.relative_path, checksums_payload) != (
        checksums_artifact.relative_path,
        checksums_artifact.size_bytes,
        checksums_artifact.sha256,
    ):
        raise ValueError("checksums.json does not match the bundle manifest")
    checksum_records = _parse_checksums(checksums_payload)
    _verify_three_way_consistency(
        installation,
        manifest,
        manifest_payload,
        checksum_records,
    )
    for artifact in manifest.artifacts:
        _verify_artifact(root, artifact)
    return manifest, manifest_sha256


def _parse_manifest(
    payload: bytes,
    installation: VerifiedModelInstallation,
) -> ImageEncoderBundleManifest:
    raw = _parse_json_object(payload, "image-encoder manifest")
    values = _closed_object(raw, _TOP_LEVEL_FIELDS, "image-encoder manifest")
    if _integer(values, "schema_version", "image-encoder manifest") != 1:
        raise ValueError("image-encoder manifest schema_version must be 1")
    repository = _string(values, "repository", "image-encoder manifest")
    revision = _string(values, "revision", "image-encoder manifest")
    license_name = _string(values, "license", "image-encoder manifest")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("image-encoder bundle revision must be a lowercase 40-character Git SHA")
    if repository != installation.model.repository:
        raise ValueError("image-encoder bundle repository does not match the trusted lock")
    if revision != installation.model.revision:
        raise ValueError("image-encoder bundle revision does not match the trusted lock")
    if license_name != installation.model.license:
        raise ValueError("image-encoder bundle license does not match the trusted lock")
    preprocessing = _parse_preprocessing(values.get("preprocessing"))
    input_contract = _parse_input(values.get("input"), preprocessing)
    output_contract = _parse_output(values.get("output"))
    embedding_space = _parse_embedding_space(values.get("embedding_space"))
    if embedding_space.model_revision != installation.model.revision:
        raise ValueError("embedding-space model revision does not match the trusted lock")
    if embedding_space.preprocessing_version != installation.model.preprocess_revision:
        raise ValueError("embedding-space preprocessing version does not match the trusted lock")
    if embedding_space.dimension != output_contract.dimension:
        raise ValueError("embedding-space dimension does not match the ONNX output")
    if embedding_space.distance_metric.value != output_contract.metric:
        raise ValueError("embedding-space metric does not match the ONNX output")
    if embedding_space.normalization is not VectorNormalization.L2:
        raise ValueError("image-encoder embedding space must declare L2 normalization")
    artifacts = _parse_artifacts(values.get("artifacts"))
    runtime = _parse_runtime(values.get("runtime"))
    network_behavior = _string(values, "network_behavior", "image-encoder manifest")
    if network_behavior != "offline":
        raise ValueError("image-encoder bundle network_behavior must be offline")
    return ImageEncoderBundleManifest(
        bundle_id=_string(values, "bundle_id", "image-encoder manifest"),
        repository=repository,
        revision=revision,
        license=license_name,
        preprocessing=preprocessing,
        input=input_contract,
        output=output_contract,
        embedding_space=embedding_space,
        artifacts=artifacts,
        runtime=runtime,
        network_behavior=network_behavior,
        schema_version=1,
    )


def _parse_preprocessing(raw: object) -> ImagePreprocessingContract:
    values = _closed_object(raw, _PREPROCESSING_FIELDS, "preprocessing contract")
    if _integer(values, "schema_version", "preprocessing contract") != 1:
        raise ValueError("preprocessing schema_version must be 1")
    color_mode = _string(values, "color_mode", "preprocessing contract")
    if color_mode != "RGB":
        raise ValueError("image preprocessing color_mode must be RGB")
    resize_strategy = _string(values, "resize_strategy", "preprocessing contract")
    if resize_strategy not in {"exact", "shortest_edge"}:
        raise ValueError("unsupported image preprocessing resize_strategy")
    resize_size = _positive_pair(values.get("resize_size"), "resize_size")
    crop_strategy = _string(values, "crop_strategy", "preprocessing contract")
    if crop_strategy not in {"none", "center"}:
        raise ValueError("unsupported image preprocessing crop_strategy")
    raw_crop_size = values.get("crop_size")
    if crop_strategy == "none":
        if raw_crop_size is not None:
            raise ValueError("crop_size must be null when crop_strategy is none")
        crop_size = None
    else:
        crop_size = _positive_pair(raw_crop_size, "crop_size")
        if any(crop > resized for crop, resized in zip(crop_size, resize_size, strict=True)):
            raise ValueError("crop_size must not exceed resize_size")
    interpolation = _string(values, "interpolation", "preprocessing contract")
    if interpolation not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("unsupported image preprocessing interpolation")
    value_range = _finite_pair(values.get("value_range"), "value_range")
    if value_range[0] >= value_range[1]:
        raise ValueError("preprocessing value_range must be increasing")
    channel_mean = _finite_triplet(values.get("channel_mean"), "channel_mean")
    channel_std = _finite_triplet(values.get("channel_std"), "channel_std")
    if any(value <= 0 for value in channel_std):
        raise ValueError("preprocessing channel_std values must be positive")
    tensor_layout = _string(values, "tensor_layout", "preprocessing contract")
    tensor_dtype = _string(values, "tensor_dtype", "preprocessing contract")
    if tensor_layout != "NCHW" or tensor_dtype != "float32":
        raise ValueError("image preprocessing must produce an NCHW float32 tensor")
    return ImagePreprocessingContract(
        color_mode=color_mode,
        apply_exif_orientation=_boolean(
            values,
            "apply_exif_orientation",
            "preprocessing contract",
        ),
        resize_strategy=resize_strategy,
        resize_size=resize_size,
        crop_strategy=crop_strategy,
        crop_size=crop_size,
        interpolation=interpolation,
        value_range=value_range,
        channel_mean=channel_mean,
        channel_std=channel_std,
        tensor_layout=tensor_layout,
        tensor_dtype=tensor_dtype,
        schema_version=1,
    )


def _parse_input(
    raw: object,
    preprocessing: ImagePreprocessingContract,
) -> OnnxTensorContract:
    values = _closed_object(raw, _INPUT_FIELDS, "ONNX input contract")
    rank = _integer(values, "rank", "ONNX input contract")
    if rank != 4:
        raise ValueError("ONNX image input rank must be 4")
    raw_shape = values.get("shape")
    if not isinstance(raw_shape, list) or len(raw_shape) != rank:
        raise ValueError("ONNX input shape must contain exactly rank entries")
    shape: list[int | str] = []
    for index, dimension in enumerate(raw_shape):
        if isinstance(dimension, bool) or not isinstance(dimension, (int, str)):
            raise ValueError("ONNX input shape entries must be positive integers or batch")
        if isinstance(dimension, int) and dimension <= 0:
            raise ValueError("ONNX input shape dimensions must be positive")
        if isinstance(dimension, str) and dimension != "batch":
            raise ValueError("ONNX input supports only the symbolic batch dimension")
        if index != 0 and isinstance(dimension, str):
            raise ValueError("ONNX input supports a symbolic dimension only for batch")
        shape.append(dimension)
    expected_size = preprocessing.crop_size or preprocessing.resize_size
    if shape[0] != "batch" or shape[1] != 3 or tuple(shape[2:]) != expected_size:
        raise ValueError("ONNX input shape does not match the preprocessing contract")
    dtype = _string(values, "dtype", "ONNX input contract")
    if dtype != "float32":
        raise ValueError("ONNX image input dtype must be float32")
    return OnnxTensorContract(
        name=_string(values, "name", "ONNX input contract"),
        rank=rank,
        shape=tuple(shape),
        dtype=dtype,
    )


def _parse_output(raw: object) -> OnnxOutputContract:
    values = _closed_object(raw, _OUTPUT_FIELDS, "ONNX output contract")
    rank = _integer(values, "rank", "ONNX output contract")
    if rank != 2:
        raise ValueError("ONNX embedding output rank must be 2")
    selection = _string(values, "selection", "ONNX output contract")
    pooling = _string(values, "pooling", "ONNX output contract")
    dtype = _string(values, "dtype", "ONNX output contract")
    if selection != "only" or pooling != "none" or dtype != "float32":
        raise ValueError("ONNX output must select the only unpooled float32 tensor")
    l2_normalized = _boolean(values, "l2_normalized", "ONNX output contract")
    if not l2_normalized:
        raise ValueError("ONNX embedding output must be L2 normalized")
    metric = _string(values, "metric", "ONNX output contract")
    if metric not in {DistanceMetric.COSINE.value, DistanceMetric.INNER_PRODUCT.value}:
        raise ValueError("unsupported ONNX embedding output metric")
    return OnnxOutputContract(
        name=_string(values, "name", "ONNX output contract"),
        rank=rank,
        selection=selection,
        pooling=pooling,
        dimension=_positive_integer(values, "dimension", "ONNX output contract"),
        dtype=dtype,
        l2_normalized=l2_normalized,
        metric=metric,
        vector_schema=_string(values, "vector_schema", "ONNX output contract"),
    )


def _parse_embedding_space(raw: object) -> EmbeddingSpaceRef:
    values = _closed_object(raw, _SPACE_FIELDS, "embedding-space contract")
    if _integer(values, "schema_version", "embedding-space contract") != 1:
        raise ValueError("embedding-space schema_version must be 1")
    query_modalities = _enum_set(
        values.get("query_modalities"),
        QueryModality,
        "query_modalities",
    )
    target_modalities = _enum_set(
        values.get("target_modalities"),
        TargetType,
        "target_modalities",
    )
    if QueryModality.IMAGE not in query_modalities or TargetType.IMAGE not in target_modalities:
        raise ValueError("image-encoder embedding space must support image-to-image retrieval")
    try:
        normalization = VectorNormalization(
            _string(values, "normalization", "embedding-space contract")
        )
        distance_metric = DistanceMetric(
            _string(values, "distance_metric", "embedding-space contract")
        )
    except ValueError as error:
        raise ValueError("embedding-space contract contains an unsupported enum value") from error
    return EmbeddingSpaceRef(
        space_id=_string(values, "space_id", "embedding-space contract"),
        provider_id=_string(values, "provider_id", "embedding-space contract"),
        model_revision=_string(values, "model_revision", "embedding-space contract"),
        preprocessing_version=_string(
            values,
            "preprocessing_version",
            "embedding-space contract",
        ),
        dimension=_positive_integer(values, "dimension", "embedding-space contract"),
        normalization=normalization,
        distance_metric=distance_metric,
        query_modalities=frozenset(query_modalities),
        target_modalities=frozenset(target_modalities),
        schema_version=1,
    )


def _parse_artifacts(raw: object) -> tuple[BundleArtifact, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("image-encoder bundle artifacts must be a non-empty array")
    artifacts: list[BundleArtifact] = []
    roles: set[str] = set()
    paths: set[str] = set()
    casefolded_paths: set[str] = set()
    for index, item in enumerate(raw):
        context = f"bundle artifact {index}"
        values = _closed_object(item, _ARTIFACT_FIELDS, context)
        role = _string(values, "role", context)
        if role not in _ARTIFACT_ROLES:
            raise ValueError(f"unsupported image-encoder bundle artifact role: {role}")
        if role in roles:
            raise ValueError(f"duplicate image-encoder bundle artifact role: {role}")
        relative_path = _safe_relative_path(_string(values, "path", context), context)
        normalized = relative_path.as_posix()
        if normalized in paths or normalized.casefold() in casefolded_paths:
            raise ValueError(f"duplicate or differently cased bundle artifact path: {normalized}")
        roles.add(role)
        paths.add(normalized)
        casefolded_paths.add(normalized.casefold())
        artifacts.append(
            BundleArtifact(
                role=role,
                relative_path=relative_path,
                size_bytes=_non_negative_integer(values, "size_bytes", context),
                sha256=_sha256(values, "sha256", context),
            )
        )
    if roles != _ARTIFACT_ROLES:
        missing = ", ".join(sorted(_ARTIFACT_ROLES - roles))
        raise ValueError(f"image-encoder bundle is missing required artifact roles: {missing}")
    checksums = _artifact_for_role(tuple(artifacts), "checksums")
    if checksums.relative_path != PurePosixPath("checksums.json"):
        raise ValueError("checksums artifact path must be checksums.json")
    return tuple(artifacts)


def _parse_runtime(raw: object) -> OnnxRuntimeContract:
    values = _closed_object(raw, _RUNTIME_FIELDS, "ONNX runtime contract")
    version_specifier = _string(values, "version_specifier", "ONNX runtime contract")
    if not _bounded_version_specifier(version_specifier):
        raise ValueError("ONNX Runtime version_specifier must have finite lower and upper bounds")
    execution_provider = _string(values, "execution_provider", "ONNX runtime contract")
    if execution_provider != "CPUExecutionProvider":
        raise ValueError("offline image encoder requires CPUExecutionProvider")
    graph_optimization = _string(values, "graph_optimization", "ONNX runtime contract")
    if graph_optimization not in {"disabled", "basic", "extended", "all"}:
        raise ValueError("unsupported ONNX Runtime graph_optimization policy")
    custom_operators_allowed = _boolean(
        values,
        "custom_operators_allowed",
        "ONNX runtime contract",
    )
    provider_fallback_allowed = _boolean(
        values,
        "provider_fallback_allowed",
        "ONNX runtime contract",
    )
    if custom_operators_allowed or provider_fallback_allowed:
        raise ValueError("ONNX runtime contract cannot allow custom operators or fallback")
    return OnnxRuntimeContract(
        version_specifier=version_specifier,
        execution_provider=execution_provider,
        graph_optimization=graph_optimization,
        custom_operators_allowed=False,
        provider_fallback_allowed=False,
    )


def _parse_checksums(payload: bytes) -> tuple[tuple[PurePosixPath, int, str], ...]:
    raw = _parse_json_object(payload, "bundle checksums")
    values = _closed_object(raw, _CHECKSUMS_FIELDS, "bundle checksums")
    if _integer(values, "schema_version", "bundle checksums") != 1:
        raise ValueError("bundle checksums schema_version must be 1")
    raw_files = values.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("bundle checksums files must be a non-empty array")
    records: list[tuple[PurePosixPath, int, str]] = []
    for index, item in enumerate(raw_files):
        context = f"bundle checksums file {index}"
        record = _closed_object(item, _CHECKSUM_FILE_FIELDS, context)
        records.append(
            (
                _safe_relative_path(_string(record, "path", context), context),
                _non_negative_integer(record, "size_bytes", context),
                _sha256(record, "sha256", context),
            )
        )
    _require_unique_record_paths(records, "bundle checksums")
    if any(path == PurePosixPath("checksums.json") for path, _, _ in records):
        raise ValueError("bundle checksums must not record checksums.json recursively")
    return tuple(records)


def _verify_three_way_consistency(
    installation: VerifiedModelInstallation,
    manifest: ImageEncoderBundleManifest,
    manifest_payload: bytes,
    checksum_records: tuple[tuple[PurePosixPath, int, str], ...],
) -> None:
    checksums_artifact = _artifact_for_role(manifest.artifacts, "checksums")
    artifact_records = {
        artifact.relative_path: (artifact.size_bytes, artifact.sha256)
        for artifact in manifest.artifacts
        if artifact.role != "checksums"
    }
    checksum_map = {path: (size, sha256) for path, size, sha256 in checksum_records}
    if checksum_map != artifact_records:
        raise ValueError("bundle checksums entries do not exactly match manifest artifacts")
    receipt = read_receipt(installation.path)
    receipt_records = _parse_receipt_files(receipt.get("files"))
    expected_receipt = dict(checksum_map)
    expected_receipt[PurePosixPath("manifest.json")] = (
        len(manifest_payload),
        hashlib.sha256(manifest_payload).hexdigest(),
    )
    expected_receipt[PurePosixPath("checksums.json")] = (
        checksums_artifact.size_bytes,
        checksums_artifact.sha256,
    )
    if receipt_records != expected_receipt:
        raise ValueError(
            "bundle receipt files do not exactly match manifest and checksums evidence"
        )


def _parse_receipt_files(raw: object) -> dict[PurePosixPath, tuple[int, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("bundle receipt files must be a non-empty array")
    records: list[tuple[PurePosixPath, int, str]] = []
    for index, item in enumerate(raw):
        context = f"bundle receipt file {index}"
        values = _closed_object(item, _RECEIPT_FILE_FIELDS, context)
        records.append(
            (
                _safe_relative_path(_string(values, "path", context), context),
                _non_negative_integer(values, "size", context),
                _sha256(values, "sha256", context),
            )
        )
    _require_unique_record_paths(records, "bundle receipt")
    return {path: (size, sha256) for path, size, sha256 in records}


def _verify_artifact(root: Path, artifact: BundleArtifact) -> None:
    size, sha256 = _hash_regular_file(root, artifact.relative_path)
    if size != artifact.size_bytes or sha256 != artifact.sha256:
        raise ValueError(
            f"bundle artifact does not match its manifest: {artifact.relative_path.as_posix()}"
        )


def _verified_root(root: Path) -> Path:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ValueError("image-encoder bundle root is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("image-encoder bundle root must be a regular directory, not a symlink")
    return root.resolve(strict=True)


def _read_regular_file(
    root: Path,
    relative_path: PurePosixPath,
    *,
    maximum_bytes: int,
) -> bytes:
    path = _checked_regular_path(root, relative_path)
    try:
        size = path.stat(follow_symlinks=False).st_size
        if size > maximum_bytes:
            raise ValueError(f"bundle control file exceeds {maximum_bytes} bytes")
        with path.open("rb") as stream:
            payload = stream.read(maximum_bytes + 1)
    except OSError as error:
        raise ValueError(f"cannot read bundle file: {relative_path.as_posix()}") from error
    if len(payload) > maximum_bytes:
        raise ValueError(f"bundle control file exceeds {maximum_bytes} bytes")
    return payload


def _hash_regular_file(root: Path, relative_path: PurePosixPath) -> tuple[int, str]:
    path = _checked_regular_path(root, relative_path)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read bundle file: {relative_path.as_posix()}") from error
    return size, digest.hexdigest()


def _checked_regular_path(root: Path, relative_path: PurePosixPath) -> Path:
    current = root
    try:
        for index, part in enumerate(relative_path.parts):
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"bundle contains a symlink: {relative_path.as_posix()}")
            if index < len(relative_path.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"bundle path parent is not a directory: {relative_path.as_posix()}"
                )
        if not stat.S_ISREG(current.lstat().st_mode):
            raise ValueError(f"bundle artifact is not a regular file: {relative_path.as_posix()}")
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"bundle artifact is missing: {relative_path.as_posix()}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"bundle artifact escapes its installation root: {relative_path}"
        ) from error
    return current


def _parse_json_object(payload: bytes, context: str) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(dict[str, object], raw)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, value in pairs:
        if name in values:
            raise ValueError(f"duplicate JSON field: {name}")
        values[name] = value
    return values


def _reject_non_finite_json_number(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _closed_object(raw: object, fields: frozenset[str], context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object")
    values = cast(dict[str, object], raw)
    unknown = sorted(set(values) - fields)
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(unknown)}")
    missing = sorted(fields - set(values))
    if missing:
        raise ValueError(f"missing {context} fields: {', '.join(missing)}")
    return values


def _string(values: dict[str, object], name: str, context: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context} {name} must be a canonical nonblank string")
    return value


def _integer(values: dict[str, object], name: str, context: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} {name} must be an integer")
    return value


def _positive_integer(values: dict[str, object], name: str, context: str) -> int:
    value = _integer(values, name, context)
    if value <= 0:
        raise ValueError(f"{context} {name} must be positive")
    return value


def _non_negative_integer(values: dict[str, object], name: str, context: str) -> int:
    value = _integer(values, name, context)
    if value < 0:
        raise ValueError(f"{context} {name} must be non-negative")
    return value


def _boolean(values: dict[str, object], name: str, context: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {name} must be a boolean")
    return value


def _sha256(values: dict[str, object], name: str, context: str) -> str:
    value = _string(values, name, context)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} {name} must be a lowercase SHA-256 digest")
    return value


def _positive_pair(raw: object, name: str) -> tuple[int, int]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw):
        raise ValueError(f"{name} must contain two positive integers")
    return cast(tuple[int, int], tuple(raw))


def _finite_pair(raw: object, name: str) -> tuple[float, float]:
    values = _finite_values(raw, name, 2)
    return cast(tuple[float, float], values)


def _finite_triplet(raw: object, name: str) -> tuple[float, float, float]:
    values = _finite_values(raw, name, 3)
    return cast(tuple[float, float, float], values)


def _finite_values(raw: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain exactly {length} finite numbers")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must contain exactly {length} finite numbers")
        values.append(normalized)
    return tuple(values)


def _safe_relative_path(value: str, context: str) -> PurePosixPath:
    if (
        "\\" in value
        or "\x00" in value
        or value.casefold().startswith("file:")
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise ValueError(f"{context} path must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{context} path must be a safe relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{context} path must be canonical")
    return path


def _enum_set(raw: object, enum_type: type[Any], name: str) -> set[Any]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"embedding-space {name} must be a non-empty array")
    values: set[Any] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"embedding-space {name} values must be strings")
        try:
            normalized = enum_type(item)
        except ValueError as error:
            raise ValueError(f"embedding-space {name} contains an unsupported value") from error
        if normalized in values:
            raise ValueError(f"embedding-space {name} must not contain duplicates")
        values.add(normalized)
    return values


def _bounded_version_specifier(value: str) -> bool:
    match = _VERSION_SPECIFIER_PATTERN.fullmatch(value)
    if match is None:
        return False
    lower = (
        int(match.group("lower_major")),
        int(match.group("lower_minor")),
        int(match.group("lower_patch") or 0),
    )
    upper = (
        int(match.group("upper_major")),
        int(match.group("upper_minor") or 0),
        int(match.group("upper_patch") or 0),
    )
    return lower < upper


def _artifact_for_role(
    artifacts: tuple[BundleArtifact, ...],
    role: str,
) -> BundleArtifact:
    matches = tuple(artifact for artifact in artifacts if artifact.role == role)
    if len(matches) != 1:
        raise ValueError(f"bundle must contain exactly one {role} artifact")
    return matches[0]


def _require_unique_record_paths(
    records: list[tuple[PurePosixPath, int, str]],
    context: str,
) -> None:
    paths = [path.as_posix() for path, _, _ in records]
    if len(set(paths)) != len(paths) or len({path.casefold() for path in paths}) != len(paths):
        raise ValueError(f"{context} contains duplicate or differently cased paths")


def _file_record(path: PurePosixPath, payload: bytes) -> tuple[PurePosixPath, int, str]:
    return path, len(payload), hashlib.sha256(payload).hexdigest()


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
