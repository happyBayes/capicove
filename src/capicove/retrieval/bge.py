"""Public local BGE adapter composition shared by retrieval and graph callers."""

import json
from pathlib import Path

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.local_assets import LocalModelAssets
from capicove.retrieval._adapters.bge import (
    BgeTextBatchEncoder,
    BgeTextEmbeddingProfile,
    bge_preprocessing_version,
)


def load_bge_m3_profile(
    root: Path, *, expected_manifest_sha256: str, max_tokens: int = 512
) -> BgeTextEmbeddingProfile:
    """Load an operator-trusted local dense bundle with explicit token bounds."""
    receipt_path = root.parent / "bge-m3.receipt.json"
    if receipt_path.is_symlink() or receipt_path.stat().st_size > 1_000_000:
        raise ValueError("invalid BGE-M3 manifest receipt")
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("asset_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("BGE-M3 trusted manifest mismatch")
    files = tuple(tuple(pair) for pair in receipt["asset_files"])
    assets = LocalModelAssets(root, files, expected_manifest_sha256)
    assets.verify()
    preprocessing = bge_preprocessing_version(
        assets, query_prefix="", document_prefix="", max_tokens=max_tokens, model_family="bge-m3"
    )
    space = EmbeddingSpaceRef(
        "bge-m3-dense-local-v1",
        "local-bge",
        "sha256:" + expected_manifest_sha256,
        preprocessing,
        1024,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT}),
        frozenset({TargetType.TEXT_CHUNK}),
    )
    return BgeTextEmbeddingProfile(assets, space, "", "", max_tokens, model_family="bge-m3")


__all__ = [
    "BgeTextBatchEncoder",
    "BgeTextEmbeddingProfile",
    "bge_preprocessing_version",
    "load_bge_m3_profile",
]
