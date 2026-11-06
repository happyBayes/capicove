"""Image adapter: the internal personal reverse-image research index.

The production image host needs provisioned DINOv2/SigLIP2 bundles, HMAC keys and
a qualification root, none of which exist offline here. This adapter therefore
evaluates the in-memory ``PersonalImageResearchIndex`` (internal research only)
with the retrieval-owned OpenCV pHash lane and two deterministic fake semantic
encoders: an 8x8 RGB grid for the instance lane and the mean palette color word
for the concept lane. Images are rendered from authored specs at run time.

Query images are corpus images under a closed set of transformations, or fresh
renders. The full ranking is exact over the corpus, so the pool is the corpus.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from capicove.evaluation.adapters._common import (
    corpus_images,
    deadline,
    render_image,
    union_labels,
)
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
)


TRANSFORMS = ("identity", "resize-half", "crop-center-80", "brighten-12", "mirror")


def _space(name: str, dimension: int) -> Any:
    from capicove.contracts.retrieval import (
        DistanceMetric,
        EmbeddingSpaceRef,
        QueryModality,
        TargetType,
        VectorNormalization,
    )

    return EmbeddingSpaceRef(
        space_id=f"capicove-deterministic-fake:{name}",
        provider_id="capicove-deterministic-fake",
        model_revision="fake-1",
        preprocessing_version=f"{name}-v1",
        dimension=dimension,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


class _GridEncoder:
    """Instance-lane fake: mean-centered 8x8 RGB thumbnail."""

    def __init__(self) -> None:
        self.space = _space("rgb-grid-8x8", 192)

    def encode(self, image: Any, *, deadline: datetime) -> Any:
        from capicove.retrieval.personal_image_research import ResearchSemanticVector

        small = image.convert("RGB").resize((8, 8))
        values = [
            float(channel) for y in range(8) for x in range(8) for channel in small.getpixel((x, y))
        ]
        mean = sum(values) / len(values)
        centered = [value - mean for value in values]
        norm = math.sqrt(sum(value * value for value in centered)) or 1.0
        if norm == 1.0 and not any(centered):
            centered[0] = 1.0
        return ResearchSemanticVector(self.space, tuple(value / norm for value in centered))


class _ColorConceptEncoder:
    """Concept-lane fake: hashed words for ``image <mean palette color>``."""

    def __init__(self) -> None:
        self.space = _space("palette-concept-64", 64)

    def encode(self, image: Any, *, deadline: datetime) -> Any:
        from capicove.evaluation.fakes import dominant_color_word, hashed_bag_of_words
        from capicove.retrieval.personal_image_research import ResearchSemanticVector

        small = image.convert("RGB").resize((8, 8))
        pixels = [small.getpixel((x, y)) for y in range(8) for x in range(8)]
        word = dominant_color_word(pixels)
        return ResearchSemanticVector(self.space, hashed_bag_of_words("image " + word, 64))


def transform(payload: bytes, name: str) -> bytes:
    """Apply one closed deterministic transformation and re-encode as PNG."""
    from PIL import Image, ImageEnhance, ImageOps

    if name not in TRANSFORMS:
        raise EvaluationError("unsupported image query transformation")
    if name == "identity":
        return payload
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
    if name == "resize-half":
        image = image.resize((image.width // 2, image.height // 2))
    elif name == "crop-center-80":
        margin_x, margin_y = image.width // 10, image.height // 10
        image = image.crop((margin_x, margin_y, image.width - margin_x, image.height - margin_y))
    elif name == "brighten-12":
        image = ImageEnhance.Brightness(image).enhance(1.12)
    else:
        image = ImageOps.mirror(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def _unit_id(image_id: str) -> str:
    return "unit:" + hashlib.sha256(image_id.encode()).hexdigest()


@dataclass(slots=True)
class _PreparedImage:
    index: Any
    rendered: Mapping[str, bytes]
    by_unit: Mapping[str, str]
    identity: Mapping[str, Any]

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        from PIL import Image

        from capicove.retrieval.personal_image_research import ResearchSearchIntent

        spec = query.payload.get("image")
        if not isinstance(spec, Mapping) or set(spec) not in ({"source", "transform"}, {"spec"}):
            raise EvaluationError("image queries require a source/transform or a spec")
        if "spec" in spec:
            payload = render_image(spec["spec"])
        else:
            if spec["source"] not in self.rendered:
                raise EvaluationError("image query source is not in the corpus")
            payload = transform(self.rendered[spec["source"]], spec["transform"])
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert("RGB")
        result = self.index.search_image(
            image,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            intent=ResearchSearchIntent(query.payload.get("intent", "balanced")),
            deadline=deadline(),
            top_k=min(100, len(self.by_unit)),
        )
        judged = set(query.judgments)
        items = []
        for hit in result.hits:
            image_id = self.by_unit[hit.unit_id]
            items.append(RankedItem(image_id, (image_id,) if image_id in judged else ()))
        return QueryOutcome(
            tuple(items[:k]),
            len(items),
            union_labels(item.labels for item in items),
            diagnostics={
                "exact_content": sorted(
                    self.by_unit[hit.unit_id] for hit in result.hits if hit.exact_content
                )
            },
        )

    def close(self) -> None:
        return None


class ImageAdapter:
    """Build the internal research index over rendered synthetic images."""

    stack = "image"
    adapter_version = "1"

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> _PreparedImage:
        from PIL import Image

        from capicove.retrieval._adapters.opencv_phash import (
            OPENCV_PHASH_64_V1_SPACE,
            OpenCvQueryPhashAdapter,
        )
        from capicove.retrieval.personal_image_research import (
            PersonalImageResearchIndex,
            PersonalImageResearchUnit,
        )

        if options.reranker != "none" or options.profile_path is not None or options.fake_dense:
            raise EvaluationError("the image adapter supports only its default policy")
        images = corpus_images(dataset)
        if not images:
            raise EvaluationError("image datasets require corpus images")
        instance, concept, phash = _GridEncoder(), _ColorConceptEncoder(), OpenCvQueryPhashAdapter()
        rendered: dict[str, bytes] = {}
        units = []
        until = deadline()
        for row in images:
            payload = render_image(row["spec"])
            rendered[row["id"]] = payload
            with Image.open(io.BytesIO(payload)) as source:
                picture = source.convert("RGB")
            units.append(
                PersonalImageResearchUnit(
                    unit_id=_unit_id(row["id"]),
                    source_sha256=hashlib.sha256(payload).hexdigest(),
                    phash_space=OPENCV_PHASH_64_V1_SPACE,
                    phash=phash.phash(picture).value,
                    instance_vector=instance.encode(picture, deadline=until),
                    concept_vector=concept.encode(picture, deadline=until),
                )
            )
        index = PersonalImageResearchIndex(
            tuple(units),
            instance_space=instance.space,
            concept_space=concept.space,
            instance_encoder=instance,
            concept_encoder=concept,
        )
        from capicove.evaluation.adapters._common import fingerprint

        manifest = fingerprint([[unit.unit_id, unit.source_sha256, unit.phash] for unit in units])
        return _PreparedImage(
            index,
            rendered,
            {_unit_id(row["id"]): row["id"] for row in images},
            {
                "profile": {
                    "id": "personal-research-balanced-v1",
                    "version": 1,
                    "fingerprint": fingerprint(
                        {
                            "fusion": "exact-first-then-rrf-k60-phash2-semantic1",
                            "phash_radius": 2,
                            "phash_space": OPENCV_PHASH_64_V1_SPACE,
                        }
                    ),
                },
                "index": {
                    "kind": "personal-image-research-index",
                    "unit_manifest_sha256": manifest,
                    "units": len(units),
                },
                "model": {
                    "kind": "deterministic-fake",
                    "components": [
                        f"instance:{instance.space.space_id}",
                        f"concept:{concept.space.space_id}",
                        f"phash:{OPENCV_PHASH_64_V1_SPACE.space_id}",
                    ],
                },
            },
        )


def adapter(stack: str) -> ImageAdapter:
    """Return the image stack adapter."""
    return ImageAdapter()
