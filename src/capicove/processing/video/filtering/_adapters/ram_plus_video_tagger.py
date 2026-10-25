"""Local RAM+-compatible adapter behind the generic VideoTagger port."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from capicove.processing.video.filtering.video_filter_contracts import (
    Aggregation,
    TagCriterion,
    VideoCriterionResult,
    VideoFilterCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    DecodeFailedFilterError,
    FrameTags,
    InvalidVideoFilterInputError,
    SampledFrame,
    VideoFilterTimeoutError,
)
from capicove.processing.video.models import AcceptedVideoSource


TagEngine = Callable[[tuple[SampledFrame, ...], datetime], tuple[FrameTags, ...]]
RamLoader = Callable[[Path, Path], TagEngine]


class RamPlusVideoTagger:
    """Generate tags with caller-provided local RAM+ assets."""

    def __init__(
        self,
        *,
        engine: TagEngine,
        weights_hash: str,
        vocabulary_hash: str,
        version: str | None,
    ) -> None:
        self._engine = engine
        self.weights_hash = weights_hash
        self.vocabulary_hash = vocabulary_hash
        self._identity = ("ram-plus", version)

    @classmethod
    def from_local(
        cls,
        weights_path: Path,
        vocabulary_path: Path,
        *,
        loader: RamLoader | None = None,
    ) -> RamPlusVideoTagger:
        """Build a tagger from explicit local files without downloads."""
        resolved_weights = weights_path.resolve()
        resolved_vocabulary = vocabulary_path.resolve()
        if not resolved_weights.is_file():
            raise BackendUnavailableFilterError(f"RAM+ weights do not exist: {resolved_weights}")
        if not resolved_vocabulary.is_file():
            raise BackendUnavailableFilterError(
                f"RAM+ vocabulary does not exist: {resolved_vocabulary}"
            )
        engine = (loader or _load_ram_engine)(resolved_weights, resolved_vocabulary)
        return cls(
            engine=engine,
            weights_hash=_hash_file(resolved_weights),
            vocabulary_hash=_hash_file(resolved_vocabulary),
            version=_package_version("ram"),
        )

    @property
    def identity(self) -> tuple[str, str | None]:
        """Return the RAM+ adapter identity."""
        return self._identity

    def tag(
        self,
        frames: tuple[SampledFrame, ...],
        deadline: datetime,
    ) -> tuple[FrameTags, ...]:
        """Return one normalized provider result per sampled frame."""
        if datetime.now(UTC) >= deadline:
            raise VideoFilterTimeoutError("The RAM+ tagging deadline expired.")
        if not frames:
            raise DecodeFailedFilterError("RAM+ tagging requires at least one frame.")
        results = self._engine(frames, deadline)
        if len(results) != len(frames):
            raise DecodeFailedFilterError("RAM+ did not return one tag result per frame.")
        return results


class RamPlusTagCriterionEvaluator:
    """Apply tag matching policy to a generic VideoTagger result."""

    def __init__(self, tagger: RamPlusVideoTagger) -> None:
        self._tagger = tagger

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        """Evaluate one tag criterion for one accepted source."""
        del source, metadata
        if not isinstance(criterion, TagCriterion):
            raise InvalidVideoFilterInputError("RAM+ requires a tag criterion.")
        frame_results = self._tagger.tag(frames, deadline)
        normalized_by_frame: list[tuple[str, ...]] = []
        merged: list[str] = []
        confidence_rows: list[tuple[float, ...]] = []
        for frame_result in frame_results:
            normalized = tuple(dict.fromkeys(tag.strip().casefold() for tag in frame_result.tags))
            normalized_by_frame.append(normalized)
            confidence_rows.append(frame_result.confidences)
            for tag in normalized:
                if tag and tag not in merged:
                    merged.append(tag)
        merged_set = set(merged)
        if criterion.contain is Aggregation.ALL:
            passed = set(criterion.tags).issubset(merged_set)
        else:
            passed = bool(set(criterion.tags) & merged_set)
        return VideoCriterionResult(
            item_id=item.item_id,
            criterion_id=criterion.criterion_id,
            passed=passed,
            tags=tuple(merged),
            details={
                "frame_tags": tuple(normalized_by_frame),
                "frame_confidences": tuple(confidence_rows),
                "contain": criterion.contain.value,
                "weights_hash": self._tagger.weights_hash,
                "vocabulary_hash": self._tagger.vocabulary_hash,
                "backend": self._tagger.identity[0],
                "backend_version": self._tagger.identity[1],
            },
        )


class RamPlusLocalTagCriterionEvaluator:
    """Resolve and cache explicit local RAM+ assets per criterion."""

    def __init__(self, *, loader: RamLoader | None = None) -> None:
        self._loader = loader
        self._cache: dict[tuple[str, str], RamPlusTagCriterionEvaluator] = {}

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        """Load local RAM+ assets and apply the generic tag criterion."""
        if not isinstance(criterion, TagCriterion):
            raise InvalidVideoFilterInputError("RAM+ requires a tag criterion.")
        if criterion.vocabulary_path is None:
            raise InvalidVideoFilterInputError(
                "RAM+ tag filtering requires an explicit local vocabulary_path."
            )
        cache_key = (criterion.model.reference, criterion.vocabulary_path)
        evaluator = self._cache.get(cache_key)
        if evaluator is None:
            tagger = RamPlusVideoTagger.from_local(
                Path(criterion.model.reference),
                Path(criterion.vocabulary_path),
                loader=self._loader,
            )
            evaluator = RamPlusTagCriterionEvaluator(tagger)
            self._cache[cache_key] = evaluator
        return evaluator.evaluate(
            item,
            source,
            metadata,
            frames,
            criterion,
            deadline,
        )


def _load_ram_engine(weights_path: Path, vocabulary_path: Path) -> TagEngine:
    tokenizer_root = vocabulary_path.parent
    tokenizer_vocabulary = tokenizer_root / "vocab.txt"
    if not tokenizer_vocabulary.is_file() or tokenizer_vocabulary.is_symlink():
        raise BackendUnavailableFilterError(
            "RAM+ requires a provisioned local BERT tokenizer vocab.txt beside its tag vocabulary."
        )
    try:
        ram = importlib.import_module("ram")
        ram_models = importlib.import_module("ram.models")
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
    except ModuleNotFoundError as error:
        raise BackendUnavailableFilterError(
            "RAM+, PyTorch, and Pillow must be installed locally for tag filtering."
        ) from error
    try:
        model = ram_models.ram_plus(
            pretrained=str(weights_path),
            image_size=384,
            vit="swin_l",
            tag_list=str(vocabulary_path),
            text_encoder_type=str(tokenizer_root),
        )
        model.eval()
        transform = ram.get_transform(image_size=384)
    except (AttributeError, OSError, RuntimeError) as error:
        raise BackendUnavailableFilterError(
            "The installed RAM+ provider could not load the local weights."
        ) from error

    def engine(
        frames: tuple[SampledFrame, ...],
        deadline: datetime,
    ) -> tuple[FrameTags, ...]:
        outputs: list[FrameTags] = []
        for frame in frames:
            if datetime.now(UTC) >= deadline:
                raise VideoFilterTimeoutError("The RAM+ tagging deadline expired.")
            try:
                rgb = frame.payload[..., ::-1]  # type: ignore[index]
                image = image_module.fromarray(rgb)
                tensor = transform(image).unsqueeze(0)
                with torch.inference_mode():
                    generated, _ = model.generate_tag(tensor)
            except (AttributeError, RuntimeError, TypeError) as error:
                raise DecodeFailedFilterError("RAM+ could not tag a sampled frame.") from error
            value = generated[0] if isinstance(generated, (list, tuple)) else generated
            tags = tuple(tag.strip() for tag in str(value).split("|") if tag.strip())
            outputs.append(FrameTags(tags))
        return tuple(outputs)

    return engine


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        metadata = importlib.import_module("importlib.metadata")
        return str(metadata.version(name))
    except Exception:
        return None
