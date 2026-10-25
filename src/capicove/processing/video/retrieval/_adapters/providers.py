"""Adapters for explicitly provisioned local model and audio services."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from capicove.contracts.retrieval import EmbeddingSpaceRef, ProcessingFingerprint
from capicove.processing.audio.models import AudioFailureCode, AudioTranscript
from capicove.processing.audio.transcription import (
    AudioTranscriptionError,
    AudioTranscriptionService,
)
from capicove.processing.video.models import AcceptedVideoSource
from capicove.processing.video.ports import BackendUnavailableError
from capicove.processing.video.retrieval.models import RetrievalKeyframe
from capicove.processing.video.retrieval.pipeline import check_deadline
from capicove.processing.video.retrieval.ports import PreviewStore


class PreviewBatchEncoder:
    """Bound batches for a caller-provisioned local image encoder.

    The composition root must verify the callable's model identity against the
    declared space. This adapter performs no model loading or network fallback.
    """

    def __init__(
        self,
        preview_store: PreviewStore,
        embedding_space: EmbeddingSpaceRef,
        encode_batch: Callable[[tuple[bytes, ...]], tuple[tuple[float, ...], ...]],
        *,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._store = preview_store
        self._space = embedding_space
        self._encode = encode_batch
        self._batch_size = batch_size

    @property
    def embedding_space(self) -> EmbeddingSpaceRef:
        """Return the explicitly bound model identity."""
        return self._space

    def encode_keyframes(
        self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        """Read verified previews and invoke bounded local batches."""
        result: list[tuple[float, ...]] = []
        for start in range(0, len(keyframes), self._batch_size):
            check_deadline(deadline)
            batch = keyframes[start : start + self._batch_size]
            vectors = self._encode(tuple(self._store.read(k.preview_resource_ref) for k in batch))
            if len(vectors) != len(batch):
                raise ValueError("encoder batch count mismatch")
            result.extend(vectors)
        check_deadline(deadline)
        return tuple(result)


class LocalASRProcessor:
    """Delegate to the existing public audio evidence/transcription operation."""

    def __init__(
        self,
        service: AudioTranscriptionService,
        store_root: Path,
        *,
        provider_fingerprint: ProcessingFingerprint,
        language: str = "auto",
        audio_stream_index: int | None = None,
    ) -> None:
        self._service = service
        self._root = store_root
        self._language = language
        self._index = audio_stream_index
        self._fingerprint = ProcessingFingerprint.from_manifest(
            {
                "provider": provider_fingerprint.sha256,
                "language": language,
                "audio_stream_index": audio_stream_index,
                "adapter": "public-audio-transcription-v1",
            }
        )

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Bind caller-verified model, options, language and stream identity."""
        return self._fingerprint

    def transcribe(
        self, source: AcceptedVideoSource, *, deadline: datetime
    ) -> AudioTranscript | None:
        """Return no units for accepted no-audio/no-speech outcomes."""
        check_deadline(deadline)
        if source.path is None:
            raise ValueError("ASR requires a materialized authorized source path")
        try:
            if self._index is None:
                result = self._service.transcribe_file(
                    source.path, self._root, language=self._language, deadline=deadline
                )
            else:
                result = self._service.transcribe_file(
                    source.path,
                    self._root,
                    language=self._language,
                    deadline=deadline,
                    audio_stream_index=self._index,
                )
        except AudioTranscriptionError as error:
            if error.code == AudioFailureCode.NO_AUDIO_STREAM:
                return None
            if error.code in {
                AudioFailureCode.MODEL_UNAVAILABLE,
                AudioFailureCode.DECODER_UNAVAILABLE,
            }:
                raise BackendUnavailableError("ASR backend unavailable") from error
            raise
        return result.transcript if result.transcript.segments else None
