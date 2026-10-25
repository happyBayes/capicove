"""Audio evidence intake and validation."""

from capicove.processing.audio.clip_models import AudioClippingError, AudioClipRequest
from capicove.processing.audio.clipping import AudioClippingService
from capicove.processing.audio.models import (
    AudioBackendLineage,
    AudioEvidence,
    AudioFailureCode,
    AudioImportResult,
    AudioStreamInventory,
    AudioStreamSelection,
    AudioTranscript,
    AudioTranscriptionResult,
    AudioTranscriptionSegment,
    AudioWarningCode,
)
from capicove.processing.audio.ports import (
    AudioStreamInspector,
    AudioTranscriber,
    AudioValidationError,
    AudioValidator,
)
from capicove.processing.audio.processor import AudioImportError, AudioProcessor
from capicove.processing.audio.transcription import (
    AudioTranscriptionError,
    AudioTranscriptionService,
)


__all__ = [
    "AudioBackendLineage",
    "AudioClipRequest",
    "AudioClippingError",
    "AudioClippingService",
    "AudioEvidence",
    "AudioFailureCode",
    "AudioImportError",
    "AudioImportResult",
    "AudioProcessor",
    "AudioStreamInspector",
    "AudioStreamInventory",
    "AudioStreamSelection",
    "AudioTranscriber",
    "AudioTranscript",
    "AudioTranscriptionError",
    "AudioTranscriptionResult",
    "AudioTranscriptionSegment",
    "AudioTranscriptionService",
    "AudioValidationError",
    "AudioValidator",
    "AudioWarningCode",
]
