"""Private audio processing adapters."""

from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
from capicove.processing.audio._adapters.filesystem import ExplicitAudioPathAuthorizer
from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber


__all__ = [
    "ExplicitAudioPathAuthorizer",
    "FfmpegAudioValidator",
    "FfprobeAudioStreamInspector",
    "WhisperCppTranscriber",
]
