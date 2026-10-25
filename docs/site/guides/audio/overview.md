# Audio processing

Audio processing has two deliberate stages:

1. import and validate local audio as immutable evidence; and
2. transcribe authorized evidence with explicit local executables and model files.

The workflow is local by default. It records source hashes, explicit audio-stream
selection, backend/model lineage, segments, warnings, and structured failures.
Capicove does not download Whisper models, guess a stream from metadata, mix
streams, or upload audio.

Continue with [evidence import](import.md) or [transcription](transcription.md).
