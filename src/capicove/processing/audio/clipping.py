"""Evidence-preserving orchestration of sample-accurate local audio clipping."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from capicove.processing.audio.clip_models import (
    AudioClipBackend,
    AudioClipGeometry,
    AudioClippingError,
    AudioClipRequest,
    remaining_seconds,
)
from capicove.processing.audio.models import AudioFailureCode
from capicove.processing.audio.processor import AudioImportError, AudioProcessor


class AudioClippingService:
    """Preserve evidence, derive one clip, and acknowledge verified publication."""

    def __init__(
        self,
        processor: AudioProcessor,
        backend: AudioClipBackend,
        *,
        max_duration_seconds: int | None = None,
    ) -> None:
        self._processor = processor
        self._backend = backend
        self._max_duration_seconds = max_duration_seconds

    def clip_file(
        self, request: AudioClipRequest, store_root: Path, *, deadline: datetime
    ) -> dict[str, object]:
        """Return path-free metadata and lineage only after secure publication."""
        from capicove.processing.audio._adapters.clip_files import (
            ClipOutput,
            admit_directory,
            hash_descriptor,
            open_directory,
            require_binding,
            same_file,
            verified_source,
            verify_wav,
        )
        from capicove.processing.audio._adapters.ffmpeg import AUDIO_DEMUXERS

        remaining_seconds(deadline)
        source, destination = request.source.absolute(), request.destination.absolute()
        if source.suffix.lower() not in AUDIO_DEMUXERS:
            raise AudioClippingError(
                AudioFailureCode.UNSUPPORTED_FORMAT, "unsupported audio extension"
            )
        version = self._backend.preflight(deadline)
        try:
            with (
                open_directory(destination.parent) as parent,
                open_directory(store_root, create=True) as store,
            ):
                for descriptor in (parent, store):
                    remaining_seconds(deadline)
                    admit_directory(descriptor)
                require_binding(store_root, store)
                try:
                    with verified_source(source, deadline) as snapshot:
                        evidence = self._processor.import_file(
                            snapshot, store_root, deadline=deadline
                        )
                except AudioImportError as error:
                    raise AudioClippingError(
                        error.code, "source evidence could not be imported"
                    ) from None
                with open_directory(evidence.stored_path.parent) as evidence_parent:
                    descriptor = os.open(
                        evidence.stored_path.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=evidence_parent,
                    )
                    try:
                        identity = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(identity.st_mode)
                            or identity.st_nlink != 1
                            or identity.st_size != evidence.size_bytes
                            or hash_descriptor(descriptor, deadline) != evidence.sha256
                        ):
                            raise AudioClippingError(
                                AudioFailureCode.STORE_CORRUPTED,
                                "immutable source evidence is invalid",
                            )
                        demuxer = AUDIO_DEMUXERS[source.suffix.lower()]
                        metadata = self._backend.inspect(descriptor, demuxer, deadline)
                        if (
                            self._max_duration_seconds is not None
                            and metadata.duration_seconds > self._max_duration_seconds
                        ):
                            raise AudioClippingError(
                                AudioFailureCode.INVALID_AUDIO,
                                "audio exceeds the configured duration limit",
                            )
                        geometry = AudioClipGeometry.resolve(request, metadata)
                        output = ClipOutput(destination, parent, geometry)
                        try:
                            self._backend.write(
                                descriptor, output.descriptor, demuxer, geometry, deadline
                            )
                            output_metadata = verify_wav(output.descriptor, geometry, deadline)
                            digest = hash_descriptor(output.descriptor, deadline)
                            size = os.fstat(output.descriptor).st_size
                            remaining_seconds(deadline)
                            output.publish()
                            if (
                                verify_wav(output.descriptor, geometry, deadline) != output_metadata
                                or hash_descriptor(output.descriptor, deadline) != digest
                            ):
                                raise AudioClippingError(
                                    AudioFailureCode.OUTPUT_CORRUPTED, "published output changed"
                                )
                            current = os.stat(
                                evidence.stored_path.name,
                                dir_fd=evidence_parent,
                                follow_symlinks=False,
                            )
                            if (
                                not same_file(current, identity)
                                or current.st_nlink != 1
                                or current.st_size != evidence.size_bytes
                                or hash_descriptor(descriptor, deadline) != evidence.sha256
                            ):
                                raise AudioClippingError(
                                    AudioFailureCode.STORE_CORRUPTED, "source evidence changed"
                                )
                            require_binding(evidence.stored_path.parent, evidence_parent)
                            require_binding(store_root, store)
                            require_binding(destination.parent, parent)
                            if not output._owned(destination.name):
                                raise AudioClippingError(
                                    AudioFailureCode.OUTPUT_CORRUPTED,
                                    "final output binding changed",
                                )
                            config = {
                                "profile": request.profile,
                                "start_frame": geometry.start_frame,
                                "end_frame": geometry.end_frame,
                                "sample_rate_hz": geometry.sample_rate_hz,
                                "channels": geometry.channels,
                                "backend_version": version,
                            }
                            result: dict[str, object] = {
                                "requested_range": {
                                    "start_seconds": request.start_seconds,
                                    "end_seconds": request.end_seconds,
                                },
                                "actual_range": {
                                    "start_frame": geometry.start_frame,
                                    "end_frame": geometry.end_frame,
                                    "start_seconds": geometry.start_frame / geometry.source_rate_hz,
                                    "end_seconds": geometry.end_frame / geometry.source_rate_hz,
                                },
                                "input": {
                                    key: value
                                    for key, value in asdict(metadata).items()
                                    if key != "duration_exact"
                                },
                                "output": output_metadata,
                                "evidence": {
                                    "evidence_id": evidence.evidence_id,
                                    "sha256": evidence.sha256,
                                    "size_bytes": evidence.size_bytes,
                                    "media_type": evidence.media_type,
                                    "source_name": source.name,
                                },
                                "artifact": {
                                    "artifact_id": f"sha256:{digest}",
                                    "sha256": digest,
                                    "size_bytes": size,
                                    "media_type": "audio/wav",
                                    "destination_name": destination.name,
                                },
                                "lineage": {
                                    "backend": "ffmpeg",
                                    "backend_version": version,
                                    "profile": request.profile,
                                    "configuration_sha256": hashlib.sha256(
                                        json.dumps(
                                            config, sort_keys=True, separators=(",", ":")
                                        ).encode()
                                    ).hexdigest(),
                                    "source_audio_stream_index": 0,
                                },
                            }
                            remaining_seconds(deadline)
                            output.acknowledged = True
                            return result
                        finally:
                            output.close()
                    finally:
                        os.close(descriptor)
        except AudioClippingError:
            raise
        except OSError as error:
            if error.errno in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
                raise AudioClippingError(
                    AudioFailureCode.BACKEND_UNAVAILABLE,
                    "secure filesystem semantics are unavailable",
                    reason="filesystem_profile_unavailable",
                ) from None
            raise AudioClippingError(
                AudioFailureCode.OUTPUT_FAILED, "audio output or storage operation failed"
            ) from None
