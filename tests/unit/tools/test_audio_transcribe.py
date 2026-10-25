from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError  # type: ignore[import-untyped]

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolErrorCode,
    ToolResources,
    ToolStatus,
)
from capicove.processing.audio import (
    AudioBackendLineage,
    AudioEvidence,
    AudioFailureCode,
    AudioImportError,
    AudioStreamSelection,
    AudioTranscript,
    AudioTranscriptionError,
    AudioTranscriptionResult,
    AudioTranscriptionSegment,
)
from capicove.processing.audio.ports import AudioPathPolicyError
from capicove.tools import ToolRegistry
from capicove.tools.audio_transcribe import (
    AUDIO_TRANSCRIBE_OUTPUT_SCHEMA,
    AUDIO_TRANSCRIBE_SPEC,
    AudioTranscribeToolHandler,
    audio_transcribe_capabilities,
    register_audio_transcribe_tool,
    sanitize_audio_transcribe_arguments,
)


class _Authorizer:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.paths: list[Path] = []

    def authorize_read(self, policy_ref: str, path: Path) -> Path:
        self.paths.append(path)
        if self.reject:
            raise AudioPathPolicyError("path denied")
        return path.resolve()


_NOT_PROVIDED = object()


class _Service:
    def __init__(self, result: AudioTranscriptionResult | None = None) -> None:
        self.result = result or _result()
        self.calls: list[tuple[Path, Path, str, datetime, object]] = []

    def transcribe_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        language: str,
        deadline: datetime,
        audio_stream_index: object = _NOT_PROVIDED,
    ) -> AudioTranscriptionResult:
        self.calls.append((source_path, store_root, language, deadline, audio_stream_index))
        return self.result


class _FailingService:
    def __init__(self, code: AudioFailureCode) -> None:
        self.code = code

    def transcribe_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        language: str,
        deadline: datetime,
        audio_stream_index: object = _NOT_PROVIDED,
    ) -> AudioTranscriptionResult:
        raise AudioTranscriptionError(self.code, "bounded audio failure")


class _ImportFailingService:
    def __init__(self, code: AudioFailureCode, *, count: int | None = None) -> None:
        self.code = code
        self.count = count

    def transcribe_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        language: str,
        deadline: datetime,
        audio_stream_index: object = _NOT_PROVIDED,
    ) -> AudioTranscriptionResult:
        raise AudioImportError(
            self.code,
            "bounded intake failure",
            preflight_audio_stream_count=self.count,
        )


def _result() -> AudioTranscriptionResult:
    segment = AudioTranscriptionSegment(0, 0.0, 1.0, "hello")
    return AudioTranscriptionResult(
        AudioEvidence(
            "sha256:" + "b" * 64,
            "b" * 64,
            10,
            "audio/wav",
            "meeting.wav",
            Path("/store/blob"),
        ),
        AudioTranscript(
            "hello",
            "en",
            (segment,),
            (),
            AudioBackendLineage(
                "whisper.cpp",
                "1.7.6",
                "tiny",
                "a" * 64,
                "cpu",
                "q5_1",
                "silero-v6.2.0",
                "c" * 64,
            ),
            1.0,
        ),
        AudioStreamSelection(1, "explicit_index", 2),
    )


def _context(permissions: frozenset[str] | None = None) -> InvocationContext:
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="agent-1",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        permissions=permissions or AUDIO_TRANSCRIBE_SPEC.required_permissions,
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def _arguments() -> dict[str, object]:
    return {"source": {"type": "path", "path": "/audio/meeting.wav"}, "language": "auto"}


def test_spec_declares_offline_confidential_security_boundary() -> None:
    assert AUDIO_TRANSCRIBE_SPEC.tool_id == "capicove.audio.transcribe"
    assert AUDIO_TRANSCRIBE_SPEC.contract_version == 2
    assert AUDIO_TRANSCRIBE_SPEC.modalities == ("audio", "text")
    assert AUDIO_TRANSCRIBE_SPEC.side_effect is SideEffect.WRITE
    assert AUDIO_TRANSCRIBE_SPEC.privacy_class is PrivacyClass.CONFIDENTIAL
    assert AUDIO_TRANSCRIBE_SPEC.network_access is False
    assert AUDIO_TRANSCRIBE_SPEC.resources == ToolResources(
        filesystem_read=True,
        filesystem_write=True,
        subprocesses=(
            "ffprobe",
            "ffmpeg",
            "whisper-cli",
            "whisper-vad-speech-segments",
        ),
    )
    properties = AUDIO_TRANSCRIBE_SPEC.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert properties["audio_stream_index"] == {"type": "integer", "minimum": 0}


def test_output_schema_closes_success_and_domain_failure_objects() -> None:
    branches = AUDIO_TRANSCRIBE_OUTPUT_SCHEMA["oneOf"]
    assert isinstance(branches, list)
    success = branches[0]
    rejected = branches[1]
    failed = branches[2]
    assert isinstance(success, Mapping)
    assert isinstance(rejected, Mapping)
    assert isinstance(failed, Mapping)
    properties = success["properties"]
    assert isinstance(properties, Mapping)

    segments = properties["segments"]
    evidence = properties["evidence"]
    lineage = properties["lineage"]
    assert isinstance(segments, Mapping)
    assert isinstance(evidence, Mapping)
    assert isinstance(lineage, Mapping)

    segment = segments["items"]
    assert isinstance(segment, Mapping)
    assert segment["additionalProperties"] is False
    assert segment["required"] == ["index", "start_seconds", "end_seconds", "text"]

    assert evidence["additionalProperties"] is False
    assert evidence["required"] == [
        "evidence_id",
        "sha256",
        "size_bytes",
        "media_type",
        "source_name",
    ]
    assert lineage["additionalProperties"] is False
    assert lineage["required"] == [
        "backend",
        "backend_version",
        "model",
        "model_sha256",
        "vad_model",
        "vad_model_sha256",
        "device",
        "precision",
        "audio_stream_selection",
    ]
    lineage_properties = lineage["properties"]
    assert isinstance(lineage_properties, Mapping)
    selection = lineage_properties["audio_stream_selection"]
    assert isinstance(selection, Mapping)
    assert selection["additionalProperties"] is False
    assert selection["required"] == [
        "schema_version",
        "selected_audio_stream_index",
        "index_semantics",
        "selection_method",
        "preflight_audio_stream_count",
        "validation_scope",
    ]
    assert rejected["additionalProperties"] is False
    assert failed["additionalProperties"] is False


def test_output_schema_validates_real_success_rejection_and_failure_results() -> None:
    Draft202012Validator.check_schema(AUDIO_TRANSCRIBE_OUTPUT_SCHEMA)
    validator = Draft202012Validator(AUDIO_TRANSCRIBE_OUTPUT_SCHEMA)
    success = AudioTranscribeToolHandler(
        _Service(),
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )(_context(), _arguments())
    rejected = AudioTranscribeToolHandler(
        _ImportFailingService(
            AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED,
            count=2,
        ),
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )(_context(), _arguments())
    failed = AudioTranscribeToolHandler(
        _ImportFailingService(AudioFailureCode.NO_AUDIO_STREAM, count=0),
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )(_context(), _arguments())

    for result in (success, rejected, failed):
        assert result.result is not None
        validator.validate(dict(result.result))

    assert success.result is not None
    with pytest.raises(ValidationError):
        validator.validate({**dict(success.result), "unexpected": True})


def test_handler_maps_success_without_exposing_server_storage_path() -> None:
    service = _Service()
    authorizer = _Authorizer()
    handler = AudioTranscribeToolHandler(
        service,
        authorizer,
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )

    result = handler(_context(), _arguments())

    assert result.status is ToolStatus.SUCCEEDED
    assert result.evidence_refs == ("sha256:" + "b" * 64,)
    assert result.result is not None
    assert result.result["status"] == "succeeded"
    assert result.result["text"] == "hello"
    lineage = result.result["lineage"]
    assert isinstance(lineage, Mapping)
    assert lineage["vad_model_sha256"] == "c" * 64
    assert lineage["audio_stream_selection"] == {
        "schema_version": 1,
        "selected_audio_stream_index": 1,
        "index_semantics": "zero_based_audio_stream_ordinal",
        "selection_method": "explicit_index",
        "preflight_audio_stream_count": 2,
        "validation_scope": "selected_audio_stream_only",
    }
    assert result.metrics["input_duration_seconds"] == 1.0
    assert result.metrics["end_to_end_real_time_factor"] >= 0
    assert "/store" not in repr(result.result)
    assert authorizer.paths == [Path("/audio/meeting.wav")]
    assert service.calls[0][0] == Path("/audio/meeting.wav").resolve()
    assert service.calls[0][2] == "auto"
    assert service.calls[0][4] is _NOT_PROVIDED


def test_handler_passes_explicit_selector_without_conflating_omission() -> None:
    service = _Service()
    handler = AudioTranscribeToolHandler(
        service,
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )
    arguments = _arguments()
    arguments["audio_stream_index"] = 1

    result = handler(_context(), arguments)

    assert result.status is ToolStatus.SUCCEEDED
    assert service.calls[0][4] == 1


def test_handler_rejects_invalid_selector_with_stable_domain_result() -> None:
    for selector in (None, True, False, 1.0, "1", -1):
        service = _Service()
        arguments = _arguments()
        arguments["audio_stream_index"] = selector
        result = AudioTranscribeToolHandler(
            service,
            _Authorizer(),
            execution_policy_ref="policy-1",
            store_root=Path("/store"),
        )(_context(), arguments)

        assert result.status is ToolStatus.FAILED
        assert result.error is not None
        assert result.error.code is ToolErrorCode.INVALID_INPUT
        assert result.result == {
            "status": "failed",
            "failure": {
                "code": "INVALID_AUDIO_STREAM_INDEX",
                "message": "audio_stream_index must be a non-negative integer",
                "retryable": False,
            },
        }
        assert service.calls == []


def test_handler_rejects_missing_permission_and_path_policy() -> None:
    service = _Service()
    missing = AudioTranscribeToolHandler(
        service,
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )(_context(frozenset({"audio:read"})), _arguments())
    assert missing.status is ToolStatus.REJECTED
    assert missing.error is not None
    assert missing.error.code is ToolErrorCode.PERMISSION_DENIED
    assert service.calls == []

    denied = AudioTranscribeToolHandler(
        service,
        _Authorizer(reject=True),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )(_context(), _arguments())
    assert denied.status is ToolStatus.REJECTED
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.PERMISSION_DENIED


def test_handler_rejects_unknown_source_types_and_fields() -> None:
    handler = AudioTranscribeToolHandler(
        _Service(),
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )
    invalid = _arguments()
    invalid["download"] = True
    result = handler(_context(), invalid)
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT

    inline = {"source": {"type": "bytes", "base64": "c2VjcmV0"}, "language": "auto"}
    result = handler(_context(), inline)
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT


def test_handler_maps_stable_backend_failures() -> None:
    expected = {
        AudioFailureCode.INVALID_AUDIO: ToolErrorCode.INVALID_INPUT,
        AudioFailureCode.UNSUPPORTED_LANGUAGE: ToolErrorCode.INVALID_INPUT,
        AudioFailureCode.MODEL_UNAVAILABLE: ToolErrorCode.UNAVAILABLE,
        AudioFailureCode.MODEL_LOAD_FAILED: ToolErrorCode.UNAVAILABLE,
        AudioFailureCode.TIMEOUT: ToolErrorCode.TIMEOUT,
        AudioFailureCode.STORE_CORRUPTED: ToolErrorCode.INTERNAL,
        AudioFailureCode.CANCELLED: ToolErrorCode.INTERNAL,
    }
    for audio_code, tool_code in expected.items():
        result = AudioTranscribeToolHandler(
            _FailingService(audio_code),
            _Authorizer(),
            execution_policy_ref="policy-1",
            store_root=Path("/store"),
        )(_context(), _arguments())
        assert result.error is not None
        assert result.error.code is tool_code
        assert result.result == {
            "status": "failed",
            "failure": {
                "code": audio_code.value,
                "message": "bounded audio failure",
                "retryable": audio_code is AudioFailureCode.TIMEOUT,
            },
        }


def test_handler_preserves_stable_intake_failure_mapping() -> None:
    for audio_code, tool_code in (
        (AudioFailureCode.INVALID_AUDIO, ToolErrorCode.INVALID_INPUT),
        (AudioFailureCode.DECODER_UNAVAILABLE, ToolErrorCode.UNAVAILABLE),
        (AudioFailureCode.TIMEOUT, ToolErrorCode.TIMEOUT),
        (AudioFailureCode.STORE_FAILED, ToolErrorCode.INTERNAL),
        (AudioFailureCode.STORE_CORRUPTED, ToolErrorCode.INTERNAL),
    ):
        result = AudioTranscribeToolHandler(
            _ImportFailingService(audio_code),
            _Authorizer(),
            execution_policy_ref="policy-1",
            store_root=Path("/store"),
        )(_context(), _arguments())
        assert result.error is not None
        assert result.error.code is tool_code


def test_handler_returns_structured_audio_stream_rejection_and_failures() -> None:
    ambiguity = AudioTranscribeToolHandler(
        _ImportFailingService(
            AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED,
            count=2,
        ),
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )(_context(), _arguments())

    assert ambiguity.status is ToolStatus.REJECTED
    assert ambiguity.error is not None
    assert ambiguity.error.code is ToolErrorCode.POLICY_REJECTED
    assert ambiguity.evidence_refs == ()
    assert ambiguity.result == {
        "status": "rejected",
        "failure": {
            "code": "AUDIO_STREAM_SELECTION_REQUIRED",
            "message": "bounded intake failure",
            "retryable": False,
            "preflight_audio_stream_count": 2,
        },
    }

    for code in (
        AudioFailureCode.INVALID_AUDIO_STREAM_INDEX,
        AudioFailureCode.NO_AUDIO_STREAM,
        AudioFailureCode.AUDIO_STREAM_INDEX_OUT_OF_RANGE,
        AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE,
    ):
        failed = AudioTranscribeToolHandler(
            _ImportFailingService(code, count=0 if code is AudioFailureCode.NO_AUDIO_STREAM else 2),
            _Authorizer(),
            execution_policy_ref="policy-1",
            store_root=Path("/store"),
        )(_context(), _arguments())
        assert failed.status is ToolStatus.FAILED
        assert failed.error is not None
        assert failed.error.code is ToolErrorCode.INVALID_INPUT
        assert failed.result is not None
        assert failed.result["status"] == "failed"
        failure = failed.result["failure"]
        assert isinstance(failure, Mapping)
        assert failure["code"] == code.value


def test_trace_sanitization_replaces_private_path_with_name_only() -> None:
    sanitized = sanitize_audio_transcribe_arguments(_arguments())

    source = sanitized["source"]
    assert isinstance(source, Mapping)
    assert source == {"type": "path", "name": "meeting.wav"}


def test_capabilities_are_runtime_bound_and_mark_unimplemented_operations() -> None:
    capabilities = audio_transcribe_capabilities(
        executable_available=True,
        vad_executable_available=True,
        decoder_available=True,
        probe_available=True,
        model_available=True,
        model_sha256="a" * 64,
        vad_model_available=True,
        vad_model_sha256="c" * 64,
        backend_version="1.7.6",
    )

    assert capabilities["component"] == "capicove.audio.transcribe"
    assert capabilities["available"] is True
    assert capabilities["network_access"] is False
    assert capabilities["vad"] == {
        "enabled": True,
        "executable_available": True,
        "model_sha256": "c" * 64,
        "streaming": False,
    }
    assert capabilities["decoder"] == {
        "available": True,
        "backend": "ffmpeg",
        "probe_available": True,
        "probe_backend": "ffprobe",
    }
    assert capabilities["audio_stream_selection"] == {
        "index_semantics": "zero_based_audio_stream_ordinal",
        "single_stream_omission": "sole_stream_implicit",
        "multi_stream_omission": "rejected",
        "validation_scope": "selected_audio_stream_only",
    }
    assert capabilities["supported_languages"] == ["zh", "en", "auto"]
    assert capabilities["unimplemented_operations"] == [
        "audio.synthesize",
        "audio.detect_voice",
        "audio.diarize",
        "audio.denoise",
        "audio.get_job",
        "audio.cancel_job",
    ]

    without_probe = audio_transcribe_capabilities(
        executable_available=True,
        vad_executable_available=True,
        decoder_available=True,
        probe_available=False,
        model_available=True,
        model_sha256="a" * 64,
        vad_model_available=True,
        vad_model_sha256="c" * 64,
        backend_version="1.7.6",
    )
    assert without_probe["available"] is False
    decoder = without_probe["decoder"]
    assert isinstance(decoder, Mapping)
    assert decoder["available"] is False


def test_registration_uses_canonical_registry_key() -> None:
    registry = ToolRegistry()
    handler = register_audio_transcribe_tool(
        registry,
        _Service(),
        _Authorizer(),
        execution_policy_ref="policy-1",
        store_root=Path("/store"),
    )

    registered = registry.resolve("capicove.audio.transcribe", 2)
    assert registered.spec is AUDIO_TRANSCRIBE_SPEC
    assert registered.handler is handler
