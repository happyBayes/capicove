"""Operator trace inspection must expose counts without trace content."""

import hashlib
import json
import os
from dataclasses import asdict, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.contracts import PrivacyClass
from capicove.export import (
    MediaMode,
    TraceExporter,
    TraceExportOptions,
    TraceValidationIssue,
    TraceValidationReport,
    ValidationSeverity,
    validate_trace_export,
)
from capicove.traces import (
    EpisodeCompleted,
    EpisodeStarted,
    EpisodeStatus,
    MediaContent,
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    MessageObserved,
    MessageRole,
    TraceModality,
    TraceRecordDraft,
)
from capicove.traces._adapters.sqlite import SqliteTraceStore
from capicove.traces.ports import DEFAULT_TRACE_QUERY, TraceQuery, TraceReadSnapshot


NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _start(
    store: SqliteTraceStore,
    episode: str,
    *,
    privacy: PrivacyClass,
    workspace_id: str = "workspace-one",
) -> None:
    store.append(
        TraceRecordDraft(
            episode_id=episode,
            trace_id=f"trace-{episode}",
            occurred_at=NOW,
            payload=EpisodeStarted(
                goal="private full message must not be returned",
                workspace_id=workspace_id,
                actor_id="actor-one",
                environment_id="local",
                environment_version=1,
                tool_catalog_sha256="a" * 64,
                active_asset_versions={},
            ),
            privacy_class=privacy,
        ),
        idempotency_key="start",
    )


def test_inspect_store_reports_only_scoped_episode_counts(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorReader

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "completed", privacy=PrivacyClass.INTERNAL)
    store.append(
        TraceRecordDraft(
            episode_id="completed",
            trace_id="trace-completed",
            occurred_at=NOW + timedelta(seconds=1),
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        ),
        idempotency_key="complete",
    )
    _start(store, "incomplete", privacy=PrivacyClass.INTERNAL)
    _start(store, "restricted", privacy=PrivacyClass.RESTRICTED)
    reader = TraceOperatorReader(stores={"local": store})

    inspection = reader.inspect_store("local")

    assert asdict(inspection) == {
        "store_alias": "local",
        "episode_count": 2,
        "incomplete_episode_count": 1,
        "omitted_restricted_episode_count": 1,
    }
    assert "private full message" not in json.dumps(asdict(inspection))
    assert str(tmp_path) not in json.dumps(asdict(inspection))


def test_inspect_store_rejects_unconfigured_alias(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorReader

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    reader = TraceOperatorReader(stores={"local": store})

    with pytest.raises(KeyError, match="trace store is unavailable"):
        reader.inspect_store("foreign")


def test_inspect_store_rejects_path_like_alias_before_exposure(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorReader

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")

    with pytest.raises(ValueError, match="aliases"):
        TraceOperatorReader(stores={str(tmp_path): store})


def test_inspect_store_does_not_echo_reader_path_or_content() -> None:
    from capicove.traces.operator import TraceOperatorReader

    class FailedReader:
        def open_snapshot(self, _query: TraceQuery = DEFAULT_TRACE_QUERY, /) -> TraceReadSnapshot:
            raise OSError("private trace /outside/store and message body")

    reader = TraceOperatorReader(stores={"local": FailedReader()})

    with pytest.raises(ValueError, match=r"^trace store inspection failed$") as error:
        reader.inspect_store("local")
    assert "/outside/store" not in str(error.value)
    assert "message body" not in str(error.value)


def test_validate_export_reports_bounded_metadata_without_record_content(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorReader

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "completed", privacy=PrivacyClass.INTERNAL)
    store.append(
        TraceRecordDraft(
            episode_id="completed",
            trace_id="trace-completed",
            occurred_at=NOW + timedelta(seconds=1),
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        ),
        idempotency_key="complete",
    )
    export = tmp_path / "export"
    TraceExporter(store, clock=lambda: NOW, export_id_factory=lambda: "export-one").export(export)
    reader = TraceOperatorReader(
        stores={},
        exports={
            "export-one": lambda verify_media: validate_trace_export(
                export, verify_media=verify_media
            )
        },
    )

    inspection = reader.validate_export("export-one", verify_media=False)

    assert asdict(inspection) == {
        "export_id": "export-one",
        "valid": True,
        "records_checked": 2,
        "episodes_checked": 1,
        "media_checked": 0,
        "findings": (),
        "findings_truncated": False,
    }
    assert "private full message" not in json.dumps(asdict(inspection))
    assert str(tmp_path) not in json.dumps(asdict(inspection))

    inspected_export = reader.inspect_export("export-one")
    assert inspected_export == inspection

    shard = next(export.glob("records-*.jsonl"))
    shard.write_bytes(b'{"episode_id":"private-episode","payload":"private full message"}\n')
    invalid = reader.validate_export("export-one", verify_media=False)
    assert not invalid.valid
    assert invalid.findings
    assert "private-episode" not in json.dumps(asdict(invalid))
    assert "private full message" not in json.dumps(asdict(invalid))
    assert str(tmp_path) not in json.dumps(asdict(invalid))


def test_validate_export_rejects_unconfigured_id_without_invoking_validator() -> None:
    from capicove.traces.operator import TraceOperatorReader

    calls: list[bool] = []

    def unexpected(verify_media: bool) -> TraceValidationReport:
        calls.append(verify_media)
        raise AssertionError("unconfigured export invoked a validator")

    reader = TraceOperatorReader(
        stores={},
        exports={"configured": unexpected},
    )

    with pytest.raises(KeyError, match="trace export is unavailable"):
        reader.validate_export("foreign", verify_media=True)
    assert calls == []


def test_validate_export_does_not_echo_validator_path_or_content() -> None:
    from capicove.traces.operator import TraceOperatorReader

    def failed_validation(_verify_media: bool) -> TraceValidationReport:
        raise OSError("private source /outside/export and message body")

    reader = TraceOperatorReader(stores={}, exports={"local": failed_validation})

    with pytest.raises(ValueError, match=r"^trace export validation failed$") as error:
        reader.validate_export("local", verify_media=True)
    assert "/outside/export" not in str(error.value)
    assert "message body" not in str(error.value)


def test_validate_export_requires_configured_mapping() -> None:
    from capicove.traces.operator import TraceOperatorReader

    with pytest.raises(ValueError, match="trace resources"):
        TraceOperatorReader(stores={}, exports=[])  # type: ignore[arg-type]


def test_inspect_rejects_ambiguous_store_and_export_alias(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorReader

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")

    with pytest.raises(ValueError, match="distinct aliases"):
        TraceOperatorReader(
            stores={"local": store},
            exports={
                "local": lambda verify_media: validate_trace_export(
                    tmp_path, verify_media=verify_media
                )
            },
        )


def test_validate_export_rejects_oversized_finding_code() -> None:
    from capicove.traces.operator import TraceOperatorReader

    report = TraceValidationReport(
        valid=False,
        records_checked=0,
        episodes_checked=0,
        media_checked=0,
        issues=(TraceValidationIssue(ValidationSeverity.ERROR, "X" * 100_000, "private"),),
    )
    reader = TraceOperatorReader(stores={}, exports={"local": lambda _verify: report})

    with pytest.raises(ValueError, match=r"^trace export validation failed$"):
        reader.validate_export("local", verify_media=True)


def test_validate_export_rejects_nonboolean_media_check_before_invoking_validator() -> None:
    from capicove.traces.operator import TraceOperatorReader

    calls: list[object] = []

    def unexpected(value: bool) -> TraceValidationReport:
        calls.append(value)
        raise AssertionError("invalid request reached validator")

    reader = TraceOperatorReader(stores={}, exports={"local": unexpected})

    with pytest.raises(ValueError, match="verify_media must be a boolean"):
        reader.validate_export("local", verify_media=1)  # type: ignore[arg-type]
    assert calls == []


def test_validate_export_rejects_invalid_configured_validator_result() -> None:
    from capicove.traces.operator import TraceOperatorReader

    def invalid_validator(_verify: bool) -> TraceValidationReport:
        return None  # type: ignore[return-value]

    reader = TraceOperatorReader(stores={}, exports={"local": invalid_validator})

    with pytest.raises(ValueError, match="invalid report"):
        reader.validate_export("local", verify_media=False)


def test_trace_reader_rejects_noncallable_validator_at_startup() -> None:
    from capicove.traces.operator import TraceOperatorReader

    with pytest.raises(ValueError, match="configured validators"):
        TraceOperatorReader(stores={}, exports={"local": None})  # type: ignore[dict-item]


def test_operator_export_uses_configured_root_and_idempotent_receipt(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "completed", privacy=PrivacyClass.INTERNAL)
    store.append(
        TraceRecordDraft(
            episode_id="completed",
            trace_id="trace-completed",
            occurred_at=NOW + timedelta(seconds=1),
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        ),
        idempotency_key="complete",
    )
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )

    first = exporter.export(
        output_root_alias="private", query=TraceQuery(), idempotency_key="request-one"
    )
    repeated = exporter.export(
        output_root_alias="private", query=TraceQuery(), idempotency_key="request-one"
    )

    assert {field.name for field in fields(first)} == {
        "export_id",
        "included_episode_count",
        "incomplete_episode_count",
        "omitted_restricted_episode_count",
        "record_count",
        "shard_count",
        "media_count",
        "missing_media_count",
        "media_mode",
    }
    assert asdict(first) == {
        "export_id": first.export_id,
        "included_episode_count": 1,
        "incomplete_episode_count": 0,
        "omitted_restricted_episode_count": 0,
        "record_count": 2,
        "shard_count": 1,
        "media_count": 0,
        "missing_media_count": 0,
        "media_mode": "reference",
    }
    assert repeated == first
    assert first.included_episode_count == 1
    assert len(tuple(output_root.iterdir())) == 1
    with pytest.raises(ValueError, match="idempotency_conflict"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(include_incomplete=True),
            idempotency_key="request-one",
        )
    with pytest.raises(KeyError, match="trace output root is unavailable"):
        exporter.export(
            output_root_alias="foreign", query=TraceQuery(), idempotency_key="request-two"
        )
    assert len(tuple(output_root.iterdir())) == 1
    package = next(output_root.iterdir())
    next(package.glob("records-*.jsonl")).write_bytes(b"tampered export shard")
    with pytest.raises(ValueError, match="trace export receipt is invalid"):
        exporter.export(
            output_root_alias="private", query=TraceQuery(), idempotency_key="request-one"
        )


def test_operator_export_denies_restricted_and_copy_before_writing(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )

    with pytest.raises(PermissionError, match="restricted trace export is unavailable"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(include_restricted=True),
            idempotency_key="restricted",
        )
    with pytest.raises(PermissionError, match="trace media copy is unavailable"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(),
            idempotency_key="copy",
            media_mode=MediaMode.COPY,
        )
    assert not tuple(output_root.iterdir())


def test_operator_export_requires_base_export_permission(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store, workspace_id="workspace-one", output_roots={"private": output_root}
    )

    with pytest.raises(PermissionError, match="trace export is unavailable"):
        exporter.export(
            output_root_alias="private", query=TraceQuery(), idempotency_key="unauthorized"
        )
    assert not tuple(output_root.iterdir())


def test_operator_export_rejects_blank_idempotency_key_before_writing(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )

    with pytest.raises(ValueError, match="trace export idempotency key is invalid"):
        exporter.export(output_root_alias="private", query=TraceQuery(), idempotency_key="   ")

    assert not tuple(output_root.iterdir())


def test_operator_export_requires_explicit_admin_and_sanitizes_reader_failure(
    tmp_path: Path,
) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    class FailedReader:
        def open_snapshot(self, _query: TraceQuery = DEFAULT_TRACE_QUERY, /) -> TraceReadSnapshot:
            raise OSError("private source /outside/trace and credential")

    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="administrative must be a boolean"):
        TraceOperatorExporter(
            reader=FailedReader(),
            workspace_id="workspace-one",
            output_roots={"private": output_root},
            administrative="yes",  # type: ignore[arg-type]
            permissions=frozenset({"trace:export_restricted"}),
        )
    exporter = TraceOperatorExporter(
        reader=FailedReader(),
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )

    with pytest.raises(ValueError, match=r"^trace export failed$") as error:
        exporter.export(output_root_alias="private", query=TraceQuery(), idempotency_key="failed")
    assert "/outside/trace" not in str(error.value)
    assert not tuple(output_root.iterdir())


def test_operator_export_restricted_scope_requires_admin_permission(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "restricted", privacy=PrivacyClass.RESTRICTED)
    store.append(
        TraceRecordDraft(
            episode_id="restricted",
            trace_id="trace-restricted",
            occurred_at=NOW + timedelta(seconds=1),
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
            privacy_class=PrivacyClass.RESTRICTED,
        ),
        idempotency_key="complete",
    )
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        administrative=True,
        permissions=frozenset({"trace:export", "trace:export_restricted"}),
    )

    manifest = exporter.export(
        output_root_alias="private",
        query=TraceQuery(include_restricted=True),
        idempotency_key="approved",
    )

    assert manifest.included_episode_count == 1


@pytest.mark.parametrize(
    ("privacy", "include_restricted"),
    [(PrivacyClass.INTERNAL, False), (PrivacyClass.RESTRICTED, True)],
)
def test_operator_export_rejects_foreign_workspace_episode(
    tmp_path: Path, privacy: PrivacyClass, include_restricted: bool
) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "foreign", privacy=privacy, workspace_id="workspace-other")
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        administrative=include_restricted,
        permissions=frozenset(
            {"trace:export", "trace:export_restricted"} if include_restricted else {"trace:export"}
        ),
    )

    with pytest.raises(ValueError, match="trace export failed"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(include_incomplete=True, include_restricted=include_restricted),
            idempotency_key="foreign-workspace",
        )

    assert not tuple(output_root.iterdir())


def test_operator_export_retry_rejects_valid_foreign_workspace_package(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "foreign", privacy=PrivacyClass.INTERNAL, workspace_id="workspace-other")
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )
    request_key = "existing-foreign"
    digest = hashlib.sha256(f"workspace-one\0private\0{request_key}".encode()).hexdigest()[:32]
    export_id = f"export:{digest}"
    query = TraceQuery(include_incomplete=True)
    TraceExporter(store, clock=lambda: NOW, export_id_factory=lambda: export_id).export(
        output_root / f"trace-{digest}", options=TraceExportOptions(query=query)
    )

    with pytest.raises(ValueError, match="trace export receipt is invalid"):
        exporter.export(output_root_alias="private", query=query, idempotency_key=request_key)

    assert len(tuple(output_root.iterdir())) == 1


def test_operator_export_mixed_store_leaves_no_partial_package(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    _start(store, "a-owned", privacy=PrivacyClass.INTERNAL)
    _start(store, "z-foreign", privacy=PrivacyClass.INTERNAL, workspace_id="workspace-other")
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )

    with pytest.raises(ValueError, match="trace export failed"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(include_incomplete=True),
            idempotency_key="mixed-store",
        )

    assert not tuple(output_root.iterdir())


def test_operator_export_copies_verified_media_and_rechecks_retry(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    media_root = tmp_path / "media-root"
    source = media_root / "inputs" / "image.png"
    source.parent.mkdir(parents=True)
    payload = b"local image evidence"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    reference = MediaReference(
        "media-1",
        TraceModality.IMAGE,
        "image/png",
        digest,
        len(payload),
        locator=MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, "inputs/image.png"),
    )
    _start(store, "completed", privacy=PrivacyClass.INTERNAL)
    store.append(
        TraceRecordDraft(
            episode_id="completed",
            trace_id="trace-completed",
            occurred_at=NOW + timedelta(seconds=1),
            payload=MessageObserved(MessageRole.USER, (MediaContent("media-1"),)),
            modalities=(TraceModality.IMAGE,),
            media=(reference,),
        ),
        idempotency_key="message",
    )
    store.append(
        TraceRecordDraft(
            episode_id="completed",
            trace_id="trace-completed",
            occurred_at=NOW + timedelta(seconds=2),
            payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        ),
        idempotency_key="complete",
    )
    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=store,
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        media_root=media_root,
        administrative=True,
        permissions=frozenset({"trace:export", "trace:copy_media"}),
    )

    first = exporter.export(
        output_root_alias="private",
        query=TraceQuery(),
        idempotency_key="approved-copy",
        media_mode=MediaMode.COPY,
    )
    package = next(output_root.iterdir())
    copied = package / f"media/sha256/{digest[:2]}/{digest}.png"
    assert copied.read_bytes() == payload
    assert first.media_count == 1
    assert first.missing_media_count == 0
    assert (
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(),
            idempotency_key="approved-copy",
            media_mode=MediaMode.COPY,
        )
        == first
    )

    copied.write_bytes(b"tampered copy")
    with pytest.raises(ValueError, match="trace export receipt is invalid"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(),
            idempotency_key="approved-copy",
            media_mode=MediaMode.COPY,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX media-root ancestor link case")
def test_operator_export_rejects_linked_media_root_parent(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    foreign = tmp_path / "foreign"
    (foreign / "media").mkdir(parents=True)
    (tmp_path / "linked").symlink_to(foreign, target_is_directory=True)

    with pytest.raises(ValueError, match="trace media root is unsafe"):
        TraceOperatorExporter(
            reader=SqliteTraceStore(tmp_path / "traces.sqlite3"),
            workspace_id="workspace-one",
            output_roots={"private": output_root},
            media_root=tmp_path / "linked" / "media",
            administrative=True,
            permissions=frozenset({"trace:export", "trace:copy_media"}),
        )
    assert not tuple(output_root.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX media-root ancestor swap case")
def test_operator_export_rechecks_media_root_parent_before_copy(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    media_parent = tmp_path / "sources"
    (media_parent / "media").mkdir(parents=True)
    exporter = TraceOperatorExporter(
        reader=SqliteTraceStore(tmp_path / "traces.sqlite3"),
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        media_root=media_parent / "media",
        administrative=True,
        permissions=frozenset({"trace:export", "trace:copy_media"}),
    )
    media_parent.rename(tmp_path / "sources-held")
    foreign = tmp_path / "foreign"
    (foreign / "media").mkdir(parents=True)
    media_parent.symlink_to(foreign, target_is_directory=True)

    with pytest.raises(ValueError, match="trace media root is unsafe"):
        exporter.export(
            output_root_alias="private",
            query=TraceQuery(),
            idempotency_key="swapped",
            media_mode=MediaMode.COPY,
        )
    assert not tuple(output_root.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX output-root swap boundary")
def test_operator_export_never_writes_through_swapped_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.traces import operator

    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    exporter = operator.TraceOperatorExporter(
        reader=SqliteTraceStore(tmp_path / "traces.sqlite3"),
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )
    original_check = operator._require_private_export_root
    swapped = False

    def swap_after_check(root: Path) -> None:
        nonlocal swapped
        original_check(root)
        if root == output_root and not swapped:
            swapped = True
            output_root.rename(tmp_path / "exports-held")
            output_root.symlink_to(foreign, target_is_directory=True)

    monkeypatch.setattr(operator, "_require_private_export_root", swap_after_check)
    with pytest.raises(ValueError, match="trace export failed"):
        exporter.export(
            output_root_alias="private", query=TraceQuery(), idempotency_key="swapped-root"
        )

    assert swapped
    assert not tuple(foreign.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX pinned output-root boundary")
def test_operator_export_keeps_output_under_pinned_root_after_late_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.traces import operator

    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    exporter = operator.TraceOperatorExporter(
        reader=SqliteTraceStore(tmp_path / "traces.sqlite3"),
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )
    original_export = TraceExporter.export
    swapped = False

    def swap_before_export(
        trace_exporter: TraceExporter,
        output_directory: Path,
        /,
        *,
        options: TraceExportOptions,
    ) -> object:
        nonlocal swapped
        if not swapped:
            swapped = True
            output_root.rename(tmp_path / "exports-held")
            output_root.symlink_to(foreign, target_is_directory=True)
        return original_export(trace_exporter, output_directory, options=options)

    monkeypatch.setattr(TraceExporter, "export", swap_before_export)
    with pytest.raises(ValueError, match="trace export failed"):
        exporter.export(
            output_root_alias="private", query=TraceQuery(), idempotency_key="late-swap"
        )

    assert swapped
    assert not tuple(foreign.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX startup root identity boundary")
def test_operator_export_rejects_replaced_private_root_before_writing(tmp_path: Path) -> None:
    from capicove.traces.operator import TraceOperatorExporter

    output_root = tmp_path / "exports"
    output_root.mkdir(mode=0o700)
    exporter = TraceOperatorExporter(
        reader=SqliteTraceStore(tmp_path / "traces.sqlite3"),
        workspace_id="workspace-one",
        output_roots={"private": output_root},
        permissions=frozenset({"trace:export"}),
    )
    output_root.rename(tmp_path / "exports-held")
    output_root.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="trace export failed"):
        exporter.export(output_root_alias="private", query=TraceQuery(), idempotency_key="new-root")

    assert not tuple(output_root.iterdir())
