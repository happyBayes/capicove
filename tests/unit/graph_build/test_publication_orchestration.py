"""Publication replay ordering without SQLite, Parquet, or inference work."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from capicove.contracts.graph_build import GraphSnapshotRef, StageCoverage, TableRef
from capicove.contracts.graph_build.snapshots import TABLE_ROLES
from capicove.graph_build import service as module
from tests.graph_build_fixtures import STAGE_ROLES
from tests.unit.contracts.test_graph_snapshots import dataset, manifest


@pytest.mark.parametrize("role", sorted(TABLE_ROLES))
@pytest.mark.parametrize("after", [False, True])
def test_publication_replay_retries_every_role_before_sealing(
    monkeypatch: pytest.MonkeyPatch, role: str, after: bool
) -> None:
    template = manifest()
    request = SimpleNamespace(
        scope=template.scope, input_snapshot_ref=dataset(), baseline_snapshot_id=None
    )
    run = SimpleNamespace(
        request=request, run_id="run:test", profile_fingerprint="d" * 64, baseline_digest=None
    )
    graph_ref = GraphSnapshotRef("graph:test", "e" * 64)
    service: Any = object.__new__(module.GraphBuildService)
    service.workspace = object()
    service.runs = Mock()
    service.runs.get.return_value = SimpleNamespace(cancelled=False)
    service.runs.runtime.return_value = b"{}"
    service.catalog = Mock()
    service.catalog.register.return_value = graph_ref
    service.stages = Mock()
    service.stages.commit_operation.side_effect = ["publication:1", "retrieval:1"]
    service.retrieval = None
    service._authorize = Mock(return_value=SimpleNamespace(evidence_refs=()))
    service._check = Mock()
    service._profile = Mock()
    service._runtime_payload = Mock(return_value={})
    service._finish = Mock()
    sealed: list[tuple[str, tuple[TableRef, ...]]] = []

    def seal(
        ref: str, scope: Any, fingerprint: str, tables: tuple[TableRef, ...], **kw: Any
    ) -> Any:
        assert scope == request.scope
        assert fingerprint == run.profile_fingerprint
        assert kw["input_snapshot_ref"] == request.input_snapshot_ref
        sealed.append((ref, tables))
        return next(stage for stage in template.stages if stage.stage_id == ref)

    service.stages.seal.side_effect = seal
    committed: dict[str, Any] = {}
    calls: list[str] = []
    crash = True

    class Crash(BaseException):
        pass

    def publish(workspace: Any, **kw: Any) -> Any:
        current = kw["role"]
        assert workspace is service.workspace
        assert kw["scope"] == request.scope
        assert kw["parent_refs"] == (request.input_snapshot_ref,)
        assert kw["processing_fingerprint"] == run.profile_fingerprint
        assert kw["rows"] == ()
        calls.append(current)
        if crash and current == role and not after:
            raise Crash()
        # Idempotency belongs to the storage adapter; this double models its contract.
        value = committed.setdefault(current, dataset(current))
        if crash and current == role and after:
            raise Crash()
        return value

    monkeypatch.setattr(module, "publish_table", publish)
    refs = {stage.stage_id: stage.stage_id for stage in template.stages}
    ordered = sorted(TABLE_ROLES)
    with pytest.raises(Crash):
        service._publish(run, {}, refs, 100.0, 7)
    assert calls == ordered[: ordered.index(role) + 1]
    assert set(committed) == set(ordered[: ordered.index(role) + int(after)])
    service.stages.seal.assert_not_called()
    service.catalog.register.assert_not_called()
    service.runs.record_stage.assert_not_called()
    service._finish.assert_not_called()

    prior = dict(committed)
    crash = False
    calls.clear()
    result = service._publish(run, {}, refs, 100.0, 7)

    assert calls == ordered
    assert all(committed[key] is value for key, value in prior.items())
    assert sealed == [
        (stage, tuple(TableRef(name, committed[name]) for name in ordered if name in roles))
        for stage, roles in STAGE_ROLES.items()
    ]
    published = service.catalog.register.call_args.args[0]
    assert published == replace(
        template,
        tables=tuple(TableRef(name, committed[name]) for name in ordered),
        stages=tuple(
            next(s for s in template.stages if s.stage_id == name) for name in STAGE_ROLES
        ),
    )
    assert result.status == "succeeded"
    assert result.graph_ref == graph_ref
    assert result.completeness == "complete"
    assert result.retrieval_binding_ref is None
    assert service.stages.commit_operation.call_args_list[0].kwargs["coverage"] == StageCoverage(
        "tables", 17, 17, 17, 0, 0, 0
    )
    assert [call.args[2:] for call in service.runs.record_stage.call_args_list] == [
        ("publication", "publication:1"),
        ("retrieval", "retrieval:1"),
    ]
    service.catalog.activate.assert_not_called()
    service._finish.assert_called_once_with(run, result, 7)
