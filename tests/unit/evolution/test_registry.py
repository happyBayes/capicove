"""Version ownership and activation gate regression tests."""

import os
from pathlib import Path

import pytest

from capicove.evolution.models import EvaluationReport, PairedResult
from capicove.evolution.registry import EvolutionRegistry


def test_registry_rejects_missing_owner_support_before_creating_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(os, "getuid", raising=False)
    root = tmp_path / "new-parent" / "registry"

    with pytest.raises(ValueError, match="evolution storage requires POSIX ownership support"):
        EvolutionRegistry(root)

    assert not root.parent.exists()


def content(name: str = "formatting") -> str:
    return f"---\nname: {name}\ndescription: Format clear answers\n---\n# Instructions\nUse short paragraphs.\n"


def evaluate(registry: EvolutionRegistry, candidate_id: str) -> None:
    candidate = registry.get_candidate(candidate_id)
    registry.record_evaluation(
        EvaluationReport(
            report_id=f"report-{candidate_id}",
            candidate_id=candidate_id,
            content_hash=candidate.content_hash,
            baseline_id=candidate.parent_id,
            evaluator_id="deterministic-test-v1",
            synthesis_episode_ids=("source",),
            replay_episode_ids=("replay", "replay2"),
            held_out_episode_ids=("held", "held2"),
            results=(
                PairedResult("replay", True, True),
                PairedResult("held", False, True),
                PairedResult("replay2", True, True),
                PairedResult("held2", True, True),
            ),
        )
    )


def test_review_requires_evaluation_and_compare_and_swap(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        second = registry.create_candidate(
            "formatting", content() + "Be concise.\n", ("source",), expected_active=None
        )
        assert registry.active("formatting") is None
        with pytest.raises(ValueError, match="evaluation"):
            registry.approve(first.candidate_id, "operator", "reviewed", expected_active=None)
        evaluate(registry, first.candidate_id)
        evaluate(registry, second.candidate_id)
        registry.approve(first.candidate_id, "operator", "reviewed", expected_active=None)
        with pytest.raises(ValueError, match="stale"):
            registry.approve(second.candidate_id, "operator", "reviewed", expected_active=None)
    with EvolutionRegistry(tmp_path) as reopened:
        assert reopened.active("formatting").candidate_id == first.candidate_id  # type: ignore[union-attr]


def test_reject_and_rollback_are_audited_and_bound_to_asset(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        evaluate(registry, first.candidate_id)
        registry.approve(first.candidate_id, "operator", "reviewed", expected_active=None)
        second = registry.create_candidate(
            "formatting",
            content() + "Be concise.\n",
            ("source",),
            expected_active=first.candidate_id,
        )
        evaluate(registry, second.candidate_id)
        registry.approve(
            second.candidate_id, "operator", "reviewed", expected_active=first.candidate_id
        )
        with pytest.raises(ValueError):
            registry.rollback(
                "different", first.candidate_id, "operator", "revert", expected_active=None
            )
        registry.rollback(
            "formatting",
            first.candidate_id,
            "operator",
            "revert",
            expected_active=second.candidate_id,
        )
        assert registry.active("formatting").candidate_id == first.candidate_id  # type: ignore[union-attr]
        third = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=first.candidate_id
        )
        registry.reject(third.candidate_id, "operator", "not useful")
        with pytest.raises(ValueError):
            evaluate(registry, third.candidate_id)


def test_overlapping_splits_and_permission_documents_fail(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        with pytest.raises(ValueError):
            registry.create_candidate(
                "formatting",
                content().replace("description:", "permissions: network\ndescription:"),
                ("source",),
                expected_active=None,
            )
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        with pytest.raises(ValueError):
            registry.record_evaluation(
                EvaluationReport(
                    "bad",
                    first.candidate_id,
                    first.content_hash,
                    None,
                    "test",
                    ("source",),
                    ("source",),
                    ("held",),
                    (PairedResult("source", True, True), PairedResult("held", True, True)),
                )
            )


def test_registry_retains_evidence_loads_and_review_history(tmp_path: Path) -> None:
    import sqlite3

    with EvolutionRegistry(tmp_path) as registry:
        registry.store_batch("batch", '{"episodes": ["source"]}')
        assert registry.list_batches() == ("batch",)
        assert '"source"' in registry.get_batch("batch")
        registry.store_batch("batch", '{"episodes": ["source"]}')
        with pytest.raises(ValueError):
            registry.store_batch("batch", "{}")
        with pytest.raises(KeyError):
            registry.get_batch("missing")
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        with pytest.raises(ValueError, match="active"):
            registry.record_use(first.candidate_id, "episode")
        evaluate(registry, first.candidate_id)
        with pytest.raises(ValueError, match="reviewer"):
            registry.approve(first.candidate_id, "", "reviewed", expected_active=None)
        registry.approve(first.candidate_id, "operator", "reviewed", expected_active=None)
        registry.record_use(first.candidate_id, "episode")
        assert registry.usage(first.candidate_id)[0]["episode_id"] == "episode"
        assert registry.history("formatting")[0]["action"] == "approve"
        assert registry.evaluation_report(first.candidate_id) is not None
        assert len(registry.list_candidates()) == 1
        with sqlite3.connect(registry.path) as db:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                db.execute("UPDATE candidates SET content='tampered'")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute("DELETE FROM reviews")


def test_latest_failed_evaluation_revokes_review_readiness(tmp_path: Path) -> None:
    from dataclasses import replace

    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        evaluate(registry, first.candidate_id)
        report = registry.latest_evaluation(first.candidate_id)
        assert report is not None
        failed = replace(
            report,
            report_id="failure",
            results=(replace(report.results[0], critical_regression=True), *report.results[1:]),
        )
        registry.record_evaluation(failed)
        with pytest.raises(ValueError, match="evaluation"):
            registry.approve(first.candidate_id, "operator", "reviewed", expected_active=None)


def test_candidate_document_and_source_validation(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        for name, document, sources in (
            ("../bad", content(), ("source",)),
            ("formatting", "plain text", ("source",)),
            ("formatting", "---\nname: formatting", ("source",)),
            ("formatting", content().replace("formatting", "different"), ("source",)),
            ("formatting", content().split("# Instructions")[0], ("source",)),
            ("formatting", content() + "\n```bash\necho unsafe\n```", ("source",)),
            ("formatting", content() + "\0", ("source",)),
            ("formatting", content(), ()),
            ("formatting", content(), ("source", "source")),
        ):
            with pytest.raises(ValueError):
                registry.create_candidate(name, document, sources, expected_active=None)


def test_pair_gate_is_derived_from_metrics(tmp_path: Path) -> None:
    from dataclasses import replace

    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        evaluate(registry, first.candidate_id)
        report = registry.latest_evaluation(first.candidate_id)
        assert report is not None and report.passed
        for change in (
            {"candidate_score": 0.0},
            {"candidate_passed": False},
            {"source_leak": True},
            {"permission_expansion": True},
            {"infrastructure_error": True},
            {"candidate_score": float("nan")},
            {"candidate_tool_errors": 1},
            {"candidate_policy_rejections": 1},
            {"baseline_tool_errors": -1},
        ):
            assert not replace(
                report, results=(replace(report.results[0], **change), *report.results[1:])
            ).passed
        assert not replace(report, results=report.results[:-1]).passed
        assert not replace(report, replay_episode_ids=("single",)).passed
        assert not replace(report, evaluator_id="").passed
        with pytest.raises(ValueError, match="binding"):
            registry.record_evaluation(replace(report, report_id="other", content_hash="0" * 64))
        with pytest.raises(ValueError, match="sources"):
            registry.record_evaluation(
                replace(report, report_id="other", synthesis_episode_ids=("other",))
            )


def test_registry_rejects_foreign_database_and_links(tmp_path: Path) -> None:
    import sqlite3

    with sqlite3.connect(tmp_path / "evolution.sqlite3") as db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(ValueError):
        EvolutionRegistry(tmp_path)
    (tmp_path / "evolution.sqlite3").unlink()
    target = tmp_path / "target"
    target.touch()
    (tmp_path / "evolution.sqlite3").symlink_to(target)
    with pytest.raises(ValueError):
        EvolutionRegistry(tmp_path)


def test_unchanged_candidate_is_idempotent(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        same = registry.create_candidate("formatting", content(), ("source",), expected_active=None)
        assert first == same
        assert len(registry.list_candidates()) == 1


def test_new_skill_requires_observed_improvement() -> None:
    report = EvaluationReport(
        "report",
        "candidate",
        "0" * 64,
        None,
        "test",
        ("source",),
        ("r1", "r2"),
        ("h1", "h2"),
        tuple(PairedResult(item, True, True) for item in ("r1", "r2", "h1", "h2")),
    )
    assert not report.passed


def test_rejection_compares_expected_baseline(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        first = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None
        )
        with pytest.raises(ValueError, match="stale"):
            registry.reject(first.candidate_id, "operator", "reviewed", expected_active="missing")
        registry.reject(first.candidate_id, "operator", "reviewed", expected_active=None)


def test_candidates_bind_exact_retained_evidence_snapshot(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        registry.store_batch("first-feedback", '{"source_episode_ids":["source"],"lesson":"first"}')
        registry.store_batch("later-feedback", '{"source_episode_ids":["source"],"lesson":"later"}')
        first = registry.create_candidate(
            "formatting",
            content(),
            ("source",),
            expected_active=None,
            source_batch_id="first-feedback",
        )
        later = registry.create_candidate(
            "formatting",
            content(),
            ("source",),
            expected_active=None,
            source_batch_id="later-feedback",
        )
        assert first.candidate_id != later.candidate_id
        assert registry.get_candidate(first.candidate_id).source_batch_id == "first-feedback"
        assert registry.get_candidate(later.candidate_id).source_batch_id == "later-feedback"
        with pytest.raises(KeyError):
            registry.create_candidate(
                "formatting",
                content(),
                ("source",),
                expected_active=None,
                source_batch_id="missing",
            )
        with pytest.raises(ValueError, match="sources"):
            registry.create_candidate(
                "formatting",
                content(),
                ("other",),
                expected_active=None,
                source_batch_id="first-feedback",
            )


def test_evaluation_binds_all_exposed_batch_sources(tmp_path: Path) -> None:
    from dataclasses import replace

    with EvolutionRegistry(tmp_path) as registry:
        registry.store_batch("batch", '{"source_episode_ids":["source","exposed"]}')
        candidate = registry.create_candidate(
            "formatting", content(), ("source",), expected_active=None, source_batch_id="batch"
        )
        report = EvaluationReport(
            "report",
            candidate.candidate_id,
            candidate.content_hash,
            None,
            "test",
            ("source",),
            ("r1", "r2"),
            ("h1", "h2"),
            tuple(PairedResult(item, False, True) for item in ("r1", "r2", "h1", "h2")),
        )
        with pytest.raises(ValueError, match="sources"):
            registry.record_evaluation(report)
        full_report = replace(report, synthesis_episode_ids=("source", "exposed"))
        registry.record_evaluation(full_report)
        registry.approve(
            candidate.candidate_id, "operator", "reviewed all exposure", expected_active=None
        )
