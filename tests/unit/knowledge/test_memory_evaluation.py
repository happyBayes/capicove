"""Evaluation hygiene: labels never enter ingestion or prediction requests."""

from pathlib import Path

import pytest

from capicove.knowledge.memory_evaluation import (
    prepare_locomo,
    prepare_longmemeval,
    summarize_paired,
)


def lme(identity: str = "q1") -> dict[str, object]:
    return {
        "question_id": identity,
        "question_type": "temporal-reasoning",
        "question": "Where?",
        "question_date": "2026/01/01",
        "answer": "SECRET_GOLD",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2025/01/01"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I moved to Paris",
                    "has_answer": True,
                    "injected": "secret",
                }
            ]
        ],
    }


def test_longmemeval_allowlist_strips_all_labels() -> None:
    campaign = prepare_longmemeval([lme()], dataset_revision="abc", seed="frozen")
    assert "SECRET_GOLD" not in str(campaign.ingestion)
    assert "has_answer" not in str(campaign.ingestion)
    assert "injected" not in str(campaign.ingestion)
    assert "Where?" not in str(campaign.ingestion)
    assert campaign.evaluation[0]["answer"] == "SECRET_GOLD"
    assert campaign.questions[0]["question"] == "Where?"
    assert campaign.manifest["profile"] == "longmemeval"


def test_duplicate_missing_or_misaligned_data_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        prepare_longmemeval([lme(), lme()], dataset_revision="a", seed="s")
    broken = lme()
    broken["haystack_dates"] = []
    with pytest.raises(ValueError):
        prepare_longmemeval([broken], dataset_revision="a", seed="s")


def test_shared_supporting_history_stays_in_one_split() -> None:
    c = prepare_longmemeval([lme("q1"), lme("q2")], dataset_revision="a", seed="s")
    entries = c.manifest["cases"]
    assert isinstance(entries, list)
    assert entries[0]["group"] == entries[1]["group"]
    assert entries[0]["split"] == entries[1]["split"]


def test_locomo_conversation_group_and_no_qa_labels_in_ingestion() -> None:
    raw = [
        {
            "sample_id": "c1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "1 Jan 2026",
                "session_1": [
                    {
                        "speaker": "A",
                        "dia_id": "D1:1",
                        "text": "A boat",
                        "img_url": "https://example.invalid/private",
                    }
                ],
            },
            "qa": [
                {"question": "What?", "answer": "SECRET", "evidence": ["D1:1"], "category": 1},
                {"question": "When?", "answer": "SECRET2", "evidence": ["D1:1"], "category": 2},
            ],
        }
    ]
    c = prepare_locomo(raw, dataset_revision="a", seed="s")
    assert len(c.ingestion) == 1
    assert "SECRET" not in str(c.ingestion) and "img_url" not in str(c.ingestion)
    assert c.manifest["text_only"] is True
    assert c.manifest["cases"][0]["split"] == c.manifest["cases"][1]["split"]


def test_paired_report_counts_missing_predictions_as_failures() -> None:
    ids = ("a", "b", "c")
    report = summarize_paired(
        ids,
        {"a": True, "b": True, "c": False},
        {"a": True},
        {"a": "g1", "b": "g2", "c": "g3"},
        seed=1,
    )
    assert report["baseline_score"] == pytest.approx(2 / 3)
    assert report["candidate_score"] == pytest.approx(1 / 3)
    assert report["missing_candidate"] == ["b", "c"]
    assert report["status"] == "INCOMPLETE"
    with pytest.raises(ValueError):
        summarize_paired(
            ids, {"a": True}, {"unexpected": True}, {"a": "g1", "b": "g2", "c": "g3"}, seed=1
        )


def test_prepare_command_writes_separate_files_and_refuses_overwrite(tmp_path: "Path") -> None:
    import json

    from capicove.knowledge.memory_evaluation import main

    source = tmp_path / "source.json"
    source.write_text(json.dumps([lme()]))
    output = tmp_path / "prepared"
    args = [
        "prepare",
        "--dataset",
        "longmemeval",
        "--input",
        str(source),
        "--output",
        str(output),
        "--revision",
        "abc",
        "--seed",
        "fixed",
    ]
    assert main(args) == 0
    assert "SECRET_GOLD" not in (output / "ingestion.json").read_text()
    assert "SECRET_GOLD" in (output / "evaluation-only.json").read_text()
    with pytest.raises(FileExistsError):
        main(args)


def test_prepare_rejects_oversized_file_before_parsing(tmp_path: Path) -> None:
    from capicove.knowledge.memory_evaluation import main

    source = tmp_path / "oversized.json"
    with source.open("wb") as stream:
        stream.truncate(384 * 1024 * 1024 + 1)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="384 MiB"):
        main(
            [
                "prepare",
                "--dataset",
                "longmemeval",
                "--input",
                str(source),
                "--output",
                str(output),
                "--revision",
                "abc",
                "--seed",
                "fixed",
            ]
        )
    assert not output.exists()


def test_official_numeric_answers_and_repeated_session_ids_are_preserved() -> None:
    sample = lme()
    sample["answer"] = 42
    sample["haystack_session_ids"] = ["s1", "s1"]
    sample["haystack_dates"] = ["2025/01/01", "2025/02/01"]
    history = [
        {"role": "user", "content": "A repeated history", "has_answer": True},
        {"role": "assistant", "content": ""},
    ]
    sample["haystack_sessions"] = [history, history]
    c = prepare_longmemeval([sample], dataset_revision="a", seed="s")
    assert c.evaluation[0]["answer"] == 42
    assert len(c.ingestion[0]["sessions"]) == 2
    assert c.ingestion[0]["sessions"][0]["session_index"] == 0
    assert c.ingestion[0]["sessions"][1]["session_index"] == 1
