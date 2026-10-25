"""Public documentation must not promote unreviewed engineering measurements."""

from pathlib import Path


def test_video_guide_keeps_unqualified_status_until_reviewed_evidence() -> None:
    root = Path(__file__).parents[2]
    guide = (root / "docs/site/guides/video/retrieval.md").read_text()
    assert "experimental, not qualified for production support" in guide
    assert "No reviewed production qualification summary" in guide
    assert "1,000,000 keyframes" not in guide
    assert "P95 below 500 ms" not in guide
    assert "public image-OCR capability" in guide
    assert "does not silently provision" in guide


def test_qualification_procedure_separates_evidence_and_approval() -> None:
    root = Path(__file__).parents[2]
    procedure = (root / "docs/development/video-retrieval-qualification.md").read_text()
    assert "authored metadata" in procedure
    assert "real-semantic-evidence" in procedure
    assert (
        "Human approval, RFC acceptance, commit, push, merge and release remain separate"
        in procedure
    )
    assert "https://github.com/happyBayes/capicove" in procedure
