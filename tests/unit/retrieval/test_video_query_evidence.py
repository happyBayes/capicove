"""Query-specific diagnostics never replace immutable source authorization."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts import FullHit, LaneScore, SearchResult, SearchStatus, TargetType
from capicove.retrieval.video_query_evidence import VideoQueryEvidenceReader
from tests.unit.retrieval.test_video_evidence import EVIDENCE, context, setup


def result() -> SearchResult:
    return SearchResult(
        SearchStatus.SUCCEEDED,
        "query-1",
        hits=(
            FullHit(
                "keyframe:1",
                TargetType.KEYFRAME,
                0.5,
                EVIDENCE,
                timestamp_ms=1000,
                lane_scores=(
                    LaneScore("keyframe_visual", 0.8, 0.5, "inner_product", "rank-v1", 1),
                ),
                provider_ids=("faiss",),
                index_generation_ids=("generation-1",),
            ),
        ),
    )


def test_query_evidence_is_bound_to_owner_and_verified_source() -> None:
    base, read, _ = setup()
    owner = context()
    reader = VideoQueryEvidenceReader(base, result(), owner=owner)
    scores = reader.read(context=owner, evidence_id=EVIDENCE, sections=("scores",))
    score_section = scores["scores"]
    assert isinstance(score_section, dict)
    assert score_section["query_id"] == "query-1"
    assert score_section["lanes"][0]["raw_score"] == 0.8
    with pytest.raises(PermissionError):
        reader.read(
            context=replace(owner, actor_id="other"), evidence_id=EVIDENCE, sections=("scores",)
        )
    with pytest.raises(PermissionError):
        reader.read(context=owner, evidence_id=EVIDENCE, sections=("provider_diagnostics",))
    privileged = replace(owner, permissions=owner.permissions | {"evidence:diagnostics"})
    data = reader.read(context=privileged, evidence_id=EVIDENCE, sections=("provider_diagnostics",))
    assert data["provider_diagnostics"] == {
        "query_id": "query-1",
        "provider_ids": ["faiss"],
        "index_generation_ids": ["generation-1"],
    }
    read.side_effect = FileNotFoundError
    with pytest.raises(LookupError):
        reader.read(context=owner, evidence_id=EVIDENCE, sections=("scores",))


def test_query_evidence_rejects_ambiguous_source_ids() -> None:
    base, _, _ = setup()
    original = result()
    with pytest.raises(ValueError, match="ambiguous"):
        VideoQueryEvidenceReader(base, replace(original, hits=original.hits * 2), owner=context())


@pytest.mark.parametrize("sections", [(), ("scores", "scores")])
def test_query_evidence_rejects_invalid_section_selection(sections: tuple[str, ...]) -> None:
    base, _, _ = setup()
    owner = context()
    reader = VideoQueryEvidenceReader(base, result(), owner=owner)
    with pytest.raises(ValueError, match="sections"):
        reader.read(context=owner, evidence_id=EVIDENCE, sections=sections)


def test_query_evidence_obeys_absolute_deadline() -> None:
    base, read, _ = setup()
    owner = context()
    reader = VideoQueryEvidenceReader(base, result(), owner=owner)
    with pytest.raises(TimeoutError):
        reader.read(
            context=replace(owner, deadline=datetime.now(UTC) - timedelta(seconds=1)),
            evidence_id=EVIDENCE,
            sections=("scores",),
        )
    read.assert_not_called()


def test_query_evidence_rejects_nonopaque_provider_identity() -> None:
    base, _, _ = setup()
    original = result()
    hit = original.hits[0]
    assert isinstance(hit, FullHit)
    with pytest.raises(ValueError, match="opaque"):
        VideoQueryEvidenceReader(
            base,
            replace(original, hits=(replace(hit, provider_ids=("private / path",)),)),
            owner=context(),
        )
