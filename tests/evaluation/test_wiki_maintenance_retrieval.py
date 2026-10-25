"""Compare frozen bilingual queries against unchanged FTS and bounded fusion."""

from collections import Counter
from pathlib import Path

from capicove.wiki._adapters.maintenance_index import ranked_lanes
from capicove.wiki.maintenance.discovery import _rows, build_discovery, discover, inventory_snapshot
from tests.wiki_maintenance_support.retrieval import fixture, fixture_digest


def test_frozen_bilingual_retrieval(tmp_path: Path) -> None:
    pages, cases = fixture()
    assert Counter(case.language for case in cases) == {"zh": 50, "en": 50, "mixed": 20}
    # Frozen synthetic fixture SHA-256, not a credential.
    assert (
        fixture_digest()
        == "1de7c44e3ee2533361c286d2edd8d02bc81a0a6f156cbd557552f4ee8c5a1c0b"  # pragma: allowlist secret
    )
    index = tmp_path.resolve() / "retrieval.sqlite"
    build_discovery(index, pages)
    baseline_hits: Counter[str] = Counter()
    fused_hits: Counter[str] = Counter()
    denominators: Counter[str] = Counter()
    for case in cases:
        result = discover(index, pages, case.query, source_ids=case.source_ids)
        assert len(result.candidates) <= 20
        observed = {candidate.page_id for candidate in result.candidates}
        baseline = ranked_lanes(
            index, inventory_snapshot(pages), _rows(pages), '"' + case.query + '"', ""
        )["fts"][:20]
        if case.relevant:
            denominators[case.language] += len(case.relevant)
            fused_hits[case.language] += len(observed & set(case.relevant))
            baseline_hits[case.language] += len(set(baseline) & set(case.relevant))
            assert set(case.relevant) <= observed, case
        else:
            assert not observed
    print(
        {
            "fixture_sha256": fixture_digest(),
            "denominators": dict(denominators),
            "fusion_hits": dict(fused_hits),
            "fts_hits": dict(baseline_hits),
        }
    )
