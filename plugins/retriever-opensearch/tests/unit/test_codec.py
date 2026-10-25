"""Tests for strict OpenSearch text mapping and hit decoding."""

from capicove_retriever_opensearch.codec import decode_text_candidates

from capicove.contracts import CharacterRange, TextChunkLocator, TextOrigin


def test_decoder_strips_highlight_markup_and_preserves_logical_generation() -> None:
    response = {
        "hits": {
            "hits": [
                {
                    "_score": 4.5,
                    "_source": {
                        "unit_id": "unit-1",
                        "evidence_ref": "evidence-1",
                        "origin": "document",
                        "start_char": 0,
                        "end_char": 8,
                        "preview": "matching passage",
                    },
                    "highlight": {"text": ["<em>matching</em> passage"]},
                }
            ]
        }
    }

    result = decode_text_candidates(
        response,
        provider_id="capicove.retriever.opensearch",
        generation_id="generation-1",
        score_semantics="opensearch_bm25_score",
    )

    assert result[0].snippet == "matching passage"
    assert result[0].text_locator == TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 8))
    assert result[0].index_generation_id == "generation-1"
