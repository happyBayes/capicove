"""Deterministic fakes are stable, labelled and never presented as models."""

from __future__ import annotations

import io
import math
from datetime import UTC, datetime
from typing import Any

import pytest
from PIL import Image

from capicove.evaluation.adapters._common import render_image
from capicove.evaluation.adapters.image import transform
from capicove.evaluation.fakes import (
    FAKE_RERANKER_MODEL_ID,
    DeterministicFakeReranker,
    FakeSharedSpaceEncoder,
    dominant_color_word,
    fake_text_encoder,
    hashed_bag_of_words,
)
from capicove.evaluation.graph_fakes import (
    ScriptedGraphGenerator,
    ScriptedTokenizer,
    unique_quote,
)
from capicove.evaluation.harness import EvaluationError


def test_fake_reranker_orders_by_query_coverage() -> None:
    reranker = DeterministicFakeReranker()
    result = reranker.rerank("red bike tyre", ("blue car", "red bike", "red bike tyre pump"))
    assert reranker.model_id == FAKE_RERANKER_MODEL_ID
    assert [item.index for item in result.results] == [2, 1, 0]
    assert result.model == FAKE_RERANKER_MODEL_ID
    assert result.input_tokens == 3 + 2 + 2 + 4
    assert reranker.rerank("a", ("a", "b"), top_n=1).results[0].index == 0
    with pytest.raises(ValueError):
        reranker.rerank("", ("a",))


def test_hashed_bag_of_words_is_normalized_and_stable() -> None:
    first = hashed_bag_of_words("Red image")
    assert first == hashed_bag_of_words("image red")
    assert math.isclose(sum(value * value for value in first), 1.0)
    assert len(hashed_bag_of_words("", 8)) == 8
    space, vectors = fake_text_encoder(["a b", "c"], "query")
    assert space.startswith("capicove/deterministic-fake")
    assert len(vectors) == 2 and len(vectors[0]) == 64


def test_rendered_images_are_deterministic_and_colored() -> None:
    spec = {"color": "green", "variant": 1, "size": 32}
    payload = render_image(spec)
    assert payload == render_image(spec)
    assert payload != render_image({**spec, "variant": 2})
    with Image.open(io.BytesIO(payload)) as image:
        pixels = [image.convert("RGB").getpixel((x, y)) for y in range(32) for x in range(32)]
    assert dominant_color_word(pixels) == "green"  # type: ignore[arg-type]
    for bad in ({"color": "teal", "variant": 1, "size": 32}, {"color": "red"}):
        with pytest.raises(EvaluationError):
            render_image(bad)
    with pytest.raises(EvaluationError):
        render_image({"color": "red", "variant": 1, "size": 8})
    with pytest.raises(EvaluationError):
        render_image({"color": "red", "variant": "1", "size": 32})


@pytest.mark.parametrize(
    "name", ["identity", "resize-half", "crop-center-80", "brighten-12", "mirror"]
)
def test_image_transforms_are_deterministic(name: str) -> None:
    payload = render_image({"color": "blue", "variant": 3, "size": 40})
    assert transform(payload, name) == transform(payload, name)
    assert (transform(payload, name) == payload) == (name == "identity")
    with pytest.raises(EvaluationError):
        transform(payload, "rotate")


def test_shared_space_fake_maps_color_words_to_images() -> None:
    from capicove.evaluation.adapters.unified import fake_space

    encoder = FakeSharedSpaceEncoder(fake_space())
    deadline = datetime.now(UTC)
    red = render_image({"color": "red", "variant": 1, "size": 32})
    image_vector = encoder.encode_images([red], deadline=deadline)[0]
    assert image_vector == encoder.encode_text_queries(["red image"], deadline=deadline)[0]

    class Clip:
        pcm = b"\x00\x01"

    assert len(encoder.encode_audio([Clip()], deadline=deadline)[0]) == 64
    encoder.close()


def test_unique_quote_extends_repeated_names() -> None:
    core = "Ann met Bo. Ann left."
    assert unique_quote(core, "Bo") == "Bo"
    assert unique_quote(core, "Ann") == "Ann m"
    assert unique_quote(core, "Zed") is None
    assert unique_quote("aa aa", "aa", limit=2) is None


def test_scripted_generator_replays_only_present_annotations() -> None:
    generator = ScriptedGraphGenerator(
        [{"name": "Ann", "type": "PERSON"}, {"name": "Bo", "type": "PERSON"}],
        [
            {"source": "Ann", "target": "Bo", "description": "met", "quote": "Ann met Bo"},
            {"source": "Ann", "target": "Cy", "description": "saw", "quote": "Ann saw Cy"},
        ],
    )

    class Request:
        def __init__(self, role: str, segments: Any) -> None:
            self.role = role
            self.input_segments = segments
            self.canonical_payload = b"x" * 40

    import json

    raw = generator.generate(Request("user", [("s", "core", "Ann met Bo.")]), control=None)
    value = json.loads(raw.payload)
    assert [row["name"] for row in value["entities"]] == ["Ann", "Bo"]
    assert [row["quote"] for row in value["relationships"]] == ["Ann met Bo"]
    task = {"sources": [{"ref": "a0", "source": "Ann", "target": "Bo", "meaning": "met"}]}
    report = json.loads(
        generator.generate(
            Request("report", [("t", "task", json.dumps(task))]), control=None
        ).payload
    )
    assert report == {
        "title": "Ann, Bo",
        "statements": [{"text": "Ann met Bo.", "references": ["a0"]}],
    }
    empty = json.loads(
        generator.generate(Request("report", [("t", "task", "{}")]), control=None).payload
    )
    assert empty["title"] == "Overview of community summaries"
    assert json.loads(generator.generate(Request("diagnostic", []), control=None).payload) == {
        "findings": []
    }
    assert generator.calls == ["user", "report", "report", "diagnostic"]
    tokenizer = ScriptedTokenizer()
    assert tokenizer.count("a b c") == 3
    assert tokenizer.count_request(Request("user", [])) == 10
