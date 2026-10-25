"""BGE reranker adapter bounds, exact usage and lifecycle with a fake runtime."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from capicove.model.local_assets import LocalModelAssets
from capicove.retrieval.bge_reranker import BgeReranker


class Tokenizer:
    def __call__(self, pairs: list[tuple[str, str]], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["truncation"] and kwargs["max_length"] == 9
        ids = [[1] * min(len(q) + len(d) + 4, 9) for q, d in pairs]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    def pad(self, values: dict[str, Any], **kwargs: Any) -> Any:
        class Batch(dict[str, Any]):
            def to(self, device: str) -> Batch:
                assert device == "cpu"
                return self

        return Batch(values)


def adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[BgeReranker, list[int]]:
    from capicove.retrieval import bge_reranker

    calls: list[int] = []

    def model(**kwargs: Any) -> Any:
        lengths = [len(row) for row in kwargs["input_ids"]]
        calls.append(len(lengths))

        class Scores:
            def reshape(self, size: int) -> Scores:
                return self

            def float(self) -> Scores:
                return self

            def cpu(self) -> Scores:
                return self

            def tolist(self) -> list[Any]:
                return [float(v) for v in lengths]

        return SimpleNamespace(logits=Scores())

    monkeypatch.setattr(LocalModelAssets, "verify", lambda self: None)
    monkeypatch.setattr(
        bge_reranker,
        "_load_runtime",
        lambda *a, **kw: (SimpleNamespace(inference_mode=nullcontext), Tokenizer(), model),
    )
    ranker = BgeReranker(tmp_path, device="cpu", max_tokens=8, batch_size=2)
    return ranker, calls


def test_batch_scores_token_usage_and_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ranker, calls = adapter(monkeypatch, tmp_path)
    result = ranker.rerank("q", ("a", "bbb", "cc"))
    assert [r.index for r in result.results] == [1, 2, 0]
    assert result.input_tokens == 21 and calls == [2, 1]
    ranker.close()
    ranker.close()
    with pytest.raises(RuntimeError, match="closed"):
        ranker.rerank("q", ("a",))


def test_overlong_pair_rejected_before_any_forward_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ranker, calls = adapter(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="rerank_input_too_long"):
        ranker.rerank("q", ("a", "too long"))
    assert calls == []


def test_full_input_is_scored_even_when_only_one_result_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ranker, calls = adapter(monkeypatch, tmp_path)
    assert ranker.rerank("q", ("a", "bbb", "cc"), top_n=1).results[0].index == 1
    assert calls == [2, 1]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_loader_owns_cpu_weights_and_uses_only_local_safe_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    import sys

    from capicove.retrieval.bge_reranker import _load_runtime

    calls: list[str] = []
    owned = object()

    class Weight:
        def clone(self) -> object:
            calls.append("copy")
            return owned

    state: dict[str, object] = {"weight": Weight()}

    class Model:
        config = SimpleNamespace(model_type="xlm-roberta", num_labels=1)

        def state_dict(self) -> dict[str, object]:
            calls.append("state")
            return state

        def to_empty(self, *, device: str) -> None:
            raise AssertionError("must preserve initialized nonpersistent position/token buffers")

        def load_state_dict(self, value: object, *, strict: bool, assign: bool) -> None:
            assert value == {"weight": owned} and strict and assign
            calls.append("assign")

        def to(self, target: str) -> None:
            assert target == device
            calls.append("move")

        def eval(self) -> None:
            calls.append("eval")

    model = Model()

    def load_model(root: str, **kwargs: Any) -> Model:
        assert root == str(tmp_path)
        assert kwargs == {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        return model

    def load_tokenizer(root: str, **kwargs: Any) -> object:
        assert root == str(tmp_path)
        assert kwargs == {"local_files_only": True, "trust_remote_code": False, "use_fast": True}
        return object()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    assert _load_runtime(tmp_path, device)[2] is model
    assert calls == (["state", "copy", "assign", "eval"] if device == "cpu" else ["move", "eval"])
