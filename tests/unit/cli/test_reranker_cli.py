"""Reranker CLI routing and early validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import main


def test_reranker_serve_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from capicove.cli.commands import reranker

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda n: object())
    monkeypatch.setattr(reranker, "serve", lambda **kwargs: calls.append(kwargs))
    assert (
        main(
            [
                "models",
                "serve",
                "bge-reranker-v2-m3",
                "--model-directory",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls[0]["model_directory"] == tmp_path and calls[0]["port"] == 8768


@pytest.mark.parametrize(
    "args",
    [
        ["--host", "0.0.0.0"],
        ["--max-tokens", "8193"],
        ["--batch-size", "0"],
        ["--port", "-1"],
        ["--device", "auto"],
    ],
)
def test_unsafe_startup_is_rejected(args: list[str], tmp_path: Path) -> None:
    assert (
        main(["models", "serve", "bge-reranker-v2-m3", "--model-directory", str(tmp_path), *args])
        == 2
    )


def test_json_rerank_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.commands import reranker
    from tests.unit.api.test_reranker_api import Ranker

    ranker = Ranker()
    monkeypatch.setattr(reranker, "build_ranker", lambda **kwargs: ranker)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"query": "panda", "documents": ["hi", "panda bear"], "top_n": 1})
    )
    assert (
        main(["models", "rerank", "--model-directory", str(tmp_path), "--input", str(request)]) == 0
    )
    assert json.loads(capsys.readouterr().out)["results"][0]["index"] == 1
    assert ranker.closed


def test_failed_warmup_closes_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from capicove.cli.commands.reranker import build_ranker
    from capicove.retrieval import bge_reranker
    from tests.unit.api.test_reranker_api import Ranker

    ranker = Ranker()
    ranker.failure = RuntimeError("warmup failed")
    monkeypatch.setattr(bge_reranker, "BgeReranker", lambda *a, **kw: ranker)
    with pytest.raises(RuntimeError, match="warmup failed"):
        build_ranker(model_directory=tmp_path, device="cpu", batch_size=4, max_tokens=512)
    assert ranker.closed


def test_listener_failure_closes_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    from capicove.cli.commands import reranker
    from tests.unit.api.test_reranker_api import Ranker

    ranker = Ranker()
    monkeypatch.setattr(reranker, "build_ranker", lambda **kw: ranker)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("listener failed")

    monkeypatch.setattr(uvicorn, "run", fail)
    with pytest.raises(OSError, match="listener failed"):
        reranker.serve(
            model_directory=tmp_path,
            device="cpu",
            batch_size=4,
            max_tokens=512,
            host="127.0.0.1",
            port=0,
            api_key=None,
        )
    assert ranker.closed
