"""The existing explicit operator bootstrap routes queries and evidence safely."""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


bootstrap = importlib.import_module("scripts.run_video_retrieval")


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--workspace",
        str(tmp_path),
        "--model-endpoint",
        "http://127.0.0.1:8000",
        "--es-endpoint",
        "https://localhost:9200",
        "--es-ca",
        str(tmp_path / "ca.crt"),
    ]


@pytest.mark.parametrize(
    "action",
    [
        ["--query", "red"],
        ["--get-evidence", "resource:sha256:" + "a" * 64, "--section", "preview"],
    ],
)
def test_bootstrap_routes_to_existing_runtime_and_closes_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: list[str]
) -> None:
    client = Mock()
    modules = {
        "elasticsearch": {"Elasticsearch": Mock(return_value=client)},
        "capicove_retriever_elasticsearch.adapter": {"ElasticsearchTextRetriever": Mock()},
        "capicove_retriever_elasticsearch.config": {"ElasticsearchConfiguration": Mock()},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    runtime = Mock()
    runtime.cli.return_value = 0
    manager = Mock()
    manager.__enter__ = Mock(return_value=runtime)
    manager.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=manager)
    monkeypatch.setattr(bootstrap, "LocalVideoSearchRuntime", factory)

    assert bootstrap.main([*_arguments(tmp_path), *action]) == 0

    expected = (
        ["search", "red", "--target", "video_segment"]
        if action[0] == "--query"
        else ["get-evidence", action[1], "--section", "preview"]
    )
    runtime.cli.assert_called_once_with(expected)
    assert factory.call_args.kwargs["authorize_evidence"] is bootstrap._authorize_operator
    runtime.build.assert_not_called()
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "action",
    [
        ["--query", "red", "--get-evidence", "evidence"],
        ["--query", "red", "--section", "preview"],
        ["--get-evidence", "evidence", "--build-video", "source.mp4"],
    ],
)
def test_bootstrap_rejects_conflicting_actions_before_provider_setup(
    tmp_path: Path, action: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as failure:
        bootstrap.main([*_arguments(tmp_path), *action])
    assert failure.value.code == 2
    assert "error:" in capsys.readouterr().err
