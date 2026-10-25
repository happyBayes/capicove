"""Qualification evidence rejects missing, substituted and skipped observations."""

import hashlib
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _api() -> ModuleType:
    assert importlib.util.find_spec("tests.remote_support.receipts") is not None
    return importlib.import_module("tests.remote_support.receipts")


def test_observation_requires_actual_artifact_bytes(tmp_path: Path) -> None:
    api = _api()
    output = tmp_path / "observation.json"
    output.write_bytes(b'{"observed_calls":1}')
    observation = api.observe(output, root=tmp_path, test_id="suite::case", profile="cpu")
    assert observation.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    api.verify(observation, root=tmp_path, test_id="suite::case", profile="cpu")
    output.write_bytes(b'{"observed_calls":2}')
    with pytest.raises(ValueError, match="digest"):
        api.verify(observation, root=tmp_path, test_id="suite::case", profile="cpu")


def test_observation_rejects_escape_or_missing_file(tmp_path: Path) -> None:
    api = _api()
    with pytest.raises(ValueError):
        api.observe(tmp_path / "missing", root=tmp_path, test_id="a", profile="b")
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    root = tmp_path / "receipts"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        api.observe(root / "link", root=root, test_id="a", profile="b")


@pytest.mark.parametrize("state", ["skipped", "not_run", "failed", "mock", "unknown"])
def test_only_observed_success_closes_a_requirement(tmp_path: Path, state: str) -> None:
    api = _api()
    with pytest.raises(ValueError):
        api.require_complete({"op/provider": state}, required={"op/provider"})
    with pytest.raises(ValueError):
        api.require_complete({}, required={"op/provider"})
    api.require_complete({"op/provider": "passed"}, required={"op/provider"})


def test_profile_and_test_substitution_is_rejected(tmp_path: Path) -> None:
    api = _api()
    path = tmp_path / "result"
    path.write_bytes(b"real observation")
    observed = api.observe(path, root=tmp_path, test_id="one", profile="model-a")
    with pytest.raises(ValueError):
        api.verify(observed, root=tmp_path, test_id="two", profile="model-a")
    with pytest.raises(ValueError):
        api.verify(observed, root=tmp_path, test_id="one", profile="model-b")
