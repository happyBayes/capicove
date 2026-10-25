"""Installed dependency audits retain every third-party package."""

import importlib.util
from pathlib import Path
from types import ModuleType


def module() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "release_smoke.py"
    spec = importlib.util.spec_from_file_location("capicove_release_smoke_test", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_official_cpu_torch_is_audited_as_its_public_release() -> None:
    requirements, aliases = module().audit_requirements(
        {"torch": "2.14.1+cpu", "numpy": "2.4.0", "capicove": "0.1.0"}
    )
    assert requirements.splitlines() == ["numpy==2.4.0", "torch==2.14.1"]
    assert aliases == {"torch": {"installed": "2.14.1+cpu", "audited": "2.14.1"}}


def test_unrecognized_local_builds_and_similar_names_are_not_skipped() -> None:
    requirements, aliases = module().audit_requirements(
        {"torch": "2.14.1+custom", "capicove-extra": "9.0", "capicove_retriever_faiss": "0.1.0"}
    )
    assert requirements.splitlines() == ["capicove-extra==9.0", "torch==2.14.1+custom"]
    assert aliases == {}
