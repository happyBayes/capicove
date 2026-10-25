from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import capicove_processor_paddleocr.runtime as runtime
import pytest
from capicove_processor_paddleocr.models import load_book_balanced_manifest
from capicove_processor_paddleocr.runtime import (
    enforce_offline_runtime_policy,
    import_verified_runtime_module,
    resolve_runtime_identity,
    verify_loaded_runtime_modules,
)
from pytest import MonkeyPatch


def _resolve(monkeypatch: MonkeyPatch, versions: dict[str, str]) -> dict[str, str]:
    def version(distribution: str) -> str:
        if distribution not in versions:
            raise metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(metadata, "version", version)
    manifest = load_book_balanced_manifest()
    return dict(resolve_runtime_identity(manifest.inference_engine))


def test_runtime_identity_records_exact_cpu_distributions(monkeypatch: MonkeyPatch) -> None:
    identity = _resolve(
        monkeypatch,
        {
            "paddleocr": "3.7.0",
            "paddlepaddle": "3.3.1",
            "paddlex": "3.7.2",
            "pypdf": "6.18.1",
            "pypdfium2": "5.13.0",
        },
    )

    assert identity == {
        "engine.name": "PP-StructureV3",
        "paddleocr.distribution": "paddleocr",
        "paddleocr.version": "3.7.0",
        "paddlepaddle.distribution": "paddlepaddle",
        "paddlepaddle.version": "3.3.1",
        "paddlex.distribution": "paddlex",
        "paddlex.version": "3.7.2",
        "pypdf.distribution": "pypdf",
        "pypdf.version": "6.18.1",
        "pypdfium2.distribution": "pypdfium2",
        "pypdfium2.version": "5.13.0",
    }


def test_runtime_identity_accepts_the_gpu_distribution(monkeypatch: MonkeyPatch) -> None:
    identity = _resolve(
        monkeypatch,
        {
            "paddleocr": "3.7.0",
            "paddlepaddle-gpu": "3.3.1",
            "paddlex": "3.7.2",
            "pypdf": "6.18.1",
            "pypdfium2": "5.13.0",
        },
    )

    assert identity["paddlepaddle.distribution"] == "paddlepaddle-gpu"


@pytest.mark.parametrize(
    "versions",
    (
        {
            "paddleocr": "3.7.0",
            "paddlex": "3.7.2",
            "pypdf": "6.18.1",
            "pypdfium2": "5.13.0",
        },
        {
            "paddleocr": "3.7.0",
            "paddlepaddle": "3.3.1",
            "paddlepaddle-gpu": "3.3.1",
            "paddlex": "3.7.2",
            "pypdf": "6.18.1",
            "pypdfium2": "5.13.0",
        },
        {
            "paddleocr": "3.7.0",
            "paddlepaddle": "3.3.1",
            "paddlex": "3.3.0",
            "pypdf": "6.18.1",
            "pypdfium2": "5.13.0",
        },
    ),
)
def test_runtime_identity_rejects_missing_ambiguous_or_incompatible_distributions(
    monkeypatch: MonkeyPatch, versions: dict[str, str]
) -> None:
    with pytest.raises(RuntimeError, match=r"unavailable|ambiguous|incompatible"):
        _resolve(monkeypatch, versions)


def test_runtime_module_shadow_is_rejected_before_use(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    installed = tmp_path / "site-packages"
    owned = installed / "paddleocr" / "__init__.py"
    owned.parent.mkdir(parents=True)
    owned.write_text("# installed\n", encoding="utf-8")
    shadow = tmp_path / "shadow" / "paddleocr.py"
    shadow.parent.mkdir()
    shadow.write_text("# shadow\n", encoding="utf-8")

    class Distribution:
        files = (Path("paddleocr/__init__.py"),)

        @staticmethod
        def locate_file(path: Path) -> Path:
            return installed / path

    def distribution(name: str) -> object:
        if name == "paddleocr":
            return Distribution()
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", distribution)
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(__name__="paddleocr", __file__=str(shadow)),
    )

    with pytest.raises(RuntimeError, match="outside its installed distribution"):
        import_verified_runtime_module("paddleocr")


def test_runtime_shadow_is_rejected_before_top_level_code_executes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    installed = tmp_path / "site-packages"
    owned = installed / "pypdf" / "__init__.py"
    owned.parent.mkdir(parents=True)
    owned.write_text("# installed\n", encoding="utf-8")
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    marker = tmp_path / "shadow-executed"
    (shadow / "pypdf.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    class Distribution:
        files = (Path("pypdf/__init__.py"),)

        @staticmethod
        def locate_file(path: Path) -> Path:
            return installed / path

    monkeypatch.setattr(metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(runtime, "verify_loaded_runtime_modules", lambda: None)
    monkeypatch.syspath_prepend(str(shadow))
    monkeypatch.delitem(sys.modules, "pypdf", raising=False)

    with pytest.raises(RuntimeError, match="outside its installed distribution"):
        import_verified_runtime_module("pypdf")
    assert not marker.exists()


def test_fileless_fake_paddle_submodule_is_rejected(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    installed = tmp_path / "site-packages"
    owned = installed / "paddle" / "__init__.py"
    owned.parent.mkdir(parents=True)
    owned.write_text("# installed\n", encoding="utf-8")

    class Distribution:
        files = (Path("paddle/__init__.py"),)

        @staticmethod
        def locate_file(path: Path) -> Path:
            return installed / path

    def distribution(name: str) -> object:
        if name == "paddlepaddle":
            return Distribution()
        raise metadata.PackageNotFoundError(name)

    paddle = ModuleType("paddle")
    paddle.__file__ = str(owned)
    fake = SimpleNamespace(__name__="paddle.inference", __file__=None)
    monkeypatch.setattr(metadata, "distribution", distribution)
    monkeypatch.setattr(runtime, "_MODULE_DISTRIBUTIONS", {"paddle": ("paddlepaddle",)})
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "paddle", paddle)
        context.setitem(sys.modules, "paddle.inference", fake)
        with pytest.raises(RuntimeError, match="outside its installed distribution"):
            verify_loaded_runtime_modules()


@pytest.mark.parametrize(("captured", "rejected"), ((False, True), ("True", False)))
def test_preloaded_paddlex_offline_flag_is_enforced(
    tmp_path: Path, monkeypatch: MonkeyPatch, captured: object, rejected: bool
) -> None:
    installed = tmp_path / "site-packages"
    files = tuple(
        Path(name)
        for name in (
            "paddlex/__init__.py",
            "paddlex/utils/__init__.py",
            "paddlex/utils/flags.py",
        )
    )
    for relative in files:
        path = installed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# installed\n", encoding="utf-8")

    owned_files = files

    class Distribution:
        files = owned_files

        @staticmethod
        def locate_file(path: Path) -> Path:
            return installed / path

    paddlex = ModuleType("paddlex")
    paddlex.__file__ = str(installed / files[0])
    utils = ModuleType("paddlex.utils")
    utils.__file__ = str(installed / files[1])
    flags = ModuleType("paddlex.utils.flags")
    flags.__file__ = str(installed / files[2])
    flags.__dict__["DISABLE_MODEL_SOURCE_CHECK"] = captured
    monkeypatch.setattr(metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(runtime, "_MODULE_DISTRIBUTIONS", {"paddlex": ("paddlex",)})
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "paddlex", paddlex)
        context.setitem(sys.modules, "paddlex.utils", utils)
        context.setitem(sys.modules, "paddlex.utils.flags", flags)
        if not rejected:
            enforce_offline_runtime_policy()
        else:
            with pytest.raises(RuntimeError, match="before offline mode"):
                enforce_offline_runtime_policy()
