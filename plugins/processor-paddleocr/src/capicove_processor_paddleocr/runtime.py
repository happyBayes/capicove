"""Side-effect-free validation of the installed parser runtime."""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType


_DISTRIBUTIONS = {
    "paddleocr": ("paddleocr",),
    "paddlepaddle": ("paddlepaddle", "paddlepaddle-gpu"),
    "paddlex": ("paddlex",),
    "pypdf": ("pypdf",),
    "pypdfium2": ("pypdfium2",),
}
_MODULE_DISTRIBUTIONS = {
    "paddleocr": ("paddleocr",),
    "paddlex": ("paddlex",),
    "paddle": ("paddlepaddle", "paddlepaddle-gpu"),
    "pypdf": ("pypdf",),
    "pypdfium2": ("pypdfium2",),
}
_OFFLINE_FLAG = "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"


def _owned_distribution_files(candidates: tuple[str, ...]) -> frozenset[Path]:
    installed: list[metadata.Distribution] = []
    for name in candidates:
        try:
            installed.append(metadata.distribution(name))
        except metadata.PackageNotFoundError:
            continue
    if len(installed) != 1:
        raise RuntimeError("runtime distribution is missing or ambiguous")
    distribution = installed[0]
    files = distribution.files
    if files is None:
        raise RuntimeError("runtime distribution has no installed-file manifest")
    return frozenset(Path(str(distribution.locate_file(path))).resolve() for path in files)


def _namespace_is_owned(module: ModuleType, owned: frozenset[Path]) -> bool:
    spec = getattr(module, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None)
    if locations is None:
        return False
    suffix = Path(*module.__name__.split("."))
    resolved = tuple(Path(str(location)).resolve() for location in locations)
    return bool(resolved) and all(
        tuple(location.parts[-len(suffix.parts) :]) == suffix.parts
        and any(location in path.parents for path in owned)
        for location in resolved
    )


def _dynamic_paddle_module_is_owned(module: ModuleType, owned: frozenset[Path]) -> bool:
    prefix = "paddle.base.libpaddle."
    if not module.__name__.startswith(prefix):
        return False
    parent = sys.modules.get("paddle.base.libpaddle")
    parent_path = getattr(parent, "__file__", None)
    if not isinstance(parent, ModuleType) or not isinstance(parent_path, str):
        return False
    if Path(parent_path).resolve() not in owned:
        return False
    current: object = parent
    for segment in module.__name__[len(prefix) :].split("."):
        current = getattr(current, segment, None)
        if current is None:
            return False
    return current is module


def _preflight_runtime_module(name: str) -> None:
    owned = _owned_distribution_files(_MODULE_DISTRIBUTIONS[name])
    spec = PathFinder.find_spec(name)
    origin = None if spec is None else spec.origin
    if (
        spec is None
        or spec.loader is None
        or not isinstance(origin, str)
        or Path(origin).resolve() not in owned
    ):
        raise RuntimeError(f"runtime module is outside its installed distribution: {name}")


def verify_loaded_runtime_modules() -> None:
    """Bind every loaded OCR runtime module to its installed distribution files."""
    for prefix, distributions in _MODULE_DISTRIBUTIONS.items():
        modules = [
            module
            for name, module in sys.modules.items()
            if module is not None and (name == prefix or name.startswith(f"{prefix}."))
        ]
        if not modules:
            continue
        owned = _owned_distribution_files(distributions)
        for module in modules:
            raw_path = getattr(module, "__file__", None)
            if isinstance(raw_path, str) and Path(raw_path).resolve() in owned:
                continue
            if isinstance(module, ModuleType) and (
                _namespace_is_owned(module, owned)
                or (prefix == "paddle" and _dynamic_paddle_module_is_owned(module, owned))
            ):
                continue
            if not isinstance(raw_path, str) or Path(raw_path).resolve() not in owned:
                raise RuntimeError(
                    f"runtime module is outside its installed distribution: {module.__name__}"
                )


def enforce_offline_runtime_policy() -> None:
    """Set PaddleX offline mode before import and reject an unsafe captured value."""
    os.environ[_OFFLINE_FLAG] = "True"
    if "paddlex" not in sys.modules:
        return
    verify_loaded_runtime_modules()
    flags = importlib.import_module("paddlex.utils.flags")
    verify_loaded_runtime_modules()
    captured = getattr(flags, "DISABLE_MODEL_SOURCE_CHECK", None)
    offline = captured is True or (
        isinstance(captured, str) and captured.strip().casefold() in {"1", "true", "yes", "on"}
    )
    if not offline:
        raise RuntimeError("PaddleX runtime was imported before offline mode was enforced")


def import_verified_runtime_module(name: str) -> ModuleType:
    """Import one declared runtime module and validate the complete loaded closure."""
    if name not in _MODULE_DISTRIBUTIONS:
        raise ValueError(f"unsupported runtime module: {name}")
    if name in {"paddleocr", "paddlex"}:
        enforce_offline_runtime_policy()
    verify_loaded_runtime_modules()
    _preflight_runtime_module(name)
    module = importlib.import_module(name)
    verify_loaded_runtime_modules()
    return module


def resolve_runtime_identity(
    inference_engine: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Validate compatible distributions and return their exact identities."""
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    requirements = dict(inference_engine)
    if set(requirements) != {"name", *_DISTRIBUTIONS}:
        raise ValueError("PaddleOCR inference engine manifest is invalid")
    if requirements["name"] != "PP-StructureV3":
        raise ValueError("unsupported PaddleOCR inference engine")
    identity = [("engine.name", requirements["name"])]
    for runtime, candidates in _DISTRIBUTIONS.items():
        installed: list[tuple[str, str]] = []
        for distribution in candidates:
            try:
                installed.append((distribution, metadata.version(distribution)))
            except metadata.PackageNotFoundError:
                continue
        if not installed:
            raise RuntimeError(f"required runtime distribution is unavailable: {runtime}")
        if len(installed) != 1:
            names = ", ".join(name for name, _ in installed)
            raise RuntimeError(f"runtime distribution is ambiguous for {runtime}: {names}")
        distribution, raw_version = installed[0]
        try:
            version = Version(raw_version)
            compatible = version in SpecifierSet(requirements[runtime])
        except (InvalidSpecifier, InvalidVersion) as error:
            raise ValueError(f"invalid runtime version policy for {runtime}") from error
        if not compatible:
            raise RuntimeError(
                f"incompatible {runtime} runtime {version}; expected {requirements[runtime]}"
            )
        identity.extend(
            (
                (f"{runtime}.distribution", distribution),
                (f"{runtime}.version", str(version)),
            )
        )
    return tuple(sorted(identity))


__all__ = [
    "enforce_offline_runtime_policy",
    "import_verified_runtime_module",
    "resolve_runtime_identity",
    "verify_loaded_runtime_modules",
]
