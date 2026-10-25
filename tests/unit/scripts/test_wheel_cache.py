"""The e2e wheel cache builds once and shares one environment per selection."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache, WheelInstallError


class FakeUv:
    """Stand in for uv: record commands and create the files uv would create."""

    def __init__(self, *, fail_installs: int = 0) -> None:
        self.commands: list[list[str]] = []
        self.fail_installs = fail_installs
        self._lock = threading.Lock()

    def __call__(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = list(arguments)
        with self._lock:
            self.commands.append(command)
        if command[:2] == ["uv", "build"]:
            out = Path(command[command.index("--out-dir") + 1])
            out.mkdir(parents=True)
            (out / "capicove-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        elif command[:2] == ["uv", "venv"]:
            environment = Path(command[-1])
            (environment / "bin").mkdir(parents=True)
            (environment / "Scripts").mkdir()
        elif command[:2] == ["uv", "export"]:
            Path(command[command.index("--output-file") + 1]).write_text("numpy==2\n")
        elif command[:3] == ["uv", "pip", "install"] and self.fail_installs:
            self.fail_installs -= 1
            return subprocess.CompletedProcess(command, 1, "", "resolution failed")
        return subprocess.CompletedProcess(command, 0, "", "")

    def count(self, *prefix: str) -> int:
        return sum(command[: len(prefix)] == list(prefix) for command in self.commands)


def test_one_build_and_one_environment_serve_repeated_requests(tmp_path: Path) -> None:
    uv = FakeUv()
    cache = WheelCache(tmp_path / "cache", run=uv)

    first = cache.install(("mcp",))
    second = cache.install(("mcp",))

    assert first == second
    assert uv.count("uv", "build") == 1
    assert uv.count("uv", "venv") == 1
    assert uv.count("uv", "pip", "install") == 1
    install = next(command for command in uv.commands if command[:3] == ["uv", "pip", "install"])
    assert install[-1].endswith("capicove-0.1.0-py3-none-any.whl[mcp]")
    assert "--constraints" in install


def test_selections_get_separate_environments_and_no_deps_skips_resolution(
    tmp_path: Path,
) -> None:
    uv = FakeUv()
    cache = WheelCache(tmp_path / "cache", run=uv)

    with_extras = cache.install(("wiki", "mcp"))
    same_extras = cache.install(("mcp", "wiki"))
    base = cache.install()
    bare = cache.install(dependencies=False)

    assert with_extras == same_extras
    assert len({with_extras.environment, base.environment, bare.environment}) == 3
    assert uv.count("uv", "build") == 1
    assert uv.count("uv", "export") == 2
    bare_install = [command for command in uv.commands if "--no-deps" in command]
    assert len(bare_install) == 1 and "--constraints" not in bare_install[0]


def test_failed_install_raises_and_is_retried_from_a_clean_environment(tmp_path: Path) -> None:
    uv = FakeUv(fail_installs=1)
    cache = WheelCache(tmp_path / "cache", run=uv)

    with pytest.raises(WheelInstallError, match="resolution failed"):
        cache.install(("mcp",))
    installed = cache.install(("mcp",))

    assert uv.count("uv", "venv") == 2
    assert Path(installed.python).parent.is_dir()


def test_concurrent_requests_share_one_build_and_environment(tmp_path: Path) -> None:
    uv = FakeUv()
    results: list[object] = []

    def request() -> None:
        results.append(WheelCache(tmp_path / "cache", run=uv).install(("mcp",)))

    threads = [threading.Thread(target=request) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert len(results) == 4 and len(set(results)) == 1
    assert uv.count("uv", "build") == 1
    assert uv.count("uv", "venv") == 1


def test_private_environment_is_created_outside_the_shared_cache(tmp_path: Path) -> None:
    uv = FakeUv()
    cache = WheelCache(tmp_path / "cache", run=uv)

    private = cache.private_environment(tmp_path / "private")

    assert private.environment == tmp_path / "private"
    assert uv.count("uv", "pip", "install") == 0
