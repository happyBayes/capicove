"""Session-shared installed-wheel environments for end-to-end tests.

Building the wheel and installing its dependencies dominates the storage and time
cost of the installed-wheel journeys. A ``WheelCache`` builds the wheel once and
creates one environment per dependency selection; every test and xdist worker in
the session shares them through file locks. Tests must treat a shared environment
as read-only. A test that changes its environment uses ``private_environment``.

The cache root must be specific to one session, because it is never invalidated
when the source tree changes.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


ROOT_VARIABLE = "CAPICOVE_TEST_WHEEL_ROOT"
_TIMEOUT_SECONDS = 600

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class WheelInstallError(RuntimeError):
    """A uv command failed; the message carries its stderr."""


@dataclass(frozen=True, slots=True)
class InstalledWheel:
    """Paths of one virtual environment and its console entry points."""

    environment: Path
    python: str
    executable: str


def environment_paths(environment: Path) -> InstalledWheel:
    """Return the interpreter and ``capicove`` command of a virtual environment."""
    windows = sys.platform == "win32"
    scripts = environment / ("Scripts" if windows else "bin")
    return InstalledWheel(
        environment,
        str(scripts / ("python.exe" if windows else "python")),
        str(scripts / ("capicove.exe" if windows else "capicove")),
    )


def _run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments), capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, check=False
    )


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold an exclusive lock shared by threads and processes on every OS."""
    with path.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            while True:
                handle.seek(0)
                try:
                    # LK_LOCK retries for about ten seconds before raising.
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class WheelCache:
    """Build the project wheel once and share environments per selection."""

    def __init__(self, root: Path, *, run: Runner = _run) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self._run = run

    def wheel(self) -> Path:
        """Return the session's wheel, building it on first use."""
        directory = self._root / "dist"
        with _exclusive(self._root / "dist.lock"):
            ready = self._root / "dist.ready"
            if not ready.exists():
                shutil.rmtree(directory, ignore_errors=True)
                self._check(["uv", "build", "--wheel", "--out-dir", str(directory)])
                ready.touch()
            (wheel,) = directory.glob("capicove-*.whl")
            return wheel

    def constraints(self, extras: Sequence[str] = ()) -> Path:
        """Return locked constraints for the base package plus ``extras``."""
        selected = tuple(sorted(set(extras)))
        path = self._root / f"constraints-{_key(selected)}.txt"
        with _exclusive(path.with_suffix(".lock")):
            if not path.exists():
                staging = path.with_suffix(".partial")
                self._check(
                    [
                        *("uv", "export", "--frozen", "--no-dev", "--no-default-groups"),
                        *(argument for name in selected for argument in ("--extra", name)),
                        *("--no-emit-project", "--no-hashes", "--output-file", str(staging)),
                    ]
                )
                staging.replace(path)
        return path

    def install(self, extras: Sequence[str] = (), *, dependencies: bool = True) -> InstalledWheel:
        """Return a shared environment with the wheel and the selected extras."""
        selected = tuple(sorted(set(extras)))
        key = _key(selected) + ("" if dependencies else "-nodeps")
        wheel = self.wheel()
        constraints = self.constraints(selected) if dependencies else None
        environment = self._root / f"env-{key}"
        with _exclusive(self._root / f"env-{key}.lock"):
            ready = self._root / f"env-{key}.ready"
            if not ready.exists():
                shutil.rmtree(environment, ignore_errors=True)
                installed = environment_paths(environment)
                self._check(["uv", "venv", "--python", "3.11", str(environment)])
                target = f"{wheel}[{','.join(selected)}]" if selected else str(wheel)
                if constraints is None:
                    options = ["--no-deps"]
                else:
                    options = ["--constraints", str(constraints)]
                self._check(
                    ["uv", "pip", "install", *options, "--python", installed.python, target]
                )
                ready.touch()
        return environment_paths(environment)

    def private_environment(self, environment: Path) -> InstalledWheel:
        """Create an empty environment the caller owns and may change."""
        self._check(["uv", "venv", "--python", "3.11", str(environment)])
        return environment_paths(environment)

    def _check(self, arguments: list[str]) -> None:
        completed = self._run(arguments)
        if completed.returncode != 0:
            raise WheelInstallError(
                f"{' '.join(arguments[:3])} failed:\n{completed.stdout}\n{completed.stderr}"
            )


def _key(extras: Sequence[str]) -> str:
    return "-".join(extras) if extras else "base"
