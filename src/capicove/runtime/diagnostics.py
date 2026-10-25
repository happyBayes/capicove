"""Side-effect-free runtime diagnostics."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Literal

from capicove.__about__ import __version__


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Describe the installed Capicove and Python runtime."""

    version: str
    python_version: str
    platform_system: str
    platform_machine: str
    status: Literal["ok"]


def collect_diagnostics() -> DiagnosticReport:
    """Collect local runtime metadata without external access or mutation."""
    return DiagnosticReport(
        version=__version__,
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        status="ok",
    )
