"""Installation and runtime diagnostics command."""

from capicove.runtime import collect_diagnostics


def run() -> int:
    """Print local installation diagnostics."""
    report = collect_diagnostics()
    print("Capicove diagnostics")
    print(f"version: {report.version}")
    print(f"python: {report.python_version}")
    print(f"platform: {report.platform_system} {report.platform_machine}")
    print(f"status: {report.status}")
    print("scope: installation metadata. Model inference has not been checked.")
    return 0
