import platform

from capicove.__about__ import __version__
from capicove.runtime import collect_diagnostics


def test_collect_diagnostics_reports_the_installed_runtime() -> None:
    report = collect_diagnostics()

    assert report.version == __version__
    assert report.python_version == platform.python_version()
    assert report.platform_system == platform.system()
    assert report.platform_machine == platform.machine()
    assert report.status == "ok"
