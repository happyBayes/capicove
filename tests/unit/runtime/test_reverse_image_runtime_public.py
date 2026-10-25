"""Public composition-root contract for reverse-image retrieval."""

from __future__ import annotations

import subprocess
import sys


def test_reverse_image_runtime_exports_remain_lazy() -> None:
    """Importing base runtime diagnostics must not import image dependencies."""
    source = """
import sys
import capicove.runtime as runtime

assert "ReverseImageRuntime" in runtime.__all__
assert "ReverseImageModelProvisioning" not in runtime.__all__
assert "build_reverse_image_runtime" in runtime.__all__
assert "capicove.runtime.reverse_image" not in sys.modules
assert "capicove.processing.image._adapters.onnx_target" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
