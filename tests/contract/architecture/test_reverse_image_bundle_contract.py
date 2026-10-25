"""Architecture contract for the offline image-encoder bundle verifier."""

from __future__ import annotations

import subprocess
import sys


def test_bundle_contract_imports_without_optional_model_runtimes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import capicove.model.bundle; "
                "blocked={'onnxruntime','torch','transformers','faiss'}; "
                "assert not blocked.intersection(sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
