"""Canonical model imports and resources after legacy package removal."""

import importlib.util
import subprocess
import sys
from importlib import resources


def test_legacy_package_is_absent() -> None:
    assert importlib.util.find_spec("capicove.model_management") is None


def test_model_import_and_factory_validation_keep_sdks_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.model; import sys; "
                "from pathlib import Path; "
                "from capicove.model import ModelProfile, ModelConfiguration, ModelFactory, ModelCapability; "
                "profile = ModelProfile('dense', 'bge-m3', ModelCapability.TEXT_EMBEDDING, Path.cwd(), 'a'*64); "
                "factory = ModelFactory(ModelConfiguration({'dense': profile}, {'text': 'dense'})); factory.close(); "
                "assert not {'torch', 'transformers', 'sentence_transformers', 'paddle', "
                "'paddleocr', 'cv2', 'PIL', 'numpy'} & sys.modules.keys()"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_resources_exist_under_canonical_package() -> None:
    for name in ("image_models.toml", "bge_m3.lock.toml"):
        assert resources.files("capicove.model").joinpath(name).read_bytes()
