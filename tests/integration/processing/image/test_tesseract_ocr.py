"""Local integration coverage for Tesseract structured image OCR."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest

from capicove.processing.image import ImageOcr, ImageSource
from capicove.processing.image._adapters.tesseract import TesseractCliOcrProvider


_FIXTURE = (
    Path(__file__).parents[3] / "datasets" / "images" / "kelly-sikkema-Kl1gC0ve620-unsplash.jpg"
)


@pytest.mark.integration
def test_tesseract_recognizes_structured_text_from_repository_fixture() -> None:
    if not _FIXTURE.is_file():
        pytest.skip(f"place the local OCR fixture at {_FIXTURE}")
    executable = shutil.which("tesseract")
    if executable is None:
        pytest.skip("install Tesseract 5 with the chi_sim and eng trained data")
    # The executable path is resolved locally and the argument vector is fixed.
    languages = subprocess.run(  # nosec B603
        (executable, "--list-langs"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    missing = {"chi_sim", "eng"}.difference(languages)
    if missing:
        pytest.skip(f"install Tesseract trained data: {', '.join(sorted(missing))}")

    result = ImageOcr(TesseractCliOcrProvider(executable)).recognize(
        (ImageSource(path=str(_FIXTURE)),)
    )[0]

    assert result.words
    assert result.full_text
    assert result.languages == ("chi_sim", "eng")
    assert result.provider_id == "tesseract-cli"
    assert result.provider_version
    assert all(word.x + word.width <= 640 for word in result.words)
    assert all(word.y + word.height <= 960 for word in result.words)
