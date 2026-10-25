"""Tests for the private Tesseract CLI image OCR adapter."""

from __future__ import annotations

import subprocess
import tempfile
from io import BytesIO

import pytest
from PIL import Image

from capicove.processing.image._adapters.tesseract import (
    TesseractCliOcrProvider,
    _parse_tsv,
    _run_process,
)
from capicove.processing.image.invocation import ImageDeadlineExceeded
from capicove.processing.image.ocr import (
    OcrProcessingError,
    OcrProviderUnavailableError,
)


_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
    "width\theight\tconf\ttext\n"
)


def test_parse_tsv_returns_unicode_words_and_normalized_confidence() -> None:
    tsv = (
        _HEADER
        + "4\t1\t1\t1\t1\t0\t0\t0\t30\t10\t-1\t\n"
        + "5\t1\t1\t1\t1\t1\t1\t2\t10\t8\t92.5\t你好\n"
        + "5\t1\t1\t1\t1\t2\t12\t2\t15\t8\t80\tworld\n"
    ).encode()

    output = _parse_tsv(tsv, (40, 20))

    assert [word.text for word in output.words] == ["你好", "world"]
    assert output.words[0].confidence == pytest.approx(0.925)
    assert output.words[1].hierarchy == (1, 1, 1, 1, 2)


def test_parse_tsv_skips_blank_and_negative_confidence_words() -> None:
    tsv = (
        _HEADER
        + "5\t1\t1\t1\t1\t1\t1\t2\t10\t8\t-1\tnoise\n"
        + "5\t1\t1\t1\t1\t2\t12\t2\t15\t8\t80\t   \n"
    ).encode()

    assert _parse_tsv(tsv, (40, 20)).words == ()


@pytest.mark.parametrize(
    ("tsv", "message"),
    [
        (b"bad\theader\n", "header"),
        ((_HEADER + "5\t1\t1\t1\t1\t1\tx\t2\t10\t8\t80\tword\n").encode(), "numeric"),
        ((_HEADER + "5\t1\t1\t1\t1\t1\t1\t2\t10\t8\t101\tword\n").encode(), "confidence"),
        ((_HEADER + "5\t1\t1\t1\t1\t1\t35\t2\t10\t8\t80\tword\n").encode(), "bounds"),
    ],
)
def test_parse_tsv_rejects_invalid_structure(tsv: bytes, message: str) -> None:
    with pytest.raises(OcrProcessingError, match=message):
        _parse_tsv(tsv, (40, 20))


def test_parse_tsv_rejects_invalid_utf8_and_oversized_output() -> None:
    with pytest.raises(OcrProcessingError, match="UTF-8"):
        _parse_tsv(b"\xff", (40, 20))
    with pytest.raises(OcrProcessingError, match="size"):
        _parse_tsv(b"x" * (16 * 1024 * 1024 + 1), (40, 20))


def test_parse_tsv_rejects_missing_text_value() -> None:
    with pytest.raises(OcrProcessingError, match="columns"):
        _parse_tsv(
            (_HEADER + "5\t1\t1\t1\t1\t1\t1\t2\t10\t8\t80\n").encode(),
            (40, 20),
        )


def test_provider_uses_fixed_command_and_png_standard_input() -> None:
    calls: list[tuple[tuple[str, ...], bytes, float]] = []

    def runner(
        arguments: tuple[str, ...],
        stdin: bytes,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((arguments, stdin, timeout))
        return subprocess.CompletedProcess(arguments, 0, _HEADER.encode(), b"")

    provider = TesseractCliOcrProvider(runner=runner, version="5.5.0")

    output = provider.recognize(Image.new("RGB", (20, 20), "white"), ("chi_sim", "eng"))

    assert output.words == ()
    assert calls[0][0] == (
        "tesseract",
        "stdin",
        "stdout",
        "-l",
        "chi_sim+eng",
        "tsv",
    )
    assert calls[0][1].startswith(b"\x89PNG\r\n\x1a\n")
    assert calls[0][2] == 300.0
    assert provider.provider_id == "tesseract-cli"
    assert provider.provider_version == "5.5.0"


def test_provider_maps_missing_executable_and_language_to_unavailable() -> None:
    def missing(
        arguments: tuple[str, ...],
        stdin: bytes,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del arguments, stdin, timeout
        raise FileNotFoundError

    with pytest.raises(OcrProviderUnavailableError, match="unavailable"):
        TesseractCliOcrProvider(runner=missing).recognize(
            Image.new("RGB", (20, 20)),
            ("eng",),
        )

    def missing_language(
        arguments: tuple[str, ...],
        stdin: bytes,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del stdin, timeout
        return subprocess.CompletedProcess(
            arguments,
            1,
            b"",
            b"Error opening data file /secret/chi_sim.traineddata\n"
            b"Failed loading language 'chi_sim'",
        )

    with pytest.raises(OcrProviderUnavailableError, match="language data") as error:
        TesseractCliOcrProvider(runner=missing_language).recognize(
            Image.new("RGB", (20, 20)),
            ("chi_sim",),
        )
    assert "/secret" not in str(error.value)


def test_provider_maps_unrelated_process_failure_without_raw_diagnostic() -> None:
    def failed(
        arguments: tuple[str, ...],
        stdin: bytes,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del stdin, timeout
        return subprocess.CompletedProcess(arguments, 2, b"", b"/secret/image: bad")

    with pytest.raises(OcrProcessingError, match="failed") as error:
        TesseractCliOcrProvider(runner=failed).recognize(
            Image.new("RGB", (20, 20)),
            ("eng",),
        )
    assert "/secret" not in str(error.value)


def test_run_process_kills_process_group_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[tuple[int, int]] = []

    class _Process:
        pid = 123
        returncode = -9

        def poll(self) -> None:
            return None

        def wait(self) -> int:
            return self.returncode

    process = _Process()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.tesseract.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "capicove.processing.image._adapters.tesseract.os.killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    with pytest.raises(ImageDeadlineExceeded):
        _run_process(("tesseract",), b"png", 0.1)
    assert killed and killed[0][0] == 123


def test_run_process_stops_oversized_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []

    class _Process:
        pid = 456
        returncode = -9

        def poll(self) -> None:
            return None

        def wait(self) -> int:
            return self.returncode

    def popen(*args: object, **kwargs: object) -> _Process:
        del args
        stdout = kwargs["stdout"]
        assert isinstance(stdout, tempfile._TemporaryFileWrapper | object)
        stdout.write(b"x" * (16 * 1024 * 1024 + 1))  # type: ignore[attr-defined]
        stdout.flush()  # type: ignore[attr-defined]
        return _Process()

    monkeypatch.setattr(
        "capicove.processing.image._adapters.tesseract.subprocess.Popen",
        popen,
    )
    monkeypatch.setattr(
        "capicove.processing.image._adapters.tesseract.os.killpg",
        lambda pid, sig: killed.append(pid),
    )

    with pytest.raises(OcrProcessingError, match="size"):
        _run_process(("tesseract",), b"png", 1.0)
    assert killed == [456]


def _png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (20, 20), "white").save(stream, format="PNG")
    return stream.getvalue()
