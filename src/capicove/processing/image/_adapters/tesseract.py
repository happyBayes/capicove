"""Private Tesseract CLI adapter for structured image OCR."""

from __future__ import annotations

import contextlib
import csv
import os
import signal
import subprocess  # nosec B404
import tempfile
import time
from collections.abc import Callable
from io import BytesIO, StringIO

from PIL import Image

from capicove.processing.image.invocation import (
    ImageDeadlineExceeded,
    remaining_image_seconds,
)
from capicove.processing.image.models import OcrWord
from capicove.processing.image.ocr import (
    OcrProcessingError,
    OcrProviderOutput,
    OcrProviderUnavailableError,
)


CommandRunner = Callable[
    [tuple[str, ...], bytes, float],
    subprocess.CompletedProcess[bytes],
]
_TSV_FIELDS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)
_MAX_TSV_BYTES = 16 * 1024 * 1024
_MAX_STDERR_BYTES = 8192
_MAX_WORDS = 100_000
_DEFAULT_TIMEOUT_SECONDS = 300.0
_VERSION_TIMEOUT_SECONDS = 5.0


class TesseractCliOcrProvider:
    """Recognize words using a fixed local Tesseract CLI invocation."""

    provider_id = "tesseract-cli"

    def __init__(
        self,
        executable: str = "tesseract",
        *,
        runner: CommandRunner | None = None,
        version: str | None = None,
    ) -> None:
        self._executable = executable
        self._runner = runner or _run_process
        self._version = version
        self._version_checked = version is not None

    @property
    def provider_version(self) -> str | None:
        """Return the bounded, lazily detected Tesseract version."""
        if not self._version_checked:
            self._version = self._read_version()
            self._version_checked = True
        return self._version

    def recognize(
        self,
        image: Image.Image,
        languages: tuple[str, ...],
    ) -> OcrProviderOutput:
        """Return structured OCR words for one normalized image."""
        stream = BytesIO()
        image.save(stream, format="PNG")
        arguments = (
            self._executable,
            "stdin",
            "stdout",
            "-l",
            "+".join(languages),
            "tsv",
        )
        try:
            completed = self._runner(
                arguments,
                stream.getvalue(),
                remaining_image_seconds(_DEFAULT_TIMEOUT_SECONDS),
            )
        except FileNotFoundError as error:
            raise OcrProviderUnavailableError("Tesseract OCR provider is unavailable.") from error
        if completed.returncode != 0:
            if _is_missing_language(completed.stderr):
                raise OcrProviderUnavailableError(
                    "Requested Tesseract language data is unavailable."
                )
            raise OcrProcessingError("Tesseract OCR process failed.")
        return _parse_tsv(completed.stdout, image.size)

    def _read_version(self) -> str | None:
        try:
            completed = self._runner(
                (self._executable, "--version"),
                b"",
                remaining_image_seconds(_VERSION_TIMEOUT_SECONDS),
            )
        except FileNotFoundError as error:
            raise OcrProviderUnavailableError("Tesseract OCR provider is unavailable.") from error
        if completed.returncode != 0 or not completed.stdout:
            return None
        first_line = completed.stdout.decode("utf-8", errors="replace").splitlines()[0]
        parts = first_line.split()
        return parts[1] if len(parts) >= 2 else None


def _parse_tsv(tsv: bytes, image_size: tuple[int, int]) -> OcrProviderOutput:
    if len(tsv) > _MAX_TSV_BYTES:
        raise OcrProcessingError("Tesseract TSV output exceeds the size limit.")
    try:
        decoded = tsv.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OcrProcessingError("Tesseract TSV output is not valid UTF-8.") from error
    reader = csv.DictReader(StringIO(decoded), delimiter="\t")
    if tuple(reader.fieldnames or ()) != _TSV_FIELDS:
        raise OcrProcessingError("Tesseract TSV header is invalid.")
    words: list[OcrWord] = []
    for row in reader:
        if (
            None in row
            or any(field not in row for field in _TSV_FIELDS)
            or any(row[field] is None for field in _TSV_FIELDS)
        ):
            raise OcrProcessingError("Tesseract TSV row has invalid columns.")
        try:
            level = int(row["level"])
        except (TypeError, ValueError) as error:
            raise OcrProcessingError("Tesseract TSV row has invalid numeric data.") from error
        if level != 5:
            continue
        try:
            confidence_percent = float(row["conf"])
            page = int(row["page_num"])
            block = int(row["block_num"])
            paragraph = int(row["par_num"])
            line = int(row["line_num"])
            word_number = int(row["word_num"])
            x = int(row["left"])
            y = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
        except (TypeError, ValueError) as error:
            raise OcrProcessingError("Tesseract TSV row has invalid numeric data.") from error
        text = row["text"].strip()
        if confidence_percent < 0 or not text:
            continue
        if confidence_percent > 100:
            raise OcrProcessingError("Tesseract TSV confidence is outside [0, 100].")
        try:
            word = OcrWord(
                text=text,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=confidence_percent / 100.0,
                page=page,
                block=block,
                paragraph=paragraph,
                line=line,
                word=word_number,
            )
        except ValueError as error:
            raise OcrProcessingError("Tesseract TSV word structure is invalid.") from error
        image_width, image_height = image_size
        if word.x + word.width > image_width or word.y + word.height > image_height:
            raise OcrProcessingError("Tesseract TSV word bounds exceed the image.")
        words.append(word)
        if len(words) > _MAX_WORDS:
            raise OcrProcessingError("Tesseract TSV output exceeds the word limit.")
    return OcrProviderOutput(tuple(words))


def _is_missing_language(stderr: bytes) -> bool:
    diagnostic = stderr.decode("utf-8", errors="replace").casefold()
    return "failed loading language" in diagnostic or "error opening data file" in diagnostic


def _run_process(
    arguments: tuple[str, ...],
    stdin: bytes,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    deadline = time.monotonic() + timeout
    with (
        tempfile.TemporaryFile() as input_file,
        tempfile.TemporaryFile() as output_file,
        tempfile.TemporaryFile() as error_file,
    ):
        input_file.write(stdin)
        input_file.seek(0)
        # The argument vector is constructed internally and never passed through a shell.
        process = subprocess.Popen(  # nosec B603
            arguments,
            stdin=input_file,
            stdout=output_file,
            stderr=error_file,
            shell=False,
            start_new_session=True,
        )
        while process.poll() is None:
            if _file_size(output_file) > _MAX_TSV_BYTES:
                _terminate_process_group(process)
                raise OcrProcessingError("Tesseract TSV output exceeds the size limit.")
            if _file_size(error_file) > _MAX_STDERR_BYTES:
                _terminate_process_group(process)
                raise OcrProcessingError("Tesseract diagnostic output exceeds the size limit.")
            if time.monotonic() >= deadline:
                _terminate_process_group(process)
                raise ImageDeadlineExceeded("Tesseract OCR invocation exceeded its deadline.")
            time.sleep(0.01)
        output_file.seek(0)
        error_file.seek(0)
        stdout = output_file.read(_MAX_TSV_BYTES + 1)
        stderr = error_file.read(_MAX_STDERR_BYTES + 1)
        if len(stdout) > _MAX_TSV_BYTES:
            raise OcrProcessingError("Tesseract TSV output exceeds the size limit.")
        if len(stderr) > _MAX_STDERR_BYTES:
            raise OcrProcessingError("Tesseract diagnostic output exceeds the size limit.")
        return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)


def _file_size(stream: object) -> int:
    return os.fstat(stream.fileno()).st_size  # type: ignore[attr-defined]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()
