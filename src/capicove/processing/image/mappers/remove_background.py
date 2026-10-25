"""Create immutable PNG images with their backgrounds removed."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from capicove.processing.image.invocation import check_image_deadline, consume_image_model_calls
from capicove.processing.image.io import load_image
from capicove.processing.image.models import DerivedImage, ImageSource
from capicove.processing.image.ports import BackgroundRemovalProvider


@dataclass(frozen=True, slots=True)
class BackgroundRemovalOptions:
    """Configure background removal and optional alpha matting."""

    alpha_matting: bool = False
    foreground_threshold: int = 240
    background_threshold: int = 10
    erode_size: int = 10
    background_color: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("foreground_threshold", self.foreground_threshold),
            ("background_threshold", self.background_threshold),
        ):
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be within [0, 255]")
        if self.erode_size < 0:
            raise ValueError("erode_size must be non-negative")
        if self.background_color is not None and (
            len(self.background_color) != 4
            or any(channel < 0 or channel > 255 for channel in self.background_color)
        ):
            raise ValueError("background_color must contain four channels within [0, 255]")


class ImageRemoveBackgroundMapper:
    """Remove backgrounds and write new content-addressed PNG artifacts."""

    def __init__(self, provider: BackgroundRemovalProvider, output_directory: Path) -> None:
        self._provider = provider
        self._output_directory = Path(output_directory)

    def map(
        self,
        sources: tuple[ImageSource, ...],
        options: BackgroundRemovalOptions | None = None,
        *,
        output_directory_fd: int | None = None,
        authorized_output_directory: Path | None = None,
    ) -> tuple[DerivedImage, ...]:
        """Map ordered sources while reusing duplicate source content."""
        if options is None:
            options = BackgroundRemovalOptions()
        owns_descriptor = output_directory_fd is None
        if output_directory_fd is None:
            self._output_directory.mkdir(parents=True, exist_ok=True)
            output_directory_fd = os.open(
                self._output_directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        try:
            processed: dict[str, DerivedImage] = {}
            results: list[DerivedImage] = []
            for source in sources:
                check_image_deadline()
                loaded = load_image(source)
                cached = processed.get(loaded.sha256)
                if cached is None:
                    consume_image_model_calls()
                    output = self._provider.remove(loaded.image.copy(), options)
                    check_image_deadline()
                    _require_directory_identity(
                        output_directory_fd,
                        authorized_output_directory or self._output_directory,
                    )
                    cached = self._persist(
                        output,
                        loaded.sha256,
                        options,
                        output_directory_fd,
                        authorized_output_directory or self._output_directory,
                    )
                    processed[loaded.sha256] = cached
                results.append(cached)
            return tuple(results)
        finally:
            if owns_descriptor:
                os.close(output_directory_fd)

    @property
    def output_directory(self) -> Path:
        """Return the configured artifact root for pre-execution authorization."""
        return self._output_directory

    def _persist(
        self,
        image: object,
        source_hash: str,
        options: BackgroundRemovalOptions,
        output_directory_fd: int,
        output_directory: Path,
    ) -> DerivedImage:
        if not hasattr(image, "save") or not hasattr(image, "size"):
            raise TypeError("background removal provider must return a Pillow image")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        check_image_deadline()
        digest = hashlib.sha256(content).hexdigest()
        destination_name = f"{digest}.png"
        destination = output_directory / destination_name
        temporary_name = f".capicove-background-{uuid.uuid4().hex}.png"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=output_directory_fd,
        )
        try:
            _write_all(descriptor, content)
        finally:
            os.close(descriptor)
        published = False
        try:
            check_image_deadline()
            try:
                os.link(
                    temporary_name,
                    destination_name,
                    src_dir_fd=output_directory_fd,
                    dst_dir_fd=output_directory_fd,
                    follow_symlinks=False,
                )
                published = True
            except FileExistsError:
                existing_digest = _digest_regular_file_at(
                    output_directory_fd,
                    destination_name,
                )
                if existing_digest != digest:
                    raise RuntimeError(
                        "existing content-addressed artifact does not match its digest"
                    ) from None
            _require_directory_identity(output_directory_fd, output_directory)
        except Exception:
            if published:
                os.unlink(destination_name, dir_fd=output_directory_fd)
            raise
        finally:
            os.unlink(temporary_name, dir_fd=output_directory_fd)
        width, height = image.size
        return DerivedImage(
            path=str(destination),
            sha256=digest,
            media_type="image/png",
            width=int(width),
            height=int(height),
            byte_size=len(content),
            source_hashes=(source_hash,),
            operation="image_remove_background_mapper",
            provider_id=self._provider.provider_id,
            provider_version=self._provider.provider_version,
            operation_version=1,
            parameters=(
                ("alpha_matting", options.alpha_matting),
                ("background_color", repr(options.background_color)),
                ("background_threshold", options.background_threshold),
                ("erode_size", options.erode_size),
                ("foreground_threshold", options.foreground_threshold),
            ),
            model_id=getattr(self._provider, "model_id", None),
            model_revision=getattr(self._provider, "model_revision", None),
            model_manifest_sha256=getattr(self._provider, "model_manifest_sha256", None),
            preprocess_revision=getattr(self._provider, "preprocess_revision", None),
            device=getattr(self._provider, "device", None),
        )


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("unable to publish background artifact")
        view = view[written:]


def _require_directory_identity(descriptor: int, expected_path: Path) -> None:
    opened = os.fstat(descriptor)
    try:
        current = os.stat(expected_path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("background output directory changed during inference") from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        raise RuntimeError("background output directory changed during inference")


def _digest_regular_file_at(directory_fd: int, name: str) -> str:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("existing content-addressed artifact is not a regular file")
        digest = hashlib.sha256()
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)
