"""Lazy rembg adapter for local background removal."""

from __future__ import annotations

import importlib

from PIL import Image

from capicove.processing.image.mappers.remove_background import BackgroundRemovalOptions


class RembgUnavailableError(RuntimeError):
    """Raised when the optional rembg provider is unavailable."""


class RembgBackgroundRemovalProvider:
    """Remove backgrounds with an explicitly provisioned local rembg session.

    The adapter never creates a rembg session because rembg downloads its default
    model when the local cache is missing. Session provisioning therefore belongs
    to an explicit application composition step outside this network-disabled
    capability.
    """

    provider_id = "rembg"
    provider_version: str | None = None

    def __init__(
        self,
        *,
        session: object,
        model_id: str,
        model_revision: str | None = None,
        preprocess_revision: str = "rembg-remove-v1",
        device: str = "session-managed",
    ) -> None:
        if session is None:
            raise ValueError("a preprovisioned local rembg session is required")
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        if model_revision is not None and not model_revision.strip():
            raise ValueError("model_revision must not be blank")
        self._session = session
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_manifest_sha256: str | None = None
        if not preprocess_revision.strip():
            raise ValueError("preprocess_revision must not be blank")
        if not device.strip():
            raise ValueError("device must not be blank")
        self.preprocess_revision = preprocess_revision
        self.device = device

    def remove(self, image: Image.Image, options: BackgroundRemovalOptions) -> Image.Image:
        """Remove one background using explicitly configured local rembg assets."""
        try:
            rembg = importlib.import_module("rembg")
        except ModuleNotFoundError as error:
            raise RembgUnavailableError(
                "rembg is unavailable; install capicove[image-rembg]"
            ) from error
        output = rembg.remove(
            image,
            alpha_matting=options.alpha_matting,
            alpha_matting_foreground_threshold=options.foreground_threshold,
            alpha_matting_background_threshold=options.background_threshold,
            alpha_matting_erode_size=options.erode_size,
            bgcolor=options.background_color,
            session=self._session,
        )
        if not isinstance(output, Image.Image):
            raise TypeError("rembg returned an unsupported output type")
        version = getattr(rembg, "__version__", None)
        if isinstance(version, str) and version.strip():
            self.provider_version = version
        return output
