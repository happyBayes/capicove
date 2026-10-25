"""Tests for lazy Ray perceptual deduplication coordination."""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import capicove.processing.image.invocation as image_invocation
from capicove.processing.image.deduplication import (
    ImagePerceptualDeduplicator,
    ImageRecord,
    RayImagePerceptualDeduplicator,
)
from capicove.processing.image.invocation import (
    ImageDeadlineExceeded,
    image_invocation_deadline,
)
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import authored_source, write_authored_image


class ImmediateRemote:
    """Execute one Ray-style remote function synchronously."""

    def __init__(self, function: Callable[..., object]) -> None:
        self._function = function

    def remote(self, *args: object) -> object:
        return self._function(*args)

    def options(self, **kwargs: object) -> ImmediateRemote:
        del kwargs
        return self


class ImmediateRay:
    """Provide the Ray calls used by the adapter without installing Ray."""

    def is_initialized(self) -> bool:
        return True

    def remote(self, function: Callable[..., object]) -> ImmediateRemote:
        return ImmediateRemote(function)

    def get(self, values: list[object], *, timeout: float | None = None) -> list[object]:
        del timeout
        return values

    def wait(
        self,
        values: list[object],
        *,
        num_returns: int,
        timeout: float | None = None,
    ) -> tuple[list[object], list[object]]:
        del num_returns, timeout
        return values, []


class WidthHasher:
    """Use decoded width to prove the same configured provider runs remotely."""

    provider_id = "width-test-hasher"
    provider_version: str | None = "1"
    hash_version = "width-v1"

    def phash(self, image: Image.Image) -> int:
        return image.width


class WorkerVersionHasher(WidthHasher):
    """Populate package lineage only inside the isolated worker copy."""

    provider_version: str | None = None

    def phash(self, image: Image.Image) -> int:
        self.provider_version = "worker-package-1"
        return super().phash(image)


class InspectingRemote(ImmediateRemote):
    """Capture the record payload crossing the simulated worker boundary."""

    def __init__(self, function: Callable[..., object], records: list[ImageRecord]) -> None:
        super().__init__(function)
        self._records = records

    def remote(self, *args: object) -> object:
        record = args[0]
        assert isinstance(record, ImageRecord)
        self._records.append(record)
        return super().remote(*args)


class InspectingRay(ImmediateRay):
    def __init__(self) -> None:
        self.records: list[ImageRecord] = []
        self.timeout: float | None = None
        self.options: dict[str, object] = {}

    def remote(self, function: Callable[..., object]) -> InspectingRemote:
        remote = InspectingRemote(function, self.records)

        def options(**kwargs: object) -> InspectingRemote:
            self.options = dict(kwargs)
            return remote

        remote.options = options  # type: ignore[method-assign]
        return remote

    def get(self, values: list[object], *, timeout: float | None = None) -> list[object]:
        self.timeout = timeout
        return values


def test_ray_perceptual_deduplicator_matches_local_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same = authored_source("landscape")
    different = authored_source("portrait")
    records = (
        ImageRecord("b", (same,)),
        ImageRecord("a", (same,)),
        ImageRecord("c", (different,)),
    )
    imported: list[str] = []

    def import_module(name: str) -> Any:
        imported.append(name)
        return ImmediateRay()

    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        import_module,
    )
    distributed = RayImagePerceptualDeduplicator()

    assert imported == []
    assert distributed.group(records) == ImagePerceptualDeduplicator().group(records)
    assert imported == ["ray"]


def test_ray_perceptual_deduplicator_materializes_paths_and_preserves_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = InspectingRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    records = (
        ImageRecord(
            "a",
            (
                ImageSource(
                    path=str(write_authored_image(tmp_path / "landscape.png")),
                ),
            ),
        ),
    )

    result = RayImagePerceptualDeduplicator(
        hasher=WidthHasher(),
        timeout_seconds=12.5,
        max_retries=2,
    ).group(records)

    assert runtime.records[0].images[0].path is None
    assert runtime.records[0].images[0].data is not None
    assert runtime.timeout == 12.5
    assert runtime.options == {"max_retries": 2}
    assert result.provider_id == "width-test-hasher"


def test_ray_perceptual_deduplicator_returns_worker_provider_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IsolatedRay(ImmediateRay):
        def remote(self, function: Callable[..., object]) -> ImmediateRemote:
            class IsolatedRemote(ImmediateRemote):
                def remote(self, *args: object) -> object:
                    return self._function(*(copy.deepcopy(argument) for argument in args))

            return IsolatedRemote(function)

    runtime = IsolatedRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    hasher = WorkerVersionHasher()

    result = RayImagePerceptualDeduplicator(hasher=hasher).group(
        (ImageRecord("a", (authored_source("landscape"),)),)
    )

    assert hasher.provider_version is None
    assert result.provider_version == "worker-package-1"


def test_ray_perceptual_deduplicator_enforces_invocation_wide_decoded_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ImmediateRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    monkeypatch.setattr(
        image_invocation,
        "_MAX_INVOCATION_DECODED_PIXELS",
        300_000,
    )
    records = (
        ImageRecord("a", (authored_source("landscape"),)),
        ImageRecord("b", (authored_source("landscape"),)),
    )

    with image_invocation_deadline(datetime.now(UTC) + timedelta(seconds=30)):
        with pytest.raises(ValueError, match="cumulative decoded-pixel"):
            RayImagePerceptualDeduplicator(hasher=WidthHasher()).group(records)


def test_ray_perceptual_deduplicator_cancels_tasks_after_wait_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRay(ImmediateRay):
        def __init__(self) -> None:
            self.cancelled: list[object] = []

        def get(self, values: list[object], *, timeout: float | None = None) -> list[object]:
            del values, timeout
            raise TimeoutError("deadline")

        def remote(self, function: Callable[..., object]) -> ImmediateRemote:
            del function

            class DeferredRemote(ImmediateRemote):
                def __init__(self) -> None:
                    pass

                def options(self, **kwargs: object) -> ImmediateRemote:
                    del kwargs
                    return self

                def remote(self, *args: object) -> object:
                    del args
                    return object()

            return DeferredRemote()

        def cancel(self, reference: object, *, force: bool = False) -> None:
            assert force is True
            self.cancelled.append(reference)

    runtime = FailingRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    records = (ImageRecord("a", (authored_source("landscape"),)),)

    with pytest.raises(TimeoutError, match="deadline"):
        RayImagePerceptualDeduplicator().group(records)

    assert len(runtime.cancelled) == 1


def test_ray_perceptual_deduplicator_cancels_submitted_tasks_after_submit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SubmitFailingRay(ImmediateRay):
        def __init__(self) -> None:
            self.cancelled: list[object] = []
            self.submissions = 0

        def remote(self, function: Callable[..., object]) -> ImmediateRemote:
            del function
            runtime = self

            class SubmitFailingRemote(ImmediateRemote):
                def __init__(self) -> None:
                    pass

                def options(self, **kwargs: object) -> ImmediateRemote:
                    del kwargs
                    return self

                def remote(self, *args: object) -> object:
                    del args
                    runtime.submissions += 1
                    if runtime.submissions == 2:
                        raise RuntimeError("submission failed")
                    return object()

            return SubmitFailingRemote()

        def cancel(self, reference: object, *, force: bool = False) -> None:
            assert force is True
            self.cancelled.append(reference)

    runtime = SubmitFailingRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    records = (
        ImageRecord("a", (authored_source("landscape"),)),
        ImageRecord("b", (authored_source("portrait"),)),
    )

    with pytest.raises(RuntimeError, match="submission failed"):
        RayImagePerceptualDeduplicator().group(records)

    assert len(runtime.cancelled) == 1


def test_ray_perceptual_deduplicator_propagates_deadline_and_stops_blocking_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Reference:
        def __init__(self) -> None:
            self.active = True

    class BlockingRay(ImmediateRay):
        def __init__(self) -> None:
            self.references: list[Reference] = []
            self.deadlines: list[datetime | None] = []
            self.forces: list[bool] = []

        def remote(self, function: Callable[..., object]) -> ImmediateRemote:
            del function
            runtime = self

            class BlockingRemote(ImmediateRemote):
                def __init__(self) -> None:
                    pass

                def options(self, **kwargs: object) -> ImmediateRemote:
                    del kwargs
                    return self

                def remote(self, *args: object) -> object:
                    runtime.deadlines.append(args[2] if len(args) > 2 else None)  # type: ignore[arg-type]
                    reference = Reference()
                    runtime.references.append(reference)
                    return reference

            return BlockingRemote()

        def get(self, values: list[object], *, timeout: float | None = None) -> list[object]:
            del values, timeout
            raise TimeoutError("deadline")

        def cancel(self, reference: object, *, force: bool = False) -> None:
            assert isinstance(reference, Reference)
            self.forces.append(force)
            reference.active = False

        def wait(
            self,
            values: list[object],
            *,
            num_returns: int,
            timeout: float | None = None,
        ) -> tuple[list[object], list[object]]:
            del num_returns, timeout
            ready = [value for value in values if not value.active]  # type: ignore[attr-defined]
            remaining = [value for value in values if value.active]  # type: ignore[attr-defined]
            return ready, remaining

    runtime = BlockingRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    deadline = datetime.now(UTC) + timedelta(seconds=1)
    records = (
        ImageRecord("a", (authored_source("landscape"),)),
        ImageRecord("b", (authored_source("portrait"),)),
    )

    with image_invocation_deadline(deadline):
        with pytest.raises(ImageDeadlineExceeded, match="deadline"):
            RayImagePerceptualDeduplicator().group(records)

    assert runtime.deadlines == [deadline, deadline]
    assert runtime.forces == [True, True]
    assert all(not reference.active for reference in runtime.references)


def test_ray_perceptual_deduplicator_attempts_every_cancel_when_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelFailingRay(ImmediateRay):
        def __init__(self) -> None:
            self.references = [object(), object()]
            self.cancel_attempts: list[object] = []

        def remote(self, function: Callable[..., object]) -> ImmediateRemote:
            del function
            runtime = self

            class DeferredRemote(ImmediateRemote):
                def __init__(self) -> None:
                    pass

                def options(self, **kwargs: object) -> ImmediateRemote:
                    del kwargs
                    return self

                def remote(self, *args: object) -> object:
                    del args
                    return runtime.references.pop(0)

            return DeferredRemote()

        def get(self, values: list[object], *, timeout: float | None = None) -> list[object]:
            del values, timeout
            raise TimeoutError("deadline")

        def cancel(self, reference: object, *, force: bool = False) -> None:
            assert force is True
            self.cancel_attempts.append(reference)
            if len(self.cancel_attempts) == 1:
                raise RuntimeError("first cancellation failed")

    runtime = CancelFailingRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    records = (
        ImageRecord("a", (authored_source("landscape"),)),
        ImageRecord("b", (authored_source("portrait"),)),
    )

    with pytest.raises(RuntimeError, match="cancellation"):
        RayImagePerceptualDeduplicator().group(records)

    assert len(runtime.cancel_attempts) == 2


def test_ray_perceptual_deduplicator_bounds_cancellation_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundedWaitRay(ImmediateRay):
        def __init__(self) -> None:
            self.wait_timeout: float | None = None

        def remote(self, function: Callable[..., object]) -> ImmediateRemote:
            del function

            class DeferredRemote(ImmediateRemote):
                def __init__(self) -> None:
                    pass

                def remote(self, *args: object) -> object:
                    del args
                    return object()

            return DeferredRemote()

        def get(self, values: list[object], *, timeout: float | None = None) -> list[object]:
            del values, timeout
            raise TimeoutError("deadline")

        def cancel(self, reference: object, *, force: bool = False) -> None:
            del reference
            assert force is True

        def wait(
            self,
            values: list[object],
            *,
            num_returns: int,
            timeout: float | None = None,
        ) -> tuple[list[object], list[object]]:
            del num_returns
            self.wait_timeout = timeout
            return values, []

    runtime = BoundedWaitRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    records = (ImageRecord("a", (authored_source("landscape"),)),)

    with pytest.raises(TimeoutError, match="deadline"):
        RayImagePerceptualDeduplicator().group(records)

    assert runtime.wait_timeout is not None
    assert 0 <= runtime.wait_timeout <= 5


def test_ray_perceptual_deduplicator_rejects_duplicate_ids_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = InspectingRay()
    monkeypatch.setattr(
        "capicove.processing.image._adapters.ray_dedup.importlib.import_module",
        lambda name: runtime if name == "ray" else None,
    )
    records = (
        ImageRecord("duplicate", (ImageSource(data=b"first"),)),
        ImageRecord("duplicate", (ImageSource(data=b"second"),)),
    )

    with pytest.raises(ValueError, match="unique"):
        RayImagePerceptualDeduplicator().group(records)

    assert runtime.records == []
