"""Tests for authorized, same-object reverse-image query resolution."""

from __future__ import annotations

import hashlib
import io
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    ArtifactSource,
    ImageQuery,
    InlineBytesSource,
    LocalPathSource,
    MountSource,
    SearchFailureCode,
)
from capicove.contracts.tools import PrivacyClass
from capicove.retrieval.search_models import InvocationAuthorizationContext, SearchFailure
from capicove.runtime.image_query import (
    AuthorizedImageQueryResolver,
    ResolvedImageArtifact,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)


def _png_bytes(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="PNG")
    return output.getvalue()


def _authorization(**changes: object) -> InvocationAuthorizationContext:
    values: dict[str, object] = {
        "request_id": "request-1",
        "workspace_id": "workspace-1",
        "actor_id": "actor-1",
        "environment_id": "environment-1",
        "operation": "capicove.search.image",
        "authorization_scope_ref": "scope-1",
        "permissions": frozenset({"retrieval:search"}),
        "allowed_data_policy_refs": frozenset({"local-only"}),
        "allowed_source_root_refs": frozenset({"mount:course-media"}),
        "privacy_class": PrivacyClass.CONFIDENTIAL,
    }
    values.update(changes)
    return InvocationAuthorizationContext(**values)  # type: ignore[arg-type]


class FakeArtifactResolver:
    def __init__(self, artifact: ResolvedImageArtifact | None = None) -> None:
        self.artifact = artifact
        self.calls: list[str] = []

    def resolve(
        self,
        artifact_ref: str,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> ResolvedImageArtifact:
        assert authorization.actor_id == "actor-1"
        assert deadline is FUTURE
        self.calls.append(artifact_ref)
        if self.artifact is None:
            raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
        return self.artifact


def _resolver(
    root: Path, artifact: FakeArtifactResolver | None = None, **kwargs: object
) -> AuthorizedImageQueryResolver:
    return AuthorizedImageQueryResolver(
        workspace_id="workspace-1",
        artifact_resolver=artifact or FakeArtifactResolver(),
        local_roots_by_policy={"local-only": {"mount:course-media": root}},
        **kwargs,  # type: ignore[arg-type]
    )


def test_mount_query_is_authorized_read_once_and_hashed(tmp_path: Path) -> None:
    payload = _png_bytes("navy")
    (tmp_path / "query.png").write_bytes(payload)
    calls = 0

    from capicove.processing.image.preparation import prepare_image

    def recording_preparer(data: bytes, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return prepare_image(data, **kwargs)  # type: ignore[arg-type]

    resolved = _resolver(tmp_path, preparer=recording_preparer).resolve(
        ImageQuery(MountSource("course-media", "query.png")),
        authorization=_authorization(),
        data_policy_ref="local-only",
        deadline=FUTURE,
    )

    assert resolved.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert resolved.image.mode == "RGB"
    assert resolved.image.size == (8, 6)
    assert calls == 1


def test_authorized_absolute_local_path_uses_same_policy_root(tmp_path: Path) -> None:
    path = tmp_path / "query.png"
    path.write_bytes(_png_bytes())

    result = _resolver(tmp_path).resolve(
        ImageQuery(LocalPathSource(str(path))),
        authorization=_authorization(),
        data_policy_ref="local-only",
        deadline=FUTURE,
    )

    assert result.encoded_byte_size == path.stat().st_size


def test_inline_query_allows_an_empty_source_root_authorization(tmp_path: Path) -> None:
    result = _resolver(tmp_path).resolve(
        ImageQuery(InlineBytesSource(_png_bytes(), "image/png")),
        authorization=_authorization(allowed_source_root_refs=frozenset()),
        data_policy_ref="local-only",
        deadline=FUTURE,
    )

    assert result.media_type == "image/png"


def test_local_path_requires_the_specific_authorized_root_ref(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    denied_root = tmp_path / "denied"
    allowed_root.mkdir()
    denied_root.mkdir()
    allowed_path = allowed_root / "query.png"
    denied_path = denied_root / "query.png"
    allowed_path.write_bytes(_png_bytes("green"))
    denied_path.write_bytes(_png_bytes("red"))
    resolver = AuthorizedImageQueryResolver(
        workspace_id="workspace-1",
        artifact_resolver=FakeArtifactResolver(),
        local_roots_by_policy={
            "local-only": {
                "local:allowed": allowed_root,
                "local:denied": denied_root,
            }
        },
    )
    authorization = _authorization(allowed_source_root_refs=frozenset({"local:allowed"}))

    allowed = resolver.resolve(
        ImageQuery(LocalPathSource(str(allowed_path))),
        authorization=authorization,
        data_policy_ref="local-only",
        deadline=FUTURE,
    )
    with pytest.raises(SearchFailure) as raised:
        resolver.resolve(
            ImageQuery(LocalPathSource(str(denied_path))),
            authorization=authorization,
            data_policy_ref="local-only",
            deadline=FUTURE,
        )

    assert allowed.source_sha256 == hashlib.sha256(allowed_path.read_bytes()).hexdigest()
    assert raised.value.code is SearchFailureCode.SOURCE_NOT_AUTHORIZED


def test_mount_ref_cannot_widen_authority_within_an_allowed_policy(tmp_path: Path) -> None:
    mount_root = tmp_path / "course"
    mount_root.mkdir()
    (mount_root / "query.png").write_bytes(_png_bytes())
    resolver = AuthorizedImageQueryResolver(
        workspace_id="workspace-1",
        artifact_resolver=FakeArtifactResolver(),
        local_roots_by_policy={"local-only": {"mount:course-media": mount_root}},
    )

    with pytest.raises(SearchFailure) as raised:
        resolver.resolve(
            ImageQuery(MountSource("course-media", "query.png")),
            authorization=_authorization(allowed_source_root_refs=frozenset({"mount:other-media"})),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.SOURCE_NOT_AUTHORIZED


def test_source_root_ref_rejects_conflicting_identities_across_policies(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    with pytest.raises(ValueError, match="stable identity"):
        AuthorizedImageQueryResolver(
            workspace_id="workspace-1",
            artifact_resolver=FakeArtifactResolver(),
            local_roots_by_policy={
                "policy-a": {"local:shared": first},
                "policy-b": {"local:shared": second},
            },
        )


def test_source_root_ref_may_reuse_the_same_canonical_identity_across_policies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    alias = tmp_path / "shared-alias"
    alias.symlink_to(root, target_is_directory=True)
    path = root / "query.png"
    path.write_bytes(_png_bytes())
    resolver = AuthorizedImageQueryResolver(
        workspace_id="workspace-1",
        artifact_resolver=FakeArtifactResolver(),
        local_roots_by_policy={
            "policy-a": {"local:shared": root},
            "policy-b": {"local:shared": alias},
        },
    )

    result = resolver.resolve(
        ImageQuery(LocalPathSource(str(path))),
        authorization=_authorization(
            allowed_data_policy_refs=frozenset({"policy-a", "policy-b"}),
            allowed_source_root_refs=frozenset({"local:shared"}),
        ),
        data_policy_ref="policy-b",
        deadline=FUTURE,
    )

    assert result.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_hash_mismatch_fails_as_source_changed(tmp_path: Path) -> None:
    payload = _png_bytes()
    artifact = FakeArtifactResolver(ResolvedImageArtifact(payload, "0" * 64, "image/png"))

    with pytest.raises(SearchFailure) as raised:
        _resolver(tmp_path, artifact).resolve(
            ImageQuery(ArtifactSource("artifact-1")),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.SOURCE_CHANGED
    assert artifact.calls == ["artifact-1"]


def test_inline_limit_accepts_boundary_and_rejects_one_more(tmp_path: Path) -> None:
    png = _png_bytes()
    boundary = png + bytes(4096 - len(png))
    resolver = _resolver(tmp_path, maximum_inline_bytes=4096)

    result = resolver.resolve(
        ImageQuery(InlineBytesSource(boundary, "image/png")),
        authorization=_authorization(),
        data_policy_ref="local-only",
        deadline=FUTURE,
    )
    assert result.encoded_byte_size == 4096

    with pytest.raises(SearchFailure) as raised:
        resolver.resolve(
            ImageQuery(InlineBytesSource(boundary + b"x", "image/png")),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )
    assert raised.value.code is SearchFailureCode.INPUT_TOO_LARGE


@pytest.mark.parametrize(
    "source",
    [
        MountSource("course-media", "../query.png"),
        MountSource("course-media", "/query.png"),
        MountSource("course-media", "nested\\query.png"),
    ],
)
def test_mount_traversal_and_non_posix_paths_fail_closed(
    tmp_path: Path, source: MountSource
) -> None:
    with pytest.raises(SearchFailure) as raised:
        _resolver(tmp_path).resolve(
            ImageQuery(source),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )
    assert raised.value.code is SearchFailureCode.SOURCE_NOT_AUTHORIZED


def test_symlink_and_non_regular_file_fail_before_read(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside.write_bytes(_png_bytes())
    (tmp_path / "link.png").symlink_to(outside)

    for path in (tmp_path / "link.png", tmp_path):
        with pytest.raises(SearchFailure) as raised:
            _resolver(tmp_path).resolve(
                ImageQuery(LocalPathSource(str(path))),
                authorization=_authorization(),
                data_policy_ref="local-only",
                deadline=FUTURE,
            )
        assert raised.value.code is SearchFailureCode.SOURCE_NOT_AUTHORIZED


def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "query.fifo"
    os.mkfifo(fifo)

    with pytest.raises(SearchFailure) as raised:
        _resolver(tmp_path).resolve(
            ImageQuery(LocalPathSource(str(fifo))),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )
    assert raised.value.code is SearchFailureCode.SOURCE_NOT_AUTHORIZED


@pytest.mark.parametrize(
    "authorization",
    [
        _authorization(workspace_id="other-workspace"),
        _authorization(permissions=frozenset()),
        _authorization(operation="retrieval:search"),
        _authorization(operation="retrieval:admin"),
        _authorization(allowed_data_policy_refs=frozenset({"other-policy"})),
    ],
)
def test_untrusted_invocation_cannot_grant_source_authority(
    tmp_path: Path,
    authorization: InvocationAuthorizationContext,
) -> None:
    with pytest.raises(SearchFailure) as raised:
        _resolver(tmp_path).resolve(
            ImageQuery(InlineBytesSource(_png_bytes(), "image/png")),
            authorization=authorization,
            data_policy_ref="local-only",
            deadline=FUTURE,
        )
    assert raised.value.code in {
        SearchFailureCode.SOURCE_NOT_AUTHORIZED,
        SearchFailureCode.PERMISSION_DENIED,
    }


def test_invalid_media_and_expired_deadline_are_distinct(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)

    with pytest.raises(SearchFailure) as invalid:
        resolver.resolve(
            ImageQuery(InlineBytesSource(b"not-an-image", "image/png")),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )
    assert invalid.value.code is SearchFailureCode.INVALID_MEDIA

    with pytest.raises(SearchFailure) as expired:
        resolver.resolve(
            ImageQuery(InlineBytesSource(_png_bytes(), "image/png")),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )
    assert expired.value.code is SearchFailureCode.DEADLINE_EXCEEDED


def test_declared_media_type_must_match_decoded_format(tmp_path: Path) -> None:
    with pytest.raises(SearchFailure) as raised:
        _resolver(tmp_path).resolve(
            ImageQuery(InlineBytesSource(_png_bytes(), "image/jpeg")),
            authorization=_authorization(),
            data_policy_ref="local-only",
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.INVALID_MEDIA
