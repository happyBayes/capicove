"""R3.01 artifact upload, resume, quota and authorized retrieval on a real store."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.remote import ArtifactRef, AuthorityKey, PrincipalKey
from capicove.runtime.remote_artifacts import ArtifactError, ArtifactService


CHUNK = 256 * 1024


def _authority(subject: str = "alice", application: str = "app-a") -> AuthorityKey:
    return AuthorityKey(
        PrincipalKey("https://issuer.test", "tenant", application, "human", subject),
        "workspace-1",
        "grant-1",
        "https://mcp.test/mcp",
    )


def _service(root: Path, **options: Any) -> ArtifactService:
    return ArtifactService(root, **options)


def _upload(
    service: ArtifactService, authority: AuthorityKey, data: bytes, key: str = "k1"
) -> ArtifactRef:
    status = service.begin(
        authority,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="application/octet-stream",
        idempotency_key=key,
    )
    for index in range(0, max(1, -(-len(data) // CHUNK))):
        service.write_chunk(
            authority, status.upload_id, index, data[index * CHUNK : (index + 1) * CHUNK]
        )
    return service.commit(authority, status.upload_id)


def test_upload_restart_resume_commit_read_matches_an_independent_hash(tmp_path: Path) -> None:
    alice = _authority()
    data = os.urandom(CHUNK * 2 + 1234)
    first = _service(tmp_path / "store")
    status = first.begin(
        alice,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="application/octet-stream",
        idempotency_key="resume",
    )
    first.write_chunk(alice, status.upload_id, 0, data[:CHUNK])
    resumed = _service(tmp_path / "store")
    again = resumed.begin(
        alice,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="application/octet-stream",
        idempotency_key="resume",
    )
    assert again.upload_id == status.upload_id
    assert [chunk[0] for chunk in resumed.status(alice, status.upload_id).chunks] == [0]
    resumed.write_chunk(alice, status.upload_id, 1, data[CHUNK : 2 * CHUNK])
    resumed.write_chunk(alice, status.upload_id, 2, data[2 * CHUNK :])
    reference = resumed.commit(alice, status.upload_id)
    assert resumed.commit(alice, status.upload_id) == reference
    assert reference.sha256 == hashlib.sha256(data).hexdigest() and reference.size == len(data)
    restarted = _service(tmp_path / "store")
    read = b"".join(
        restarted.read_chunk(alice, reference.artifact_id, offset, CHUNK)
        for offset in range(0, len(data), CHUNK)
    )
    assert hashlib.sha256(read).digest() == hashlib.sha256(data).digest()
    assert str(tmp_path) not in repr(reference)
    assert restarted.usage(alice) == {"reserved": 0, "stored": len(data)}


def test_wrong_subject_hash_size_and_chunk_conflicts_fail(tmp_path: Path) -> None:
    service = _service(tmp_path / "store")
    alice, bob = _authority(), _authority("bob")
    data = b"exact payload"
    status = service.begin(
        alice,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        media_type="text/plain",
        idempotency_key="k",
    )
    with pytest.raises(ArtifactError) as error:
        service.write_chunk(bob, status.upload_id, 0, data)
    assert error.value.code == "not_found"
    service.write_chunk(alice, status.upload_id, 0, data)
    service.write_chunk(alice, status.upload_id, 0, data)
    for index, chunk, code in ((0, b"changed bytes", "conflict"), (1, b"x", "invalid_input")):
        with pytest.raises(ArtifactError) as error:
            service.write_chunk(alice, status.upload_id, index, chunk)
        assert error.value.code == code
    with pytest.raises(ArtifactError) as error:
        service.begin(
            alice,
            size=len(data) + 1,
            sha256=hashlib.sha256(data).hexdigest(),
            media_type="text/plain",
            idempotency_key="k",
        )
    assert error.value.code == "conflict"
    wrong = service.begin(
        alice, size=len(data), sha256="0" * 64, media_type="text/plain", idempotency_key="w"
    )
    service.write_chunk(alice, wrong.upload_id, 0, data)
    with pytest.raises(ArtifactError) as error:
        service.commit(alice, wrong.upload_id)
    assert error.value.code == "integrity_mismatch"
    reference = service.commit(alice, status.upload_id)
    for caller in (bob, _authority(application="app-b")):
        with pytest.raises(ArtifactError) as error:
            service.stat(caller, reference.artifact_id)
        assert error.value.code == "not_found"
    with pytest.raises(ArtifactError) as error:
        service.stat(bob, "artifact:" + "0" * 32)
    assert error.value.code == "not_found"
    # Same bytes uploaded by another user are an independent artifact.
    other = _upload(service, bob, data)
    assert other.artifact_id != reference.artifact_id


def test_overquota_misleading_types_and_unsafe_inputs_fail_before_writes(tmp_path: Path) -> None:
    service = _service(tmp_path / "store", max_artifact_bytes=1024, application_quota_bytes=1500)
    alice = _authority()
    _upload(service, alice, b"a" * 1000, key="first")
    for size, code in ((1025, "too_large"), (600, "quota_exceeded")):
        with pytest.raises(ArtifactError) as error:
            service.begin(
                alice,
                size=size,
                sha256="0" * 64,
                media_type="text/plain",
                idempotency_key=f"s{size}",
            )
        assert error.value.code == code
    # Another subject of the same application shares the aggregate application quota.
    with pytest.raises(ArtifactError) as error:
        service.begin(
            _authority("bob"),
            size=600,
            sha256="0" * 64,
            media_type="text/plain",
            idempotency_key="bob",
        )
    assert error.value.code == "quota_exceeded"
    for media in ("../etc/passwd", "text/plain; charset=../x", ""):
        with pytest.raises(ArtifactError):
            service.begin(
                alice, size=4, sha256="0" * 64, media_type=media, idempotency_key="m" + media
            )
    fake_png = b"not really a png"
    status = service.begin(
        alice,
        size=len(fake_png),
        sha256=hashlib.sha256(fake_png).hexdigest(),
        media_type="image/png",
        idempotency_key="png",
    )
    service.write_chunk(alice, status.upload_id, 0, fake_png)
    with pytest.raises(ArtifactError) as error:
        service.commit(alice, status.upload_id)
    assert error.value.code == "media_mismatch"
    assert service.usage(alice)["stored"] == 1000


def test_crash_retries_do_not_double_reserve_or_publish_partial_data(tmp_path: Path) -> None:
    alice = _authority()
    data = b"crash-safe payload" * 100
    digest = hashlib.sha256(data).hexdigest()

    class Crash(BaseException):
        pass

    def barrier(name: str) -> None:
        if name == "after_publish_file":
            raise Crash

    crashing = _service(tmp_path / "store", barrier=barrier)
    status = crashing.begin(
        alice,
        size=len(data),
        sha256=digest,
        media_type="application/octet-stream",
        idempotency_key="crash",
    )
    crashing.write_chunk(alice, status.upload_id, 0, data)
    with pytest.raises(Crash):
        crashing.commit(alice, status.upload_id)
    recovered = _service(tmp_path / "store")
    assert recovered.list_artifacts(alice) == ()
    again = recovered.begin(
        alice,
        size=len(data),
        sha256=digest,
        media_type="application/octet-stream",
        idempotency_key="crash",
    )
    assert again.upload_id == status.upload_id
    assert recovered.usage(alice) == {"reserved": len(data), "stored": 0}
    reference = recovered.commit(alice, status.upload_id)
    assert recovered.usage(alice) == {"reserved": 0, "stored": len(data)}
    assert recovered.list_artifacts(alice) == (reference,)
    with pytest.raises(ArtifactError) as error:
        recovered.abort(alice, status.upload_id)
    assert error.value.code == "conflict"
    aborted = recovered.begin(
        alice, size=4, sha256="0" * 64, media_type="text/plain", idempotency_key="abort"
    )
    recovered.abort(alice, aborted.upload_id)
    recovered.abort(alice, aborted.upload_id)
    assert recovered.usage(alice) == {"reserved": 0, "stored": len(data)}


def test_abort_racing_commit_settles_exactly_one_outcome(tmp_path: Path) -> None:
    alice = _authority()
    data = os.urandom(CHUNK + 7)
    for attempt in range(8):
        service = _service(tmp_path / f"store-{attempt}")
        status = service.begin(
            alice,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            media_type="application/octet-stream",
            idempotency_key="race",
        )
        service.write_chunk(alice, status.upload_id, 0, data[:CHUNK])
        service.write_chunk(alice, status.upload_id, 1, data[CHUNK:])
        outcomes: list[str] = []

        def run(
            action: str,
            service: ArtifactService = service,
            outcomes: list[str] = outcomes,
            upload_id: str = status.upload_id,
        ) -> None:
            try:
                getattr(service, action)(alice, upload_id)
                outcomes.append(action)
            except ArtifactError as error:
                outcomes.append(error.code)

        threads = [threading.Thread(target=run, args=(name,)) for name in ("commit", "abort")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        usage = service.usage(alice)
        if "commit" in outcomes:
            assert sorted(outcomes) == ["commit", "conflict"]
            assert usage == {"reserved": 0, "stored": len(data)}
        else:
            assert sorted(outcomes) == ["abort", "conflict"]
            assert usage == {"reserved": 0, "stored": 0}
            assert service.list_artifacts(alice) == ()


def test_revoked_and_deleted_artifacts_stay_denied_after_restart(tmp_path: Path) -> None:
    alice = _authority()
    revoked: set[AuthorityKey] = set()

    def authorize(authority: AuthorityKey) -> None:
        if authority in revoked:
            raise PermissionError("revoked")

    service = _service(tmp_path / "store", authorize=authorize)
    reference = _upload(service, alice, b"secret bytes")
    revoked.add(alice)
    with pytest.raises(ArtifactError) as error:
        service.read_chunk(alice, reference.artifact_id, 0, 10)
    assert error.value.code == "permission_denied"
    revoked.clear()
    service.delete(alice, reference.artifact_id)
    content = tmp_path / "store" / "artifacts"
    assert not any(path.is_file() for path in content.rglob("*"))
    restarted = _service(tmp_path / "store", authorize=authorize)
    calls: tuple[Callable[[], object], ...] = (
        lambda: restarted.stat(alice, reference.artifact_id),
        lambda: restarted.read_chunk(alice, reference.artifact_id, 0, 10),
    )
    for call in calls:
        with pytest.raises(ArtifactError) as error:
            call()
        assert error.value.code == "not_found"
    assert restarted.usage(alice) == {"reserved": 0, "stored": 0}


def test_store_files_are_private_and_symlinked_roots_are_refused(tmp_path: Path) -> None:
    service = _service(tmp_path / "store")
    reference = _upload(service, _authority(), b"private")
    for path in (tmp_path / "store").rglob("*"):
        assert path.stat().st_mode & 0o077 == 0, path
    assert reference.artifact_id.startswith("artifact:")
    (tmp_path / "real").mkdir(mode=0o700)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises((ArtifactError, ValueError)):
        _service(tmp_path / "link")
