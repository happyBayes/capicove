"""Tests for trusted local preview-qualification receipts."""

from __future__ import annotations

import json
import os
import secrets
import stat
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.retrieval._adapters.sqlite_image_generations import (
    GenerationQualificationError,
    PreviewActivationQualification,
    PublishedGenerationRecord,
    SqliteGenerationCatalog,
)
from capicove.runtime import retrieval_qualification
from capicove.runtime.retrieval_qualification import LocalPreviewQualificationStore


KEY = b"local-preview-qualification-test-key-0001"
OTHER_KEY = b"local-preview-qualification-test-key-0002"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _published(
    tmp_path: Path,
    *,
    activation_set_id: str = "set-a",
    manifest_sha256: str = SHA_A,
    profile_ref: str = "personal-lite",
) -> PublishedGenerationRecord:
    root = tmp_path / "published" / activation_set_id
    root.mkdir(parents=True, exist_ok=True)
    return PublishedGenerationRecord(
        activation_set_id=activation_set_id,
        manifest_sha256=manifest_sha256,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="personal-photos",
        retrieval_profile_ref=profile_ref,
        published_at=datetime.now(UTC) - timedelta(seconds=5),
    )


def _store(
    tmp_path: Path,
    *,
    key: bytes = KEY,
    limit: int = 16 * 1024,
) -> LocalPreviewQualificationStore:
    return LocalPreviewQualificationStore(
        tmp_path / "qualifications",
        hmac_key=key,
        maximum_evidence_bytes=limit,
    )


def _issue(
    store: LocalPreviewQualificationStore,
    published: PublishedGenerationRecord,
    *,
    deadline: datetime,
) -> PreviewActivationQualification:
    return store.issue(
        published,
        report_sha256=SHA_C,
        policy_sha256=SHA_D,
        verification_receipt_sha256=SHA_E,
        verified_at=datetime.now(UTC),
        deadline=deadline,
    )


def _receipt_path(
    store: LocalPreviewQualificationStore,
    qualification: PreviewActivationQualification,
) -> Path:
    return store.trusted_root / f"{qualification.qualification_id}.json"


def _canonical(document: object) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _resign(
    store: LocalPreviewQualificationStore,
    document: dict[str, object],
) -> bytes:
    signed = {name: document[name] for name in retrieval_qualification._SIGNED_FIELDS}
    document["hmac_sha256"] = store._sign(signed)
    return _canonical(document)


def test_issue_writes_private_canonical_evidence_and_verify_binds_all_fields(
    tmp_path: Path,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)

    qualification = _issue(store, published, deadline=future)
    verified = store.verify(qualification, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    raw = receipt.read_bytes()
    document = json.loads(raw)

    assert raw == _canonical(document)
    assert stat.S_IMODE(store.trusted_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert verified.qualification_id == qualification.qualification_id
    assert verified.activation_set_id == "set-a"
    assert verified.manifest_sha256 == SHA_A
    assert verified.profile_ref == "personal-lite"
    assert verified.report_sha256 == SHA_C
    assert verified.policy_sha256 == SHA_D
    assert verified.verification_receipt_sha256 == SHA_E
    assert verified.verified_at.tzinfo is not None


def test_issue_is_only_evidence_and_does_not_activate_publication(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    catalog = SqliteGenerationCatalog.open(tmp_path / "lifecycle.sqlite3", verifier=store)
    catalog.register_published(published, deadline=future)

    _issue(store, published, deadline=future)

    assert catalog.resolve_active("workspace-1", "personal-lite", deadline=future) is None


def test_tampered_signed_field_is_rejected(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    document = json.loads(receipt.read_bytes())
    document["report_sha256"] = SHA_B
    receipt.write_bytes(_canonical(document))
    receipt.chmod(0o600)

    with pytest.raises(GenerationQualificationError, match="signature"):
        store.verify(qualification, published, deadline=future)


def test_noncanonical_or_wrong_schema_evidence_is_rejected(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    document = json.loads(receipt.read_bytes())
    receipt.write_text(json.dumps(document, indent=2), encoding="utf-8")
    receipt.chmod(0o600)

    with pytest.raises(GenerationQualificationError, match="canonical"):
        store.verify(qualification, published, deadline=future)

    document["schema_version"] = 2
    receipt.write_bytes(_canonical(document))
    with pytest.raises(GenerationQualificationError, match="schema"):
        store.verify(qualification, published, deadline=future)


@pytest.mark.parametrize(
    "different_publication",
    [
        {"activation_set_id": "set-b"},
        {"manifest_sha256": SHA_B},
        {"profile_ref": "personal-lite-v2"},
    ],
)
def test_receipt_is_bound_to_exact_publication(
    tmp_path: Path,
    different_publication: dict[str, str],
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    changed = _published(tmp_path, **different_publication)

    with pytest.raises(GenerationQualificationError, match="publication"):
        store.verify(qualification, changed, deadline=future)


def test_wrong_startup_key_cannot_verify_existing_evidence(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    qualification = _issue(_store(tmp_path), published, deadline=future)
    wrong_key_store = _store(tmp_path, key=OTHER_KEY)

    with pytest.raises(GenerationQualificationError, match="signature"):
        wrong_key_store.verify(qualification, published, deadline=future)


def test_receipt_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    outside = tmp_path / "outside.json"
    outside.write_bytes(receipt.read_bytes())
    outside.chmod(0o600)
    receipt.unlink()
    receipt.symlink_to(outside)

    with pytest.raises(GenerationQualificationError, match="safely read"):
        store.verify(qualification, published, deadline=future)


def test_oversize_and_non_owner_only_receipts_fail_closed(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path, limit=1_024)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    receipt.write_bytes(receipt.read_bytes() + b"x" * 1_024)

    with pytest.raises(GenerationQualificationError, match="size limit"):
        store.verify(qualification, published, deadline=future)

    receipt.write_bytes(b"{}\n")
    receipt.chmod(0o644)
    with pytest.raises(GenerationQualificationError, match="private regular file"):
        store.verify(qualification, published, deadline=future)


def test_expired_deadlines_fail_before_issue_or_verification(tmp_path: Path) -> None:
    published = _published(tmp_path)
    store = _store(tmp_path)
    past = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(GenerationQualificationError, match="deadline"):
        _issue(store, published, deadline=past)

    future = datetime.now(UTC) + timedelta(minutes=1)
    qualification = _issue(store, published, deadline=future)
    with pytest.raises(GenerationQualificationError, match="deadline"):
        store.verify(qualification, published, deadline=past)


def test_forged_locator_and_future_or_prepublication_evidence_are_rejected(
    tmp_path: Path,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=2)
    published = _published(tmp_path)
    store = _store(tmp_path)

    with pytest.raises(GenerationQualificationError, match="locator"):
        store.verify(
            PreviewActivationQualification("../../request-verdict"),
            published,
            deadline=future,
        )
    with pytest.raises(GenerationQualificationError, match="future"):
        store.issue(
            published,
            report_sha256=SHA_C,
            policy_sha256=SHA_D,
            verification_receipt_sha256=SHA_E,
            verified_at=datetime.now(UTC) + timedelta(minutes=1),
            deadline=future,
        )
    with pytest.raises(GenerationQualificationError, match="predates"):
        store.issue(
            published,
            report_sha256=SHA_C,
            policy_sha256=SHA_D,
            verification_receipt_sha256=SHA_E,
            verified_at=published.published_at - timedelta(seconds=1),
            deadline=future,
        )


def test_changed_root_permissions_are_rejected(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    os.chmod(store.trusted_root, 0o755)

    with pytest.raises(GenerationQualificationError, match="owner-only"):
        store.verify(qualification, published, deadline=future)


def test_publication_dataclass_rejects_request_controlled_profile_replacement(
    tmp_path: Path,
) -> None:
    """The locator alone cannot replace the trusted publication at verification."""
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    request_claim = replace(published, retrieval_profile_ref="request-claimed-profile")

    with pytest.raises(GenerationQualificationError, match="publication"):
        store.verify(qualification, request_claim, deadline=future)


def test_store_constructor_rejects_unsafe_roots_keys_and_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        LocalPreviewQualificationStore(Path("relative"), hmac_key=KEY)
    with pytest.raises(ValueError, match="32 bytes"):
        LocalPreviewQualificationStore(tmp_path / "short-key", hmac_key=b"short")
    with pytest.raises(ValueError, match="32 bytes"):
        LocalPreviewQualificationStore(
            tmp_path / "wrong-key-type",
            hmac_key="x" * 32,  # type: ignore[arg-type]
        )
    for invalid_limit in (True, 1_023, 1024 * 1024 + 1):
        with pytest.raises(ValueError, match="maximum_evidence_bytes"):
            LocalPreviewQualificationStore(
                tmp_path / f"limit-{invalid_limit}",
                hmac_key=KEY,
                maximum_evidence_bytes=invalid_limit,
            )

    regular_file = tmp_path / "regular-file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="initialize"):
        LocalPreviewQualificationStore(regular_file, hmac_key=KEY)

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        LocalPreviewQualificationStore(symlink, hmac_key=KEY)


def test_issue_and_verify_reject_wrong_types_hashes_and_naive_times(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)

    with pytest.raises(TypeError, match="PublishedGenerationRecord"):
        store.issue(
            object(),  # type: ignore[arg-type]
            report_sha256=SHA_C,
            policy_sha256=SHA_D,
            verification_receipt_sha256=SHA_E,
            verified_at=datetime.now(UTC),
            deadline=future,
        )
    with pytest.raises(GenerationQualificationError, match="report_sha256"):
        store.issue(
            published,
            report_sha256="C" * 64,
            policy_sha256=SHA_D,
            verification_receipt_sha256=SHA_E,
            verified_at=datetime.now(UTC),
            deadline=future,
        )
    with pytest.raises(GenerationQualificationError, match="timezone-aware"):
        store.issue(
            published,
            report_sha256=SHA_C,
            policy_sha256=SHA_D,
            verification_receipt_sha256=SHA_E,
            verified_at=datetime.now(),
            deadline=future,
        )
    with pytest.raises(GenerationQualificationError, match="deadline"):
        store.issue(
            published,
            report_sha256=SHA_C,
            policy_sha256=SHA_D,
            verification_receipt_sha256=SHA_E,
            verified_at=datetime.now(UTC),
            deadline=datetime.now(),
        )

    qualification = _issue(store, published, deadline=future)
    with pytest.raises(TypeError, match="PreviewActivationQualification"):
        store.verify(object(), published, deadline=future)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PublishedGenerationRecord"):
        store.verify(qualification, object(), deadline=future)  # type: ignore[arg-type]


def test_issue_rejects_evidence_larger_than_configured_bound(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path, profile_ref="profile-" + "x" * 2_000)
    store = _store(tmp_path, limit=1_024)

    with pytest.raises(GenerationQualificationError, match="size limit"):
        _issue(store, published, deadline=future)
    assert tuple(store.trusted_root.iterdir()) == ()


def test_valid_signature_with_different_issuer_key_id_is_rejected(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    document = json.loads(receipt.read_bytes())
    document["issuer_key_sha256"] = SHA_B
    receipt.write_bytes(_resign(store, document))

    with pytest.raises(GenerationQualificationError, match="issuer key"):
        store.verify(qualification, published, deadline=future)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\n", "UTF-8 JSON"),
        (b"not-json\n", "UTF-8 JSON"),
        (_canonical([]), "JSON object"),
    ],
)
def test_malformed_receipt_documents_are_rejected(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    receipt.write_bytes(payload)
    receipt.chmod(0o600)

    with pytest.raises(GenerationQualificationError, match=message):
        store.verify(qualification, published, deadline=future)


def test_exact_schema_rejects_missing_or_extra_fields_before_signature(
    tmp_path: Path,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    original = json.loads(receipt.read_bytes())

    for mutation in ("missing", "extra"):
        document = dict(original)
        if mutation == "missing":
            document.pop("policy_sha256")
        else:
            document["request_verdict"] = "approved"
        receipt.write_bytes(_canonical(document))
        with pytest.raises(GenerationQualificationError, match="schema fields"):
            store.verify(qualification, published, deadline=future)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qualification_id", " ", "qualification_id is invalid"),
        ("activation_set_id", None, "activation_set_id is invalid"),
        ("profile_ref", "x" * 513, "profile_ref is invalid"),
        ("report_sha256", "C" * 64, "lowercase SHA-256"),
        ("verified_at", "not-a-date", "verified_at is invalid"),
        ("verified_at", "2026-08-27T12:00:00", "timezone-aware"),
        ("verified_at", "2026-08-27T12:00:00Z", "canonical UTC"),
    ],
)
def test_resigned_invalid_field_values_still_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    document = json.loads(receipt.read_bytes())
    document[field] = value
    receipt.write_bytes(_resign(store, document))

    with pytest.raises(GenerationQualificationError, match=message):
        store.verify(qualification, published, deadline=future)


def test_receipt_hardlink_and_directory_are_rejected(tmp_path: Path) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)

    hardlink_store = _store(tmp_path / "hardlink")
    qualification = _issue(hardlink_store, published, deadline=future)
    receipt = _receipt_path(hardlink_store, qualification)
    os.link(receipt, tmp_path / "second-link.json")
    with pytest.raises(GenerationQualificationError, match="private regular file"):
        hardlink_store.verify(qualification, published, deadline=future)

    directory_store = _store(tmp_path / "directory")
    qualification = _issue(directory_store, published, deadline=future)
    receipt = _receipt_path(directory_store, qualification)
    receipt.unlink()
    receipt.mkdir(mode=0o700)
    with pytest.raises(GenerationQualificationError, match="private regular file"):
        directory_store.verify(qualification, published, deadline=future)


def test_missing_platform_no_follow_support_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0)

    with pytest.raises(GenerationQualificationError, match="O_NOFOLLOW"):
        store.verify(qualification, published, deadline=future)


def test_duplicate_qualification_id_preserves_first_receipt_and_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    monkeypatch.setattr(secrets, "token_hex", lambda _count: "1" * 32)

    first = _issue(store, published, deadline=future)
    first_payload = _receipt_path(store, first).read_bytes()
    with pytest.raises(GenerationQualificationError, match="atomically publish"):
        _issue(store, published, deadline=future)

    assert _receipt_path(store, first).read_bytes() == first_payload
    assert tuple(path.name for path in store.trusted_root.iterdir()) == (
        f"{first.qualification_id}.json",
    )


def test_receipt_short_write_and_link_failure_are_closed_and_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    short_store = _store(tmp_path / "short")
    monkeypatch.setattr(os, "write", lambda *_args: 0)

    with pytest.raises(GenerationQualificationError, match="short write"):
        _issue(short_store, published, deadline=future)
    assert tuple(short_store.trusted_root.iterdir()) == ()

    monkeypatch.undo()
    link_store = _store(tmp_path / "link")

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("synthetic link denial")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(GenerationQualificationError, match="atomically publish"):
        _issue(link_store, published, deadline=future)
    assert tuple(link_store.trusted_root.iterdir()) == ()


@pytest.mark.parametrize("mutation", ["grow", "touch"])
def test_receipt_mutation_during_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    receipt = _receipt_path(store, qualification)
    original_read = os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            if mutation == "grow":
                with receipt.open("ab") as stream:
                    stream.write(b"x")
            else:
                metadata = receipt.stat()
                os.utime(
                    receipt,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
                )
        return chunk

    monkeypatch.setattr(os, "read", mutate_after_read)
    with pytest.raises(
        GenerationQualificationError,
        match=r"changed while read|changed while being",
    ):
        store.verify(qualification, published, deadline=future)


def test_small_read_chunks_still_verify_complete_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = datetime.now(UTC) + timedelta(minutes=1)
    published = _published(tmp_path)
    store = _store(tmp_path)
    qualification = _issue(store, published, deadline=future)
    monkeypatch.setattr(retrieval_qualification, "_READ_CHUNK_BYTES", 16)

    assert store.verify(qualification, published, deadline=future).qualification_id == (
        qualification.qualification_id
    )
