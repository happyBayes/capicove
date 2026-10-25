"""Integration tests for verified local dataset publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.contracts.data import DatasetSchema, SchemaField
from capicove.processing._adapters.filesystem import (
    DatasetPublicationError,
    LocalDatasetPublisher,
    PublicationRequest,
)


def _request(staging: Path) -> PublicationRequest:
    return PublicationRequest(
        run_id="run:one",
        dataset_id="dataset:curated-traces",
        version=1,
        staging_path=staging,
        schema=DatasetSchema((SchemaField("episode_id", "string", False),)),
        processing_fingerprint="c" * 64,
        parent_snapshot_ids=("snapshot:" + "d" * 64,),
    )


def test_publication_hashes_parts_and_atomically_commits(tmp_path: Path) -> None:
    publisher = LocalDatasetPublisher((tmp_path / "datasets").resolve())
    staging = publisher.stage("run:one")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")

    snapshot = publisher.publish(_request(staging), validator=lambda _: True)

    committed = Path(snapshot.uri.removeprefix("file://"))
    assert committed == (tmp_path / "datasets" / "curated-traces" / "v1").resolve()
    assert (committed / "manifest.json").is_file()
    assert (committed / ".capicove-pending.json").is_file()
    assert snapshot.parent_snapshot_ids == ("snapshot:" + "d" * 64,)
    assert not staging.exists()

    publisher.finalize("run:one", snapshot)

    assert not (committed / ".capicove-pending.json").exists()


def test_failed_publication_does_not_create_committed_directory(tmp_path: Path) -> None:
    publisher = LocalDatasetPublisher((tmp_path / "datasets").resolve())
    staging = publisher.stage("run:one")
    (staging / "part-00000.parquet").write_bytes(b"not parquet")

    with pytest.raises(DatasetPublicationError, match="validator"):
        publisher.publish(_request(staging), validator=lambda _: False)

    assert not (tmp_path / "datasets" / "curated-traces" / "v1").exists()


def test_finalize_rejects_a_changed_manifest(tmp_path: Path) -> None:
    publisher = LocalDatasetPublisher((tmp_path / "datasets").resolve())
    staging = publisher.stage("run:one")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = publisher.publish(_request(staging), validator=lambda _: True)
    committed = Path(snapshot.uri.removeprefix("file://"))
    (committed / "manifest.json").write_bytes(b"changed\n")

    with pytest.raises(DatasetPublicationError, match="manifest"):
        publisher.finalize("run:one", snapshot)

    assert (committed / ".capicove-pending.json").is_file()


def test_finalize_rejects_a_different_run_identity(tmp_path: Path) -> None:
    publisher = LocalDatasetPublisher((tmp_path / "datasets").resolve())
    staging = publisher.stage("run:one")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = publisher.publish(_request(staging), validator=lambda _: True)

    with pytest.raises(DatasetPublicationError, match="identity"):
        publisher.finalize("run:different", snapshot)


def test_retract_removes_only_the_exact_published_snapshot(tmp_path: Path) -> None:
    publisher = LocalDatasetPublisher((tmp_path / "datasets").resolve())
    staging = publisher.stage("run:one")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = publisher.publish(_request(staging), validator=lambda _: True)

    publisher.retract(snapshot)

    assert not Path(snapshot.uri.removeprefix("file://")).exists()


def test_publication_rejects_symlinked_dataset_destination(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    root.mkdir()
    (root / "curated-traces").symlink_to(outside, target_is_directory=True)
    publisher = LocalDatasetPublisher(root)
    staging = publisher.stage("run:one")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")

    with pytest.raises(DatasetPublicationError, match="symbolic link"):
        publisher.publish(_request(staging), validator=lambda _: True)

    assert not (outside / "v1").exists()


def test_publication_rejects_symlinked_part(tmp_path: Path) -> None:
    publisher = LocalDatasetPublisher((tmp_path / "datasets").resolve())
    staging = publisher.stage("run:one")
    target = tmp_path / "outside.parquet"
    target.write_bytes(b"PAR1-authored-test-PAR1")
    (staging / "part-00000.parquet").symlink_to(target)

    with pytest.raises(DatasetPublicationError, match="regular"):
        publisher.publish(_request(staging), validator=lambda _: True)


def test_publisher_rejects_symlinked_staging_root(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    (root / ".staging").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DatasetPublicationError, match="staging"):
        LocalDatasetPublisher(root)
