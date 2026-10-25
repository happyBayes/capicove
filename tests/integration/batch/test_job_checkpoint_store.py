"""Job checkpoint files must not alias unrelated local files."""

import os
from pathlib import Path

import pytest

from capicove.batch._adapters.sqlite import SqliteBatchStore
from capicove.batch.job_checkpoint_store import JobCheckpointStores


@pytest.mark.skipif(os.name != "posix", reason="POSIX hard-link ownership case")
def test_existing_checkpoint_hard_link_is_rejected_before_sqlite_open(tmp_path: Path) -> None:
    cache_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    checkpoint_root = tmp_path / "checkpoints-jobs"
    checkpoint_root.mkdir(mode=0o700)
    target = tmp_path / "private-target"
    target.write_bytes(b"private target bytes")
    target.chmod(0o600)
    os.link(target, checkpoint_root / ("a" * 24 + ".sqlite3"))

    with pytest.raises(ValueError, match="private regular file"):
        JobCheckpointStores(cache_store).for_job("job:" + "a" * 24)

    assert target.read_bytes() == b"private target bytes"
