"""Tests for closed FAISS HNSW startup configuration."""

from pathlib import Path

import pytest
from capicove_retriever_faiss.config import FaissConfiguration


def test_configuration_accepts_only_absolute_bounded_values(tmp_path: Path) -> None:
    configuration = FaissConfiguration.from_mapping(
        {
            "index_root": str(tmp_path.resolve()),
            "threads": 2,
            "maximum_keyframes": 50,
            "maximum_segments": 75,
            "hnsw_m": 32,
            "ef_construction": 200,
            "ef_search": 64,
        }
    )

    assert configuration.index_root == tmp_path.resolve()
    assert configuration.threads == 2
    assert configuration.maximum_keyframes == 50
    assert configuration.maximum_segments == 75


def test_configuration_rejects_unknown_keys_relative_roots_and_missing_threads(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported FAISS configuration key"):
        FaissConfiguration.from_mapping(
            {"index_root": str(tmp_path.resolve()), "threads": 1, "gpu": True}
        )
    with pytest.raises(ValueError, match="absolute"):
        FaissConfiguration(index_root=Path("relative"), threads=1)
    with pytest.raises(ValueError, match="threads must be provided"):
        FaissConfiguration.from_mapping({"index_root": str(tmp_path.resolve())})


@pytest.mark.parametrize("field", ["maximum_keyframes", "maximum_segments"])
def test_configuration_rejects_capacity_above_one_million(tmp_path: Path, field: str) -> None:
    values: dict[str, object] = {
        "index_root": tmp_path.resolve(),
        "threads": 1,
        field: 1_000_001,
    }
    with pytest.raises(ValueError, match=field):
        FaissConfiguration.from_mapping(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hnsw_m", 1),
        ("hnsw_m", 129),
        ("ef_construction", 0),
        ("ef_construction", 4_097),
        ("ef_search", 0),
        ("ef_search", 4_097),
        ("threads", 0),
        ("threads", 257),
    ],
)
def test_configuration_rejects_values_outside_closed_bounds(
    tmp_path: Path, field: str, value: int
) -> None:
    values: dict[str, object] = {
        "index_root": tmp_path.resolve(),
        "threads": 1,
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        FaissConfiguration.from_mapping(values)
