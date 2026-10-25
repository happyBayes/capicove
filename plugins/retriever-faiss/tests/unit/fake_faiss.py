"""Small persistence-capable FAISS double for plugin contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


class FakeHnsw:
    """Expose the native HNSW settings used by the adapter."""

    def __init__(self, m: int) -> None:
        self.m = m
        self.efConstruction = 40
        self.efSearch = 16

    def nb_neighbors(self, level: int) -> int:
        return self.m * 2 if level == 0 else self.m


class FakeIndex:
    """Implement the narrow IndexHNSWFlat surface consumed by the plugin."""

    def __init__(self, dimension: int, m: int, metric_type: int) -> None:
        self.d = dimension
        self.metric_type = metric_type
        self.hnsw = FakeHnsw(m)
        self._vectors = np.empty((0, dimension), dtype=np.float32)
        self.search_override: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = np.concatenate((self._vectors, np.asarray(vectors, dtype=np.float32)))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.search_override is not None:
            return self.search_override
        scores = np.asarray(queries, dtype=np.float32) @ self._vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        return np.take_along_axis(scores, order, axis=1), order.astype(np.int64)

    def reconstruct_n(self, start: int, count: int) -> np.ndarray:
        return self._vectors[start : start + count].copy()


class FakeFaiss:
    """Persist fake indexes without importing or requiring native FAISS."""

    METRIC_INNER_PRODUCT = 0

    def __init__(self) -> None:
        self.thread_calls: list[int] = []
        self.last_created: FakeIndex | None = None
        self.fail_write = False
        self.fail_search = False
        self.search_override: tuple[np.ndarray, np.ndarray] | None = None

    def IndexHNSWFlat(self, dimension: int, m: int, metric_type: int) -> FakeIndex:
        index = FakeIndex(dimension, m, metric_type)
        self.last_created = index
        return index

    def omp_set_num_threads(self, threads: int) -> None:
        self.thread_calls.append(threads)

    def write_index(self, index: FakeIndex, path: str) -> None:
        if self.fail_write:
            raise RuntimeError("native token that must not escape")
        with Path(path).open("wb") as output:
            self._write_payload(output, index)

    def read_index(self, path: str) -> FakeIndex:
        with np.load(path, allow_pickle=False) as payload:
            dimension = int(payload["dimension"][0])
            m = int(payload["m"][0])
            metric_type = int(payload["metric_type"][0])
            index = FakeIndex(dimension, m, metric_type)
            index.hnsw.efConstruction = int(payload["ef_construction"][0])
            index.hnsw.efSearch = int(payload["ef_search"][0])
            index.add(np.asarray(payload["vectors"], dtype=np.float32))
        if self.fail_search:
            index.search = self._failed_search  # type: ignore[method-assign]
        index.search_override = self.search_override
        return index

    @staticmethod
    def _write_payload(output: BinaryIO, index: FakeIndex) -> None:
        np.savez(
            output,
            dimension=np.asarray([index.d], dtype=np.int64),
            m=np.asarray([index.hnsw.m], dtype=np.int64),
            metric_type=np.asarray([index.metric_type], dtype=np.int64),
            ef_construction=np.asarray([index.hnsw.efConstruction], dtype=np.int64),
            ef_search=np.asarray([index.hnsw.efSearch], dtype=np.int64),
            vectors=index._vectors,
        )

    @staticmethod
    def _failed_search(queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        del queries, top_k
        raise RuntimeError("native token that must not escape")
