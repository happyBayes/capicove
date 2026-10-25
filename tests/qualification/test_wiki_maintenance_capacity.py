"""Explicit local capacity measurements; semantic and OS support are separate."""

import hashlib
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import pytest

from capicove.wiki.maintenance.discovery import build_discovery, discover
from tests.unit.wiki.maintenance.test_capture import experimental_workspace


@pytest.mark.qualification
@pytest.mark.parametrize(("count", "size_mib"), [(1000, 20), (10000, 256)])
def test_capacity_profile(tmp_path: Path, count: int, size_mib: int) -> None:
    service = experimental_workspace(tmp_path.resolve())
    total = size_mib * 1024 * 1024
    quotient, remainder = divmod(total, count)
    for index in range(count):
        header = (
            f"---\nid: page:CAP{index:09d}\ntitle: specimen{index:09d}\nkind: concept\n"
            "created: 2026-10-23\nupdated: 2026-10-23\nsources: []\naliases: []\n---\n"
        ).encode()
        payload = header + b"x" * (quotient + (index < remainder) - len(header))
        (service.wiki / "pages" / f"{index:09d}.md").write_bytes(payload)
    files = tuple((service.wiki / "pages").glob("*.md"))
    assert len(files) == count
    assert sum(path.stat().st_size for path in files) == total
    pages = service._inventory()
    index_path = service.wiki / "_derived/maintenance.sqlite"
    started = time.perf_counter()
    build_discovery(index_path, pages)
    rebuild = time.perf_counter() - started
    samples = []
    cold = None
    for request in range(111):
        started = time.perf_counter()
        result = discover(index_path, pages, f"specimen{request % count:09d}")
        elapsed = time.perf_counter() - started
        assert result.candidates[0].page_id == f"page:CAP{request % count:09d}"
        if request == 0:
            cold = elapsed
        elif request > 10:
            samples.append(elapsed)
    assert len(samples) == 100
    p95 = sorted(samples)[94]
    report = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "page_count": count,
        "page_bytes": total,
        "warmups": 10,
        "measured": 100,
        "concurrency": 1,
        "cold_seconds": cold,
        "warm_p50_seconds": statistics.median(samples),
        "warm_p95_seconds": p95,
        "rebuild_seconds": rebuild,
        "index_bytes": index_path.stat().st_size,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "small_discovery_slo_passed": p95 <= 2 if count == 1000 else None,
        "durable_sha256": hashlib.sha256(
            b"".join(path.read_bytes() for path in sorted(files))
        ).hexdigest(),
        "limitations": [
            "synthetic repeated body",
            "discovery only; preview/apply SLO not established",
            "warm OS cache; not controlled cold cache",
            "RSS is process lifetime peak",
        ],
    }
    destination = os.environ.get("CAPICOVE_WIKI_QUALIFICATION_REPORT")
    if destination:
        Path(destination + f"-{count}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    if count == 1000:
        assert p95 <= 2, report
