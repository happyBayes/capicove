"""Export frozen synthetic cases or score independently reviewed observations."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tests.wiki_maintenance_support.scenarios import (
    Observation,
    evaluate,
    fixture_digest,
    scenarios,
)


def main() -> int:
    """Run offline only; never choose or contact a generation provider implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.observations is None:
        result = {
            "fixture_sha256": fixture_digest(),
            "scenarios": [asdict(case) for case in scenarios()],
            "qualification": "not_run",
        }
    else:
        if arguments.observations.stat().st_size > 1048576:
            raise ValueError("observations exceed 1 MiB")
        records = json.loads(arguments.observations.read_text())
        if not isinstance(records, list) or len(records) > 40:
            raise ValueError("expected at most 40 held-out observations")
        observations = tuple(
            Observation(**{**record, "affected_pages": tuple(record["affected_pages"])})
            for record in records
        )
        result = evaluate(observations)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
