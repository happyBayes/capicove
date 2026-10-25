"""Provider-neutral rrf-v1 scoring shared by text and memory retrieval."""

from collections.abc import Iterable


def normalized_rrf(
    lanes: Iterable[tuple[float, tuple[tuple[str, int], ...]]], *, k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    maximum = 0.0
    for weight, ranks in lanes:
        if weight <= 0 or not ranks:
            continue
        maximum += weight / (k + 1)
        seen: set[str] = set()
        for identity, rank in ranks:
            if identity in seen or rank < 1:
                raise ValueError("invalid ranked lane")
            seen.add(identity)
            scores[identity] = scores.get(identity, 0.0) + weight / (k + rank)
    return (
        {identity: min(1.0, max(0.0, score / maximum)) for identity, score in scores.items()}
        if maximum
        else {}
    )
