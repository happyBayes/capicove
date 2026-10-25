"""Deterministic binary LSH candidate search with exact verification."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.processing.image.invocation import check_image_deadline


_HASH_BITS = 64
_MAX_HASH = 2**_HASH_BITS
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class HashRecord:
    """Identify one ordered collection of compatible 64-bit hashes."""

    record_id: str
    hashes: tuple[int, ...]
    text_hash: str | None = None


@dataclass(frozen=True, slots=True)
class LshMatch:
    """Describe one exact Hamming-distance match returned through LSH."""

    record_id: str
    hashes: tuple[int, ...]
    distance: int


@dataclass(frozen=True, slots=True)
class LshSearchResult:
    """Return stable, exactly verified matches and search observability."""

    matches: tuple[LshMatch, ...]
    hash_version: str
    bit_width: int
    max_hamming_distance: int
    candidate_strategy: str
    candidate_count: int
    comparison_count: int


@dataclass(frozen=True, slots=True)
class CandidatePairResult:
    """Return exactly verified record pairs and candidate statistics."""

    pairs: tuple[tuple[str, str], ...]
    strategy: str
    candidate_count: int
    comparison_count: int


class ImageLshSearch:
    """Search invocation-local 64-bit hashes using deterministic LSH bands."""

    def search(
        self,
        query_hashes: tuple[int, ...],
        candidates: tuple[HashRecord, ...],
        *,
        hash_version: str,
        query_text_hash: str | None = None,
        max_hamming_distance: int = 0,
        top_k: int = 100,
    ) -> LshSearchResult:
        """Return exact Hamming neighbors in stable distance and ID order."""
        _validate_hashes(query_hashes, "query_hashes", allow_empty=False)
        _validate_non_blank(hash_version, "hash_version")
        _validate_text_hash(query_text_hash)
        _validate_options(max_hamming_distance, top_k)
        _validate_records(candidates)

        compatible = tuple(
            record
            for record in candidates
            if len(record.hashes) == len(query_hashes) and record.text_hash == query_text_hash
        )
        bit_width = len(query_hashes) * _HASH_BITS
        if max_hamming_distance >= bit_width:
            strategy = "exhaustive"
            selected = compatible
        else:
            strategy = "lsh"
            selected = _query_candidates(
                query_hashes,
                compatible,
                max_hamming_distance,
            )

        matches: list[LshMatch] = []
        for record in selected:
            check_image_deadline()
            distance = _distance(query_hashes, record.hashes)
            if distance <= max_hamming_distance:
                matches.append(LshMatch(record.record_id, record.hashes, distance))
        matches.sort(key=lambda match: (match.distance, match.record_id))
        return LshSearchResult(
            matches=tuple(matches[:top_k]),
            hash_version=hash_version,
            bit_width=bit_width,
            max_hamming_distance=max_hamming_distance,
            candidate_strategy=strategy,
            candidate_count=len(selected),
            comparison_count=len(selected),
        )


def candidate_pairs(
    records: tuple[HashRecord, ...],
    max_hamming_distance: int,
) -> CandidatePairResult:
    """Return all verified near-neighbor pairs without duplicate comparisons."""
    _validate_options(max_hamming_distance, 1)
    _validate_records(records)
    partitions: dict[tuple[int, str | None], list[HashRecord]] = {}
    for record in records:
        if record.hashes:
            partitions.setdefault((len(record.hashes), record.text_hash), []).append(record)

    candidate_ids: set[tuple[str, str]] = set()
    used_lsh = False
    for partition in partitions.values():
        check_image_deadline()
        ordered = tuple(sorted(partition, key=lambda record: record.record_id))
        bit_width = len(ordered[0].hashes) * _HASH_BITS
        if max_hamming_distance >= bit_width:
            candidate_ids.update(_all_pair_ids(ordered))
            continue
        used_lsh = True
        buckets: dict[tuple[int, int, int], list[str]] = {}
        for record in ordered:
            for key in _band_keys(record.hashes, max_hamming_distance):
                buckets.setdefault(key, []).append(record.record_id)
        for bucket_ids in buckets.values():
            for left_index, left_id in enumerate(bucket_ids):
                for right_id in bucket_ids[left_index + 1 :]:
                    candidate_ids.add((left_id, right_id))

    by_id = {record.record_id: record for record in records}
    verified: list[tuple[str, str]] = []
    for left_id, right_id in sorted(candidate_ids):
        check_image_deadline()
        if _distance(by_id[left_id].hashes, by_id[right_id].hashes) <= max_hamming_distance:
            verified.append((left_id, right_id))
    return CandidatePairResult(
        pairs=tuple(verified),
        strategy="lsh" if used_lsh else "exhaustive",
        candidate_count=len(candidate_ids),
        comparison_count=len(candidate_ids),
    )


def _query_candidates(
    query_hashes: tuple[int, ...],
    candidates: tuple[HashRecord, ...],
    radius: int,
) -> tuple[HashRecord, ...]:
    buckets: dict[tuple[int, int, int], list[HashRecord]] = {}
    for record in candidates:
        check_image_deadline()
        for key in _band_keys(record.hashes, radius):
            buckets.setdefault(key, []).append(record)
    selected: dict[str, HashRecord] = {}
    for key in _band_keys(query_hashes, radius):
        for record in buckets.get(key, ()):
            selected[record.record_id] = record
    return tuple(selected[record_id] for record_id in sorted(selected))


def _band_keys(hashes: tuple[int, ...], radius: int) -> tuple[tuple[int, int, int], ...]:
    bit_width = len(hashes) * _HASH_BITS
    band_count = radius + 1
    base_width, wider_bands = divmod(bit_width, band_count)
    packed = 0
    for value in hashes:
        packed = (packed << _HASH_BITS) | value

    keys: list[tuple[int, int, int]] = []
    remaining = bit_width
    for band_index in range(band_count):
        width = base_width + int(band_index < wider_bands)
        remaining -= width
        mask = (1 << width) - 1
        keys.append((band_index, width, (packed >> remaining) & mask))
    return tuple(keys)


def _all_pair_ids(records: tuple[HashRecord, ...]) -> set[tuple[str, str]]:
    return {
        (left.record_id, right.record_id)
        for left_index, left in enumerate(records)
        for right in records[left_index + 1 :]
    }


def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(
        (left_hash ^ right_hash).bit_count()
        for left_hash, right_hash in zip(left, right, strict=True)
    )


def _validate_records(records: tuple[HashRecord, ...]) -> None:
    record_ids: set[str] = set()
    for record in records:
        _validate_non_blank(record.record_id, "record_id")
        if record.record_id in record_ids:
            raise ValueError("record_id values must be unique")
        record_ids.add(record.record_id)
        _validate_hashes(record.hashes, "hashes", allow_empty=True)
        _validate_text_hash(record.text_hash)


def _validate_hashes(values: tuple[int, ...], name: str, *, allow_empty: bool) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= _MAX_HASH
        for value in values
    ):
        raise ValueError(f"{name} must contain unsigned 64-bit integers")


def _validate_text_hash(value: str | None) -> None:
    if value is None:
        return
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("text_hash must be a lowercase SHA-256 hexadecimal string")


def _validate_options(max_hamming_distance: int, top_k: int) -> None:
    if (
        not isinstance(max_hamming_distance, int)
        or isinstance(max_hamming_distance, bool)
        or max_hamming_distance < 0
        or max_hamming_distance > _HASH_BITS
    ):
        raise ValueError("max_hamming_distance must be within [0, 64]")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or top_k > 1_000:
        raise ValueError("top_k must be within [1, 1000]")


def _validate_non_blank(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
