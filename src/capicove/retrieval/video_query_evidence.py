"""Explicit, short-lived query snapshots for authorized score expansion."""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from capicove.contracts import FullHit, InvocationContext, SearchResult, SearchStatus
from capicove.retrieval.video_evidence import VideoEvidenceReader


class VideoQueryEvidenceReader:
    """Bind diagnostics to one result and owner, never a mutable latest-query cache.

    A runtime explicitly creates this reader from a FULL search result and routes
    the owner's evidence requests to that snapshot. Nothing is written to disk;
    raw queries, vectors and provider responses are not retained. Every read still
    checks the immutable source reader, so revocation/erasure takes effect.
    """

    def __init__(
        self, reader: VideoEvidenceReader, result: SearchResult, *, owner: InvocationContext
    ) -> None:
        if result.status is not SearchStatus.SUCCEEDED or len(result.hits) > 100:
            raise ValueError("a bounded successful result is required")
        hits: dict[str, FullHit] = {}
        for hit in result.hits:
            if not isinstance(hit, FullHit):
                raise ValueError("FULL search results are required for query evidence")
            if hit.evidence_id in hits:
                raise ValueError("ambiguous evidence ID in query result")
            values = (result.query_id, *hit.provider_ids, *hit.index_generation_ids)
            if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,511}", v) is None for v in values):
                raise ValueError("diagnostic identities must be opaque")
            if max(len(hit.lane_scores), len(hit.provider_ids), len(hit.index_generation_ids)) > 32:
                raise ValueError("diagnostic item capacity exceeded")
            hits[hit.evidence_id] = hit
        self._reader = reader
        self._result = result
        self._hits = hits
        self._owner = (owner.workspace_id, owner.actor_id, owner.environment_id)
        self._expires_at = datetime.now(UTC) + timedelta(minutes=5)

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]:
        """Return only requested diagnostics after current source authorization."""
        if (context.workspace_id, context.actor_id, context.environment_id) != self._owner:
            raise PermissionError("query evidence owner mismatch")
        if datetime.now(UTC) >= min(context.deadline, self._expires_at):
            raise TimeoutError("query evidence expired")
        if "provider_diagnostics" in sections and "evidence:diagnostics" not in context.permissions:
            raise PermissionError("diagnostics permission required")
        if not sections or len(set(sections)) != len(sections):
            raise ValueError("invalid evidence sections")
        ordinary = tuple(s for s in sections if s not in {"scores", "provider_diagnostics"})
        # Even a diagnostic-only request must reauthorize and rehash source evidence.
        source = self._reader.read(
            context=context, evidence_id=evidence_id, sections=ordinary or ("metadata",)
        )
        output = dict(source) if ordinary else {}
        hit = self._hits.get(evidence_id)
        if hit is None:
            raise LookupError("evidence absent from this query snapshot")
        if "scores" in sections:
            output["scores"] = {
                "query_id": self._result.query_id,
                "brief_score": hit.brief_score,
                "lanes": [asdict(score) for score in hit.lane_scores],
            }
        if "provider_diagnostics" in sections:
            output["provider_diagnostics"] = {
                "query_id": self._result.query_id,
                "provider_ids": list(hit.provider_ids),
                "index_generation_ids": list(hit.index_generation_ids),
            }
        if any(
            len(json.dumps(value, allow_nan=False).encode()) > 4096 for value in output.values()
        ):
            raise LookupError("query evidence section capacity exceeded")
        return output


__all__ = ["VideoQueryEvidenceReader"]
