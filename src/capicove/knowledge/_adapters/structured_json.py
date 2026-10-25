"""Offline extraction of explicit structured assertions, with no inference or network."""

import json
from collections.abc import Callable
from datetime import datetime

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import MemoryEpisode, TimeInterval, TypedLiteral
from capicove.knowledge.models import EntityMention, ExtractedFact


class StructuredJsonExtractor:
    extractor_id = "structured-json"
    revision = "1"

    def extract(
        self, episode: MemoryEpisode, read_text: Callable[[int, int], str]
    ) -> tuple[ExtractedFact, ...]:
        if episode.episode_type != "json":
            raise ValueError("structured extractor requires a json Episode")
        payload = json.loads(read_text(0, episode.content_length))
        if not isinstance(payload, dict) or set(payload) - {"facts", "text"}:
            raise ValueError("invalid structured document")
        facts = payload.get("facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 100:
            raise ValueError("invalid structured facts")
        result = []
        for fact in facts:
            if not isinstance(fact, dict) or set(fact) - {
                "subject_name",
                "subject_type",
                "predicate",
                "object",
                "valid_from",
                "valid_to",
            }:
                raise ValueError("invalid structured fact")
            obj = fact["object"]
            if not isinstance(obj, dict):
                raise ValueError("invalid structured object")
            value: EntityMention | TypedLiteral
            if obj.get("kind") == "entity":
                if set(obj) != {"kind", "name", "entity_type"}:
                    raise ValueError("invalid entity mention")
                value = EntityMention(obj["name"], obj["entity_type"])
            else:
                if set(obj) != {"kind", "value"}:
                    raise ValueError("invalid literal")
                value = TypedLiteral(obj["kind"], obj["value"])
            result.append(
                ExtractedFact(
                    EntityMention(fact["subject_name"], fact["subject_type"]),
                    fact["predicate"],
                    value,
                    TimeInterval(
                        *(
                            None if fact.get(key) is None else datetime.fromisoformat(fact[key])
                            for key in ("valid_from", "valid_to")
                        )
                    ),
                    (EvidenceSpan(episode.evidence_refs[0], 0, episode.content_length),),
                )
            )
        return tuple(result)
