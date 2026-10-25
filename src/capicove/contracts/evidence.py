"""Immutable content-addressed references; source owns payload authorization."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.content_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.content_sha256
        ):
            raise ValueError("invalid Evidence digest")

    @property
    def evidence_id(self) -> str:
        return "evidence:" + self.content_sha256


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """Half-open Unicode code-point offsets into the exact immutable source."""

    reference: EvidenceRef
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference, EvidenceRef)
            or type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise ValueError("invalid Evidence span")
