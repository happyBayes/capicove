"""Bounded lexical planning for plain English questions, never query-language evaluation."""

from __future__ import annotations

import re
import unicodedata


_QUESTION_STARTS = frozenset(
    "what which who whom whose when where why how do does did is are was were "
    "can could would should tell".split()
)
_FUNCTION_WORDS = frozenset(
    "a an the and or to of in on at is are was were be been being what which who "
    "whom whose when where why how do does did has have had i my me we our us "
    "you your he his him she her they their them it its for with from by about "
    "as that this these those tell can could would should please s t".split()
)


def question_terms(text: str) -> tuple[str, ...] | None:
    """Return content terms, or None to preserve the existing literal/keyword path.

    Symbols such as percent, underscore and plus remain literal. An identified
    question with no content terms returns an empty tuple, never a browse query.
    No dates, identities or access filters are inferred from question text.
    """
    # Validate before normalization: symbols such as the Kelvin sign fold to letters.
    for character in text:
        if character.isascii():
            if re.fullmatch(r"[A-Za-z0-9\s?'\".,!():;-]", character) is None:
                return None
        elif not (
            (character.isalpha() and unicodedata.name(character, "").startswith("LATIN "))
            or unicodedata.category(character) == "Mn"
        ):
            return None
    # Normalize only the planning copy; evidence and literal queries stay exact.
    text = unicodedata.normalize("NFC", text)
    if any(unicodedata.category(character) == "Mn" for character in text):
        return None
    words = re.findall(r"[^\W_]+", text.lower())
    if not words or not (
        text.rstrip().endswith("?") or (len(words) > 1 and words[0] in _QUESTION_STARTS)
    ):
        return None
    return tuple(dict.fromkeys(word for word in words if word not in _FUNCTION_WORDS))[:64]
