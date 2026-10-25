"""Distinguish repairable wire framing from non-repairable evidence failures."""


class StructuralOutputError(ValueError):
    """Reject malformed JSON or schema shape, without accepting its evidence."""
