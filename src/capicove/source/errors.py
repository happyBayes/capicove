"""Sanitized Evidence failures."""


class EvidenceUnavailable(RuntimeError):
    """Source content is inaccessible, erased, corrupt or out of bounds."""
