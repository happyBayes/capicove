"""Generator ports owned by Wiki, composed only by the host runtime."""

from typing import Protocol

from capicove.contracts.wiki_maintenance import AnswerTask, CompilationTask


class CompilationGenerator(Protocol):
    """Return untrusted wire bytes, never a review or publication grant."""

    def generate(self, task: CompilationTask) -> bytes:
        """Generate one bounded submission for host validation."""
        ...


class AnswerGenerator(Protocol):
    """Return untrusted answer bytes for a pinned read-only task."""

    def generate(self, task: AnswerTask) -> bytes:
        """Generate an answer without modifying editorial state."""
        ...
