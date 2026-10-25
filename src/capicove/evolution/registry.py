"""Local composition entry point for the SQLite-backed evolution registry."""

from capicove.evolution._adapters.sqlite import EvolutionRegistry


__all__ = ["EvolutionRegistry"]
