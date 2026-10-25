# Architecture overview

Capicove is a capability-first Python modular monolith. Canonical contracts define
stable data and Tool boundaries; capability packages own their workflows, ports,
and adapters; composition roots connect those pieces to CLI and MCP transports.

```text
CLI / MCP / Python
        |
canonical Tool and dataset contracts
        |
video / image / audio / governed processing
        |
explicit local adapters, optional models, and execution plugins
```

Source evidence is immutable. Derived media, metrics, transcripts, indexes, and
other projections retain hashes and lineage back to their inputs.

Execution plugins depend on the public processing ports. The core validates and
hashes logical plans without importing a backend; an explicitly selected backend
owns runtime execution and publishes only verified snapshots.
