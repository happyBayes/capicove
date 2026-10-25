# Local Wiki API

The public Wiki package exposes immutable result values, stable failures, and one
`WikiService` for an offline, file-backed workspace. SQLite and PyYAML remain lazy
implementation details, and the Wiki never accesses Temporal Memory private state.

::: capicove.wiki
