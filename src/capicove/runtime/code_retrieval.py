"""Model-free local code search (RFC-0033 S1, ``local-code-lexical-v1``).

Configuration is ``capabilities.retrieval`` schema version 2. Repository discovery
uses an explicitly selected Git executable only to enumerate candidates; every
selected file passes the closed intake policy and is verified again at build.
Builds seal a ``capicove-local-hybrid-v1`` lexical generation (no vectors) and
activate it atomically. Search and Evidence read only that sealed generation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import subprocess  # nosec B404
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import (
    CharacterRange,
    CompactHit,
    SearchFailureCode,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
)
from capicove.retrieval.code_ast import (
    AST_POLICY_VERSION,
    ParseStatus,
    ast_chunks,
    ast_language,
    parse_source,
    parser_identity,
)
from capicove.retrieval.code_symbols import (
    CHUNKER_VERSION,
    CLASSIFIER_VERSION,
    DEFINITION_METHOD,
    PATH_NORMALIZATION,
    POLICY_VERSION,
    IntakeRejection,
    QueryKind,
    chunk_code,
    classify_query,
    extract_symbols,
    identifier_components,
    normalize_identifier,
    path_terms,
    screen_source,
)
from capicove.retrieval.snippets import preview_snippet
from capicove.runtime.mcp_config import WorkspaceHostConfiguration


if TYPE_CHECKING:
    from capicove.model.factory import ModelFactory


FORMAT_VERSION = "capicove-local-hybrid-v1"
FORMAT_VARIANT = "lexical-code-v1"
PROFILE_ID = "local-code-lexical-v1"
HYBRID_PROFILE_ID = "local-code-hybrid-v1"
HYBRID_POLICY = "code-hybrid-v1"
RELATIONS_POLICY = "code-relations-v1"
MAX_SEEDS = 5
MAX_EXPANSION = 20
MAX_SNIPPET_BUDGET = 16 * 1024
MAX_EVIDENCE_RELATIONS = 16
EXPANSION_KINDS = ("calls", "contains")
CONTEXT_FORMAT = "path-header-v1"
MAX_VECTOR_BYTES = 268_435_456
DENSE_BATCH_ROWS = 1_024
MAX_CANDIDATES = 10_000
MAX_SELECTED_BYTES = 64 * 1024 * 1024
MAX_GIT_OUTPUT = 8 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 30
IDENTIFIER_CANDIDATES = 1_000
LANE_CANDIDATES = 100
RRF_K = 60
MAX_SNIPPET_CHARACTERS = 2_048
MAX_EVIDENCE_DEFINITIONS = 16
_FIELDS = frozenset(
    {
        "schema_version",
        "source_roots",
        "index_root",
        "corpus_ref",
        "retrieval_profile_ref",
        "max_source_bytes",
        "model_config_path",
        "model_config_sha256",
        "model_binding",
        "max_units",
        "max_vector_bytes",
        "code_policy_ref",
    }
)
_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_GENERATION = re.compile(r"^cg-[0-9a-f]{64}$")
_EVIDENCE = re.compile(r"^cev-[0-9a-f]{64}$")
# Fixed field weights per query kind; request text cannot change them.
_WEIGHTS = {
    QueryKind.IDENTIFIER: {"identifier_terms": 1.0, "body": 1.0, "path": 0.5},
    QueryKind.PATH: {"path": 3.0, "body": 1.0, "identifier_terms": 0.5},
    QueryKind.NATURAL_LANGUAGE: {"body": 1.0, "identifier_terms": 0.5, "path": 0.5},
}
_POLICIES = frozenset({POLICY_VERSION, AST_POLICY_VERSION})
# Documentation and configuration formats are never AST candidates.
_NON_CODE_LANGUAGES = frozenset({"markdown", "rst", "toml", "yaml", "sql", "shell"})
# Dense weight per query kind; identifier classes still decide identifier queries.
_DENSE_WEIGHTS = {
    QueryKind.IDENTIFIER: 0.5,
    QueryKind.PATH: 0.5,
    QueryKind.NATURAL_LANGUAGE: 1.0,
}
_EXCLUSION_BUCKETS = {
    "unsupported_type": "binary_or_unsupported",
    "binary": "binary_or_unsupported",
}


class CodeRetrievalError(ValueError):
    """Report a stable, path-free operator failure."""


@dataclass(frozen=True, slots=True)
class CodeSettings:
    """Validated startup-owned code retrieval authority."""

    workspace_id: str
    host_profile: str
    source_roots: Mapping[str, Path]
    index_root: Path
    corpus_ref: str
    max_source_bytes: int
    max_units: int
    code_policy: str = POLICY_VERSION
    profile: str = PROFILE_ID
    model_config: Path | None = None
    model_config_sha256: str | None = None
    model_binding: str | None = None
    max_vector_bytes: int = 0

    @property
    def hybrid(self) -> bool:
        """Return whether this profile adds the dense lane."""
        return self.profile == HYBRID_PROFILE_ID


def is_code_configuration(config: WorkspaceHostConfiguration) -> bool:
    """Return whether ``capabilities.retrieval`` selects schema version 2."""
    raw = config.capabilities.get("retrieval")
    return raw is not None and raw.get("schema_version") == 2


def code_settings(config: WorkspaceHostConfiguration) -> CodeSettings:
    """Validate the closed schema-2 object for the lexical code profile."""
    if os.name != "posix":
        raise CodeRetrievalError("code retrieval requires POSIX no-follow file APIs")
    raw = config.capabilities.get("retrieval")
    if raw is None or set(raw) != _FIELDS:
        raise CodeRetrievalError(
            "capabilities.retrieval schema 2 requires exactly the documented fields"
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise CodeRetrievalError("capabilities.retrieval schema_version must be 2")
    profile, policy = raw["retrieval_profile_ref"], raw["code_policy_ref"]
    model_fields = ("model_config_path", "model_config_sha256", "model_binding")
    vector_budget = raw["max_vector_bytes"]
    model_config: Path | None = None
    model_sha256: str | None = None
    model_binding: str | None = None
    if profile == PROFILE_ID:
        if policy not in _POLICIES:
            raise CodeRetrievalError(
                "local-code-lexical-v1 requires code-symbols-v1 or code-ast-v1"
            )
        if any(raw[name] is not None for name in model_fields):
            raise CodeRetrievalError("the lexical code profile requires null model fields")
        if type(vector_budget) is not int or vector_budget != 0:
            raise CodeRetrievalError("the lexical code profile requires max_vector_bytes = 0")
    elif profile == HYBRID_PROFILE_ID:
        if policy not in {HYBRID_POLICY, RELATIONS_POLICY}:
            raise CodeRetrievalError(
                "local-code-hybrid-v1 requires code-hybrid-v1 or code-relations-v1"
            )
        if any(raw[name] is None for name in model_fields):
            raise CodeRetrievalError("the hybrid code profile requires all model fields")
        if type(vector_budget) is not int or not 1 <= vector_budget <= MAX_VECTOR_BYTES:
            raise CodeRetrievalError("max_vector_bytes must be an integer from 1 to 268435456")
        model_config = _absolute(raw["model_config_path"], "model_config_path")
        binding = raw["model_binding"]
        if not isinstance(binding, str) or re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", binding) is None:
            raise CodeRetrievalError("model_binding must name a configured binding")
        digest = raw["model_config_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise CodeRetrievalError("model_config_sha256 must be lowercase SHA-256 hex")
        model_sha256, model_binding = digest, binding
        try:
            if _file_sha256(model_config) != digest:
                raise CodeRetrievalError(
                    "model_config_sha256 does not match the configuration bytes"
                )
        except OSError:
            raise CodeRetrievalError("model_config_path is not a readable file") from None
    else:
        raise CodeRetrievalError("unsupported code retrieval profile")
    units = raw["max_units"]
    if type(units) is not int or not 1 <= units <= 10_000:
        raise CodeRetrievalError("max_units must be an integer from 1 to 10000")
    limit = raw["max_source_bytes"]
    if type(limit) is not int or not 1 <= limit <= 16 * 1024 * 1024:
        raise CodeRetrievalError("max_source_bytes must be an integer from 1 to 16 MiB")
    root = config.workspace_root
    index_root = _absolute(raw["index_root"], "index_root")
    if index_root == root or not index_root.is_relative_to(root):
        raise CodeRetrievalError("index_root must be inside workspace_root")
    sources = raw["source_roots"]
    if not isinstance(sources, Mapping) or not 1 <= len(sources) <= 16:
        raise CodeRetrievalError("source_roots must contain 1 to 16 aliases")
    roots: dict[str, Path] = {}
    for alias, value in sources.items():
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise CodeRetrievalError("source root aliases must be lowercase identifiers")
        path = _absolute(value, "source_roots")
        if path.resolve() != path or not path.is_dir():
            raise CodeRetrievalError("source roots must be existing unlinked directories")
        if path == index_root or index_root.is_relative_to(path) or path.is_relative_to(index_root):
            raise CodeRetrievalError("source roots must be disjoint from index_root")
        roots[alias] = path
    corpus = raw["corpus_ref"]
    if not isinstance(corpus, str) or not corpus.strip() or len(corpus) > 256:
        raise CodeRetrievalError("corpus_ref must be a bounded nonblank string")
    return CodeSettings(
        config.workspace_id,
        config.profile,
        roots,
        index_root,
        corpus,
        limit,
        units,
        str(policy),
        str(profile),
        model_config,
        model_sha256,
        model_binding,
        vector_budget,
    )


def _absolute(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise CodeRetrievalError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise CodeRetrievalError(f"{field} must be an absolute path")
    return path


def _writable(config: WorkspaceHostConfiguration) -> CodeSettings:
    settings = code_settings(config)
    if settings.host_profile not in {"write", "admin"}:
        raise PermissionError("code import and build require a write or admin profile")
    return settings


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = path.lstat()
    if not stat.S_ISDIR(identity.st_mode) or identity.st_uid != os.geteuid():
        raise CodeRetrievalError("index directories must be owned unlinked directories")


@contextmanager
def _locked(settings: CodeSettings) -> Iterator[None]:
    _private_directory(settings.index_root)
    descriptor = os.open(
        settings.index_root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _ledger(settings: CodeSettings) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(settings.index_root / "code-ledger.sqlite3", isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " source_ref TEXT NOT NULL, event TEXT NOT NULL CHECK (event IN ('import', 'forget')),"
            " sha256 TEXT, language TEXT, recorded_at TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS receipts (seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " receipt TEXT NOT NULL)"
        )
        yield connection
    finally:
        connection.close()


def _latest(connection: sqlite3.Connection, source_ref: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = connection.execute(
        "SELECT * FROM events WHERE source_ref = ? ORDER BY seq DESC LIMIT 1", (source_ref,)
    ).fetchone()
    return row


def _read_source(root: Path, relative_path: str, limit: int) -> bytes:
    """Read one file below root through no-follow descriptors on every component."""
    parts = PurePosixPath(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts) or relative_path.startswith("/"):
        raise CodeRetrievalError("relative path must stay below its source root")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    except OSError:
        os.close(directory)
        raise CodeRetrievalError("source is not a readable unlinked regular file") from None
    os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        identity = os.fstat(stream.fileno())
        if not stat.S_ISREG(identity.st_mode):
            raise CodeRetrievalError("source is not a regular file")
        if identity.st_size > limit:
            raise CodeRetrievalError("too_large")
        return stream.read(limit + 1)


def _git_environment(home: str) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": home,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    }


def _run_git(
    git: str, root: Path, arguments: list[str], home: str, stdin: bytes | None = None
) -> tuple[int, bytes]:
    command = [
        git,
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "protocol.allow=never",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        completed = subprocess.run(  # nosec B603
            command,
            input=stdin,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=_git_environment(home),
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise CodeRetrievalError("git enumeration failed or timed out") from None
    if len(completed.stdout) > MAX_GIT_OUTPUT:
        raise CodeRetrievalError("git output exceeds the enumeration limit")
    return completed.returncode, completed.stdout


def _git_identity(git: str) -> dict[str, str]:
    path = Path(git)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise CodeRetrievalError("an absolute executable --git path is required")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"git_path": str(path), "git_sha256": digest}


def import_repository(
    config: WorkspaceHostConfiguration, alias: str, *, git: str
) -> dict[str, Any]:
    """Discover, screen and record every eligible file of one configured repository root."""
    settings = _writable(config)
    root = settings.source_roots.get(alias)
    if root is None:
        raise CodeRetrievalError("unknown source root alias")
    identity = _git_identity(git)
    with tempfile.TemporaryDirectory(prefix="capicove-git-home-") as home:
        status, top = _run_git(git, root, ["rev-parse", "--show-toplevel"], home)
        if status != 0 or Path(top.decode("utf-8", "replace").strip()).resolve() != root:
            raise CodeRetrievalError("the source root alias must be a Git repository root")
        status, listed = _run_git(
            git, root, ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], home
        )
        if status != 0:
            raise CodeRetrievalError("git ls-files failed")
        candidates = sorted(
            {item.decode("utf-8", "surrogateescape") for item in listed.split(b"\0") if item}
        )
        if len(candidates) > MAX_CANDIDATES:
            raise CodeRetrievalError("repository candidates exceed the 10000-file limit")
        status, ignored_output = _run_git(
            git,
            root,
            ["check-ignore", "--no-index", "-z", "--stdin"],
            home,
            stdin=b"\0".join(c.encode("utf-8", "surrogateescape") for c in candidates) + b"\0",
        )
        if status not in (0, 1):
            raise CodeRetrievalError("git check-ignore failed")
        ignored = {
            item.decode("utf-8", "surrogateescape") for item in ignored_output.split(b"\0") if item
        }
    excluded: dict[str, int] = {}
    selected: list[tuple[str, str, str]] = []
    total = 0
    for relative in candidates:
        reason = None
        if relative in ignored:
            reason = "ignored"
        else:
            path = root / relative
            try:
                mode = path.lstat().st_mode
            except OSError:
                mode = 0
            if stat.S_ISLNK(mode):
                reason = "symlink"
            elif not stat.S_ISREG(mode):
                reason = "not_regular"
        if reason is None:
            try:
                payload = _read_source(root, relative, settings.max_source_bytes)
                screened = screen_source(relative, payload)
                total += len(payload)
                if total > MAX_SELECTED_BYTES:
                    raise CodeRetrievalError("selected bytes exceed the 64 MiB import limit")
                selected.append((relative, hashlib.sha256(payload).hexdigest(), screened.language))
            except IntakeRejection as rejection:
                reason = rejection.reason
            except CodeRetrievalError as error:
                if str(error) != "too_large":
                    raise
                reason = "too_large"
        if reason is not None:
            bucket = _EXCLUSION_BUCKETS.get(reason, reason)
            excluded[bucket] = excluded.get(bucket, 0) + 1
    receipt = {
        **identity,
        "policy": POLICY_VERSION,
        "candidates": len(candidates),
        "selection_sha256": hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode()
        ).hexdigest(),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    imported: list[str] = []
    with _locked(settings), _ledger(settings) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for relative, sha256, language in selected:
                source_ref = f"{alias}:{relative}"
                latest = _latest(connection, source_ref)
                if latest is None or latest["event"] != "import" or latest["sha256"] != sha256:
                    connection.execute(
                        "INSERT INTO events (source_ref, event, sha256, language, recorded_at)"
                        " VALUES (?, 'import', ?, ?, ?)",
                        (source_ref, sha256, language, receipt["recorded_at"]),
                    )
                    imported.append(source_ref)
            connection.execute("INSERT INTO receipts (receipt) VALUES (?)", (json.dumps(receipt),))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return {
        "imported": imported,
        "selected": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "receipt": {key: value for key, value in receipt.items() if key != "git_path"},
    }


def import_file(
    config: WorkspaceHostConfiguration, alias: str, relative_path: str
) -> dict[str, Any]:
    """Record one explicitly selected file under exactly the discovery policy."""
    settings = _writable(config)
    root = settings.source_roots.get(alias)
    if root is None:
        raise CodeRetrievalError("unknown source root alias")
    payload = _read_source(root, relative_path, settings.max_source_bytes)
    try:
        screened = screen_source(relative_path, payload)
    except IntakeRejection as rejection:
        raise CodeRetrievalError(f"source rejected: {rejection.reason}") from None
    sha256 = hashlib.sha256(payload).hexdigest()
    source_ref = f"{alias}:{relative_path}"
    with _locked(settings), _ledger(settings) as connection:
        latest = _latest(connection, source_ref)
        changed = latest is None or latest["event"] != "import" or latest["sha256"] != sha256
        if changed:
            connection.execute(
                "INSERT INTO events (source_ref, event, sha256, language, recorded_at)"
                " VALUES (?, 'import', ?, ?, ?)",
                (source_ref, sha256, screened.language, datetime.now(UTC).isoformat()),
            )
    return {"source_ref": source_ref, "imported": changed, "sha256": sha256}


def forget_file(
    config: WorkspaceHostConfiguration, alias: str, relative_path: str
) -> dict[str, Any]:
    """Retire one source for the next build; repeated forget is a no-op."""
    settings = _writable(config)
    source_ref = f"{alias}:{relative_path}"
    with _locked(settings), _ledger(settings) as connection:
        latest = _latest(connection, source_ref)
        if latest is None or latest["event"] != "import":
            return {"source_ref": source_ref, "forgotten": False}
        connection.execute(
            "INSERT INTO events (source_ref, event, recorded_at) VALUES (?, 'forget', ?)",
            (source_ref, datetime.now(UTC).isoformat()),
        )
    return {"source_ref": source_ref, "forgotten": True}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _policy_identity(policy: str = POLICY_VERSION) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "policy": policy,
        "symbols": POLICY_VERSION,
        "chunker": CHUNKER_VERSION,
        "definitions": DEFINITION_METHOD,
        "classifier": CLASSIFIER_VERSION,
        "path_normalization": PATH_NORMALIZATION,
        "tokenizer": "fts5-unicode61-v1",
        "ranking": f"identifier-classes-then-weighted-rrf-k{RRF_K}",
    }
    if policy in {AST_POLICY_VERSION, HYBRID_POLICY, RELATIONS_POLICY}:
        identity["parser"] = parser_identity()
    if policy in {HYBRID_POLICY, RELATIONS_POLICY}:
        identity["context"] = CONTEXT_FORMAT
    if policy == RELATIONS_POLICY:
        identity["relations"] = {
            "policy": RELATIONS_POLICY,
            "expansion_kinds": list(EXPANSION_KINDS),
            "max_seeds": MAX_SEEDS,
            "max_expansion": MAX_EXPANSION,
            "snippet_budget_bytes": MAX_SNIPPET_BUDGET,
        }
    return identity


def _factory(settings: CodeSettings, models: ModelFactory | None) -> ModelFactory:
    if models is not None:
        return models
    from capicove.model.configuration import load_configuration
    from capicove.model.factory import ModelFactory

    assert settings.model_config is not None
    if _file_sha256(settings.model_config) != settings.model_config_sha256:
        raise CodeRetrievalError("model configuration changed since startup")
    return ModelFactory(load_configuration(settings.model_config))


def build_index(
    config: WorkspaceHostConfiguration, *, models: ModelFactory | None = None
) -> dict[str, Any]:
    """Verify live sources, seal one generation and activate it atomically."""
    settings = _writable(config)
    if not settings.hybrid:
        return _build(settings, None)
    factory = _factory(settings, models)
    try:
        return _build(settings, factory)
    finally:
        if models is None:
            factory.close()


def _build(settings: CodeSettings, factory: ModelFactory | None) -> dict[str, Any]:
    use_ast = settings.code_policy in {AST_POLICY_VERSION, HYBRID_POLICY, RELATIONS_POLICY}
    texts: dict[str, tuple[str, str]] = {}
    outcomes: dict[str, int] = {}
    with _locked(settings):
        with _ledger(settings) as connection:
            live = list(
                connection.execute(
                    "SELECT e.* FROM events e JOIN (SELECT source_ref, MAX(seq) AS seq FROM events"
                    " GROUP BY source_ref) latest ON e.seq = latest.seq WHERE e.event = 'import'"
                    " ORDER BY e.source_ref"
                )
            )
        units: list[dict[str, Any]] = []
        identifiers: list[tuple[str, str, str, str, int]] = []
        for row in live:
            alias, relative = row["source_ref"].split(":", 1)
            payload = _read_source(
                settings.source_roots[alias], relative, settings.max_source_bytes
            )
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise CodeRetrievalError(
                    f"source changed since import; re-import or forget {row['source_ref']}"
                )
            text = payload.decode("utf-8")
            texts[relative] = (row["language"], text)
            layout = _layout(text, relative, row["language"], use_ast)
            if use_ast:
                bucket = (
                    layout.status if layout.reason is None else f"{layout.status}:{layout.reason}"
                )
                outcomes[bucket] = outcomes.get(bucket, 0) + 1
            for start_char, end_char, start_line, end_line, structure in layout.pieces:
                body = text[start_char:end_char]
                symbols = extract_symbols(row["language"], body, first_line=start_line)
                chunk_definitions = [
                    d for d in layout.definitions if start_line <= d["line"] <= end_line
                ]
                unit_id = (
                    "cu-"
                    + hashlib.sha256(
                        _canonical([row["sha256"], start_char, end_char, settings.code_policy])
                    ).hexdigest()[:40]
                )
                focus = chunk_definitions[0] if chunk_definitions else None
                units.append(
                    {
                        "unit_id": unit_id,
                        "source_ref": row["source_ref"],
                        "relative_path": relative,
                        "language": row["language"],
                        "source_sha256": row["sha256"],
                        "start_char": start_char,
                        "end_char": end_char,
                        "start_line": start_line,
                        "end_line": end_line,
                        "structure": structure,
                        "parse_status": layout.status,
                        "parse_reason": layout.reason,
                        "text": body,
                        "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
                        "focus_line": None if focus is None else focus["line"],
                        "focus_name": None if focus is None else focus["name"],
                        "focus_kind": None if focus is None else focus["kind"],
                        "focus_method": None if focus is None else focus["method"],
                        "definitions": json.dumps(chunk_definitions[:MAX_EVIDENCE_DEFINITIONS]),
                        "path_terms": " ".join(path_terms(relative)),
                        "identifier_terms": " ".join(
                            dict.fromkeys(
                                component
                                for item in symbols.identifiers
                                for component in identifier_components(item.raw)
                            )
                        ),
                    }
                )
                definition_names = {(d["name"], d["line"]) for d in chunk_definitions}
                for item in symbols.identifiers:
                    kind = "definition" if (item.raw, item.line) in definition_names else "use"
                    identifiers.append((unit_id, item.raw, item.normalized, kind, item.line))
                for definition in chunk_definitions:
                    for name in sorted({definition["name"], definition["qualified_name"]}):
                        identifiers.append(
                            (
                                unit_id,
                                name,
                                normalize_identifier(name),
                                "definition",
                                definition["line"],
                            )
                        )
        if len(units) > settings.max_units:
            raise CodeRetrievalError("searchable units exceed max_units")
        units.sort(key=lambda unit: unit["unit_id"])
        edges: list[tuple[str, str | None, str, str, str]] = []
        relation_stats: dict[str, int] | None = None
        if settings.code_policy == RELATIONS_POLICY:
            edges, relation_stats = _relation_edges(units, texts)
        space_payload: dict[str, Any] | None = None
        if factory is not None:
            space_payload = _attach_vectors(settings, factory, units)
        else:
            for unit in units:
                unit["vector"], unit["vector_sha256"] = None, None
        descriptor_body = {
            "format": FORMAT_VERSION,
            "variant": FORMAT_VARIANT,
            "workspace_id": settings.workspace_id,
            "corpus_ref": settings.corpus_ref,
            "profile": PROFILE_ID,
            "policy": _policy_identity(settings.code_policy),
            "embedding_spaces": [] if space_payload is None else [space_payload],
            "vector_count": sum(unit["vector"] is not None for unit in units),
            "model_config_sha256": settings.model_config_sha256,
            "relations_sha256": None
            if relation_stats is None
            else hashlib.sha256(_canonical(sorted(edges, key=lambda e: (e[0], e[4])))).hexdigest(),
            "units": [
                [
                    u["unit_id"],
                    u["text_sha256"],
                    u["source_ref"],
                    u["start_line"],
                    u["end_line"],
                    u["structure"],
                    u["parse_status"],
                    u["parse_reason"],
                    u["vector_sha256"],
                ]
                for u in units
            ],
        }
        generation_id = "cg-" + hashlib.sha256(_canonical(descriptor_body)).hexdigest()
        generations = settings.index_root / "generations"
        _private_directory(generations)
        target = generations / generation_id
        if not target.exists():
            _publish(
                settings,
                generations,
                target,
                generation_id,
                descriptor_body,
                units,
                identifiers,
                edges,
            )
        _validate(target, generation_id)
        _activate(settings, target, generation_id)
    result: dict[str, Any] = {
        "generation_id": generation_id,
        "units": len(units),
        "activated": True,
    }
    if settings.hybrid:
        result["vectors"] = descriptor_body["vector_count"]
    if relation_stats is not None:
        result["relations"] = relation_stats
    if use_ast:
        result["parse_outcomes"] = dict(sorted(outcomes.items()))
    return result


def _space_payload(space: Any) -> dict[str, Any]:
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
    }


def _document(unit: dict[str, Any]) -> str:
    """Provider-neutral path context (``path-header-v1``) for document encoding."""
    return f"{unit['relative_path']}\n{unit['text']}"


def _attach_vectors(
    settings: CodeSettings, factory: ModelFactory, units: list[dict[str, Any]]
) -> dict[str, Any]:
    """Encode every unit (reusing cached vectors) and enforce the vector budget."""
    import numpy as np

    assert settings.model_binding is not None
    with factory.text_encoder(settings.model_binding) as encoder:
        space = encoder.space
        payload = _space_payload(space)
        if space.dimension * 4 * len(units) > settings.max_vector_bytes:
            raise CodeRetrievalError("declared vectors exceed max_vector_bytes")
        cache_key = hashlib.sha256(_canonical(payload)).hexdigest()
        cache = sqlite3.connect(settings.index_root / "code-vectors.sqlite3")
        try:
            cache.execute(
                "CREATE TABLE IF NOT EXISTS vectors (space TEXT, document TEXT, vector BLOB,"
                " sha256 TEXT, PRIMARY KEY (space, document))"
            )
            pending: list[int] = []
            for index, unit in enumerate(units):
                key = hashlib.sha256(_document(unit).encode()).hexdigest()
                row = cache.execute(
                    "SELECT vector, sha256 FROM vectors WHERE space = ? AND document = ?",
                    (cache_key, key),
                ).fetchone()
                if row is not None and hashlib.sha256(row[0]).hexdigest() == row[1]:
                    unit["vector"] = bytes(row[0])
                else:
                    pending.append(index)
            batch = max(1, int(getattr(encoder, "max_batch", 8)))
            for start in range(0, len(pending), batch):
                chunk = pending[start : start + batch]
                embeddings = encoder.encode(
                    tuple(_document(units[index]) for index in chunk), space=space, role="document"
                )
                if len(embeddings) != len(chunk):
                    raise CodeRetrievalError("encoder returned a wrong vector count")
                for index, embedding in zip(chunk, embeddings, strict=True):
                    vector = np.asarray(embedding.vector, dtype="<f4")
                    norm = float(np.linalg.norm(vector))
                    if (
                        vector.shape != (space.dimension,)
                        or not np.isfinite(vector).all()
                        or abs(norm - 1.0) > 1e-3
                    ):
                        raise CodeRetrievalError("encoder returned an invalid vector")
                    blob = vector.tobytes()
                    units[index]["vector"] = blob
                    cache.execute(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
                        (
                            cache_key,
                            hashlib.sha256(_document(units[index]).encode()).hexdigest(),
                            blob,
                            hashlib.sha256(blob).hexdigest(),
                        ),
                    )
                cache.commit()
        finally:
            cache.close()
    for unit in units:
        unit["vector_sha256"] = hashlib.sha256(unit["vector"]).hexdigest()
    return payload


def _relation_edges(
    units: list[dict[str, Any]], texts: Mapping[str, tuple[str, str]]
) -> tuple[list[tuple[str, str | None, str, str, str]], dict[str, int]]:
    """Map extracted relations to source/target units; budgets fail the build."""
    from capicove.retrieval.code_relations import RelationBudgetError, extract_relations

    code = {path: value for path, value in texts.items() if ast_language(path) is not None}
    try:
        relations = extract_relations(code)
    except RelationBudgetError as error:
        raise CodeRetrievalError(str(error)) from None
    by_path: dict[str, list[dict[str, Any]]] = {}
    definition_unit: dict[str, str] = {}
    for unit in units:
        by_path.setdefault(unit["relative_path"], []).append(unit)
        for definition in json.loads(unit["definitions"]):
            definition_unit.setdefault(
                f"{unit['relative_path']}:{definition['qualified_name']}", unit["unit_id"]
            )
    edges: list[tuple[str, str | None, str, str, str]] = []
    stats: dict[str, int] = {}
    for relation in relations:
        source = next(
            (
                unit["unit_id"]
                for unit in by_path.get(relation.source_path, ())
                if unit["start_char"] <= relation.start_char < unit["end_char"]
            ),
            None,
        )
        if source is None:
            continue
        target = (
            definition_unit.get(relation.target or "") if relation.status == "resolved" else None
        )
        edges.append(
            (
                source,
                target,
                relation.kind,
                relation.status,
                json.dumps(relation.payload(), sort_keys=True),
            )
        )
        stats[relation.status] = stats.get(relation.status, 0) + 1
    return edges, dict(sorted(stats.items()))


def _expand(
    connection: sqlite3.Connection, ordered: list[str], eligible: Mapping[str, Any], top_k: int
) -> list[str]:
    """One-hop expansion from the first seeds; seeds stay first, filters bound targets."""
    seeds = ordered[:MAX_SEEDS]
    if top_k <= len(seeds):
        return ordered
    added: list[str] = []
    for seed in seeds:
        rows = connection.execute(
            "SELECT target_unit, kind FROM edges WHERE source_unit = ? AND status = 'resolved'"
            " AND target_unit IS NOT NULL",
            (seed,),
        ).fetchall()
        for row in sorted(
            rows,
            key=lambda r: (
                EXPANSION_KINDS.index(r["kind"]) if r["kind"] in EXPANSION_KINDS else 99,
                r["target_unit"],
            ),
        ):
            target = row["target_unit"]
            if (
                row["kind"] in EXPANSION_KINDS
                and target in eligible
                and target not in seeds
                and target not in added
                and len(added) < MAX_EXPANSION
            ):
                added.append(target)
    rest = [unit_id for unit_id in ordered[len(seeds) :] if unit_id not in added]
    return seeds + added + rest


def _unit_relations(connection: sqlite3.Connection, unit_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT payload FROM edges WHERE source_unit = ? ORDER BY payload LIMIT ?",
        (unit_id, MAX_EVIDENCE_RELATIONS),
    ).fetchall()
    relations = []
    for row in rows:
        payload = json.loads(row["payload"])
        relations.append(
            {key: payload[key] for key in ("kind", "target", "status", "source_line", "method")}
        )
    return relations


@dataclass(frozen=True, slots=True)
class _Layout:
    """Chunk spans, definitions and parse outcome for one verified source."""

    pieces: list[tuple[int, int, int, int, str]]
    definitions: list[dict[str, Any]]
    status: str
    reason: str | None


def _layout(text: str, relative: str, language: str, use_ast: bool) -> _Layout:
    """Choose AST or S1 structure; fallback facts keep the heuristic method label."""
    definitions = [
        {
            "name": d.name,
            "qualified_name": d.qualified_name,
            "kind": d.kind,
            "line": d.line,
            "method": d.method,
        }
        for d in extract_symbols(language, text).definitions
    ]
    lines = [
        (c.start_char, c.end_char, c.start_line, c.end_line, "lines") for c in chunk_code(text)
    ]
    if not use_ast:
        return _Layout(lines, definitions, "not_run", None)
    if language in _NON_CODE_LANGUAGES:
        return _Layout(lines, definitions, "not_applicable", None)
    outcome = parse_source(ast_language(relative), text)
    pieces = [
        (c.start_char, c.end_char, c.start_line, c.end_line, c.structure)
        for c in ast_chunks(text, outcome)
    ]
    if outcome.status is ParseStatus.PARSED:
        definitions = [
            {
                "name": s.name,
                "qualified_name": s.qualified_name,
                "kind": s.kind,
                "line": s.start_line,
                "method": s.method,
                "signature": s.signature,
            }
            for s in outcome.symbols
        ]
    return _Layout(pieces, definitions, outcome.status.value, outcome.reason)


def _publish(
    settings: CodeSettings,
    generations: Path,
    target: Path,
    generation_id: str,
    body: dict[str, Any],
    units: list[dict[str, Any]],
    identifiers: list[tuple[str, str, str, str, int]],
    edges: Sequence[tuple[str, str | None, str, str, str]] = (),
) -> None:
    staging = generations / f".staging-{secrets.token_hex(8)}"
    staging.mkdir(mode=0o700)
    try:
        database = staging / "units.sqlite3"
        with closing(sqlite3.connect(database)) as db:
            db.execute(
                "CREATE TABLE units (unit_id TEXT PRIMARY KEY, evidence_id TEXT UNIQUE NOT NULL,"
                " source_ref TEXT, relative_path TEXT, language TEXT, source_sha256 TEXT,"
                " start_char INTEGER, end_char INTEGER, start_line INTEGER, end_line INTEGER,"
                " structure TEXT, parse_status TEXT, parse_reason TEXT,"
                " text TEXT, text_sha256 TEXT, focus_line INTEGER, focus_name TEXT, focus_kind TEXT,"
                " focus_method TEXT,"
                " definitions TEXT NOT NULL, vector BLOB, vector_sha256 TEXT)"
            )
            db.execute(
                "CREATE TABLE identifiers (unit_id TEXT, raw TEXT, normalized TEXT, kind TEXT,"
                " line INTEGER)"
            )
            db.execute("CREATE INDEX identifiers_raw ON identifiers (raw)")
            db.execute("CREATE INDEX identifiers_normalized ON identifiers (normalized)")
            for lane in ("body", "path", "identifier_terms"):
                db.execute(
                    f"CREATE VIRTUAL TABLE lane_{lane} USING fts5(unit_id UNINDEXED, content,"
                    " tokenize='unicode61')"
                )
            for unit in units:
                evidence_id = (
                    "cev-"
                    + hashlib.sha256(
                        _canonical(
                            [
                                settings.workspace_id,
                                generation_id,
                                unit["unit_id"],
                                unit["text_sha256"],
                            ]
                        )
                    ).hexdigest()
                )
                db.execute(
                    "INSERT INTO units VALUES"
                    " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        unit["unit_id"],
                        evidence_id,
                        unit["source_ref"],
                        unit["relative_path"],
                        unit["language"],
                        unit["source_sha256"],
                        unit["start_char"],
                        unit["end_char"],
                        unit["start_line"],
                        unit["end_line"],
                        unit["structure"],
                        unit["parse_status"],
                        unit["parse_reason"],
                        unit["text"],
                        unit["text_sha256"],
                        unit["focus_line"],
                        unit["focus_name"],
                        unit["focus_kind"],
                        unit["focus_method"],
                        unit["definitions"],
                        unit["vector"],
                        unit["vector_sha256"],
                    ),
                )
                db.execute("INSERT INTO lane_body VALUES (?, ?)", (unit["unit_id"], unit["text"]))
                db.execute(
                    "INSERT INTO lane_path VALUES (?, ?)", (unit["unit_id"], unit["path_terms"])
                )
                db.execute(
                    "INSERT INTO lane_identifier_terms VALUES (?, ?)",
                    (unit["unit_id"], unit["identifier_terms"]),
                )
            db.executemany("INSERT INTO identifiers VALUES (?, ?, ?, ?, ?)", identifiers)
            db.execute(
                "CREATE TABLE edges (source_unit TEXT NOT NULL, target_unit TEXT, kind TEXT NOT NULL,"
                " status TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX edges_source ON edges (source_unit)")
            db.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", list(edges))
            db.commit()
        descriptor = {
            **body,
            "generation_id": generation_id,
            "units_sha256": _file_sha256(database),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (staging / "descriptor.json").write_bytes(_canonical(descriptor))
        for item in staging.iterdir():
            os.chmod(item, 0o400)
        os.chmod(staging, 0o500)
        os.rename(staging, target)
    except BaseException:
        if staging.exists():
            os.chmod(staging, 0o700)
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate(directory: Path, generation_id: str) -> dict[str, Any]:
    try:
        descriptor = json.loads((directory / "descriptor.json").read_bytes())
    except (OSError, ValueError):
        raise CodeRetrievalError("index_corrupt") from None
    body = {
        k: v
        for k, v in descriptor.items()
        if k not in {"generation_id", "units_sha256", "created_at"}
    }
    if (
        descriptor.get("format") != FORMAT_VERSION
        or descriptor.get("variant") != FORMAT_VARIANT
        or descriptor.get("generation_id") != generation_id
        or "cg-" + hashlib.sha256(_canonical(body)).hexdigest() != generation_id
        or _file_sha256(directory / "units.sqlite3") != descriptor.get("units_sha256")
    ):
        raise CodeRetrievalError("index_corrupt")
    return dict(descriptor)


def _activate(settings: CodeSettings, directory: Path, generation_id: str) -> None:
    payload = _canonical(
        {
            "generation_id": generation_id,
            "descriptor_sha256": _file_sha256(directory / "descriptor.json"),
        }
    )
    pointer = settings.index_root / f".active-{secrets.token_hex(8)}"
    descriptor = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(pointer, settings.index_root / "code-active.json")


@contextmanager
def _lease(settings: CodeSettings) -> Iterator[tuple[str, dict[str, Any], sqlite3.Connection]]:
    try:
        pointer = json.loads((settings.index_root / "code-active.json").read_bytes())
        generation_id = pointer["generation_id"]
    except (OSError, ValueError, KeyError, TypeError):
        raise CodeRetrievalError("index_generation_unavailable") from None
    if not isinstance(generation_id, str) or _GENERATION.fullmatch(generation_id) is None:
        raise CodeRetrievalError("index_corrupt")
    directory = settings.index_root / "generations" / generation_id
    try:
        if _file_sha256(directory / "descriptor.json") != pointer.get("descriptor_sha256"):
            raise CodeRetrievalError("index_corrupt")
    except OSError:
        raise CodeRetrievalError("index_corrupt") from None
    descriptor = _validate(directory, generation_id)
    if (
        descriptor["workspace_id"] != settings.workspace_id
        or descriptor["corpus_ref"] != settings.corpus_ref
    ):
        raise CodeRetrievalError("index_corrupt")
    connection = sqlite3.connect(
        f"file:{directory / 'units.sqlite3'}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        yield generation_id, descriptor, connection
    finally:
        connection.close()


_MEDIA_TYPES = {"markdown": "text/markdown", "rst": "text/x-rst"}


def media_type(language: str) -> str:
    """Return the media type used by the canonical media-type filter."""
    return _MEDIA_TYPES.get(language, f"text/x-{language}")


class CodeSearchOperation:
    """Class-ordered identifier, path and body ranking over the pinned generation."""

    def __init__(self, settings: CodeSettings, models: ModelFactory | None = None) -> None:
        self._settings = settings
        self._models = models
        self._owned: ModelFactory | None = None
        self._degraded = False

    def close(self) -> None:
        """Release an owned resident encoder."""
        owned, self._owned = self._owned, None
        if owned is not None:
            owned.close()

    def _query_vector(self, text: str, descriptor: Mapping[str, Any]) -> Any:
        """Encode the query; only encoder-execution unavailability returns None."""
        import numpy as np

        from capicove.model.multimodal_embedding import MultimodalEmbeddingError

        if self._models is None and self._owned is None:
            self._owned = _factory(self._settings, None)
        factory = self._models or self._owned
        assert factory is not None and self._settings.model_binding is not None
        try:
            with factory.text_encoder(self._settings.model_binding) as encoder:
                if [_space_payload(encoder.space)] != descriptor["embedding_spaces"]:
                    raise _Fatal(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
                (embedding,) = encoder.encode((text,), space=encoder.space, role="query")
        except MultimodalEmbeddingError as error:
            if error.code == "unavailable":
                return None
            raise _Fatal(
                SearchFailureCode.DEADLINE_EXCEEDED
                if error.code == "deadline_exceeded"
                else SearchFailureCode.INVALID_QUERY
            ) from None
        return np.asarray(embedding.vector, dtype="<f4")

    def search(self, request: SearchRequest) -> SearchResult:
        """Return canonical text-chunk hits or a stable failure code."""
        query_id = f"query-{uuid.uuid4().hex}"
        if (
            not isinstance(request.query, TextQuery)
            or request.targets != frozenset({TargetType.TEXT_CHUNK})
            or request.retrieval_profile_ref not in {self._settings.profile, "local-default"}
            or request.cursor is not None
        ):
            return SearchResult(
                SearchStatus.FAILED, query_id, failure_code=SearchFailureCode.INVALID_QUERY
            )
        self._degraded = False
        try:
            with _lease(self._settings) as (_generation, descriptor, connection):
                hits = self._rank(request, connection, descriptor)
        except _Fatal as failure:
            return SearchResult(SearchStatus.FAILED, query_id, failure_code=failure.code)
        except CodeRetrievalError as error:
            code = (
                SearchFailureCode.INDEX_GENERATION_UNAVAILABLE
                if str(error) == "index_generation_unavailable"
                else SearchFailureCode.INDEX_CORRUPT
            )
            return SearchResult(SearchStatus.FAILED, query_id, failure_code=code)
        except sqlite3.DatabaseError:
            return SearchResult(
                SearchStatus.FAILED, query_id, failure_code=SearchFailureCode.INDEX_CORRUPT
            )
        warnings = [SearchWarningCode.RETRIEVAL_PREVIEW]
        if self._degraded:
            warnings.append(SearchWarningCode.TEXT_DENSE_EMBEDDING_UNAVAILABLE)
        return SearchResult(
            SearchStatus.SUCCEEDED,
            query_id,
            degraded=self._degraded,
            warning_codes=tuple(warnings),
            hits=hits,
        )

    def _dense(
        self,
        connection: sqlite3.Connection,
        vector: Any,
        eligible: Mapping[str, Any],
        dimension: int,
    ) -> list[str]:
        """Exact dot product over eligible vectors in bounded row batches."""
        import numpy as np

        scored: list[tuple[float, str]] = []
        ids = sorted(eligible)
        for start in range(0, len(ids), DENSE_BATCH_ROWS):
            batch = ids[start : start + DENSE_BATCH_ROWS]
            marks = ", ".join("?" for _ in batch)
            rows = connection.execute(
                f"SELECT unit_id, vector, vector_sha256 FROM units WHERE unit_id IN ({marks})",
                batch,
            ).fetchall()
            if len(rows) != len(batch):
                raise CodeRetrievalError("index_corrupt")
            for row in rows:
                blob = row["vector"]
                if blob is None or hashlib.sha256(blob).hexdigest() != row["vector_sha256"]:
                    raise CodeRetrievalError("index_corrupt")
                values = np.frombuffer(blob, dtype="<f4")
                if values.shape != (dimension,):
                    raise CodeRetrievalError("index_corrupt")
                scored.append((float(values @ vector), row["unit_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [unit_id for _, unit_id in scored[:LANE_CANDIDATES]]

    def _eligible(
        self, request: SearchRequest, connection: sqlite3.Connection
    ) -> dict[str, sqlite3.Row]:
        filters = request.filters
        if (
            filters.time_range is not None
            or (filters.corpus_refs and self._settings.corpus_ref not in filters.corpus_refs)
            or (filters.text_origins and TextOrigin.DOCUMENT not in filters.text_origins)
        ):
            return {}
        rows = {}
        for row in connection.execute("SELECT * FROM units"):
            if (
                filters.artifact_refs
                and f"sha256:{row['source_sha256']}" not in filters.artifact_refs
            ):
                continue
            if filters.media_types and media_type(row["language"]) not in filters.media_types:
                continue
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
                raise CodeRetrievalError("index_corrupt")
            rows[row["unit_id"]] = row
        return rows

    def _lane(
        self,
        connection: sqlite3.Connection,
        lane: str,
        terms: list[str],
        eligible: Mapping[str, Any],
    ) -> list[str]:
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:64])
        rows = connection.execute(
            f"SELECT unit_id, bm25(lane_{lane}) AS score FROM lane_{lane} WHERE lane_{lane} MATCH ?"
            " ORDER BY score, unit_id LIMIT ?",
            (expression, MAX_CANDIDATES),
        )
        return [row["unit_id"] for row in rows if row["unit_id"] in eligible][:LANE_CANDIDATES]

    def _rank(
        self,
        request: SearchRequest,
        connection: sqlite3.Connection,
        descriptor: Mapping[str, Any],
    ) -> tuple[CompactHit, ...]:
        assert isinstance(request.query, TextQuery)
        text = request.query.text.strip()
        kind = classify_query(text)
        eligible = self._eligible(request, connection)
        if not eligible:
            return ()
        words = re.findall(r"\w+", text)
        components = list(dict.fromkeys(c for word in words for c in identifier_components(word)))
        lanes = {
            "body": self._lane(
                connection, "body", list(dict.fromkeys(words + components)), eligible
            ),
            "path": self._lane(
                connection,
                "path",
                list(
                    dict.fromkeys(
                        path_terms(text)
                        if kind is QueryKind.PATH
                        else [w.casefold() for w in words] + components
                    )
                ),
                eligible,
            ),
            "identifier_terms": self._lane(connection, "identifier_terms", components, eligible),
        }
        weights = dict(_WEIGHTS[kind])
        if self._settings.hybrid:
            vector = self._query_vector(text, descriptor)
            if vector is None:
                self._degraded = True
            else:
                dimension = int(descriptor["embedding_spaces"][0]["dimension"])
                lanes["dense"] = self._dense(connection, vector, eligible, dimension)
                weights["dense"] = _DENSE_WEIGHTS[kind]
        fused: dict[str, float] = {}
        for lane, ranked in lanes.items():
            for rank, unit_id in enumerate(ranked, start=1):
                fused[unit_id] = fused.get(unit_id, 0.0) + weights[lane] / (RRF_K + rank)
        classes: dict[str, int] = {}
        if kind is QueryKind.PATH:
            target = text.strip("/").casefold()
            for unit_id, row in eligible.items():
                path = row["relative_path"].casefold()
                if path == target or path.endswith("/" + target):
                    classes[unit_id] = 0
                    fused.setdefault(unit_id, 0.0)
        if kind is QueryKind.IDENTIFIER:
            normalized = normalize_identifier(text)
            for column, value, offset in (("raw", text, 0), ("normalized", normalized, 2)):
                rows = connection.execute(
                    f"SELECT unit_id, kind FROM identifiers WHERE {column} = ? LIMIT ?",
                    (value, IDENTIFIER_CANDIDATES),
                )
                for row in rows:
                    if row["unit_id"] not in eligible:
                        continue
                    rank_class = offset + (0 if row["kind"] == "definition" else 1)
                    classes[row["unit_id"]] = min(classes.get(row["unit_id"], 4), rank_class)
                    fused.setdefault(row["unit_id"], 0.0)
        ordered = sorted(
            fused, key=lambda unit_id: (classes.get(unit_id, 4), -fused[unit_id], unit_id)
        )
        if self._settings.code_policy == RELATIONS_POLICY:
            ordered = _expand(connection, ordered, eligible, request.top_k)
        hits = []
        for rank, unit_id in enumerate(ordered[: request.top_k], start=1):
            row = eligible[unit_id]
            hits.append(
                CompactHit(
                    unit_id,
                    TargetType.TEXT_CHUNK,
                    1.0 / rank,
                    row["evidence_id"],
                    text_locator=TextChunkLocator(
                        TextOrigin.DOCUMENT, CharacterRange(row["start_char"], row["end_char"])
                    ),
                    snippet=preview_snippet(row["text"])
                    if row["text"].strip()
                    else row["relative_path"],
                )
            )
        if self._settings.code_policy == RELATIONS_POLICY:
            # Drop whole tail hits to fit the context budget; cited text is never cut.
            budget, kept = MAX_SNIPPET_BUDGET, []
            for hit in hits:
                size = len((hit.snippet or "").encode("utf-8"))
                if size > budget:
                    break
                budget -= size
                kept.append(hit)
            hits = kept
        return tuple(hits)


class CodeEvidenceReader:
    """Return bounded, sealed chunk evidence; never reopen the live source."""

    def __init__(self, settings: CodeSettings) -> None:
        self._settings = settings

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, Any]:
        """Reauthorize the workspace and resolve one evidence ID in the active generation."""
        if context.workspace_id != self._settings.workspace_id:
            raise PermissionError("evidence belongs to another workspace")
        if _EVIDENCE.fullmatch(evidence_id) is None:
            raise LookupError("evidence is unavailable")
        if set(sections) - {"metadata", "lineage"}:
            raise LookupError("unsupported evidence section")
        try:
            with _lease(self._settings) as (generation_id, descriptor, connection):
                row = connection.execute(
                    "SELECT * FROM units WHERE evidence_id = ?", (evidence_id,)
                ).fetchone()
                relations = [] if row is None else _unit_relations(connection, row["unit_id"])
        except (CodeRetrievalError, sqlite3.DatabaseError):
            raise LookupError("no consistent active generation") from None
        if row is None or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise LookupError("evidence is unavailable")
        result: dict[str, Any] = {}
        if "metadata" in sections:
            metadata: dict[str, Any] = {
                "kind": "text_chunk",
                "source_ref": row["source_ref"],
                "relative_path": row["relative_path"],
                "language": row["language"],
                "media_type": media_type(row["language"]),
                "source_sha256": row["source_sha256"],
                "start_char": row["start_char"],
                "end_char": row["end_char"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
                "relations": relations,
                "structure": row["structure"],
                "parse_status": row["parse_status"],
                "parse_reason": row["parse_reason"],
                "corpus_ref": self._settings.corpus_ref,
                "generation_id": generation_id,
            }
            if row["focus_line"] is not None:
                metadata.update(
                    definition_focus_line=row["focus_line"],
                    definition_name=row["focus_name"],
                    definition_kind=row["focus_kind"],
                    definition_method=row["focus_method"],
                    definitions=json.loads(row["definitions"]),
                )
            result["metadata"] = metadata
        if "lineage" in sections:
            result["lineage"] = {
                "generation_id": generation_id,
                "format": descriptor["format"],
                "variant": descriptor["variant"],
                "policy": descriptor["policy"],
                "text_sha256": row["text_sha256"],
            }
        return result

    def read_evidence(
        self, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, Any]:
        """Positional convenience for in-process callers."""
        return self.read(context=context, evidence_id=evidence_id, sections=sections)


@dataclass(frozen=True, slots=True)
class CodeRetrievalRuntime:
    """Search operation and Evidence reader bound to startup settings."""

    operation: CodeSearchOperation
    reader: CodeEvidenceReader

    def search(self, request: SearchRequest) -> SearchResult:
        """Delegate to the bound operation."""
        return self.operation.search(request)

    def read_evidence(
        self, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, Any]:
        """Delegate to the bound evidence reader."""
        return self.reader.read_evidence(context, evidence_id, sections)

    def close(self) -> None:
        """Release resident encoder resources."""
        self.operation.close()


def open_code_retrieval(
    config: WorkspaceHostConfiguration, *, models: ModelFactory | None = None
) -> CodeRetrievalRuntime:
    """Compose read-only code Search and Evidence; never builds or creates storage.

    A hybrid profile loads its encoder lazily on the first query and keeps it until
    ``close``; startup and catalog composition never instantiate a model.
    """
    settings = code_settings(config)
    return CodeRetrievalRuntime(CodeSearchOperation(settings, models), CodeEvidenceReader(settings))


__all__ = [
    "FORMAT_VERSION",
    "HYBRID_PROFILE_ID",
    "PROFILE_ID",
    "CodeEvidenceReader",
    "CodeRetrievalError",
    "CodeRetrievalRuntime",
    "CodeSearchOperation",
    "CodeSettings",
    "build_index",
    "code_settings",
    "forget_file",
    "import_file",
    "import_repository",
    "is_code_configuration",
    "media_type",
    "open_code_retrieval",
]


class _Fatal(Exception):
    """A non-degradable dense-lane failure that fails the whole query."""

    def __init__(self, code: SearchFailureCode) -> None:
        super().__init__(code.value)
        self.code = code
