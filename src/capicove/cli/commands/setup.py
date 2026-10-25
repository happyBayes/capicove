"""Create and inspect an explicit offline workspace configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register private local workspace setup commands."""
    parser = parent.add_parser("setup", help="Initialize and inspect a local workspace.")
    commands = parser.add_subparsers(dest="setup_command", required=True)
    initialize = commands.add_parser("init", help="Create an offline starter workspace.")
    initialize.add_argument("workspace", type=Path)
    initialize.add_argument("--use-existing", action="store_true")
    initialize.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="ALIAS=DIR",
        help=(
            "Also enable offline search over an existing local directory (repeatable). "
            "Selects the write profile so sources can be imported; nothing is read yet."
        ),
    )
    doctor = commands.add_parser("doctor", help="Check selected local Tool readiness.")
    doctor.add_argument("--config", type=Path, required=True)
    generation = commands.add_parser(
        "check-llm", help="Explicitly call a configured compatible LLM with an authored probe."
    )
    generation.add_argument("--config", type=Path, required=True)


def run(arguments: argparse.Namespace) -> int:
    """Run a setup operation without implicit network or model access."""
    try:
        # Interactive commands anchor relative paths at the caller's working directory.
        # absolute() keeps symlinks unresolved so the no-follow checks still apply.
        if arguments.setup_command == "init":
            return _initialize(
                arguments.workspace.absolute(),
                use_existing=arguments.use_existing,
                source_roots=_parse_source_roots(arguments.source_root),
            )
        if arguments.setup_command == "check-llm":
            from capicove.graph_build.remote_generation import (
                load_remote_generation_profile,
                probe_remote_generation,
            )

            report = probe_remote_generation(
                load_remote_generation_profile(arguments.config.absolute())
            )
            print(json.dumps(report, sort_keys=True))
            return 0
        return _doctor(arguments.config.absolute())
    except (TimeoutError, RuntimeError):
        print("Setup failed: generation service unavailable or deadline exceeded", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 2


def _parse_source_roots(values: list[str]) -> dict[str, Path]:
    """Parse repeated ``ALIAS=DIR`` options; relative directories use the caller's cwd."""
    roots: dict[str, Path] = {}
    for value in values:
        alias, separator, directory = value.partition("=")
        if not separator or not alias or not directory:
            raise ValueError("--source-root must be ALIAS=DIR")
        if alias in roots:
            raise ValueError("source root aliases must be unique")
        roots[alias] = Path(directory).absolute()
    return roots


def _starter_body(root: Path, source_roots: Mapping[str, Path]) -> dict[str, object]:
    """Return the starter config: image inspection, plus offline search when requested."""
    body: dict[str, object] = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": root.name,
        "actor_id": "local-owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "read",
        "permissions": ["image:read"],
        "enabled_tools": ["capicove.image.filter.shape"],
        "trace_store": None,
        "capabilities": {"image": {"schema_version": 1, "read_roots": [str(root / "inputs")]}},
        "network": {"enabled": False, "source_hosts": []},
    }
    if source_roots:
        # Importing sources is an operator write (RFC-0018), so search needs `write`.
        body["profile"] = "write"
        body["permissions"] = ["evidence:read", "evidence:search", "image:read"]
        body["enabled_tools"] = [
            "capicove.get_evidence",
            "capicove.image.filter.shape",
            "capicove.search",
        ]
        body["capabilities"] = {
            "image": {"schema_version": 1, "read_roots": [str(root / "inputs")]},
            "retrieval": {
                "schema_version": 1,
                "source_roots": {alias: str(path) for alias, path in source_roots.items()},
                "index_root": str(root / "index"),
                "corpus_ref": f"corpus:{root.name}",
                "retrieval_profile_ref": "local-default",
                "max_source_bytes": 1024 * 1024,
            },
        }
    return body


def _validate_starter(body: Mapping[str, object]) -> None:
    """Reject a starter config the host would refuse, before anything is created."""
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config

    config = parse_mcp_host_config(json.dumps(body).encode("utf-8"))
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ValueError("starter config must be a version-2 workspace config")
    if "retrieval" in config.capabilities:
        from capicove.runtime.local_retrieval import (
            RetrievalDependencyError,
            retrieval_settings,
        )

        try:
            retrieval_settings(config)
        except RetrievalDependencyError as error:
            # An ImportError subclass; report it like any other unusable starter.
            raise ValueError(str(error)) from None


def _next_commands(config_path: Path, source_roots: Mapping[str, Path]) -> list[list[str]]:
    config = str(config_path)
    if not source_roots:
        return [
            ["capicove", "setup", "doctor", "--config", config],
            ["capicove", "mcp", "catalog", "--config", config],
        ]
    alias = next(iter(source_roots))
    return [
        ["capicove", "setup", "doctor", "--config", config],
        [
            *("capicove", "source", "import", "--runtime-config", config),
            *("--root", alias, "--relative-path", "RELATIVE_PATH"),
        ],
        ["capicove", "index", "build", "--runtime-config", config],
        ["capicove", "search", "QUERY", "--runtime-config", config],
    ]


def _initialize(root: Path, *, use_existing: bool, source_roots: Mapping[str, Path]) -> int:
    if not root.is_absolute():
        raise ValueError("workspace path must be absolute")
    if not root.name.strip() or len(root.name.encode("utf-8")) > 128:
        raise ValueError("workspace directory name must be 1 to 128 UTF-8 bytes")
    body = _starter_body(root, source_roots)
    _validate_starter(body)
    next_commands = _next_commands(root / "host.json", source_roots)
    if os.name == "nt":
        from capicove._windows_files import reject_windows_reparse_components

        reject_windows_reparse_components(root, include_self=False)
    elif any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("workspace path must not contain symlinks")
    if os.name == "posix":
        return _initialize_posix(
            root, use_existing=use_existing, body=body, next_commands=next_commands
        )
    created_root = False
    created_inputs = False
    inputs = root / "inputs"
    try:
        if root.exists():
            if not use_existing or not root.is_dir() or any(root.iterdir()):
                raise ValueError(
                    "workspace already exists; use --use-existing for an empty directory"
                )
        else:
            root.mkdir(mode=0o700)
            created_root = True
        if os.name == "nt":
            from capicove._windows_files import validate_windows_path

            validate_windows_path(root, directory=True, private_acl=True)
        elif root.stat().st_mode & 0o077:
            raise ValueError("workspace must be owner-private")
        inputs.mkdir(mode=0o700)
        created_inputs = True
        if os.name == "nt":
            validate_windows_path(inputs, directory=True, private_acl=True)
        config_path = root / "host.json"
        staging_path = root / f".host-{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(staging_path, flags, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(body, output, sort_keys=True, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            if os.name == "nt":
                validate_windows_path(staging_path, directory=False, private_acl=True)
            os.link(staging_path, config_path)
        finally:
            staging_path.unlink(missing_ok=True)
    except Exception:
        if created_inputs:
            try:
                inputs.rmdir()
            except OSError:
                pass
        if created_root:
            try:
                root.rmdir()
            except OSError:
                pass
        raise
    print(
        json.dumps(
            {
                "config": str(config_path),
                "workspace": str(root),
                "next_commands": next_commands,
            },
            sort_keys=True,
        )
    )
    return 0


def _open_posix_parent(path: Path) -> int:
    """Open every existing parent component without following symlinks."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:-1]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_posix_workspace_path(root: Path, parent_fd: int, root_fd: int) -> None:
    """Reject a renamed parent or workspace before publishing its path."""
    fresh_parent = _open_posix_parent(root)
    try:
        parent_info = os.fstat(parent_fd)
        fresh_info = os.fstat(fresh_parent)
        if (fresh_info.st_dev, fresh_info.st_ino) != (parent_info.st_dev, parent_info.st_ino):
            raise ValueError("workspace path changed during initialization")
        fresh_root = os.open(
            root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fresh_parent
        )
        try:
            root_info = os.fstat(root_fd)
            fresh_root_info = os.fstat(fresh_root)
            if (fresh_root_info.st_dev, fresh_root_info.st_ino) != (
                root_info.st_dev,
                root_info.st_ino,
            ):
                raise ValueError("workspace path changed during initialization")
        finally:
            os.close(fresh_root)
    finally:
        os.close(fresh_parent)


def _remove_created_posix_root(parent_fd: int, root_name: str, root_fd: int) -> None:
    """Remove only the pinned new root through a random quarantine name."""
    quarantine = f".workspace-rollback-{uuid.uuid4().hex}"
    try:
        os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError:
        return
    else:
        return
    moved = False
    try:
        os.rename(root_name, quarantine, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        moved = True
        observed = os.stat(quarantine, dir_fd=parent_fd, follow_symlinks=False)
        pinned = os.fstat(root_fd)
        if (observed.st_dev, observed.st_ino) != (pinned.st_dev, pinned.st_ino):
            return
        os.rmdir(quarantine, dir_fd=parent_fd)
        moved = False
    except OSError:
        pass
    finally:
        if moved:
            try:
                os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.rename(quarantine, root_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                except OSError:
                    pass
            except OSError:
                pass


def _initialize_posix(
    root: Path,
    *,
    use_existing: bool,
    body: Mapping[str, object],
    next_commands: list[list[str]],
) -> int:
    """Create a starter workspace through pinned POSIX directory handles."""
    parent_fd = _open_posix_parent(root)
    root_fd: int | None = None
    created_root = False
    created_inputs = False
    created_config = False
    staging_name = ".host-" + uuid.uuid4().hex + ".tmp"
    try:
        try:
            root_fd = os.open(
                root.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
            created_root = True
            root_fd = os.open(
                root.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        else:
            if not use_existing or os.listdir(root_fd):
                raise ValueError(
                    "workspace already exists; use --use-existing for an empty directory"
                )
        if os.fstat(root_fd).st_mode & 0o077:
            raise ValueError("workspace must be owner-private")
        _verify_posix_workspace_path(root, parent_fd, root_fd)
        os.mkdir("inputs", mode=0o700, dir_fd=root_fd)
        created_inputs = True
        descriptor = os.open(
            staging_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(body, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        _verify_posix_workspace_path(root, parent_fd, root_fd)
        os.link(
            staging_name,
            "host.json",
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )
        created_config = True
        os.fsync(root_fd)
        _verify_posix_workspace_path(root, parent_fd, root_fd)
    except BaseException:
        if root_fd is not None:
            try:
                os.unlink(staging_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            if created_config:
                try:
                    os.unlink("host.json", dir_fd=root_fd)
                except OSError:
                    pass
            if created_inputs:
                try:
                    os.rmdir("inputs", dir_fd=root_fd)
                except OSError:
                    pass
        if created_root and root_fd is not None:
            _remove_created_posix_root(parent_fd, root.name, root_fd)
        raise
    finally:
        if root_fd is not None:
            try:
                os.unlink(staging_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            os.close(root_fd)
        os.close(parent_fd)
    config_path = root / "host.json"
    print(
        json.dumps(
            {
                "config": str(config_path),
                "workspace": str(root),
                "next_commands": next_commands,
            },
            sort_keys=True,
        )
    )
    return 0


def _doctor(path: Path) -> int:
    from capicove.runtime.mcp_config import (
        WorkspaceHostConfiguration,
        load_mcp_host_config,
        tool_capability,
    )
    from capicove.runtime.mcp_host import compose_mcp_host

    config = load_mcp_host_config(path)
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ValueError("setup doctor requires a version-2 workspace config")
    tool_status: dict[str, str] = {}
    diagnostics: list[dict[str, str]] = []
    for capability in sorted(config.capabilities):
        selected = tuple(
            tool_id for tool_id in config.enabled_tools if tool_capability(tool_id) == capability
        )
        if not selected:
            continue
        for tool_id in selected:
            try:
                with compose_mcp_host(
                    replace(config, enabled_tools=(tool_id,)), catalog_only=True
                ) as bindings:
                    if len(bindings) != 1 or bindings[0].spec.tool_id != tool_id:
                        raise ValueError("selected Tool is unavailable")
                    tool_status[tool_id] = "ready"
            except Exception as error:
                code, hint = _readiness_diagnostic(
                    capability, config.capabilities[capability], error
                )
                tool_status[tool_id] = code
                diagnostic = {"capability": capability, "code": code, "hint": hint}
                if diagnostic not in diagnostics:
                    diagnostics.append(diagnostic)
    report = {
        "schema_version": 1,
        "ready": all(status == "ready" for status in tool_status.values()),
        "tools": [
            {"tool_id": tool_id, "status": tool_status[tool_id]} for tool_id in config.enabled_tools
        ],
    }
    if diagnostics:
        report["diagnostics"] = diagnostics
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ready"] else 2


def _readiness_diagnostic(
    capability: str, configured: Mapping[str, object], error: Exception
) -> tuple[str, str]:
    """Return a stable, non-sensitive readiness code and remediation hint."""
    if isinstance(error, ImportError):
        return "missing_dependency", "Install the selected capability's optional dependencies."
    if "trace_store" in str(error):
        return (
            "unsafe_path",
            "Configure trace_store below an existing owner-private local directory.",
        )
    if capability == "image" and "requires only read_roots" in str(error):
        return (
            "not_configured",
            "Configure existing local directories in capabilities.image.read_roots.",
        )
    if capability == "image" and "read_roots" in str(error):
        return (
            "unsafe_path",
            "Configure existing local directories in capabilities.image.read_roots.",
        )
    if capability == "retrieval" and isinstance(error, ValueError):
        if "documented fields" in str(error):
            return (
                "not_configured",
                "Configure exactly the documented capabilities.retrieval fields.",
            )
        if any(field in str(error) for field in ("index_root", "source_roots")):
            return (
                "unsafe_path",
                "Use existing local source_roots and a private index_root inside the workspace.",
            )
    from capicove.runtime.media_handoff import AssetIntegrityError

    if isinstance(error, AssetIntegrityError):
        return "asset_integrity", "Restore the model file or update its pinned sha256."
    if capability == "audio" and isinstance(error, ValueError):
        if "documented fields" in str(error) or "requires" in str(error):
            return "not_configured", "Configure the documented capabilities.audio fields."
        if any(field in str(error) for field in ("read_roots", "staging_root", "evidence_root")):
            return (
                "unsafe_path",
                "Use existing read_roots and private staging/evidence roots inside the workspace.",
            )
    if capability == "video" and isinstance(error, ValueError):
        if "documented fields" in str(error) or "required when" in str(error):
            return "not_configured", "Configure the documented capabilities.video fields."
        if any(field in str(error) for field in ("read_roots", "staging_root", "output_root")):
            return (
                "unsafe_path",
                "Use existing read_roots and private staging/output roots inside the workspace.",
            )
    if capability in {"batch", "trace", "model"} and isinstance(error, ValueError):
        message = str(error)
        if "network" in message:
            return (
                "network_required",
                "Model install needs the admin profile, network.enabled, and allowlisted hosts.",
            )
        if "capabilities.video" in message:
            return "not_configured", "Batch video operations also need capabilities.video."
        if any(
            word in message
            for word in ("documented fields", "not configured", "schema_version", "allowlist")
        ):
            return "not_configured", f"Configure the documented capabilities.{capability} fields."
        if any(
            word in message
            for word in (
                "roots",
                "root",
                "private",
                "symlink",
                "disjoint",
                "stores",
                "destinations",
            )
        ):
            return (
                "unsafe_path",
                f"Use existing inputs and owner-private {capability} roots inside the workspace.",
            )
    if capability == "memory" and "memory root" in str(error) and "root" not in configured:
        return "not_configured", "Set capabilities.memory.root to a usable local directory."
    if capability == "memory" and "memory scopes" in str(error) and "scopes" not in configured:
        return "not_configured", "Set capabilities.memory.scopes to local scope names."
    if capability == "memory" and "memory root" in str(error):
        return "unsafe_path", "Set capabilities.memory.root to a usable local directory."
    return "unavailable", "Inspect the selected local Tool IDs and capability configuration."
