"""Audit planned remote operations without granting runtime qualification.

Source discovery is conservative: public symbols are candidates, not accepted
Tools. Source fingerprints intentionally include implementation bodies, so changes
require review even when a Python signature happens to remain unchanged. This
script neither loads plugin entry points nor creates providers or downloads assets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import re
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = "docs/development/remote-mcp-capability-coverage.md"
# Every public function/method in these domain surfaces is a review candidate.
# CLI/API/Tools are scanned globally; new files cannot hide behind a frozen count.
DOMAINS = {
    "C01": ("processing/image",),
    "C02": ("retrieval/image", "runtime/reverse_image.py"),
    "C03": (
        "processing/video/filtering",
        "processing/video/splitter.py",
        "processing/video/strategies",
    ),
    "C04": ("processing/video/retrieval", "runtime/video_search.py"),
    "C05": ("processing/audio",),
    "C06": ("runtime/local_retrieval.py",),
    "C07": ("retrieval/service.py", "retrieval/reranking.py"),
    "C08": ("runtime/native_knowledge.py", "runtime/knowledge_mcp.py"),
    "C09": ("knowledge/service.py",),
    "C10": ("agent/context_assembly.py", "agent/memory_service.py", "agent/memory_policy.py"),
    "C11": ("documents/service.py",),
    "C12": ("books/service.py",),
    "C13": ("books/chunk_service.py", "books/chunk_result_service.py"),
    "C14": ("wiki/service.py",),
    "C15": ("graph_build/service.py", "graph_build/snapshots.py"),
    "C16": (
        "processing/planner.py",
        "processing/local.py",
        "processing/ports.py",
        "processing/operators",
    ),
    "C18": ("batch/job_service.py",),
    "C19": ("export", "traces/service.py"),
    "C20": (
        "model/provisioning.py",
        "model/cli.py",
        "model/installers.py",
        "model/reranker_assets.py",
    ),
    "C21": (
        "model/registry.py",
        "model/ports.py",
        "model/factory.py",
        "model/serving.py",
        "model/vlm.py",
        "model/vlm_service.py",
        "model/vlm_tools.py",
        "model/providers",
    ),
    "C22": (
        "evolution/registry.py",
        "evolution/learning.py",
        "evolution/evaluation.py",
        "evolution/generation.py",
    ),
    "C23": ("plugins", "runtime/setup.py"),
    "C25": ("processing/video/deduplication",),
    "C26": (
        "runtime/retrieval_build.py",
        "retrieval/generation_service.py",
        "retrieval/image_generations.py",
        "retrieval/video_generations.py",
        "retrieval/video_publication.py",
    ),
}
SURFACE_FAMILIES = {
    "knowledge_mcp": "C08",
    "image": "C01",
    "image_search": "C02",
    "video_split": "C03",
    "video_filter": "C03",
    "video_deduplicate": "C25",
    "search_video": "C04",
    "audio": "C05",
    "audio_import": "C05",
    "audio_clip": "C05",
    "audio_transcribe": "C05",
    "local_retrieval": "C06",
    "search": "C07",
    "evidence": "C07",
    "knowledge_base": "C08",
    "memory": "C09",
    "memory_tools": "C09",
    "documents": "C11",
    "book": "C12",
    "book_chunks": "C13",
    "book_chunk_results": "C13",
    "wiki": "C14",
    "batch_jobs": "C18",
    "jobs": "C18",
    "operator_traces": "C19",
    "operator_models": "C20",
    "models": "C20",
    "evolution": "C22",
    "model_server": "C21",
    "model_serve": "C21",
    "vlm": "C21",
    "reranker": "C21",
    "bge_m3": "C21",
    "siglip2": "C21",
    "doctor": "C23",
    "setup": "C23",
    "mcp": "C24",
}
# Additional proposed business modes from the mandatory expansion. These do not
# pretend to be implemented public Python APIs or accepted remote contracts.
PROPOSED = {
    "C01": "model-provider-artifact-handoff",
    "C02": "reverse-search pinned-index-search",
    "C03": "explicit-file-grouping filter-criteria long-dispatch",
    "C04": "prepare-keyframes prepare-asr prepare-ocr temporal-relations evidence-range",
    "C05": "upload-import-transcribe-clip-artifact",
    "C06": "source-import source-forget offline-index-build",
    "C07": "grant-aware-search grant-aware-evidence rerank",
    "C08": "bounded-initialize status doctor rebuild update forget search ask evidence",
    "C09": "current-facts entity-alias entity-merge resolve-entity projection-maintenance",
    "C10": "context-assembly governed-memory-admission",
    "C11": "parse-page-selection page-retrieval block-retrieval coordinate-retrieval receipt-retrieval rejected-document partial-document",
    "C12": "prepare summary-tasks summary-next structure-tasks structure-submit summary-submit status publish",
    "C13": "chunk-preview chunk-create chunk-read parent-context chunk-tasks chunk-status chunk-submit chunk-publish publication-manifest",
    "C14": "initialize scan ingest lint build doctor query migration-preview migration-apply",
    "C15": "plan build resume cancel run-discovery status result",
    "C16": "plan validate execute status result cancel snapshot-lookup snapshot-validate snapshot-publish snapshot-retract nested-schema",
    "C17": "plan validate execute status result active-cancel nested-schema",
    "C18": "plan enqueue status cancel resume worker-recovery",
    "C19": "trace-inspect trace-validate export-redacted export-media-copy artifact-retrieve",
    "C20": "model-inventory profile-plan profile-verify profile-install metadata legacy-image-acquisition",
    "C21": "text-embedding-float text-embedding-base64 text-rerank image-embedding text-image-embedding image-text-similarity ocr document-parse transcription vlm-generate vlm-count_tokens model-list model-lookup vlm-json-response vlm-text-response function-definitions function-selection returned-function-calls usage-accounting",
    "C22": "initialize status active distill generate evaluate run feedback review-show review-approve review-reject rollback",
    "C23": "runtime-diagnostics catalog-readiness selected-profile-verification compatible-llm-probe bounded-setup",
    "C24": "upload-start upload-status upload-write upload-abort upload-commit artifact-stat artifact-read artifact-delete operation-submit operation-status operation-result operation-cancel operation-resume approval-request approval-review policy-inspect audit-read worker-isolation",
    "C25": "directory-scan manifest-export duplicate-move duplicate-delete keeper-selection partial-action-receipt",
    "C26": "text-create text-validate text-activate text-reconcile image-stage image-validate image-publish image-activate video-register video-validate video-activate video-resolve prepared-video-publish read-lease eviction-safe-cleanup generation-inspect",
}


TASKS = {
    "C01": "R4.01",
    "C02": "R4.04",
    "C03": "R4.02",
    "C04": "R4.04",
    "C05": "R4.03",
    "C06": "R4.06",
    "C07": "R4.04",
    "C08": "R4.07",
    "C09": "R4.05",
    "C10": "R4.05",
    "C11": "R5.02",
    "C12": "R5.03",
    "C13": "R5.03",
    "C14": "R5.04",
    "C15": "R5.05",
    "C16": "R5.06",
    "C17": "R5.06",
    "C18": "R3.01",
    "C19": "R5.07",
    "C20": "R5.01",
    "C21": "R5.01",
    "C22": "R5.08",
    "C23": "R6.01",
    "C24": "R2.01",
    "C25": "R4.02",
    "C26": "R4.06",
}


def provider_sources(root):
    """Enumerate source-defined adapter variants without loading dependencies."""
    files = {}
    for family, prefix in {
        "C01": "processing/image",
        "C03": "processing/video/filtering",
        "C04": "processing/video/retrieval",
        "C05": "processing/audio",
    }.items():
        for path in (root / "src/capicove" / prefix).rglob("_adapters/*.py"):
            files[path] = family
    for path in (root / "plugins").glob("*/src/**/*.py"):
        if "__pycache__" not in path.parts:
            files[path] = (
                "C17"
                if "execution-spark" in path.parts
                else ("C11" if "processor-paddleocr" in path.parts else "C07")
            )
    return files


def additional_surfaces(root):
    """Inventory provider variants and explicit CLI commands/effect switches."""
    rows = []
    for path, family in sorted(provider_sources(root).items()):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                anchor = f"{path.relative_to(root)}:{node.name}"
                rows.append(
                    row(
                        family,
                        "provider:" + anchor,
                        anchor,
                        node.name,
                        source_digest=digest(ast.dump(node, include_attributes=False)),
                        blocking_reason="Source adapter exists; exact operation-to-provider binding, supported parameter modes, profile pins and real runtime evidence require owner review.",
                    )
                )
    for path in sorted((root / "src/capicove/cli/commands").glob("*.py")):
        family = SURFACE_FAMILIES.get(path.stem, "C23")
        found = {}
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or not node.args
            ):
                continue
            if not isinstance(node.args[0], ast.Constant) or not isinstance(
                node.args[0].value, str
            ):
                continue
            name = node.args[0].value
            if node.func.attr == "add_parser" or (
                node.func.attr == "add_argument"
                and name in {"--move-duplicates", "--delete-duplicates", "--write-manifest"}
            ):
                found[name] = node
        for name, node in sorted(found.items()):
            anchor = f"{path.relative_to(root)}:{name}"
            candidate = row(
                family,
                "cli:" + anchor,
                anchor,
                source_digest=digest(ast.dump(node, include_attributes=False)),
            )
            if name in {"--delete-duplicates", "--move-duplicates"}:
                candidate.update(
                    classification="admin_tool",
                    classification_rationale="Existing explicit CLI effect requires a separately accepted remote approval and safe-handle contract.",
                )
            rows.append(candidate)
    return rows


def digest(value):
    """Hash canonical JSON using stable mapping and sequence conversion."""

    def plain(item):
        if isinstance(item, Mapping):
            return {key: plain(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(val) for val in item]
        return item

    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_files(root):
    """Discover sources from public entry directories and domain service paths."""
    files = {}
    base = root / "src/capicove"
    for directory in ("cli/commands", "api", "tools"):
        for path in (base / directory).glob("*.py"):
            files[path] = SURFACE_FAMILIES.get(path.stem, "C23")
    for family, prefixes in DOMAINS.items():
        for prefix in prefixes:
            path = base / prefix
            candidates = [path] if path.is_file() else path.rglob("*.py")
            for candidate in candidates:
                if candidate.is_file() and not any(
                    part.startswith("_") for part in candidate.relative_to(base).parts[:-1]
                ):
                    files[candidate] = family
    for path in (root / "plugins/execution-spark/src").rglob("*.py"):
        files[path] = "C17"
    return files


def source_manifest(root):
    """Freeze public source, transport mapping, registry and plugin definitions."""
    paths = set(source_files(root)) | set(provider_sources(root))
    # Fingerprint the full source/configuration boundary. A discovered symbol
    # list alone cannot detect changed registry choices or model asset locks.
    # Hidden directories (.venv, .tox, .git, caches) hold local environments,
    # never reviewed source; scanning them makes the manifest checkout-specific.
    for base in (root / "src", root / "plugins"):
        paths.update(
            path
            for path in base.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".toml", ".json"}
            and "__pycache__" not in path.parts
            and not any(part.startswith(".") for part in path.relative_to(base).parts)
        )
    paths.update((root / "src/capicove/mcp").glob("*.py"))
    paths.update((root / "src/capicove/runtime").glob("*_mcp.py"))
    paths.update((root / "plugins").glob("*/pyproject.toml"))
    paths.add(root / COVERAGE)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def discover_public_surfaces(root):
    """Return public symbol candidates, including undecorated CLI/API helpers."""
    found = {}
    for path, family in sorted(source_files(root).items()):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            candidates = [(node, "")]
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                candidates = [(member, node.name + ".") for member in node.body]
            for member, prefix in candidates:
                if not isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or member.name.startswith("_"):
                    continue
                anchor = f"{path.relative_to(root)}:{prefix}{member.name}"
                found[anchor] = {
                    "family": family,
                    "fingerprint": digest(ast.dump(member, include_attributes=False)),
                }
    return found


def ensure_source_root(root):
    """Prevent source fingerprints and live schemas from different checkouts.

    Shared environments may install an editable .pth pointing elsewhere. Pin the
    selected source directory before first import; never replace already loaded
    foreign modules, because their objects could survive in other modules.
    """
    source = (root / "src").resolve()
    package = (source / "capicove").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "capicove" or name.startswith("capicove."):
            origin = getattr(module, "__file__", None)
            locations = (
                [origin] if isinstance(origin, str) else list(getattr(module, "__path__", ()))
            )
            if not locations or any(
                not Path(location).resolve().is_relative_to(package) for location in locations
            ):
                raise ValueError(f"imported {name} does not match selected source root")
    path = str(source)
    if not sys.path or sys.path[0] != path:
        sys.path.insert(0, path)
    importlib.invalidate_caches()


def tool_specs(root):
    """Read immutable Tool metadata; never construct handlers or model providers."""
    ensure_source_root(root)
    from capicove.contracts import ToolSpec

    found = {}

    def visit(value, anchor):
        if isinstance(value, ToolSpec):
            found[value.registry_key] = (value, anchor)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, ToolSpec):
                    visit(item, f"{anchor}[{key}]")
        elif isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, ToolSpec):
                    visit(item, anchor)

    for path in sorted((root / "src/capicove/tools").glob("*.py")):
        module = importlib.import_module(f"capicove.tools.{path.stem}")
        declarations = {
            target.id
            for statement in ast.parse(path.read_text()).body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            for target in (
                statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            )
            if isinstance(target, ast.Name)
        }
        for name, value in vars(module).items():
            if name.isupper() and name in declarations:
                visit(value, f"{path.relative_to(root)}:{name}")
    from capicove.runtime.knowledge_mcp import specs

    for spec in specs(generation=False).values():
        found[spec.registry_key] = (spec, "src/capicove/runtime/knowledge_mcp.py:specs")
    return found


def local_wire_names(root):
    """Collect existing literal names plus dependency-light canonical mappings."""
    ensure_source_root(root)
    from capicove.mcp.book import BOOK_MCP_NAMES
    from capicove.mcp.image import MCP_IMAGE_TOOL_NAMES
    from capicove.tools.book import BOOK_TOOL_SPECS
    from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
    from capicove.tools.memory_tools import MEMORY_SPECS

    names = {
        IMAGE_TOOL_SPECS_BY_NAME[key].tool_id: value for key, value in MCP_IMAGE_TOOL_NAMES.items()
    }
    names.update({BOOK_TOOL_SPECS[key].tool_id: value for key, value in BOOK_MCP_NAMES.items()})
    names.update({spec.tool_id: "capicove_memory_" + key for key, spec in MEMORY_SPECS.items()})
    for domain, operations in {
        "audio": ("import", "clip", "transcribe"),
        "knowledge": ("status", "search", "ask", "evidence"),
    }.items():
        names.update(
            {
                f"capicove.{domain}.{operation}": f"capicove_{domain}_{operation}"
                for operation in operations
            }
        )
    literals = set()
    for path in list((root / "src/capicove/mcp").glob("*.py")) + list(
        (root / "src/capicove/runtime").glob("*_mcp.py")
    ):
        literals.update(
            node.value
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("capicove_")
        )
    for tool_id, _ in tool_specs(root):
        candidate = tool_id.replace(".", "_").replace("-", "_")
        if candidate in literals or tool_id.startswith(
            (
                "capicove.book.",
                "capicove.batch.",
                "capicove.model.",
                "capicove.trace.",
                "capicove.video.",
            )
        ):
            names[tool_id] = candidate
    return names


def row(family, operation, source, profile="builtin", **extra):
    """Create an explicitly blocked candidate; defaults are never authorization."""
    result = {
        "capability_id": family,
        "operation_id": operation,
        "source_entrypoints": [source],
        "classification": "business_tool",
        "classification_rationale": "Conservative public-operation candidate; owner classification and remote contract review remain required.",
        "classification_review": "pending",
        "canonical_tool_id": None,
        "contract_version": None,
        "mcp_name": None,
        "input_schema_digest": None,
        "output_schema_digest": None,
        "schema_scope": "no_accepted_remote_schema",
        "provider_profile": profile,
        "runtime_requirements": {"provider": profile, "pins": None, "qualification": "not_run"},
        "permissions": [],
        "resource_policy": "unreviewed; deny remote invocation",
        "approval_policy": "unreviewed; deny effects",
        "execution_mode": "unresolved",
        "deadline": None,
        "idempotency_policy": "unreviewed; deny remote mutation",
        "task_ids": [
            "R0.02",
            *(
                ["R3.02", "R3.04"]
                if family == "C18"
                else [
                    f"R{stage}.{number:02}"
                    for stage, total in ((1, 4), (2, 4), (3, 5))
                    for number in range(1, total + 1)
                ]
                if family == "C24"
                else [TASKS[family]]
            ),
        ],
        "planning": None,
        "provider_binding_status": "candidate_unreviewed",
        "test_ids": [],
        "acceptance_receipt": None,
        "state": "inventoried",
        "blocking_reason": "Remote contract, authority policy, pinned runtime profile and real acceptance evidence remain unqualified.",
    }
    result.update(extra)
    return result


def expected_rows(root):
    """Discover independently required operation/provider pairs, not family counts."""
    ensure_source_root(root)
    rows = additional_surfaces(root)
    for anchor, metadata in discover_public_surfaces(root).items():
        candidate = row(
            metadata["family"], "public:" + anchor, anchor, source_digest=metadata["fingerprint"]
        )
        method = anchor.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
        if (
            method
            in {
                "to_dict",
                "from_dict",
                "to_json",
                "from_json",
                "as_dict",
                "encode",
                "decode",
                "close",
                "dispose",
                "execution_lock",
                "acquire",
                "operator_specs",
            }
            or "/tools/" in anchor
        ):
            # Tool modules expose registration/handler support; their business
            # operation has its own exact canonical Tool row below.
            candidate.update(
                classification="internal_helper",
                classification_rationale=f"{anchor} is a serialization/resource/registration primitive; enclosing canonical or service operation owns remote behavior. Owner review remains pending.",
            )
        if "/ports.py:" in anchor or "/model/vlm.py:" in anchor:
            candidate["implementation_status"] = "contract_only"
        rows.append(candidate)
    wires = local_wire_names(root)
    for (tool_id, version), (spec, anchor) in sorted(tool_specs(root).items()):
        family = SURFACE_FAMILIES.get(anchor.split("/")[-1].split(".")[0], "C23")
        profiles = ["builtin"]
        if tool_id == "capicove.search":
            # The dispatcher supports modality-specific ports. A plugin name
            # alone is not proof of an operation/provider pairing.
            profiles = ["pending-exact-provider-binding"]
        elif "perceptual-deduplicate-ray" in tool_id:
            profiles = ["ray-gpu"]
        rows.extend(
            row(
                family,
                f"tool:{tool_id}@{version}",
                anchor,
                profile,
                canonical_tool_id=tool_id,
                contract_version=version,
                mcp_name=wires.get(tool_id),
                input_schema_digest=digest(spec.input_schema),
                output_schema_digest=digest(spec.output_schema),
                schema_scope="existing_local_canonical_only",
                permissions=sorted(spec.required_permissions),
                deadline=spec.timeout_seconds,
            )
            for profile in profiles
        )
    for path in sorted((root / "plugins").glob("*/pyproject.toml")):
        project = tomllib.loads(path.read_text())["project"]
        for group, entries in project.get("entry-points", {}).items():
            for name, target in entries.items():
                family = {
                    "capicove.retrievers": "C07",
                    "capicove.execution_backends": "C17",
                    "capicove.document_parsers": "C11",
                    "capicove.model_installers": "C20",
                }.get(group, "C23")
                rows.append(
                    row(
                        family,
                        f"plugin:{group}:{name}",
                        f"{path.relative_to(root)}:{group}.{name}={target}",
                        project["name"],
                        source_digest=digest(project),
                        runtime_requirements={
                            "distribution": project["name"],
                            "version": project["version"],
                            "dependencies": project.get("dependencies", []),
                            "pins": None,
                            "qualification": "not_run",
                        },
                    )
                )
    from capicove.model.serving import ROUTES

    provider_path = root / "src/capicove/model/providers/__init__.py"
    tree = ast.parse(provider_path.read_text())
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "ProviderDefinition"
        ):
            provider, version = (ast.literal_eval(arg) for arg in call.args[:2])
            capability = call.args[2].attr
            for cap, routes in ROUTES.items():
                if cap.name == capability:
                    for route in routes:
                        rows.append(
                            row(
                                "C21",
                                "route:" + route,
                                "src/capicove/model/serving.py:ROUTES",
                                provider,
                                source_digest=digest([capability, route, version]),
                                runtime_requirements={
                                    "provider": provider,
                                    "version": version,
                                    "pins": None,
                                    "qualification": "not_run",
                                },
                            )
                        )
    # Operator identities are read without importing Spark/Java or any backend.
    for path in sorted(
        list((root / "src/capicove/processing").rglob("*.py"))
        + list((root / "plugins/execution-spark/src").rglob("*contracts.py"))
    ):
        ids = sorted(
            {
                node.value
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(("capicove.data.", "capicove.agent-trace."))
            }
        )
        if ids:
            family, profile = (
                ("C17", "capicove-execution-spark") if "plugins" in path.parts else ("C16", "local")
            )
            for operator in ids:
                rows.append(
                    row(
                        family,
                        f"operator:{operator}:{path.relative_to(root)}",
                        str(path.relative_to(root)),
                        profile,
                        source_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    for family, operations in PROPOSED.items():
        for operation in operations.split():
            rows.append(
                row(
                    family,
                    f"proposed:{family}:{operation}",
                    COVERAGE,
                    implementation_status="contract_only",
                    source_digest=hashlib.sha256((root / COVERAGE).read_bytes()).hexdigest(),
                )
            )
    return sorted(rows, key=lambda item: (item["operation_id"], item["provider_profile"]))


def _specified(value):
    """Reject absent values and explicit placeholders in planning metadata."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized) and not any(
            word in normalized
            for word in ("pending", "unreviewed", "unresolved", "not_run", "tbd", "unknown")
        )
    if isinstance(value, dict):
        return bool(value) and all(
            _specified(key) and _specified(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return bool(value) and all(_specified(item) for item in value)
    return type(value) in {int, float} and value > 0


def planning_errors(item, known):
    """Validate separately authored planning mappings, never execution evidence.

    Reviewed metadata lives under ``planning``. Discovery defaults and local
    schemas remain immutable evidence; they are not remote contract acceptance.
    Review references are declared provenance, not authenticated human approval.
    """
    plan = item.get("planning")
    label = item["operation_id"]
    if not isinstance(plan, dict):
        return [
            f"planning missing: {label}: owner classification and operation/provider mapping required"
        ]
    errors = []

    def require(condition, field):
        if not condition:
            errors.append(f"planning incomplete: {label}: {field}")

    classification = plan.get("classification")
    require(
        classification in ("business_tool", "admin_tool", "internal_helper", "transport_lifecycle"),
        "classification",
    )
    require(_specified(plan.get("classification_rationale")), "classification_rationale")
    review = plan.get("review")
    require(
        isinstance(review, dict)
        and all(_specified(review.get(field)) for field in ("owner", "reference")),
        "review",
    )
    if classification in ("internal_helper", "transport_lifecycle"):
        enclosing = plan.get("enclosing_operation")
        matches = [
            candidate
            for (operation, _), candidate in known.items()
            if operation == enclosing and operation != label
        ]
        require(
            any(
                isinstance(candidate.get("planning"), dict)
                and candidate["planning"].get("classification") in ("business_tool", "admin_tool")
                for candidate in matches
            ),
            "reviewed enclosing business operation",
        )
        return errors
    contract = plan.get("contract")
    require(isinstance(contract, dict), "contract")
    if isinstance(contract, dict):
        for field in ("tool_id", "mcp_name", "review_reference"):
            require(_specified(contract.get(field)), "contract." + field)
        require(
            type(contract.get("version")) is int and contract["version"] > 0, "contract.version"
        )
        for field in ("input_schema_digest", "output_schema_digest"):
            require(
                isinstance(contract.get(field), str)
                and re.fullmatch(r"[0-9a-f]{64}", contract[field]) is not None,
                "contract." + field,
            )
    for field in ("permissions", "test_ids"):
        value = plan.get(field)
        require(
            isinstance(value, list)
            and bool(value)
            and all(isinstance(entry, str) and _specified(entry) for entry in value),
            field,
        )
    for field in ("resource_policy", "approval_policy", "idempotency_policy"):
        require(isinstance(plan.get(field), dict) and _specified(plan[field]), field)
    require(plan.get("execution_mode") in ("sync", "durable"), "execution_mode")
    deadline = plan.get("deadline")
    require(
        type(deadline) in {int, float}
        and deadline > 0
        and (type(deadline) is int or math.isfinite(deadline)),
        "deadline",
    )
    bindings = plan.get("operation_provider_bindings")
    require(isinstance(bindings, list) and bool(bindings), "operation_provider_bindings")
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, dict):
                require(False, "operation_provider_binding object")
                continue
            operation = binding.get("operation_id")
            provider = binding.get("provider_profile")
            source = binding.get("provider_source")
            valid_names = all(
                isinstance(value, str) and _specified(value)
                for value in (operation, provider, source)
            )
            require(valid_names, "binding identities")
            if valid_names:
                require(
                    (operation, provider) in known
                    and not operation.startswith(("provider:", "plugin:")),
                    "binding business operation/provider variant",
                )
                require(
                    provider == item["provider_profile"],
                    "binding matches inventoried provider variant",
                )
                require((source, provider) in known, "binding provider source/profile")
                require(label in {operation, source}, "binding belongs to inventoried candidate")
            profile = binding.get("profile")
            require(
                isinstance(profile, dict)
                and _specified(profile.get("id"))
                and isinstance(profile.get("pins"), dict)
                and _specified(profile["pins"]),
                "binding pinned profile",
            )
    return errors


CORE_FAMILIES = frozenset(
    {"C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C25", "C26"}
)
CORE_REQUIRED_OPERATIONS = frozenset(
    {
        "capicove.search.text",
        "capicove.search.video",
        "capicove.search.image",
        "capicove.get_evidence",
        "capicove.video.prepare",
        "capicove.video.directory.scan",
        "capicove.video.duplicates.manifest",
        "capicove.video.duplicates.move",
        "capicove.video.duplicates.delete",
        "capicove.source.import",
        "capicove.source.forget",
        "capicove.index.build",
        "capicove.context.assemble",
        "capicove.context.admit_episode",
        "capicove.memory.current_facts",
        "capicove.memory.add_entity_alias",
        "capicove.memory.merge_entities",
        "capicove.memory.rebuild_projections",
        "capicove.memory.check_entity_resolution",
        "capicove.knowledge.initialize",
        "capicove.knowledge.update",
        "capicove.knowledge.rebuild",
        "capicove.knowledge.forget",
        "capicove.generation.text.create",
        "capicove.generation.text.validate",
        "capicove.generation.text.activate",
        "capicove.generation.text.reconcile_active",
        "capicove.generation.image.build",
        "capicove.generation.image.validate",
        "capicove.generation.image.publish",
        "capicove.generation.image.abort",
        "capicove.generation.video.register",
        "capicove.generation.video.activate",
        "capicove.generation.video.resolve_active",
        "capicove.generation.video.can_evict",
        "capicove.generation.video.publish_prepared",
    }
)


def _source_anchor_exists(anchor, root):
    """Check exact Python symbol anchors without importing capability providers."""
    if not isinstance(anchor, str):
        return False
    relative, _, symbol = anchor.partition(":")
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        return False
    if not symbol or path.suffix != ".py":
        return True
    symbol = symbol.split("[", 1)[0]
    nodes = ast.parse(path.read_text()).body
    for part in symbol.split("."):
        match = next((node for node in nodes if getattr(node, "name", None) == part), None)
        if match is not None:
            nodes = getattr(match, "body", [])
            continue
        declarations = [
            target.id
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        ]
        return part == symbol and part in declarations
    return True


def _schema_has_unconstrained_node(schema):
    """Reject unconstrained schema nodes, including nested property/item schemas."""
    if schema is False:
        return False
    if not isinstance(schema, dict) or not schema:
        return True
    for key in ("properties", "$defs", "patternProperties"):
        children = schema.get(key, {})
        if isinstance(children, dict) and any(
            _schema_has_unconstrained_node(child) for child in children.values()
        ):
            return True
    for key in ("items", "contains", "propertyNames", "not", "if", "then", "else"):
        if key in schema and _schema_has_unconstrained_node(schema[key]):
            return True
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict) and _schema_has_unconstrained_node(additional):
        return True
    for key in ("oneOf", "anyOf", "allOf", "prefixItems"):
        if any(_schema_has_unconstrained_node(child) for child in schema.get(key, [])):
            return True
    return False


def _payload_reference_exists(reference, document):
    """Resolve only local bounded payload definitions, never remote URLs."""
    if not isinstance(reference, str):
        return False
    if "#" not in reference:
        return reference in document.get("$defs", {})
    prefix, pointer = reference.split("#", 1)
    if prefix not in {"", "remote-mcp-core-payloads.json"} or not pointer.startswith("/"):
        return False
    current = document
    for part in pointer[1:].split("/"):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return isinstance(current, dict) and bool(current)


def validate_core_manifest(manifest, root):
    """Validate concrete draft contract mappings, not acceptance or execution.

    All profiles remain obligations, including unprovisioned optional profiles.
    Source pins bind inspected code; they never stand in for installed assets.
    """
    from jsonschema import Draft202012Validator, SchemaError

    errors = []
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != 1
        or not isinstance(manifest.get("operations"), list)
    ):
        return ["invalid core contract manifest"]
    if (
        manifest.get("approval_state") != "draft_for_review"
        or manifest.get("runtime_qualification") is not False
    ):
        errors.append("core manifest cannot assert unrecorded approval or qualification")
    payload_relative = "docs/development/remote-mcp-core-payloads.json"
    payload_path = root / payload_relative
    payload_document = {}
    try:
        payload_bytes = payload_path.read_bytes()
        expected_payload_link = {
            "path": payload_relative,
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        if manifest.get("payload_schema_file") != expected_payload_link:
            errors.append("core payload schema linkage missing or changed")
        payload_document = json.loads(payload_bytes)
        Draft202012Validator.check_schema(payload_document)
        if (
            payload_document.get("approval_state") != "draft_for_review"
            or payload_document.get("runtime_implemented") is not False
        ):
            errors.append("core payload schema cannot assert acceptance or implementation")
    except (OSError, ValueError, SchemaError):
        errors.append("core payload schema missing or invalid")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or any(
        not isinstance(profile, dict) or not isinstance(profile.get("profile_id"), str)
        for profile in profiles
    ):
        return ["invalid core provider profiles"]
    profile_map = {profile["profile_id"]: profile for profile in profiles}
    if len(profile_map) != len(profiles):
        errors.append("duplicate core provider profile")
    for profile in profiles:
        if (
            profile.get("enabled_by_default") is not False
            or profile.get("provisioning_state") != "not_verified"
            or profile.get("qualification_receipt") is not None
            or profile.get("final_coverage_obligation") is not True
        ):
            errors.append("core profile cannot invent readiness or waive final coverage")
    sources = manifest.get("source_manifest")
    if not isinstance(sources, dict):
        return ["missing core source manifest"]
    for relative, expected in sources.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"core source drift: {relative}")
    specs = tool_specs(root)
    required_local = {
        tool_id
        for tool_id, _ in specs
        if tool_id.startswith(
            (
                "capicove.image.",
                "capicove.audio.",
                "capicove.memory.",
                "capicove.knowledge.",
                "capicove.video.",
            )
        )
    }
    seen, covered, families = set(), set(), set()
    tool_names, wire_names = set(), set()
    for item in [*profiles, *manifest["operations"]]:
        if not isinstance(item, dict):
            errors.append("invalid core operation/profile")
            continue
        anchors = item.get("source_entrypoints")
        if not isinstance(anchors, list) or not anchors:
            errors.append("missing core source entrypoints")
        else:
            for anchor in anchors:
                if (
                    not _source_anchor_exists(anchor, root)
                    or anchor.split(":", 1)[0] not in sources
                ):
                    errors.append(f"invalid core source anchor: {anchor}")
    for operation in manifest["operations"]:
        if not isinstance(operation, dict) or not isinstance(operation.get("operation_id"), str):
            errors.append("invalid core operation")
            continue
        identity = operation["operation_id"]
        if identity in seen:
            errors.append(f"duplicate core operation: {identity}")
        seen.add(identity)
        for field, names in (("proposed_tool_id", tool_names), ("mcp_name", wire_names)):
            name = operation.get(field)
            if not isinstance(name, str) or not name or name in names:
                errors.append(f"invalid or duplicate core {field}: {identity}")
            else:
                names.add(name)
        parameters = operation.get("input_schema", {}).get("properties", {}).get("parameters")
        if (
            not isinstance(parameters, dict)
            or parameters.get("type") != "object"
            or parameters.get("additionalProperties") is not False
            or not isinstance(parameters.get("properties"), dict)
        ):
            errors.append(f"unbounded core parameters: {identity}")
        families.add(operation.get("capability_id"))
        local_id, version = operation.get("local_tool_id"), operation.get("local_contract_version")
        if local_id is not None:
            local = (
                specs.get((local_id, version))
                if isinstance(local_id, str) and type(version) is int
                else None
            )
            if local is None:
                errors.append(f"unknown local Tool: {identity}")
            else:
                covered.add(local_id)
                for direction in ("input", "output"):
                    if operation.get(f"local_{direction}_schema_digest") != digest(
                        getattr(local[0], f"{direction}_schema")
                    ):
                        errors.append(f"local schema drift: {identity}: {direction}")
        for direction in ("input", "output"):
            schema = operation.get(direction + "_schema")
            if not isinstance(schema, dict):
                errors.append(f"missing core schema: {identity}: {direction}")
                continue
            if _schema_has_unconstrained_node(schema):
                errors.append(f"unconstrained nested core schema: {identity}: {direction}")
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError:
                errors.append(f"invalid core schema: {identity}: {direction}")
            if schema.get("additionalProperties") is not False or operation.get(
                direction + "_schema_digest"
            ) != digest(schema):
                errors.append(f"core schema digest/closure drift: {identity}: {direction}")
        for reference in operation.get("payload_contracts", {}).values():
            if not _payload_reference_exists(reference, payload_document):
                errors.append(f"unresolved core payload reference: {identity}: {reference}")
        for references in operation.get("section_payload_contracts", {}).values():
            if not isinstance(references, list) or any(
                not _payload_reference_exists(ref, payload_document) for ref in references
            ):
                errors.append(f"unresolved core section payload reference: {identity}")
        output = operation.get("output_schema", {})
        if not {"operation_id", "result", "artifacts", "lineage"} <= set(
            output.get("properties", {})
        ):
            errors.append(f"missing core result envelope: {identity}")
        audit = operation.get("semantic_audit")
        if not isinstance(audit, dict) or audit.get("state") not in {
            "pending",
            "blocked",
            "source_fields_projected",
        }:
            errors.append(f"missing core semantic audit: {identity}")
        else:
            if audit.get("state") == "source_fields_projected":
                evidence = audit.get("field_projection", audit.get("projection_evidence"))
                if not evidence or audit.get("unresolved_fields"):
                    errors.append(f"empty or unresolved core semantic projection: {identity}")
            result_fields = (
                operation.get("output_schema", {})
                .get("properties", {})
                .get("result", {})
                .get("properties", {})
            )
            if {"resource_id", "state", "changed", "warnings"} <= set(result_fields) and (
                audit.get("state") != "blocked" or not audit.get("unresolved_fields")
            ):
                errors.append(
                    f"generic core projection cannot claim semantic completeness: {identity}"
                )
        bindings = operation.get("provider_bindings")
        if not isinstance(bindings, list) or not bindings:
            errors.append(f"missing core provider binding: {identity}")
        else:
            for binding in bindings:
                profile = (
                    profile_map.get(binding.get("profile_id"))
                    if isinstance(binding, dict) and isinstance(binding.get("profile_id"), str)
                    else None
                )
                if (
                    profile is None
                    or binding.get("source_entrypoints") != profile["source_entrypoints"]
                    or not binding.get("supported_modes")
                ):
                    errors.append(f"unresolved core provider binding: {identity}")
        for field in (
            "permissions",
            "resource_policy",
            "approval_policy",
            "idempotency_policy",
            "transport_limits",
            "planned_test_suites",
            "test_ids",
        ):
            if not operation.get(field):
                errors.append(f"missing core obligation: {identity}: {field}")
        if (
            operation.get("approval_state") != "draft_for_review"
            or operation.get("acceptance_receipt") is not None
        ):
            errors.append(f"unrecorded core acceptance: {identity}")
    if families != CORE_FAMILIES:
        errors.append("core capability mapping is incomplete")
    for missing in sorted((CORE_REQUIRED_OPERATIONS - seen) | (required_local - covered)):
        errors.append(f"missing core operation: {missing}")
    try:
        inventory_data = json.loads(
            (root / "docs/development/remote-mcp-operations.json").read_text()
        )
        recorded_mappings = {
            (row["operation_id"], row["provider_profile"]): row.get("contract_mapping")
            for row in inventory_data["operations"]
        }
    except (OSError, ValueError, KeyError, TypeError):
        recorded_mappings = {}
        errors.append("core ownership inventory unavailable")
    mappings = manifest.get("source_mappings")
    if not isinstance(mappings, list):
        errors.append("missing core source reconciliation")
    else:
        for mapping in mappings:
            if (
                not isinstance(mapping, dict)
                or not mapping.get("classification_rationale")
                or not mapping.get("enclosing_operations")
            ):
                errors.append("invalid core source reconciliation")
                continue
            key = (mapping.get("candidate_operation_id"), mapping.get("candidate_provider_profile"))
            recorded = recorded_mappings.get(key)
            if not isinstance(recorded, dict) or recorded.get(
                "enclosing_operations"
            ) != mapping.get("enclosing_operations"):
                errors.append(f"core source ownership linkage drift: {key}")
            if any(target not in seen for target in mapping["enclosing_operations"]):
                errors.append("unknown core enclosing operation")
    expected_candidates = {
        (row["operation_id"], row["provider_profile"]): row
        for row in expected_rows(root)
        if row["capability_id"] in CORE_FAMILIES
    }
    partition = set()
    unresolved = manifest.get("unresolved_candidates")
    if not isinstance(unresolved, list):
        errors.append("missing core unresolved obligations")
        unresolved = []
    for kind, entries in (
        ("mapped", mappings if isinstance(mappings, list) else []),
        ("unresolved", unresolved),
    ):
        for entry in entries:
            if not isinstance(entry, dict):
                errors.append("invalid core candidate")
                continue
            prefix = "candidate_" if kind == "mapped" else ""
            key = (entry.get(prefix + "operation_id"), entry.get(prefix + "provider_profile"))
            if not all(isinstance(value, str) for value in key):
                errors.append("invalid core candidate identity")
                continue
            original = expected_candidates.get(key)
            if (
                original is None
                or entry.get("source_entrypoints") != original["source_entrypoints"]
            ):
                errors.append(f"unknown or changed core candidate: {key}")
            if key in partition:
                errors.append(f"duplicate core candidate reconciliation: {key}")
            partition.add(key)
    if partition != set(expected_candidates):
        errors.append("core source reconciliation does not cover the exact candidate partition")
    return errors


def validate_extended_manifest(manifest, root):
    """Verify the exact remaining source partition without granting acceptance."""
    errors = []
    if not isinstance(manifest, dict) or any(
        not isinstance(manifest.get(key), expected)
        for key, expected in {
            "operations": list,
            "source_mappings": list,
            "unresolved_candidates": list,
            "payload_schemas": dict,
            "source_manifest": dict,
        }.items()
    ):
        return ["invalid extended manifest structure"]
    scope = {f"C{number:02d}" for number in range(11, 25)}
    if set(manifest.get("scope", [])) != scope:
        errors.append("extended scope cannot omit a capability")
    if (
        manifest.get("approval_state") != "draft_for_review"
        or manifest.get("runtime_qualification") != "not_run"
    ):
        errors.append("extended draft cannot invent acceptance or qualification")
    sources = manifest["source_manifest"]
    for path, expected_hash in sources.items():
        candidate = root / path
        if (
            not candidate.is_file()
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_hash
        ):
            errors.append(f"extended source drift: {path}")
    expected = {
        (row["operation_id"], row["provider_profile"])
        for row in expected_rows(root)
        if row["capability_id"] in scope
    }
    partition = []
    identities, tool_ids, wire_names = set(), set(), set()
    schemas = manifest["payload_schemas"]
    jsonschema = importlib.import_module("jsonschema")
    validator = jsonschema.Draft202012Validator
    for name, schema in schemas.items():
        if not isinstance(schema, dict) or _schema_has_unconstrained_node(schema):
            errors.append(f"unbounded extended payload: {name}")
            continue
        try:
            validator.check_schema(schema)
        except jsonschema.exceptions.SchemaError:
            errors.append(f"invalid extended payload: {name}")
        if manifest.get("payload_schema_sha256", {}).get(name) != digest(schema):
            errors.append(f"extended payload digest drift: {name}")
    local_specs = tool_specs(root)
    for operation in manifest["operations"]:
        if not isinstance(operation, dict):
            errors.append("invalid extended operation")
            continue
        for field, seen in (
            ("operation_id", identities),
            ("proposed_tool_id", tool_ids),
            ("mcp_name", wire_names),
        ):
            value = operation.get(field)
            if not isinstance(value, str) or not value or value in seen:
                errors.append(f"invalid or duplicate extended {field}")
            else:
                seen.add(value)
        if (
            operation.get("approval_state") != "draft_for_review"
            or operation.get("implementation_state") != "adapter_required"
            or operation.get("acceptance_receipt") is not None
        ):
            errors.append("extended operation cannot invent approval or implementation")
        if operation.get("capability_id") not in scope:
            errors.append("invalid extended operation family")
        anchors = operation.get("source_entrypoints")
        if (
            not isinstance(anchors, list)
            or not anchors
            or any(
                not _source_anchor_exists(anchor, root) or anchor.split(":", 1)[0] not in sources
                for anchor in anchors
            )
        ):
            errors.append(f"invalid extended source anchors: {operation.get('operation_id')}")
        local_id = operation.get("local_tool_id")
        if local_id is not None:
            version = operation.get("local_contract_version")
            local = (
                local_specs.get((local_id, version))
                if isinstance(local_id, str) and type(version) is int
                else None
            )
            if local is None:
                errors.append("unknown extended canonical Tool identity")
            else:
                for direction in ("input", "output"):
                    if operation.get(f"local_{direction}_schema_digest") != digest(
                        getattr(local[0], f"{direction}_schema")
                    ):
                        errors.append("extended local Tool schema identity drift")
        for key in ("parameters_schema", "result_schema"):
            schema = operation.get(key)
            if not isinstance(schema, dict) or _schema_has_unconstrained_node(schema):
                errors.append(f"unbounded extended schema: {operation.get('operation_id')}: {key}")
                continue
            try:
                validator.check_schema(schema)
            except jsonschema.exceptions.SchemaError:
                errors.append(f"invalid extended schema: {operation.get('operation_id')}: {key}")
            if operation.get(key + "_sha256") != digest(schema):
                errors.append("extended operation schema digest drift")
        for reference in operation.get("payload_contracts", {}).values():
            if not isinstance(reference, str) or reference not in schemas:
                errors.append("unresolved extended payload contract")
    profiles = manifest.get("provider_profiles", [])
    declared_profiles = {}
    if not isinstance(profiles, list):
        errors.append("invalid extended provider profiles")
        profiles = []
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("profile_id"), str):
            errors.append("invalid extended provider profile")
            continue
        name = profile["profile_id"]
        if name in declared_profiles:
            errors.append("duplicate extended provider profile")
        declared_profiles[name] = profile
        if (
            profile.get("enabled_by_default") is not False
            or profile.get("provisioning_state") != "not_verified"
            or profile.get("qualification_receipt") is not None
            or profile.get("final_coverage_obligation") is not True
        ):
            errors.append("extended profile cannot invent readiness or waive coverage")
        anchors = profile.get("source_entrypoints")
        if (
            not isinstance(anchors, list)
            or not anchors
            or any(
                not _source_anchor_exists(anchor, root) or anchor.split(":", 1)[0] not in sources
                for anchor in anchors
            )
        ):
            errors.append("invalid extended profile source")
        asset = profile.get("asset_manifest")
        if asset is not None and (
            not isinstance(asset, dict) or sources.get(asset.get("path")) != asset.get("sha256")
        ):
            errors.append("extended profile asset lock drift")
    actual_pairs = set()
    for operation in manifest["operations"]:
        if not isinstance(operation, dict):
            continue
        for binding in operation.get("provider_profile_bindings", []):
            if not isinstance(binding, dict) or not isinstance(binding.get("profile_id"), str):
                errors.append("invalid extended profile binding")
                continue
            name = binding["profile_id"]
            profile = declared_profiles.get(name)
            pair = (operation.get("operation_id"), name)
            if (
                pair in actual_pairs
                or profile is None
                or operation.get("operation_id") not in profile.get("operation_ids", [])
            ):
                errors.append("extended operation/profile pairing mismatch")
            actual_pairs.add(pair)
            if (
                not isinstance(binding.get("modes"), list)
                or not binding["modes"]
                or any(not isinstance(mode, str) or not mode for mode in binding["modes"])
            ):
                errors.append("missing extended provider modes")
    for name, profile in declared_profiles.items():
        if (
            not isinstance(profile.get("operation_ids"), list)
            or not profile["operation_ids"]
            or any((identity, name) not in actual_pairs for identity in profile["operation_ids"])
        ):
            errors.append("unbound extended provider operation")
    for row in manifest["source_mappings"]:
        if not isinstance(row, dict):
            errors.append("invalid extended source mapping")
            continue
        key = row.get("candidate_operation_id"), row.get("candidate_provider_profile")
        partition.append(key)
        targets = row.get("enclosing_operations")
        if (
            not isinstance(targets, list)
            or not targets
            or any(target not in identities for target in targets)
        ):
            errors.append("unknown extended enclosing operation")
        if (
            not row.get("classification_rationale")
            or row.get("review_state") != "source_reconciled_not_human_approved"
        ):
            errors.append("unexplained extended source mapping")
    for row in manifest["unresolved_candidates"]:
        if not isinstance(row, dict):
            errors.append("invalid extended unresolved candidate")
            continue
        partition.append((row.get("operation_id"), row.get("provider_profile")))
    if len(set(partition)) != len(partition) or set(partition) != expected:
        errors.append("extended candidate/provider partition missing, duplicate or foreign")
    return errors


def validate(data, root, mode, receipts):
    """Return diagnostics; planned success is structural, never qualification."""
    errors = []
    if (
        not isinstance(data, dict)
        or type(data.get("format_version")) is not int
        or data["format_version"] != 1
        or not isinstance(data.get("operations"), list)
    ):
        return ["invalid inventory: format_version=1 and operations array required"]
    if data.get("source_manifest") != source_manifest(root):
        errors.append(
            "source manifest drift: public API/CLI/Tool/provider/plugin declarations changed"
        )
    core_path = "docs/development/remote-mcp-contracts-core.json"
    core_file = root / core_path
    core_mappings = {}
    try:
        core = json.loads(core_file.read_text())
        expected_link = {
            "path": core_path,
            "sha256": hashlib.sha256(core_file.read_bytes()).hexdigest(),
        }
        if data.get("core_contract_manifest") != expected_link:
            errors.append("core contract manifest linkage missing or changed")
        errors.extend(validate_core_manifest(core, root))
        if mode in {"planned", "qualified"}:
            for operation in core.get("operations", []):
                if operation.get("semantic_audit", {}).get("state") != "source_fields_projected":
                    errors.append(
                        f"core semantic contract incomplete: {operation.get('operation_id')}"
                    )
        core_mappings = {
            (mapping["candidate_operation_id"], mapping["candidate_provider_profile"]): mapping
            for mapping in core["source_mappings"]
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"core contract manifest unavailable or invalid: {error}")
    extended_path = "docs/development/remote-mcp-contracts-extended.json"
    extended_file = root / extended_path
    extended_mappings = {}
    try:
        extended = json.loads(extended_file.read_text())
        if data.get("extended_contract_manifest") != {
            "path": extended_path,
            "sha256": hashlib.sha256(extended_file.read_bytes()).hexdigest(),
        }:
            errors.append("extended contract manifest linkage missing or changed")
        errors.extend(validate_extended_manifest(extended, root))
        extended_mappings = {
            (row["candidate_operation_id"], row["candidate_provider_profile"]): row
            for row in extended["source_mappings"]
        }
        if mode in {"planned", "qualified"}:
            if extended.get("unresolved_candidates"):
                errors.append("extended source reconciliation incomplete")
            for operation in extended["operations"]:
                audit = operation.get("semantic_audit", {})
                if audit.get("state") != "source_fields_projected" or not audit.get(
                    "field_projection"
                ):
                    errors.append(
                        f"extended semantic contract incomplete: {operation.get('operation_id')}"
                    )
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"extended contract manifest unavailable or invalid: {error}")
    expected = {
        (item["operation_id"], item["provider_profile"]): item for item in expected_rows(root)
    }
    submitted = {
        (item["operation_id"], item["provider_profile"]): item
        for item in data["operations"]
        if isinstance(item, dict)
        and isinstance(item.get("operation_id"), str)
        and isinstance(item.get("provider_profile"), str)
    }
    seen = set()
    for item in data["operations"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("operation_id"), str)
            or not isinstance(item.get("provider_profile"), str)
        ):
            errors.append("invalid operation/provider row")
            continue
        key = item["operation_id"], item["provider_profile"]
        if key in seen:
            errors.append(f"duplicate operation/provider: {key}")
        seen.add(key)
        if item.get("contract_mapping") != core_mappings.get(key):
            errors.append(f"core source mapping linkage drift: {key}")
        if item.get("extended_contract_mapping") != extended_mappings.get(key):
            errors.append(f"extended source mapping linkage drift: {key}")
        reference = expected.get(key)
        if reference is None:
            errors.append(f"unrecognized operation/provider: {key}")
            continue
        # Source-derived candidates remain frozen; separately reviewed planning
        # mappings can evolve without rewriting discovery evidence or this code.
        for field, value in reference.items():
            if field == "planning":
                continue
            if field not in item or type(item[field]) is not type(value) or item[field] != value:
                errors.append(f"source/schema/policy drift: {key}: {field}")
        if mode in {"planned", "qualified"}:
            errors.extend(planning_errors(item, submitted))
        if mode == "qualified":
            errors.append(f"not qualified: {key}: {item.get('blocking_reason')}")
    for key in sorted(set(expected) - seen):
        errors.append(f"missing operation/provider: {key}")
    if mode not in {"inventoried", "planned", "qualified"}:
        errors.append("mode must be inventoried, planned or qualified")
    if mode == "qualified":
        errors.append(
            "Qualification evidence verifier is not configured; receipt JSON is not trusted runtime proof."
        )
        if not isinstance(receipts, dict) or not receipts.get("receipts"):
            errors.append("missing qualification receipts")
    return errors


def main():
    """Run the explicit planned or fail-closed qualification gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--mode", choices=("inventoried", "planned", "qualified"), required=True)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.inventory.read_text())
        receipts = json.loads(args.receipts.read_text()) if args.receipts else None
        errors = validate(data, ROOT, args.mode, receipts)
    except (OSError, ValueError, TypeError, ImportError) as error:
        errors = [f"coverage discovery failed: {error}"]
    if errors:
        print(json.dumps({"status": "blocked", "mode": args.mode, "errors": errors}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "discovery_consistent"
                if args.mode == "inventoried"
                else "planning_mappings_complete",
                "g0_complete": False,
                "mode": args.mode,
                "qualified": False,
                "operations": len(data["operations"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
