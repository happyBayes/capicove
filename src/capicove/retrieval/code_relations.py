"""Bounded, build-time code relation facts for S4 (RFC-0033, ``code-relations-v1``).

Only containment, repository-internal imports and direct calls whose target is
unambiguous under this versioned resolver become ``resolved`` edges. Calls through
dynamic dispatch, shadowed names, builtins, external packages or names defined in
several unimported places stay ``unresolved``, ``external`` or ``ambiguous``: name
similarity alone never creates a verified call edge. No project code is executed.
"""

from __future__ import annotations

import builtins
import hashlib
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from capicove.retrieval.code_ast import (
    AST_METHOD,
    ParseStatus,
    ast_language,
    parse_source,
    parse_tree,
)


RELATIONS_POLICY_VERSION = "code-relations-v1"
MAX_EDGES = 100_000
MAX_METADATA_BYTES = 32 * 1024 * 1024
_PYTHON_BUILTINS = frozenset(dir(builtins))
_JS_GLOBALS = frozenset(
    {
        "console",
        "Promise",
        "JSON",
        "Math",
        "Object",
        "Array",
        "String",
        "Number",
        "Date",
        "setTimeout",
        "clearTimeout",
        "parseInt",
        "parseFloat",
        "require",
        "fetch",
        "Error",
    }
)
_SCRIPT_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


class RelationBudgetError(ValueError):
    """Edge or metadata budget exceeded; publication must fail without dropping facts."""


@dataclass(frozen=True, slots=True)
class Relation:
    """One relation fact with its exact source site and resolution status."""

    kind: str
    source_path: str
    source_symbol: str
    target: str | None
    status: str
    source_line: int
    start_char: int
    end_char: int
    source_sha256: str
    method: str = AST_METHOD
    policy: str = RELATIONS_POLICY_VERSION

    @property
    def target_path(self) -> str | None:
        """Return the target file of a resolved symbol or import edge."""
        if self.status != "resolved" or self.target is None:
            return None
        return self.target.split(":", 1)[0]

    @property
    def target_symbol(self) -> str | None:
        """Return the qualified target symbol of a resolved symbol edge."""
        if self.status != "resolved" or self.target is None or ":" not in self.target:
            return None
        return self.target.split(":", 1)[1]

    def payload(self) -> dict[str, Any]:
        """Return the bounded JSON projection stored with the generation."""
        return {
            "kind": self.kind,
            "source_path": self.source_path,
            "source_symbol": self.source_symbol,
            "target": self.target,
            "status": self.status,
            "source_line": self.source_line,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source_sha256": self.source_sha256,
            "method": self.method,
            "policy": self.policy,
        }


class _Offsets:
    def __init__(self, text: str) -> None:
        self._starts: list[int] = []
        position = 0
        for character in text:
            self._starts.append(position)
            position += len(character.encode("utf-8"))
        self._starts.append(position)
        self._index = {byte: char for char, byte in enumerate(self._starts)}

    def char(self, byte: int) -> int:
        return self._index[byte]


def _text(node: Any, payload: bytes) -> str:
    return payload[node.start_byte : node.end_byte].decode("utf-8")


def _module_file(module: str, files: Mapping[str, Any]) -> str | None:
    base = module.replace(".", "/")
    for candidate in (f"{base}.py", f"{base}/__init__.py"):
        if candidate in files:
            return candidate
    return None


def _script_file(importer: str, specifier: str, files: Mapping[str, Any]) -> str | None:
    if not specifier.startswith("."):
        return None
    base = posixpath.normpath(posixpath.join(posixpath.dirname(importer), specifier))
    for candidate in [
        base,
        *(base + suffix for suffix in _SCRIPT_SUFFIXES),
        *(f"{base}/index{suffix}" for suffix in _SCRIPT_SUFFIXES),
    ]:
        if candidate in files:
            return candidate
    return None


@dataclass
class _FileFacts:
    language: str
    text: str
    sha256: str
    symbols: dict[str, Any]
    top_level: set[str]
    tree: Any


def extract_relations(
    files: Mapping[str, tuple[str, str]], *, max_edges: int = MAX_EDGES
) -> tuple[Relation, ...]:
    """Extract relations from ``{relative_path: (language, text)}``; deterministic order."""
    facts: dict[str, _FileFacts] = {}
    for path, (language, text) in sorted(files.items()):
        grammar = ast_language(path)
        outcome = parse_source(grammar, text)
        tree = (
            parse_tree(grammar, text.encode("utf-8"))
            if outcome.status is ParseStatus.PARSED
            else None
        )
        if tree is None:
            continue
        symbols = {symbol.qualified_name: symbol for symbol in outcome.symbols}
        facts[path] = _FileFacts(
            language,
            text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            symbols,
            {s.qualified_name for s in outcome.symbols if s.parent is None},
            tree,
        )
    names: dict[str, list[str]] = {}
    for path, item in facts.items():
        for qualified in item.symbols:
            names.setdefault(qualified.rsplit(".", 1)[-1], []).append(f"{path}:{qualified}")
    relations: list[Relation] = []
    for path, item in facts.items():
        offsets = _Offsets(item.text)
        payload = item.text.encode("utf-8")
        for qualified, symbol in item.symbols.items():
            if symbol.parent is not None:
                relations.append(
                    Relation(
                        "contains",
                        path,
                        symbol.parent,
                        f"{path}:{qualified}",
                        "resolved",
                        symbol.start_line,
                        symbol.start_char,
                        symbol.end_char,
                        item.sha256,
                    )
                )
        if item.language == "python":
            bindings = _python_imports(path, item, payload, offsets, facts, relations)
            _python_calls(path, item, payload, offsets, facts, bindings, names, relations)
        else:
            bindings = _script_imports(path, item, payload, offsets, facts, relations)
            _script_calls(path, item, payload, offsets, bindings, relations)
        if len(relations) > max_edges:
            raise RelationBudgetError("relation edges exceed the generation budget")
    total = sum(len(repr(relation.payload())) for relation in relations)
    if total > MAX_METADATA_BYTES:
        raise RelationBudgetError("relation metadata exceeds the generation budget")
    return tuple(
        sorted(relations, key=lambda r: (r.source_path, r.start_char, r.kind, r.target or ""))
    )


def _site(node: Any, offsets: _Offsets) -> tuple[int, int, int]:
    return node.start_point[0] + 1, offsets.char(node.start_byte), offsets.char(node.end_byte)


def _python_imports(
    path: str,
    item: _FileFacts,
    payload: bytes,
    offsets: _Offsets,
    facts: Mapping[str, _FileFacts],
    relations: list[Relation],
) -> dict[str, tuple[str, str | None]]:
    """Return name -> (kind, target); kind is module, symbol or external."""
    bindings: dict[str, tuple[str, str | None]] = {}
    for node in item.tree.root_node.named_children:
        line, start, end = _site(node, offsets)
        if node.type == "import_statement":
            for child in node.named_children:
                module = _text(child.child_by_field_name("name") or child, payload)
                alias_node = child.child_by_field_name("alias")
                target = _module_file(module, facts)
                status = "resolved" if target else "external"
                relations.append(
                    Relation(
                        "imports", path, "", target or module, status, line, start, end, item.sha256
                    )
                )
                bound = (
                    _text(alias_node, payload) if alias_node is not None else module.split(".")[0]
                )
                bindings[bound] = ("module", target) if target else ("external", module)
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            module = _text(module_node, payload) if module_node is not None else ""
            module_target = _module_file(module, facts)
            imported_any = False
            for child in node.named_children:
                if child == module_node or child.type not in {"dotted_name", "aliased_import"}:
                    continue
                name_node = (
                    child.child_by_field_name("name") if child.type == "aliased_import" else child
                )
                alias_node = (
                    child.child_by_field_name("alias") if child.type == "aliased_import" else None
                )
                name = _text(name_node, payload)
                bound = _text(alias_node, payload) if alias_node is not None else name
                submodule = _module_file(f"{module}.{name}", facts)
                if submodule is not None:
                    relations.append(
                        Relation(
                            "imports",
                            path,
                            "",
                            submodule,
                            "resolved",
                            line,
                            start,
                            end,
                            item.sha256,
                        )
                    )
                    bindings[bound] = ("module", submodule)
                    imported_any = True
                elif module_target is not None and name in facts[module_target].top_level:
                    bindings[bound] = ("symbol", f"{module_target}:{name}")
                elif module_target is None:
                    bindings[bound] = ("external", f"{module}.{name}")
            if module_target is not None and not imported_any:
                relations.append(
                    Relation(
                        "imports",
                        path,
                        "",
                        module_target,
                        "resolved",
                        line,
                        start,
                        end,
                        item.sha256,
                    )
                )
            elif module_target is None and not imported_any:
                relations.append(
                    Relation("imports", path, "", module, "external", line, start, end, item.sha256)
                )
    return bindings


def _enclosing(item: _FileFacts, start_char: int) -> str | None:
    best = None
    for qualified, symbol in item.symbols.items():
        if (
            symbol.kind in {"function", "method"}
            and symbol.start_char <= start_char < symbol.end_char
        ):
            if best is None or symbol.start_char >= item.symbols[best].start_char:
                best = qualified
    return best


def _local_names(node: Any, payload: bytes) -> set[str]:
    """Names assigned or bound as parameters inside one function body (shadowing)."""
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in {"assignment", "augmented_assignment"}:
            left = current.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                found.add(_text(left, payload))
        if current.type == "parameters":
            for child in current.named_children:
                if child.type == "identifier":
                    found.add(_text(child, payload))
        if current.type in {"function_definition", "class_definition"} and current is not node:
            continue
        stack.extend(current.named_children)
    return found


def _python_calls(
    path: str,
    item: _FileFacts,
    payload: bytes,
    offsets: _Offsets,
    facts: Mapping[str, _FileFacts],
    bindings: Mapping[str, tuple[str, str | None]],
    names: Mapping[str, list[str]],
    relations: list[Relation],
) -> None:
    stack = [item.tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call":
            line, start, end = _site(node, offsets)
            caller = _enclosing(item, start)
            if caller is not None:
                target, status = _resolve_python_call(
                    node.child_by_field_name("function"),
                    payload,
                    path,
                    item,
                    caller,
                    facts,
                    bindings,
                    names,
                    _local_names(_function_node(item, caller, offsets), payload),
                )
                relations.append(
                    Relation("calls", path, caller, target, status, line, start, end, item.sha256)
                )
        stack.extend(reversed(node.named_children))


def _function_node(item: _FileFacts, qualified: str, offsets: _Offsets) -> Any:
    """Return the definition node of one qualified function, or the module root."""
    symbol = item.symbols[qualified]
    stack = [item.tree.root_node]
    while stack:
        node = stack.pop()
        if (
            node.type == "function_definition"
            and offsets.char(node.start_byte) >= symbol.start_char
            and offsets.char(node.end_byte) <= symbol.end_char
        ):
            name = node.child_by_field_name("name")
            if name is not None and name.text is not None and name.text.decode() == symbol.name:
                return node
        stack.extend(node.named_children)
    return item.tree.root_node


def _resolve_python_call(
    function: Any,
    payload: bytes,
    path: str,
    item: _FileFacts,
    caller: str,
    facts: Mapping[str, _FileFacts],
    bindings: Mapping[str, tuple[str, str | None]],
    names: Mapping[str, list[str]],
    local: set[str],
) -> tuple[str | None, str]:
    if function is None:
        return None, "unresolved"
    if function.type == "identifier":
        name = _text(function, payload)
        if name in local:
            return name, "unresolved"
        if name in item.top_level:
            return f"{path}:{name}", "resolved"
        if name in bindings:
            kind, target = bindings[name]
            if kind == "symbol" and target is not None:
                return target, "resolved"
            return (target or name), "external" if kind == "external" else "unresolved"
        if name in _PYTHON_BUILTINS:
            return name, "external"
        return name, "ambiguous" if len(names.get(name, ())) > 1 else "unresolved"
    if function.type == "attribute":
        owner = function.child_by_field_name("object")
        attribute = function.child_by_field_name("attribute")
        if owner is None or attribute is None:
            return None, "unresolved"
        member = _text(attribute, payload)
        dotted = _text(function, payload)
        if owner.type == "identifier":
            owner_name = _text(owner, payload)
            if owner_name == "self":
                scope = caller.rsplit(".", 1)[0] if "." in caller else None
                if scope is not None and f"{scope}.{member}" in item.symbols:
                    return f"{path}:{scope}.{member}", "resolved"
                return dotted, "unresolved"
            if owner_name in local:
                return dotted, "unresolved"
            binding = bindings.get(owner_name)
            if binding is not None:
                kind, target = binding
                if kind == "module" and target is not None:
                    if member in facts[target].top_level:
                        return f"{target}:{member}", "resolved"
                    return dotted, "unresolved"
                if kind == "external":
                    return dotted, "external"
        if owner.type == "attribute" and owner.child_by_field_name("object") is not None:
            root = owner
            while root.type == "attribute" and root.child_by_field_name("object") is not None:
                root = root.child_by_field_name("object")
            if (
                root.type == "identifier"
                and bindings.get(_text(root, payload), ("", None))[0] == "external"
            ):
                return dotted, "external"
        return dotted, "unresolved"
    return None, "unresolved"


def _script_imports(
    path: str,
    item: _FileFacts,
    payload: bytes,
    offsets: _Offsets,
    facts: Mapping[str, _FileFacts],
    relations: list[Relation],
) -> dict[str, tuple[str, str | None]]:
    bindings: dict[str, tuple[str, str | None]] = {}
    for node in item.tree.root_node.named_children:
        if node.type != "import_statement":
            continue
        source = node.child_by_field_name("source")
        if source is None:
            continue
        specifier = _text(source, payload).strip("'\"")
        target = _script_file(path, specifier, facts)
        line, start, end = _site(node, offsets)
        relations.append(
            Relation(
                "imports",
                path,
                "",
                target or specifier,
                "resolved" if target else "external",
                line,
                start,
                end,
                item.sha256,
            )
        )
        clause = next(
            (child for child in node.named_children if child.type == "import_clause"), None
        )
        if clause is None:
            continue
        stack = [clause]
        while stack:
            current = stack.pop()
            if current.type == "import_specifier":
                name_node = current.child_by_field_name("name")
                alias_node = current.child_by_field_name("alias")
                if name_node is None:
                    continue
                name = _text(name_node, payload)
                bound = _text(alias_node, payload) if alias_node is not None else name
                if target is not None and name in facts[target].top_level:
                    bindings[bound] = ("symbol", f"{target}:{name}")
                else:
                    bindings[bound] = ("external", f"{specifier}.{name}")
                continue
            stack.extend(current.named_children)
    return bindings


def _script_calls(
    path: str,
    item: _FileFacts,
    payload: bytes,
    offsets: _Offsets,
    bindings: Mapping[str, tuple[str, str | None]],
    relations: list[Relation],
) -> None:
    stack = [item.tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call_expression":
            function = node.child_by_field_name("function")
            line, start, end = _site(node, offsets)
            caller = _enclosing(item, start)
            if caller is not None and function is not None:
                if function.type == "identifier":
                    name = _text(function, payload)
                    if name in item.top_level:
                        target, status = f"{path}:{name}", "resolved"
                    elif name in bindings:
                        kind, bound = bindings[name]
                        target, status = (
                            (bound, "resolved")
                            if kind == "symbol" and bound is not None
                            else (name, "external")
                        )
                    elif name in _JS_GLOBALS:
                        target, status = name, "external"
                    else:
                        target, status = name, "unresolved"
                else:
                    target, status = _text(function, payload)[:256], "unresolved"
                relations.append(
                    Relation("calls", path, caller, target, status, line, start, end, item.sha256)
                )
        stack.extend(reversed(node.named_children))


__all__ = [
    "MAX_EDGES",
    "MAX_METADATA_BYTES",
    "RELATIONS_POLICY_VERSION",
    "Relation",
    "RelationBudgetError",
    "extract_relations",
]
