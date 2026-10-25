"""Immutable function definitions and calls, independent of agent execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast


def _name(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("invalid function name")


@dataclass(frozen=True, slots=True)
class VlmTool:
    """A function schema stored as JSON so callers cannot mutate its identity."""

    name: str
    parameters_json: str
    description: str = ""
    strict: bool | None = None

    def __post_init__(self) -> None:
        from capicove.model.vlm import parse_json

        _name(self.name)
        if not isinstance(self.parameters_json, str) or len(self.parameters_json.encode()) > 65536:
            raise ValueError("tool schema exceeds 64 KiB")
        schema = parse_json(self.parameters_json.encode())
        if schema.get("type") != "object":
            raise ValueError("function parameters must be an object schema")
        if not isinstance(self.description, str) or len(self.description.encode()) > 8192:
            raise ValueError("tool description exceeds 8 KiB")
        if self.strict is not None and type(self.strict) is not bool:
            raise ValueError("invalid strict option")

    def to_openai(self) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": self.name,
            "parameters": json.loads(self.parameters_json),
        }
        if self.description:
            function["description"] = self.description
        if self.strict is not None:
            function["strict"] = self.strict
        return {"type": "function", "function": function}

    @classmethod
    def from_openai(cls, raw: Any) -> VlmTool:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"type", "function"}
            or raw["type"] != "function"
        ):
            raise ValueError("expected function tool")
        function = raw["function"]
        if not isinstance(function, dict) or set(function) - {
            "name",
            "parameters",
            "description",
            "strict",
        }:
            raise ValueError("invalid function fields")
        name = function.get("name")
        _name(name)
        return cls(
            cast(str, name),
            json.dumps(
                function.get("parameters", {"type": "object", "properties": {}}), allow_nan=False
            ),
            function.get("description", ""),
            function.get("strict"),
        )


@dataclass(frozen=True, slots=True)
class VlmToolCall:
    """Model-proposed arguments; callers validate and authorize actual execution."""

    id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        _name(self.name)
        if not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", self.id):
            raise ValueError("invalid tool call id")
        if not isinstance(self.arguments, str) or len(self.arguments.encode()) > 65536:
            raise ValueError("tool arguments exceed 64 KiB")

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_openai(cls, raw: Any) -> VlmToolCall:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"id", "type", "function"}
            or raw["type"] != "function"
        ):
            raise ValueError("invalid tool call")
        function = raw["function"]
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise ValueError("invalid tool call function")
        return cls(raw["id"], function["name"], function["arguments"])


@dataclass(frozen=True, slots=True)
class VlmToolChoice:
    """Select a named function, including names such as auto or required."""

    name: str

    def __post_init__(self) -> None:
        _name(self.name)

    def to_openai(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name}}
