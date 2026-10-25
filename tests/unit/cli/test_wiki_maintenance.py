"""W4.03: every maintenance operation/permission and CLI zero-effect rejections."""

import json
from pathlib import Path

import pytest

from capicove.cli.app import main
from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


# Design section 10, independently restated: operation -> (permissions, side effect).
EXPECTED = {
    "capture": ({"wiki:write"}, "write"),
    "task": ({"wiki:read", "wiki:write"}, "write"),
    "context": ({"wiki:read", "wiki:write"}, "write"),
    "submit": ({"wiki:write"}, "write"),
    "preview": ({"wiki:read"}, "read"),
    "review": ({"wiki:review"}, "write"),
    "apply": ({"wiki:write", "wiki:publish"}, "write"),
    "status": ({"wiki:read"}, "read"),
    "recover": ({"wiki:publish"}, "write"),
    "withdraw": ({"wiki:publish"}, "write"),
    "answer-task": ({"wiki:read"}, "read"),
    "answer-submit": ({"wiki:read"}, "read"),
    "ask": ({"wiki:read"}, "read"),
}
# Compensation and follow-up are mutation-catalog extensions beyond section 10.
EXTENSIONS = {
    "follow-up": ({"wiki:read", "wiki:write"}, "write"),
    "rollback": ({"wiki:read", "wiki:write"}, "write"),
}


def test_operation_permission_and_catalog_table() -> None:
    from capicove.tools.wiki_maintenance import maintenance_operations, maintenance_specs

    assert set(maintenance_operations()) == set(EXPECTED) | set(EXTENSIONS)
    mutating = {spec.tool_id: spec for spec in maintenance_specs(enable_mutations=True)}
    for operation, (permissions, effect) in {**EXPECTED, **EXTENSIONS}.items():
        public = operation if operation in {"answer-task", "answer-submit", "ask"} else None
        spec = mutating["capicove.wiki." + (public or "maintenance-" + operation)]
        assert set(spec.required_permissions) == permissions, operation
        assert (spec.side_effect.value == "read") == (effect == "read"), operation
        assert not spec.network_access, operation
    default = {spec.tool_id for spec in maintenance_specs()}
    assert default == {
        tool_id for tool_id, spec in mutating.items() if spec.side_effect.value == "read"
    }


def _cli(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, dict[str, object]]:
    code = main(["wiki", "maintain", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_cli_rejections_have_zero_effect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service, _, _ = prepared(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("Outside the authorized roots.")
    requests = tmp_path / "requests"
    requests.mkdir()
    cases = {
        "unknown": ("status", {"unexpected": True}),
        "workspace": ("status", {"workspace_id": "other"}),
        "reviewer": ("preview", {"proposal_id": "proposal:" + "0" * 64, "reviewer_id": "admin"}),
        "escape": ("capture", {"source": str(outside)}),
        "oversized": ("capture", {"source": "x" * 70000}),
    }
    before = snapshot(service.wiki)
    for name, (operation, request) in cases.items():
        path = requests / f"{name}.json"
        path.write_text(json.dumps(request))
        code, value = _cli(capsys, operation, str(tmp_path), "--request", str(path))
        assert code == 2, name
        assert value.get("status") != "succeeded", name
        assert snapshot(service.wiki) == before, name
    duplicate = requests / "duplicate.json"
    duplicate.write_text('{"abort": false, "abort": true}')
    code, value = _cli(capsys, "recover", str(tmp_path), "--request", str(duplicate))
    assert code == 2 and value["error"]["code"] == "wiki_invalid_request"  # type: ignore[index]
    assert snapshot(service.wiki) == before
    outside.unlink()
