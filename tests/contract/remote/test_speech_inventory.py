"""Speech discovery records pending candidates without inventing qualification."""

from tests.contract.remote.test_operation_coverage import ROOT, checker


def test_speech_tools_have_audio_ownership_and_actual_mcp_names() -> None:
    module = checker()
    rows = {
        row["canonical_tool_id"]: row
        for row in module.expected_rows(ROOT)
        if row["canonical_tool_id"] in {"capicove.audio.synthesize", "capicove.audio.clone"}
    }
    assert set(rows) == {"capicove.audio.synthesize", "capicove.audio.clone"}
    for tool_id, row in rows.items():
        assert row["capability_id"] == "C05"
        assert row["mcp_name"] == tool_id.replace(".", "_")
        assert row["runtime_requirements"]["qualification"] == "not_run"
        assert row["acceptance_receipt"] is None
