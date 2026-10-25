from pathlib import Path

import pytest


RFC = Path("docs/rfcs/0002-multimodal-retrieval-contracts.md")
ADR = Path("docs/adr/0020-intent-routed-reverse-image-search.md")
ADR_INDEX = Path("docs/adr/README.md")
MODULE_BOUNDARIES = Path("docs/architecture/module-boundaries.md")
IMPLEMENTATION_PLAN = Path("docs/architecture/reverse-image-lifecycle-requirements.md")

pytestmark = pytest.mark.contract


def test_reverse_image_stage2_amendment_is_pending_and_complete() -> None:
    rfc = RFC.read_text(encoding="utf-8")

    assert "- Status: Accepted" in rfc
    assert "- Reverse-image Stage 2 amendment status: Pending review" in rfc
    assert "This amendment is not accepted" in rfc
    assert "does not authorize production implementation" in rfc
    assert "search_intent" in rfc
    assert all(intent in rfc for intent in ("instance", "concept", "balanced"))
    assert "HashSpaceRef" in rfc
    assert "GenerationLifecycleEvent" in rfc
    assert "manifest contains no mutable lifecycle state" in rfc
    assert "SearchFailureCode" in rfc
    assert "SearchWarningCode" in rfc
    assert "personal-lite" in rfc and "Stage 2" in rfc
    assert "enterprise-disk" in rfc and "Stage 3" in rfc


def test_reverse_image_stage2_security_boundary_is_explicit() -> None:
    rfc = RFC.read_text(encoding="utf-8")
    boundaries = MODULE_BOUNDARIES.read_text(encoding="utf-8")

    for text in (rfc, boundaries):
        assert "InvocationAuthorizationContext" in text
        assert "cannot grant or widen" in text
        assert "descriptor-relative" in text
        assert "no-follow" in text
        assert "same opened object" in text
    assert "request or transport parsing cannot" in rfc
    assert "calling Agent selects retrieval intent" in boundaries
    assert "server-side intent classifier" in boundaries


def test_reverse_image_adr_is_proposed_and_indexed_as_proposed() -> None:
    assert ADR.exists()
    adr = ADR.read_text(encoding="utf-8")
    adr_index = ADR_INDEX.read_text(encoding="utf-8")

    assert "- Status: Proposed" in adr
    assert "- Status: Accepted" not in adr
    assert "personal-lite" in adr and "Stage 2" in adr
    assert "enterprise-disk" in adr and "Stage 3" in adr
    assert "InvocationAuthorizationContext" in adr
    assert "No production implementation or activation is authorized" in adr
    accepted, proposed = adr_index.split("## Proposed decisions", maxsplit=1)
    assert "0020-intent-routed-reverse-image-search.md" not in accepted
    assert "0020-intent-routed-reverse-image-search.md" in proposed


def test_reverse_image_plan_closes_lifecycle_and_source_races() -> None:
    plan = IMPLEMENTATION_PLAN.read_text(encoding="utf-8")

    assert "Prepare the Intent and Generation Contract Revision for Review" in plan
    assert "ADR-0020 remains Proposed" in plan
    assert "def renew(" in plan
    assert "expected_expires_at" in plan
    assert "owner_session_epoch" in plan
    assert "application_id" in plan
    assert "PRAGMA user_version" in plan
    assert "newer schema version" in plan
    assert "migration is transactional" in plan
    assert "rollback" in plan
    assert "InvocationAuthorizationContext" in plan
    assert "allowed_source_root_refs" in plan
    assert "AuthorizedImageHandle" in plan
    assert "open_authorized" in plan
    assert "descriptor-relative" in plan
    assert "no-follow" in plan
    assert "same opened object" in plan
