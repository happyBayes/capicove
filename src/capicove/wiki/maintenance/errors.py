"""Stable maintenance error vocabulary; no private paths in wire details."""

from enum import StrEnum


class MaintenanceErrorCode(StrEnum):
    """Errors shared by planned application and transport operations."""

    INVALID_SUBMISSION = "invalid_submission"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    EVIDENCE_DIGEST_MISMATCH = "evidence_digest_mismatch"
    CITATION_OUT_OF_SCOPE = "citation_out_of_scope"
    TASK_SUPERSEDED = "task_superseded"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    PROPOSAL_STALE = "proposal_stale"
    MANUAL_EDIT_CONFLICT = "manual_edit_conflict"
    REVIEW_REQUIRED = "review_required"
    REVIEW_STALE = "review_stale"
    IMPACT_INCOMPLETE = "impact_incomplete"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    PUBLICATION_PENDING = "publication_pending"
    RECOVERY_CONFLICT = "recovery_conflict"
    SCHEMA_UPGRADE_REQUIRED = "schema_upgrade_required"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    GENERATION_REJECTED = "generation_rejected"
    POLICY_CHANGED = "policy_changed"
