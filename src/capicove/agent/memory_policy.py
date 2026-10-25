"""Agent admission policy, independent of transports and persistence."""


def validate_admission(sensitivity: str, pinned: bool) -> None:
    if sensitivity not in {"public", "internal", "confidential"}:
        raise ValueError("sensitivity_rejected")
    if type(pinned) is not bool:
        raise ValueError("invalid pin")


def request_error(error: Exception) -> str:
    """Translate known validation failures without disclosing caller/provider values."""
    messages = {
        "time must be explicit UTC": "invalid_request: use an ISO-8601 UTC timestamp with Z or +00:00",
        "invalid predicate domain": "invalid_request: predicate or subject type is not in the configured ontology",
        "invalid predicate range": "invalid_request: object kind/type does not match the predicate",
        "static predicate cannot have validity bounds": "invalid_request: static predicates require unbounded validity",
        "empty interval": "invalid_request: valid_to must be after valid_from",
        "sensitivity_rejected": "invalid_request: this sensitivity cannot be admitted",
        "invalid context budgets": "invalid_request: context budgets must be integers between 1 and 1000000",
    }
    return messages.get(str(error), "invalid_request")
