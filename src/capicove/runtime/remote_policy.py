"""Pure permission intersection for server-resolved remote authority."""

from capicove.contracts.remote import AuthorityKey, AuthorizationDecision


def authorize(
    authority: AuthorityKey,
    *,
    required: frozenset[str],
    deployment: frozenset[str],
    application: frozenset[str],
    token: frozenset[str],
    user: frozenset[str],
    workspace: frozenset[str],
    resource: frozenset[str],
    policy_epoch: int,
    source_epoch: int,
) -> AuthorizationDecision:
    """Intersect all mandatory tiers without wildcard or missing-tier expansion.

    Callers resolve these tiers from verified tokens and server-owned grants.
    A service identity still requires an explicit user-tier ceiling selected by
    server policy; omission never implies unrestricted access.
    """
    tiers = (required, deployment, application, token, user, workspace, resource)
    if any(not isinstance(tier, frozenset) for tier in tiers):
        raise ValueError("Policy tiers must be immutable explicit permission sets")
    ceiling = deployment.intersection(application, token, user, workspace, resource)
    if not required.issubset(ceiling):
        raise PermissionError("Operation is not authorized")
    return AuthorizationDecision(authority, ceiling, policy_epoch, source_epoch)
