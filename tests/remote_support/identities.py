"""Synthetic identities for cross-tenant/application/delegated-user tests."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FixtureSubject:
    """Represent a synthetic caller before binding a real token verifier."""

    issuer: str
    tenant: str
    application: str
    kind: str
    subject: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return self.issuer, self.tenant, self.application, self.kind, self.subject


def principal_matrix() -> tuple[FixtureSubject, ...]:
    """Include two apps per tenant, same-app users and distinct service callers."""
    return tuple(
        FixtureSubject(f"https://{tenant}.identity.test", tenant, app, kind, subject)
        for tenant in ("tenant-a", "tenant-b")
        for app in ("app-a", "app-b")
        for kind, subject in (("human", "alice"), ("human", "bob"), ("service", "worker"))
    )
