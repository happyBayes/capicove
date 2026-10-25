"""Mechanical reverse dependencies with explicit bounded follow-up work."""

from dataclasses import dataclass

from capicove.wiki.maintenance.discovery import DiscoveryPage


@dataclass(frozen=True, slots=True)
class ImpactResult:
    """Known affected pages, without claiming semantic closure."""

    page_ids: tuple[str, ...]
    follow_up_page_ids: tuple[str, ...]
    coverage: str


def affected_pages(
    pages: tuple[DiscoveryPage, ...],
    *,
    source_ids: tuple[str, ...] = (),
    page_ids: tuple[str, ...] = (),
    claim_ids: tuple[str, ...] = (),
) -> ImpactResult:
    """Traverse known backlinks, shared sources/claims and identical aliases."""
    known = {p.page_id for p in pages}
    if not set(page_ids) <= known:
        raise ValueError("citation_out_of_scope")
    affected = set(page_ids)
    changed = True
    while changed:
        aliases = {
            a.casefold() for p in pages if p.page_id in affected for a in (p.title, *p.aliases)
        }
        additions = {
            p.page_id
            for p in pages
            if set(p.sources) & set(source_ids)
            or set(p.claim_ids) & set(claim_ids)
            or set(p.links) & affected
            or aliases & {a.casefold() for a in (p.title, *p.aliases)}
        }
        changed = not additions <= affected
        affected.update(additions)
    ordered = tuple(sorted(affected))
    if len(ordered) > 100:
        ordered = tuple(sorted(set(page_ids))) + tuple(sorted(affected - set(page_ids)))
    return ImpactResult(
        ordered[:100], ordered[100:], "partial" if len(ordered) > 100 else "complete_inventory"
    )
