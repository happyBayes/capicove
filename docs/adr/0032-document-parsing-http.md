# ADR-0032: Reuse canonical document parsing behind the local OCR listener

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- RFC: [RFC-0028](../rfcs/0028-document-parsing-http.md)

Add a PDF route to the existing model HTTP listener rather than creating another
model-specific server or representation. Inject the public document parser port
and startup-owned profile identity. Discover the implementation through
`capicove.plugins` and obtain the declared profile version through the public
model installer protocol. Use the existing document JSON codec so downstream
consumers retain normalized geometry, reading order and immutable lineage.

Stage binary uploads in private server-selected temporary directories because
the existing parser accepts an authorized local source. Bound upload size,
selected page count and page dimensions, reject encryption, and share image/PDF
admission to avoid overlapping native model work. Keep PDF and Paddle runtime
imports optional. Reuse listener authentication and offline asset verification;
requests cannot expand source or model authority.

Preserve the existing backend's document eligibility checks and per-invocation
pipeline lifetime. This transport does not introduce scan support, advanced table
or formula models, persistent Book jobs, workspace authorization or a worker
sandbox. Native inference timeouts and process isolation remain separate work.


Refine the implementation so `documents.service` owns profile binding, PDF
inspection, budget checks and validated backend dispatch. It exposes typed
rejection and availability failures instead of interpreting all ValueErrors as
client errors. Keep `api.documents` as an APIRouter with typed wire models and
private upload ownership. The existing canonical codec remains authoritative;
wire schemas retain normalized box arrays. Preserve the old binding import path.
Cancelled native work drains before its upload or admission slot is released.
