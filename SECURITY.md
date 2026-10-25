# Security Policy

Capicove processes untrusted local and remote content and may integrate with
external models, storage systems, and agents. Security and privacy reports are
treated as first-class engineering work.

## Supported versions

Security fixes target `main` and the latest published 0.1.x patch release.
The 0.1.x support entry takes effect when 0.1.0 is published.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Latest published `0.1.x` | Yes, after first publication |
| Unreleased snapshots and forks | No guarantee |

## Reporting a vulnerability

Do not create an issue, discussion, pull request, or public disclosure for a
suspected vulnerability.

Use [GitHub private vulnerability reporting](https://github.com/happyBayes/capicove/security/advisories/new).
Maintainers must enable and verify this route before making the repository public.
Until that gate is complete, authorized collaborators should contact a repository
administrator through the existing private collaboration channel. Do not include
vulnerability details in a public request for a reporting channel.

Include, when available:

- the affected commit or version;
- the affected component or plugin;
- reproduction steps or a minimal proof of concept;
- expected and observed impact;
- whether sensitive user data or credentials may be exposed;
- suggested mitigations;
- any planned disclosure date.

Remove personal data, production credentials, proprietary source documents, model
weights, and unrelated secrets from the report.

## Response targets

- Acknowledge receipt within three business days.
- Provide an initial severity assessment within ten business days.
- Share remediation progress at least every fourteen days while the report remains
  active.
- Aim for coordinated remediation and disclosure within ninety days, adjusted for
  severity and ecosystem impact.

These are response targets, not guarantees.

## Security-sensitive areas

Reports are especially valuable for arbitrary code execution, path traversal,
unsafe archive extraction, server-side request forgery, credential leakage,
authorization bypass, malicious document parsing, dependency confusion, plugin
isolation failures, provenance forgery, privacy-policy bypass, and unintended
network or telemetry behavior.

## Safe harbor

Good-faith research that avoids privacy violations, data destruction, service
degradation, social engineering, and unauthorized access beyond what is necessary
to demonstrate the issue will be handled constructively. Give maintainers a
reasonable opportunity to remediate before disclosure.
