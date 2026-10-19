# Security

This is an authenticated portfolio prototype, not a production security certification. The corpus consists of public AWS documentation; credentials and private question history are still protected assets.

## Implemented controls

- Cognito authorization code flow with PKCE and one-use state; access tokens remain in browser memory.
- Registration credentials travel directly to Cognito. Email verification precedes conditional first-login membership creation.
- API Gateway JWT validation plus application checks for issuer, client, token type, expiry, scopes, active membership and ownership.
- Cognito lookup uses the verified token username; the returned profile subject must match the verified token subject before enrollment. Disabled or invalid memberships are not revived.
- History is tenant-and-user scoped. Source access is bound to an owned stored request and its allowed evidence, with fresh authorization before delivery/export.
- Private S3 artifacts, immutable versions, checksums, bounded payloads, trusted evidence metadata and allowlisted outbound provider/source routes.
- Durable idempotency, atomic quota/cost reservations and retained receipts for failed or uncertain operations. No automatic model retry after ambiguous completion.
- Browser text rendering, restricted source links, CSP and explicit loading/error states.

## Important boundaries

Frontend validation improves usability; backend/provider checks enforce trust. A valid token is not sufficient without authorized membership. A genuine source quote is not sufficient to establish semantic support for every claim.

Shared query and model-accounting limits are application controls, not a hard cap on all AWS or identity-provider charges. Public registration introduces account/email-abuse exposure. UI resend cooldown is not a complete server-side abuse defense.

Broader penetration testing, independent semantic prompt-injection review, capacity testing and complete native-component auditing remain open. Existing offline fixtures and small live samples do not establish general resistance to every attack.

## Running and reporting safely

Do not commit `.env` files, AWS credentials, keys, tokens, passwords, provider responses or raw user history. Use synthetic fixtures for development and tests. Live calls and infrastructure mutations are explicit opt-in operations and need your own authorized accounts and budget controls.

Do not post a discovered credential or detailed exploit publicly. If private vulnerability reporting is available on the repository, use it; otherwise request a private reporting channel without including sensitive details in a public issue.
