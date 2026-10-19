# Architecture

The system separates slow document preparation from online question answering. It uses a small AWS deployment with bounded local indexes, rather than an always-on managed graph or vector database.

## Data preparation

1. Fetch a configured allowlist of official AWS documentation pages.
2. Normalize text while preserving section, code and table boundaries; assign content-derived document and chunk identities.
3. Prepare pinned BGE-small-en-v1.5 ONNX assets and generate normalized 384-dimensional embeddings.
4. Build vector and BM25 baselines. Extract source-backed relation assertions for separate local Neo4j experiments.
5. Validate row alignment, dimensions, provenance and checksums before publishing an immutable snapshot.

Raw documents, embeddings, local database volumes and model weights remain outside Git. Corpus manifests contain source URLs, not downloaded archives.

## Online query

The browser completes Cognito authorization code flow with PKCE. API Gateway validates the JWT; the application validates identity, scopes and active membership. Query requests cannot choose tenant, model endpoint, snapshot URI or arbitrary resource identifiers.

Two evidence paths are available:

- **Pilot snapshot:** NumPy vector search and BM25, combined using reciprocal rank fusion over a fixed corpus.
- **AWS official live search:** bounded, allowlisted search and fetched evidence. Each answer pins the passages and hashes that support its citations.

Generation receives bounded evidence and returns structured claims. Citation identifiers and trusted evidence spans are resolved by the server. Missing evidence produces partial or abstained answers; an exact quote alone does not establish that a claim is entailed.

## Identity and ownership

Registration collects separate username, email and website password and sends credentials directly to Cognito. Confirmation codes verify email. The chatbot API never handles passwords.

On the first authenticated status request, the server reads the Cognito profile using the username claim from the verified access token. It checks Enabled, CONFIRMED, verified email and equality between profile `sub` and token `sub`. A conditional write creates only a missing membership. Existing disabled or invalid memberships are never overwritten.

History and source authorization use the stable subject and server-selected tenant, rather than username or email. Source reads require an owned stored request that actually references the requested evidence.

## State and failure boundaries

DynamoDB stores memberships, owner-scoped query records, idempotency state, admissions and model-operation receipts. Conditional writes and transactions protect concurrent admission. Expiration is checked in application code; TTL deletion is cleanup.

The model gateway reserves a conservative token-cost bound before dispatch. Unknown completion retains its reservation and receipt. Reusing a completed idempotency key returns the authorized stored result; changing the payload under the same key is rejected.

S3 stores private, versioned artifacts. Lambda reuses model/snapshot state within a warm execution environment. Cold and warm latency must be reported separately.

## Deliberate scope

- AWS hosts authentication, the frontend, query handling, state and artifacts; generation calls an external provider.
- Neo4j and graph extraction are local experiments. Graph answering is gated by evaluation results.
- Web uploads, OCR, streaming, a managed graph database and a full AWS documentation archive are outside the current serving scope.
- Infrastructure synthesis is safe without credentials. A full deployment additionally needs explicitly prepared artifacts, model access, secrets, budget controls and reviewed change sets.

The HTTP contract is documented in [OpenAPI](api/openapi.json). Security controls and remaining gaps are described in [Security](security-overview.md).
