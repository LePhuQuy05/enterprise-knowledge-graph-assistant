# Development and reproduction

## Environment

Use Python 3.12, Node.js 22 or 24, Git and uv. Python dependencies are pinned in `uv.lock`; the CDK CLI is pinned in `package-lock.json`.

```sh
uv sync --frozen
npm ci --ignore-scripts
uv run python -m app.cli prepare-tokenizer
uv run pytest -q
uv run ruff check app infra scripts tests
npm run ui:test
npm run infra:synth
```

Preparation downloads public packages and a pinned tokenizer. Tests and synthesis do not require an AWS account, private results, model API keys or paid inference. Docker is optional for Neo4j protocol checks and Linux-native Lambda packaging.

## Synthetic demo

```sh
uv run python -m app.cli fixture-demo --output-root results/fixture-demo
uv run python -m scripts.preview_web_upgrade --serve --port 8765
```

The CLI fixture demonstrates snapshot validation and retrieval mechanics. The browser preview serves synthetic records and displays a banner identifying its scope. It cannot submit paid questions. For the account-state preview, run `uv run python -m scripts.preview_accounts --port 8766`; use synthetic credentials only.

## Real documentation and embeddings

```sh
uv run python -m app.cli ingest --limit 20
uv run python -m scripts.prepare_bge --download
uv run python -m scripts.build_bge_baseline --help
```

The ingestion command reads `configs/corpus/aws-r72-v1.json`. Model preparation verifies pinned files and does not execute remote model code. Use each script's help output for required artifact paths. Actual baseline, extraction and graph builds need the preceding ingestion artifacts; a clone does not contain them.

Keep vector and graph experiments tied to the same snapshot, embedding recipe and context budget. Do not regenerate a benchmark or silently replace evidence labels merely to improve results.

## Optional providers and AWS

DeepSeek credentials are supplied by the operator. Never commit keys, credential files, bearer tokens, Cognito passwords or model receipts. Local key setup and cloud Secrets Manager configuration are separate from the fixture path. Paid scripts require explicit execution and initialized durable budget state.

Cloud code uses the project profile `kg-dev` in `us-east-1` and deliberately rejects unexpected deployment identities. Configure a non-root identity in your own account before any cloud operation. The included IAM preflight policy grants metadata reads; it is not a deployment administrator policy.

`infra:synth` produces templates only. The default CDK entry point provides the foundation and setup tracer, not a one-command recreation of the live answer service. A complete query deployment additionally requires:

1. A validated corpus snapshot and pinned Linux-compatible model assets.
2. Private S3 uploads with checksum readback.
3. Provider secrets and durable funding/admission state.
4. Cognito configuration, callback URLs, runtime permissions and a query package.
5. A reviewed change set followed by code/assets/authentication/ownership checks.

Inspect `scripts/query_cloud.py`, `scripts/account_rollout.py` and `scripts/build_query_package.py` before using their guarded commands. They intentionally depend on operator-generated artifacts in `results/`; missing artifacts must fail instead of triggering inference or fabricating successful results. These are deployment tools for the current project layout, not a general multi-account installer.

## Tests and recorded data

Unit tests construct synthetic datasets and snapshots under temporary directories. Replaying the original development experiment is a separate opt-in operation requiring its source artifacts. The committed dashboard JSON is a bounded, attributed display projection; it is not a replacement for original corpus or retrieval receipts.

Browser scripts use an installed Microsoft Edge and isolated CDP profiles on Windows. Run `node scripts/check_accounts_browser.mjs` or `node scripts/check_web_upgrade_browser.mjs` for local synthetic flow checks. They create ignored screenshots/reports and do not authenticate against production. CI runs portable Python and Node unit suites rather than claiming those Windows browser checks ran on Linux.
