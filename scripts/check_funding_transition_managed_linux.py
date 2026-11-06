"""Offline deployed-ZIP gate check with fresh managed rows, no credentials."""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    source = Path("/input")
    rows = json.loads((source / "managed-counter-fixture.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="funding-managed-") as directory:
        with zipfile.ZipFile(source / "query-funding-transition.zip") as archive:
            archive.extractall(directory)
        sys.path.insert(0, directory)
        from app import query_handler
        from app.funding_policy import TRANSITION_PROFILE

        class ReadOnlyStore:
            def get(self, key):
                return next((rows[name] for name in ("funding", "run", "audit")
                             if (rows[name]["PK"], rows[name]["SK"]) == key), None)
        class SafeBoundary(Exception):
            pass
        def stop(*args, **kwargs):
            raise SafeBoundary()
        query_handler.clients = lambda: {"dynamodb": None, "secretsmanager": None}
        query_handler.DynamoLedgerStore = lambda *args: ReadOnlyStore()
        query_handler.read_deepseek_secret = stop
        os.environ.update(MODEL_ENABLED="true", KG_DEEPSEEK_SECRET_ARN="synthetic-unused-secret",
                          KG_FUNDING_POLICY=TRANSITION_PROFILE)
        try:
            query_handler.gateway(1)
        except SafeBoundary:
            pass
        else:
            raise AssertionError("MANAGED_FUNDING_GATE_DID_NOT_REACH_SAFE_SECRET_BOUNDARY")
        print(json.dumps({"status": "DEPLOYED_ZIP_MANAGED_8_USD_ROWS_PASS", "model_calls": 0,
            "secret_reads": 0, "aws_calls": 0, "counter_changes": 0,
            "fixture_origin": "Fresh managed funding/run/audit readbacks after approved allocation",
            "limitation": "Exact deployed ZIP under Linux using managed row fixtures; not a production gateway invocation"}))


if __name__ == "__main__":
    main()
