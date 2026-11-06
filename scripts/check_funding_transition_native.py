"""Offline Linux check of the exact proposed ZIP; mounted inputs are read-only."""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def main():
    package = Path("/input/query-funding-transition.zip")
    with tempfile.TemporaryDirectory(prefix="kg-funding-") as directory:
        with zipfile.ZipFile(package) as archive:
            if archive.testzip() is not None:
                raise ValueError("FUNDING_NATIVE_ZIP_CRC")
            if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts or "\\" in n for n in archive.namelist()):
                raise ValueError("FUNDING_NATIVE_ZIP_PATH")
            archive.extractall(directory)
        sys.path.insert(0, directory)
        from app import query_handler
        from app.funding_policy import COMPATIBLE_AUTHORIZATION_ID, TRANSITION_PROFILE
        from app.ledger import Change, Ledger, MemoryLedgerStore
        from app.query_auth import ApiError
        from app.query_service import RUN

        ledger = Ledger(MemoryLedgerStore())
        ledger.initialize_external_funding("kg-r72", cap_micro_usd=2000000)
        ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
        key = ledger.external_funding_key("kg-r72")
        os.environ.update(MODEL_ENABLED="true", KG_DEEPSEEK_SECRET_ARN="fixture-no-secret-access")
        os.environ.pop("KG_FUNDING_POLICY", None)
        query_handler.clients = lambda: {"dynamodb": None, "secretsmanager": None}
        query_handler.DynamoLedgerStore = lambda *args: ledger.store
        class SafeBoundary(Exception):
            pass
        def secret_boundary(*args, **kwargs):
            raise SafeBoundary()
        query_handler.read_deepseek_secret = secret_boundary

        def check(allowed):
            try:
                query_handler.gateway(1)
            except SafeBoundary:
                if not allowed:
                    raise AssertionError("UNAPPROVED_NATIVE_FUNDING_ADMITTED") from None
            except ApiError as exc:
                if allowed or exc.code != "BUDGET_LIMIT":
                    raise
            else:
                raise AssertionError("PROVIDER_BOUNDARY_NOT_REACHED")

        def update(**fields):
            row = ledger.store.get(key)
            ledger.store.commit([Change(key, row["version"], row | fields | {"version": row["version"] + 1})])
        check(True)  # Existing 2 USD, no opt-in.
        os.environ["KG_FUNDING_POLICY"] = TRANSITION_PROFILE
        check(True)  # First release remains compatible before any cap mutation.
        update(cap_micro_usd=8000000)
        check(False)  # Transition flag alone is insufficient.
        audit_key = (key[0], "ALLOCATION#" + COMPATIBLE_AUTHORIZATION_ID)
        audit = {"PK": audit_key[0], "SK": audit_key[1], "version": 1, "project_id": "kg-r72",
            "record_type": "AUTHORIZED_ALLOCATION_CHANGE", "authorization_id": COMPATIBLE_AUTHORIZATION_ID,
            "old_cap": 2000000, "new_cap": 8000000, "maximum_authorized_cap": 8000000}
        ledger.store.commit([Change(audit_key, None, audit)])
        check(True)
        update(halted=True)
        check(False)
        update(halted=False, cap_micro_usd=9000000)
        check(False)
        update(cap_micro_usd=8000000)
        os.environ.pop("KG_FUNDING_POLICY")
        check(False)  # A legacy profile never silently starts accepting 8 USD.
        print(json.dumps({"status": "LINUX_PROPOSED_ZIP_FUNDING_GATES_PASS", "checks": 7,
            "provider_calls": 0, "secret_reads": 0, "aws_calls": 0, "counter_changes": "FIXTURE_ONLY"}))


if __name__ == "__main__":
    main()
