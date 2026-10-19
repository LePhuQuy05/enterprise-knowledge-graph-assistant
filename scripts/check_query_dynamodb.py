"""Verify query transactions and pagination on isolated DynamoDB Local, no AWS/model calls."""

import json
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import boto3
from botocore.config import Config

from app.ledger import Change
from app.query_auth import ApiError, AuthContext
from app.query_state import DynamoQueryStore, QueryRepository
from scripts.query_cloud import ROOT


def main():
    image = json.loads((ROOT / "configs/runtime_images.json").read_bytes())["dynamodb_local"]
    name = "kg-r72-query-local-" + uuid4().hex[:12]
    started = False
    report = {"status": "FAIL", "managed_aws_calls": 0, "paid_model_calls": 0, "image": image}

    def docker(args):
        result = subprocess.run(["docker", *args], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=90)
        if result.returncode:
            raise ValueError("LOCAL_DYNAMODB_CONTAINER_COMMAND_FAILED")
        return result.stdout.strip()

    try:
        docker(["run", "--detach", "--rm", "--name", name, "--memory", "768m", "--cpus", "1",
            "--publish", "127.0.0.1::8000", image, "-jar", "DynamoDBLocal.jar", "-inMemory", "-sharedDb"])
        started = True
        address = docker(["port", name, "8000/tcp"])
        if not address.startswith("127.0.0.1:") or not address.rsplit(":", 1)[-1].isdigit():
            raise ValueError("LOCAL_DYNAMODB_MUST_BIND_LOOPBACK")
        client = boto3.client("dynamodb", endpoint_url="http://" + address, region_name="us-east-1",
            aws_access_key_id="localfixture", aws_secret_access_key="localfixture",
            config=Config(retries={"total_max_attempts": 1}, connect_timeout=1, read_timeout=3))
        for _ in range(30):
            try:
                client.list_tables()
                break
            except Exception:
                time.sleep(0.3)
        table = "QueryFixture" + uuid4().hex[:12]
        client.create_table(TableName=table, BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": x, "AttributeType": "S"} for x in ("PK", "SK")],
            KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}])
        store = DynamoQueryStore(client, table)
        clock = [datetime(2026, 10, 18, tzinfo=UTC)]
        repo = QueryRepository(store, clock=lambda: clock[0])
        auth = AuthContext("demo", "11111111-1111-1111-1111-111111111111")
        release = {"snapshot_version": "snap_" + "0" * 64, "manifest_sha256": "0" * 64}
        payload, key = {"question": "Local transaction fixture", "mode": "hybrid"}, str(uuid4())
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: repo.admit(auth, key, payload, release, "0" * 64), range(8)))
        assert sum(fresh for _, fresh in results) == 1 and repo.quota_remaining() == 299
        rid = results[0][0]["request_id"]
        repo.update(auth, rid, status="FAILED", error_code="SYNTHETIC_NO_PROVIDER_CALL")
        fresh_reader = QueryRepository(DynamoQueryStore(client, table), clock=lambda: clock[0])
        assert fresh_reader.get(auth, rid)["status"] == "FAILED"
        for _ in range(2):
            clock[0] += timedelta(seconds=1)
            repo.admit(auth, str(uuid4()), payload, release, "0" * 64)
        first, cursor = repo.history(auth, limit=1)
        second, _ = repo.history(auth, limit=1, cursor=cursor)
        assert len(first) == len(second) == 1 and first[0]["request_id"] != second[0]["request_id"]
        quota_key = repo.quota_key("2026-10")
        counter = store.get(quota_key)
        store.commit([Change(quota_key, counter["version"], counter | {"count": 299, "version": counter["version"] + 1})])

        def competing(_):
            try:
                return repo.admit(auth, str(uuid4()), payload, release, "0" * 64)[1]
            except ApiError as exc:
                assert exc.code in {"QUERY_QUOTA_EXCEEDED", "STATE_BUSY"}
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert sum(pool.map(competing, range(8))) == 1
        assert repo.quota_remaining() == 0
        clock[0] += timedelta(days=8)
        try:
            repo.admit(auth, key, payload, release, "0" * 64)
        except ApiError as exc:
            assert exc.code == "IDEMPOTENCY_KEY_EXPIRED"
        else:
            raise AssertionError("Expired key cannot restart a request")
        report.update(status="PASS", duplicate_admission_one_winner=True, quota_299_one_winner=True,
                      history_query_pagination=True, strong_reader=True, expired_key_no_resume=True)
    except Exception as exc:
        report["error_category"] = type(exc).__name__
        report["error_line"] = traceback.extract_tb(exc.__traceback__)[-1].lineno
    finally:
        if started:
            try:
                docker(["stop", "--timeout", "5", name])
                report["temporary_container_stopped"] = True
            except Exception:
                report["temporary_container_stopped"] = False
        report["checked_at"] = datetime.now(UTC).isoformat()
        (ROOT / "results/query_dynamodb_local.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" and report.get("temporary_container_stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
