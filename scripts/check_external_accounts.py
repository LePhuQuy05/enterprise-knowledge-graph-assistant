"""Read-only account checks; no inference, top-up, subscriptions or key disclosure."""

import json
from datetime import UTC, datetime
from pathlib import Path

from app.external_http import (
    ExternalHttpError,
    OfficialHttpsTransport,
    deepseek_preflight,
    load_private_key,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {"checked_at": datetime.now(UTC).isoformat(), "source": "LOCAL_READ_ONLY_PROBE",
              "providers": {}, "inference_calls": 0, "balance_mutations": 0}
    for provider in ("DEEPSEEK", "VOYAGE"):
        entry = {"key_status": "UNVERIFIED", "inference_verified": False}
        try:
            key = load_private_key(ROOT, provider)
            entry["key_status"] = "STORED_AND_PARSED"
            if provider == "DEEPSEEK":
                entry.update(deepseek_preflight(OfficialHttpsTransport({provider: key})))
                entry["status"] = "READ_ONLY_VERIFIED"
            else:
                entry["status"] = "KEY_READY_INFERENCE_SMOKE_PENDING"
                entry["note"] = "No public balance probe selected; do not infer free allowance or quota."
        except ExternalHttpError as exc:
            entry.update(status="BLOCKED", category=exc.category)
        report["providers"][provider] = entry
    output = ROOT / "results/external_account_preflight.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
