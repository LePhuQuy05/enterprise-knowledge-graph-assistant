"""Freeze existing V10 observations and explicit synthetic syntax controls."""

import json

from app.contracts import canonical_json, sha256_bytes, stable_id
from scripts.run_clause_graph_qualification import immutable
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs
from scripts.run_source_actions_qualification import DIRECTORY as NATIVE

OUTPUT = ROOT / "results/source-syntax-v1"
CONTROLS = (
    ("S3 internally uses TLS.", "POSITIVE_DIRECT_ADVERB"),
    ("To configure S3, you can use TLS.", "NEGATIVE_FALSE_AGENT"),
    ("S3 clients use TLS.", "NEGATIVE_SERVICE_SUBSTRING"),
    ("CloudTrail captures API calls and delivers log files to an S3 bucket.", "POSITIVE_COORDINATED_ACTOR"),
    ("CloudTrail captures API calls and Lambda delivers log files to an S3 bucket.", "NEGATIVE_DIFFERENT_COORDINATED_ACTOR"),
    ("CloudTrail needs permission to deliver log files to an S3 bucket.", "NEGATIVE_OBLIGATION_NOT_ACTUAL_ACTION"),
    ("CloudTrail does not need to deliver log files to an S3 bucket.", "NEGATIVE_NECESSITY_SCOPE"),
    ("CloudTrail does not deliver log files to an S3 bucket.", "POSITIVE_NEGATION_WITNESS"),
    ("A trail enables CloudTrail to deliver log files to an S3 bucket.", "POSITIVE_ENABLED_ACTION"),
    ("CloudTrail can continue to send in-flight records to the data stream.", "POSITIVE_ASPECT_MODAL"),
    ("CloudTrail uses some or all of the cookies in viewer requests.", "POSITIVE_QUANTIFIER_PRESERVATION"),
    ("These two calls use taskToken.", "NEGATIVE_UNRESOLVED_COUNTED_REFERENCE"),
    ("CloudTrail encrypts logs with a KMS key.", "POSITIVE_KEY_INSTRUMENT_NOT_DATA"),
    ("CloudTrail delivers log files to the bucket that you specify.", "POSITIVE_RESOURCE_RESTRICTION"),
    ("CloudTrail delivers log files to the bucket in account A.", "POSITIVE_ACCOUNT_SCOPE"),
    ("CloudTrail delivers log files to the bucket in account B.", "NEGATIVE_ACCOUNT_EQUIVALENCE"),
    ("CloudTrail uses a cached result for 60 seconds unless disabled.", "POSITIVE_TTL_EXCEPTION_SCOPE"),
    ("CloudTrail uses a cached result for 600 seconds unless disabled.", "NEGATIVE_TTL_EQUIVALENCE"),
)


def main():
    collection, chunks, _ = load_collection_inputs()
    plan_raw = (NATIVE / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    rows, texts = [], set()
    for unit in plan["selection"]:
        raw = (NATIVE / "units" / (unit["v10_unit_id"] + ".json")).read_bytes()
        row = json.loads(raw)
        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        if sha256_bytes(text.encode()) != row["text_sha256"]:
            raise ValueError("SYNTAX_QUEUE_SOURCE_CHANGED")
        output = json.loads(row["receipt"]["raw_output"])
        records = []
        for kind, key, accepted in (("RELATION", "relationships", "accepted_relationships"),
                ("PERMISSION", "permission_requirements", "accepted_permissions"), ("ACTION", "actions", "accepted_actions"),
                ("DEFERRED", "deferred_statements", "deferred_statements")):
            for index, record in enumerate(output[key]):
                clause = record["clause_quote"]
                texts.add(clause)
                records.append({"kind": kind, "index": index, "record": record,
                    "clause_sha256": sha256_bytes(clause.encode()),
                    "native_guard_admitted": any(c["index"] == index for c in row[accepted]),
                    "native_rejections": [r for r in row["rejected_records"] if r["kind"] == kind and r["index"] == index]})
        rows.append({"unit_id": row["unit_id"], "cohort": row["cohort"], "source_text": text, "source_sha256": row["text_sha256"],
            "receipt_sha256": sha256_bytes(canonical_json(row["receipt"])), "native_unit_sha256": sha256_bytes(raw),
            "coverage_note_model_reported": row["coverage_note"], "occurrences": row["occurrences"], "observations": records})
    controls = [{"control_id": stable_id("source-syntax-control", text=text, expected=label), "text": text,
        "expected_structural_case": label, "synthetic_not_aws_fact": True} for text, label in CONTROLS]
    texts.update(c["text"] for c in controls)
    queue = {"version": "source-syntax-queue-v1", "source_manifest_sha256": collection.manifest_sha256,
        "native_plan_sha256": sha256_bytes(plan_raw), "native_recipe_sha256": sha256_bytes(canonical_json(plan["recipe"])),
        "rows": rows, "controls": controls, "human_reviewed": 0,
        "scope": "KNOWN_RECEIPTS_AND_SYNTHETIC_SYNTAX_CONTROLS_NOT_INDEPENDENT_GOLD"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    immutable(OUTPUT / "queue.json", canonical_json(queue) + b"\n")
    installation = json.loads((ROOT / "data/tools/source-parser/installation.json").read_bytes())
    immutable(OUTPUT / "parse-request.json", canonical_json({"installation": installation, "texts": sorted(texts)}) + b"\n")
    print(json.dumps({"frozen_native_sources": len(rows), "controls": len(controls), "distinct_parse_texts": len(texts), "model_api_calls": 0}))


if __name__ == "__main__":
    main()
