"""Read-only source/path/pin audit of the local authorization release."""

import json
from collections import Counter
from xml.etree import ElementTree

from app.authorization_evidence import load_authorization_facts
from app.contracts import canonical_json, sha256_bytes
from app.reviewed_source_graph import load_reviewed_source_graph
from scripts.prepare_authorization_evidence import ROOT, load_sources


def audit():
    collection, chunks, documents = load_sources()
    root = ROOT / "results/authorization-runtime-v2"
    auth_selection = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    authorization_path = ROOT / auth_selection["release_path"]
    auth = load_authorization_facts(authorization_path, collection, chunks, documents,
        trusted_sha256=auth_selection["release_sha256"], allow_experimental=True)
    selection = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    old = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    records = {r["assertion_id"]: r for r in auth.records()}
    source_chunks = {e.assertion_id: e.chunk_id for e in old.edges}
    source_chunks.update({k: r["candidate"]["chunk_id"] for k, r in records.items()})
    negatives = {e.assertion_id for e in old.edges if e.polarity != "POSITIVE"}
    raw = (root / "receipts.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    report = json.loads((root / "report.json").read_bytes())
    if sha256_bytes(raw) != report["receipts_sha256"]:
        raise ValueError("AUTH_AUDIT_RECEIPTS_CHECKSUM")
    seen, two_fact = Counter(), []
    for row in rows:
        outcome, trace, packed = row["outcome"], row["trace"], row["packed"]
        ids = set(outcome["context_assertion_ids"])
        if (ids & negatives or row["generation_executed"] is not False or trace["hidden_fallbacks"] != 0
                or outcome["effective_method"] != outcome["requested_method"]
                or tuple(s["chunk_id"] for s in packed["untrusted_sources"]) != tuple(outcome["context_chunk_ids"])
                or {r["assertion_id"] for r in packed["untrusted_graph_assertions"]} != ids):
            raise ValueError("AUTH_AUDIT_CONTEXT_OR_METHOD_BINDING")
        for source in packed["untrusted_sources"]:
            chunk = chunks[source["chunk_id"]]
            if (source["untrusted_text"] != chunk.text or source["document_version_id"] != chunk.document_version_id
                    or source["start_char"] != chunk.start_char or source["end_char"] != chunk.end_char):
                raise ValueError("AUTH_AUDIT_PACKED_SOURCE")
        for fact in packed["untrusted_graph_assertions"]:
            if fact["untrusted_evidence_quote"] != chunks[source_chunks[fact["assertion_id"]]].text:
                raise ValueError("AUTH_AUDIT_FULL_SOURCE")
            if fact["assertion_id"] in records:
                original = records[fact["assertion_id"]]
                if (fact["record"] != original["candidate"] or fact["nodes"] != original["nodes"]
                        or fact["permission_granted"] is not False or fact["configuration_verified"] is not False):
                    raise ValueError("AUTH_AUDIT_REQUIREMENT_CHANGED")
                seen[fact["assertion_id"]] += 1
        for path in trace["packed_paths"]:
            keys = path["assertion_ids"]
            supports = list(dict.fromkeys(source_chunks[k] for k in keys))
            if (len(set(keys)) != path["hop"] or not 1 <= path["hop"] <= int(outcome["requested_method"][6])
                    or not set(keys) <= ids or path["supporting_chunk_ids"] != supports
                    or not set(supports) <= set(outcome["context_chunk_ids"])
                    or path["condition_compatibility"] != "NOT_INFERRED"):
                raise ValueError("AUTH_AUDIT_PATH_SUPPORT")
            if path["hop"] == 2:
                two_fact.append({"control": row["control"], "assertion_ids": keys,
                    "source_document_count": len({chunks[k].document_version_id for k in supports})})
    if set(seen) != set(records):
        raise ValueError("AUTH_AUDIT_NEW_FACT_NEVER_PACKED")
    scope = json.loads((ROOT / "results/reviewed-source-answers-v1/scope.json").read_bytes())
    pin_matches = {p: sha256_bytes((ROOT / p).read_bytes()) == digest for p, digest in scope["code_sha256"].items()}
    if not all(pin_matches.values()):
        raise ValueError("AUTH_AUDIT_FROZEN_PAID_CODE_CHANGED")
    replay = json.loads((ROOT / "results/reviewed-source-answers-v1/replay.json").read_bytes())
    source_replay = json.loads((root / "replay.json").read_bytes())
    if (replay["status"] != "EXACT_REPLAY_MATCH" or replay["provider_calls_this_invocation"] != 0
            or source_replay["status"] != "EXACT_REPLAY_MATCH"):
        raise ValueError("AUTH_AUDIT_REPLAY_REQUIRED")
    xml = ElementTree.parse(ROOT / "results/authorization-evidence-v1/related-tests.xml")
    suites = list(xml.iter("testsuite"))
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if tests["failures"] or tests["errors"]:
        raise ValueError("AUTH_AUDIT_TEST_FAILURES")
    old_docs = {e.source_record()["document_version_id"] for e in old.edges}
    auth_docs = {r["candidate"]["document_version_id"] for r in records.values()}
    result = {"status": "SOURCE_PATH_REPLAY_AND_FROZEN_CODE_PINS_PASS_NOT_SEMANTIC_ACCURACY",
        "source_manifest_sha256": collection.manifest_sha256, "new_facts": len(records),
        "base_stored_records": len(old.edges), "negative_records_excluded": len(negatives),
        "traversable_records": len(old.edges) - len(negatives) + len(records),
        "base_stored_source_documents": len(old_docs), "new_fact_source_documents": len(auth_docs),
        "additional_source_documents": len(auth_docs - old_docs), "combined_source_documents": len(old_docs | auth_docs),
        "receipts": len(rows), "new_fact_packed_appearances": dict(seen),
        "two_fact_paths": two_fact, "two_fact_path_control_count": len({r["control"] for r in two_fact}),
        "frozen_paid_code_pin_matches": pin_matches, "old_answer_cache_replay": len(replay["answers"]),
        "related_tests": tests, "new_model_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0,
        "human_reviewed": 0, "serving_published": False, "independent_quality_evaluation_completed": False}
    (root / "audit.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"two_fact_paths", "frozen_paid_code_pin_matches", "new_fact_packed_appearances"}}))


if __name__ == "__main__":
    audit()
