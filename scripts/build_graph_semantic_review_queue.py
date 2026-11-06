"""Preserve every candidate and explicit AI review objection; no gold labels or publication."""

import json
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from scripts.reconcile_collection_graph_pilot import load_inputs, require
from scripts.run_collection_graph_pilot import ROOT

# Observed source/meaning objections. These are AI review notes, never human labels.
OBJECTIONS = {
    "21843e42": ("SUBJECT_NOT_ESTABLISHED", "Nguồn nói một service dùng SDK rồi S3 chuyển tracing headers; chưa thiết lập S3 chính là service dùng SDK."),
    "c287068f": ("OPTIONAL_FEATURE_NOT_REQUIREMENT", "Version ID chỉ có khi bật Versioning; một object không bắt buộc phụ thuộc S3 Versioning để tồn tại."),
    "46076284": ("CLASSIFICATION_NOT_COMPONENT", "Log archive account là một AWS account có vai trò lưu log, không phải thành phần nằm trong một AWS account khác."),
    "d0b3da50": ("UNRESOLVED_PRONOUN", "Đầu mút this field là tham chiếu chưa được giải; quote có thật chưa xác định được field nào."),
    "7e5fc029": ("PERMISSION_NOT_USAGE", "Không có quyền gọi API trong một ví dụ không đồng nghĩa quan hệ USES bị phủ định trong mọi tình huống."),
    "8a2b1619": ("OBJECT_NOT_ESTABLISHED", "CloudFormation uses the modified list; MyEC2Instance là tài nguyên được liên kết security groups, không phải object của động từ uses."),
    "10c0058": ("DEPENDENCY_POLARITY_INVERTED", "Mất quyền truy cập khi tắt key không phủ định sự phụ thuộc vào key. NEGATIVE DEPENDS_ON đảo nghĩa nguồn."),
}


def build():
    _, plan, docs, chunks = load_inputs()
    audit_raw = (ROOT / "results/collection-graph/reconciliation-final-v1.json").read_bytes()
    audit = json.loads(audit_raw)
    require(audit["status"] == "PASS_BOTH_GRAPH_EXPERIMENTS_SOURCE_RECEIPT_USAGE_MANAGED_LEDGER", "GRAPH_REVIEW_RECONCILIATION")
    queue, matched = [], set()
    for name in ("pilot-v1.json", "pilot-continuation-v1.json", "anchored-v4-check.json"):
        report_raw = (ROOT / "results/collection-graph" / name).read_bytes()
        report = json.loads(report_raw)
        content = Path(report["canonical_path"]).read_bytes()
        if "canonical_sha256" in report:
            require(sha256_bytes(content) == report["canonical_sha256"], "GRAPH_REVIEW_CANONICAL_HASH")
        for row in map(json.loads, content.splitlines()):
            entities = {r["entity_id"]: r for r in row["entities"]}
            for assertion in row["assertions"]:
                d, c = docs[assertion["document_version_id"]], chunks[assertion["chunk_id"]]
                evidence = assertion["evidence"]
                require(d.normalized_text[evidence["start_char"]:evidence["end_char"]] == evidence["quote"]
                    and c.start_char <= evidence["start_char"] < evidence["end_char"] <= c.end_char, "GRAPH_REVIEW_SOURCE_SLICE")
                matches = [suffix for suffix in OBJECTIONS if assertion["assertion_id"].endswith(suffix)]
                require(len(matches) <= 1 and not set(matches) & matched, "GRAPH_REVIEW_OBJECTION_AMBIGUOUS")
                matched.update(matches)
                reason, note = OBJECTIONS[matches[0]] if matches else (None, None)
                queue.append({"assertion": assertion, "subject": entities[assertion["subject_id"]]["canonical_name"],
                    "object": entities[assertion["object_id"]]["canonical_name"], "source_uri": d.source_uri,
                    "source_text_sha256": c.text_sha256, "origin_report": name, "origin_report_sha256": sha256_bytes(report_raw),
                    "reviewer": "CODEX_AI_SOURCE_REVIEW", "decision": "AI_OBJECTION_DO_NOT_PUBLISH" if matches else "PENDING",
                    "reason_code": reason, "note_vi": note, "human_reviewed": False})
    require(matched == set(OBJECTIONS) and len(queue) == len({r["assertion"]["assertion_id"] for r in queue}),
        "GRAPH_REVIEW_MISSING_OR_DUPLICATE_ASSERTION")
    return {"status": "REVIEW_QUEUE_NOT_APPROVED_GRAPH", "collection_manifest_sha256": plan["collection_manifest_sha256"],
        "financial_audit_sha256": sha256_bytes(audit_raw), "candidates": len(queue), "ai_objections": len(matched),
        "pending": len(queue) - len(matched), "approved_for_serving": 0, "human_reviewed": False,
        "model_calls": 0, "aws_calls": 0, "items": queue}


def main():
    path = ROOT / "results/collection-graph/semantic-review-v1.json"
    require(not path.exists(), "GRAPH_REVIEW_QUEUE_EXISTS")
    result = build()
    with path.open("xb") as handle:
        handle.write(canonical_json(result) + b"\n")
    print(json.dumps({key: result[key] for key in ("status", "candidates", "ai_objections", "pending", "approved_for_serving")}))


if __name__ == "__main__":
    main()
