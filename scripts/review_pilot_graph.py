"""Record explicit AI source-review judgments; preserve raw graph and pending human queue."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST = "a24bab418b4ce86081f97a75c09db0dfdce5dc6e4c65ebadbad2c799400b00d0"
JUDGMENTS = {
    "ast_24b50b13c49be23796a84a4fb90aa2c707ffb1b94af5c62855081eae7504c91b": (
        "REJECTED", "CONDITION_OMISSION", "Quan hệ chưa giữ phạm vi KMS key và yêu cầu đồng thời policy ở tài khoản ngoài; không dùng như dependency không điều kiện."),
    "ast_338c093115a452d9047509bdafc176cdac2b2b49e15a3e52797417fff5a79300": (
        "REJECTED", "PREDICATE_DIRECTION", "Nguồn nói người dùng dùng Lambda để xử lý sự kiện S3; chưa khẳng định Lambda USES S3 theo nghĩa explicit usage của ontology."),
    "ast_4c33a50eac713072e832f012eda19a38a698c9c968a684b45101c36d63d1f1eb": (
        "SUPPORTED_CANDIDATE", "EXPLICIT_CAPABILITY", "Nguồn nêu capability hỗ trợ policy fragments và qualifier giữ giới hạn hành động trên tài nguyên được chia sẻ; cần human review type/thuật ngữ."),
    "ast_55da9ab77f8fe43f1e87e1bf16d64d51be7a5e1b378aa00c3e3f8608b5b3d7f1": (
        "REJECTED", "PREDICATE_MISMATCH", "Controls access không tương đương SUPPORTS theo định nghĩa capability; scope Region có giữ nhưng không sửa được predicate sai."),
    "ast_57855fcfb31358db4c376396749819102c06ecb92d11c26bbf8fc4dee8915a08": (
        "SUPPORTED_CANDIDATE", "EXPLICIT_CAPABILITY", "Nguồn nêu trực tiếp S3 hỗ trợ nhóm event types/destinations; entity khá rộng nên chưa dùng như fact cụ thể hơn."),
    "ast_650bbb48614f0851f7c23857aab79786beadf1055f6b18e3006e4c0f1af2ceee": (
        "REJECTED", "ENTITY_TYPE_AND_PREDICATE", "Tên functions chỉ tài nguyên nhưng bị gán AWS_SERVICE; cấp quyền truy cập cũng chưa tương đương SUPPORTS capability."),
    "ast_6cf9f39f9cab58fb8a7880ecdca7aa87e69b660f73c8d8c712d9565a3fd4c125": (
        "SUPPORTED_CANDIDATE", "EXPLICIT_REQUIREMENT", "Nguồn nêu mỗi KMS key bắt buộc có key policy; giới hạn exactly one đã giữ. Cần review identity trước merge cross-chunk."),
    "ast_959ed142dba391a7782ff8fa2b362008d2e701c87f976a65f9a1d55519070c75": (
        "NEEDS_REVIEW", "SPLIT_GROUP_SCOPE", "Câu nguồn nhóm SCP/RCP với principals/resources; khi tách cần kiểm scope áp dụng của từng loại policy, không suy cả hai áp dụng giống nhau."),
    "ast_a4c177dd24c040ba8453993df1438cb835abb73edc53914de915d6a6a9331993": (
        "REJECTED", "PREDICATE_MISMATCH", "Không thể dùng policy để cấp access trong một điều kiện không nên đổi thành phủ định capability SUPPORTS tài nguyên. Negation/qualifier nguyên văn vẫn chưa đủ."),
    "ast_bf681f67b0b915d396a6b2a82a0b4bc73ded048791e2f009f3cae531ced0d1da": (
        "REJECTED", "ENABLED_STATE_NOT_CAPABILITY", "Nguồn nói default key policy enables IAM policies; ontology phân biệt enabled state với capability SUPPORTS."),
    "ast_c2e436c8d6ace74c2b6fbda1d96751e3a462cc388a64f70c08dffe2f2e6a9548": (
        "REJECTED", "ENTITY_TYPE", "Endpoint Lambda Event Function là function resource theo câu nguồn, không là tên service identity đã được dictionary duyệt; không tự merge sang AWS Lambda."),
    "ast_db21cd6dd8654c7f701920f886567eae72a272ed6e3cdad98ca1bcba44cecf0c": (
        "NEEDS_REVIEW", "OWNERSHIP_VS_PART_OF", "Policy dành cho key không tự đủ để khẳng định quan hệ PART_OF theo nghĩa containment; cần định nghĩa domain rõ hơn."),
    "ast_dd35aadceb512eb1c43a8b7738533f1f0c123bc977b6ea49f22e9fc4704c2fa7": (
        "NEEDS_REVIEW", "SPLIT_GROUP_SCOPE", "Cần phân biệt scope principals/resources khi tách câu nhóm policy; chưa chấp nhận suy rộng cho SCP riêng."),
    "ast_f54b35fc679fae896f86e9cf64f5ca5d32dbf54dc17da0e8988d0e01cb03bba3": (
        "REJECTED", "CONDITION_OMISSION", "Không giữ yêu cầu IAM policy ở external user's account và policy đồng thời ở key account; graph rời hai điều kiện có thể gây hiểu sai."),
}


def main():
    receipt = json.loads((ROOT / "results/pilot_graph_snapshot.json").read_bytes())
    if receipt["manifest_sha256"] != PINNED_MANIFEST:
        raise ValueError("Review is bound to the original frozen pilot, not new graph content")
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=PINNED_MANIFEST)
    if set(JUDGMENTS) != {a.assertion_id for a in snapshot.content.assertions}:
        raise ValueError("Explicit judgments must cover exactly the frozen assertion IDs")
    records = []
    for assertion in snapshot.content.assertions:
        verdict, category, reason = JUDGMENTS[assertion.assertion_id]
        records.append({"assertion_id": assertion.assertion_id, "assertion_payload_sha256": sha256_bytes(
            canonical_json(assertion.model_dump(mode="json"))), "verdict": verdict, "category": category,
            "reason_vi": reason, "reviewer_type": "AI_SOURCE_REVIEW", "human_review": "PENDING",
            "source_uri": snapshot.documents[assertion.document_version_id].source_uri})
    report = {"version": "pilot-v3-ai-semantic-review-v1", "checked_at": datetime.now(UTC).isoformat(),
        "status": "PRELIMINARY_NOT_GOLD", "snapshot_manifest_sha256": PINNED_MANIFEST,
        "counts": dict(Counter(r["verdict"] for r in records)), "records": records,
        "model_calls": 0, "raw_graph_mutations": 0, "human_queue_mutations": 0,
        "release_decision": "HYBRID_DEFAULT_GRAPH_REMAINS_EXPERIMENTAL",
        "note": "Judgments concern support in frozen source and ontology, not independent AWS truth verification. No precision metric or human sign-off."}
    output = ROOT / "results/pilot_graph_ai_semantic_review.json"
    if output.exists():
        previous = json.loads(output.read_bytes())
        if previous["records"] != records:
            raise ValueError("Existing review differs; create a new version without overwriting it")
    else:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "model_calls": 0,
                      "release_decision": report["release_decision"]}, indent=2))


if __name__ == "__main__":
    main()
