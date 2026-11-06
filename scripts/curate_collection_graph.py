"""Build an explicit AI-reviewed experimental subset; original outputs stay immutable.

No model/AWS calls. A selection is not an accuracy benchmark or human approval.
"""

import json
from pathlib import Path

from app.collection_graph import graph_fingerprint, literal_mentions, service_entities
from app.collection_graph_runtime import CollectionGraphOverlayV1, validate_graph_sources
from app.contracts import Entity, Mention, RelationAssertion, canonical_json, sha256_bytes
from scripts.reconcile_collection_graph_pilot import load_inputs, require
from scripts.run_collection_graph_pilot import ROOT

# These notes are the current agent's source review, not labels from another person.
SELECTION = {
    "8b10460d": "Nguồn nói rõ ECS Exec requires a task IAM role; chỉ giữ dependency của feature này, không suy toàn ECS cần cùng role.",
    "f023f37a": "Nguồn có động từ integrates with giữa X-Ray và S3; integration không chứng minh đã bật trong tài khoản.",
    "2fb4ab15": "Enables trace messages cho S3 event notifications hỗ trợ capability tracing; không khẳng định mọi notification đã được trace.",
    "f2ccf98e": "Every object is contained in a bucket hỗ trợ PART_OF. Hai khái niệm giữ source scope, không gộp mọi object/bucket toàn AWS.",
    "113f428b": "Listeners được thêm vào load balancer; PART_OF được hiểu là cấu trúc cấu hình trong tài liệu ALB, không là identity instance.",
    "e302137f": "Rules được định nghĩa cho listener; chỉ giữ quan hệ thành phần trong source ALB, không tự gộp listener với listeners.",
    "d7096033": "Managed external secrets cung cấp monitoring through CloudTrail logs; USES chỉ biểu diễn integration monitoring được nói rõ.",
    "b6fa2f29": "Cùng câu nêu monitoring through CloudWatch metrics; không suy secret value được gửi vào metrics.",
    "01c6b6cc": "Nguồn nói nguyên văn Amazon SNS uses the publisher-subscriber model; quan hệ không thay danh sách quyền truy cập phía sau dấu hai chấm.",
    "cf243fb6": "Phân phối FIFO topic data across partitions chỉ trong With high throughput topics; qualifier này được giữ nguyên.",
    "554198ca": "Partition là allocation of capacity for a topic; PART_OF mô tả cấu trúc, không tự suy cấu hình replication của tài khoản.",
    "6b90e85f": "Hướng dẫn To use PrivateLink, create a VPC endpoint hỗ trợ dependency để truy cập service/resource; không khẳng định endpoint đã tồn tại.",
    "11fcb52f": "VPC endpoint service được mô tả powered by PrivateLink. USES là quan hệ cơ chế, không là quyền cross-account đã được cấp.",
    "7e9b2ba2": "NAT64 chỉ được nêu For IPv6 traffic; giữ qualifier, không suy NAT64 dùng cho mọi loại traffic.",
    "fac11850": "Nguồn nêu nguyên văn NAT gateway supports TCP; capability không chứng minh route hoặc firewall cho phép kết nối.",
    "fd156ee9": "Cùng danh sách hỗ trợ UDP; không thêm quan hệ transitive với các protocols khác.",
    "45636e75": "Cùng danh sách hỗ trợ ICMP; không suy mọi ICMP packet đều đi được trong tài khoản.",
}


def write_immutable(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, "CURATED_GRAPH_IMMUTABLE_CONTENT_CHANGED")
    else:
        with path.open("xb") as handle:
            handle.write(raw)


def main():
    _, plan, docs, chunks = load_inputs()
    queue_path = ROOT / "results/collection-graph/semantic-review-v1.json"
    queue_raw = queue_path.read_bytes()
    queue = json.loads(queue_raw)
    decisions, matched, selected_ids = [], set(), set()
    for row in queue["items"]:
        assertion = row["assertion"]
        matches = [suffix for suffix in SELECTION if assertion["assertion_id"].endswith(suffix)]
        require(len(matches) <= 1 and not matched.intersection(matches), "CURATED_GRAPH_SELECTION_AMBIGUOUS")
        if matches:
            require(row["decision"] == "PENDING", "CURATED_GRAPH_CANNOT_OVERRIDE_OBJECTION")
            matched.update(matches)
            selected_ids.add(assertion["assertion_id"])
        decisions.append({"assertion_id": assertion["assertion_id"], "assertion_payload_sha256": sha256_bytes(canonical_json(assertion)),
            "decision": "AI_SUPPORTED_FOR_EXPERIMENT" if matches else row["decision"],
            "reason_vi": SELECTION[matches[0]] if matches else row["note_vi"], "source_uri": row["source_uri"],
            "source_text_sha256": row["source_text_sha256"]})
    require(matched == set(SELECTION) and len(selected_ids) == 17, "CURATED_GRAPH_SELECTION_INCOMPLETE")
    curation = {"schema_version": 1, "policy": "AI_SOURCE_REVIEW_EXPERIMENTAL_SUBSET_V1", "reviewer": "CODEX_AI_SOURCE_REVIEW",
        "collection_manifest_sha256": plan["collection_manifest_sha256"], "upstream_review_queue_sha256": sha256_bytes(queue_raw),
        "human_reviewed": False, "quality_metrics": None, "eligible_for_production": False,
        "selected": len(selected_ids), "rejected": queue["ai_objections"], "pending": queue["pending"] - len(selected_ids),
        "selection_reason": "Narrow source support for an explicit experimental graph; no predicate/type/qualifier or raw-output edits.",
        "decisions": decisions}
    curation_raw = canonical_json(curation) + b"\n"
    curation_path = ROOT / "configs/graph/collection316-curation-v1.json"
    write_immutable(curation_path, curation_raw)
    entities, mentions, assertions = {}, {}, {}
    for name in ("pilot-v1.json", "pilot-continuation-v1.json", "anchored-v4-check.json"):
        receipt = json.loads((ROOT / "results/collection-graph" / name).read_bytes())
        content = Path(receipt["canonical_path"]).read_bytes()
        if "canonical_sha256" in receipt:
            require(sha256_bytes(content) == receipt["canonical_sha256"], "CURATED_GRAPH_CANONICAL_SOURCE_CHANGED")
        for row in map(json.loads, content.splitlines()):
            accepted = [RelationAssertion.model_validate_json(canonical_json(a)) for a in row["assertions"] if a["assertion_id"] in selected_ids]
            own_ids = {getattr(a, key) for a in accepted for key in ("subject_id", "object_id")}
            for raw in row["entities"]:
                if raw["entity_id"] in own_ids:
                    entity = Entity.model_validate_json(canonical_json(raw))
                    previous = entities.get(entity.entity_id)
                    require(previous is None or previous == entity, "CURATED_GRAPH_ENTITY_PAYLOAD_CONFLICT")
                    entities[entity.entity_id] = entity
            for raw in row["mentions"]:
                if raw["entity_id"] in own_ids:
                    mention = Mention.model_validate_json(canonical_json(raw))
                    mentions[mention.mention_id] = mention
            assertions.update((a.assertion_id, a) for a in accepted)
    require(set(assertions) == selected_ids, "CURATED_GRAPH_MISSING_SELECTED_ASSERTION")
    for decision in decisions:
        if decision["assertion_id"] in assertions:
            assertion = assertions[decision["assertion_id"]]
            require(sha256_bytes(canonical_json(assertion.model_dump(mode="json"))) == decision["assertion_payload_sha256"],
                "CURATED_GRAPH_SELECTED_PAYLOAD_CHANGED")
            require(chunks[assertion.chunk_id].text_sha256 == decision["source_text_sha256"]
                and docs[assertion.document_version_id].source_uri == decision["source_uri"], "CURATED_GRAPH_SOURCE_CHANGED")
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    all_entities = service_entities(pipeline["service_counts"], tuple(entities.values()))
    # Only dictionary/service identities can have global literal mention expansion.
    global_entities = tuple(e for e in all_entities if e.entity_type == "AWS_SERVICE" and e.canonical_key.startswith("aws/"))
    for mention in literal_mentions(global_entities, tuple(chunks.values()), canonicalization_version="collection-curated-literal-v1"):
        mentions[mention.mention_id] = mention
    own_entities = tuple(sorted(all_entities, key=lambda e: e.entity_id))
    own_mentions = tuple(sorted(mentions.values(), key=lambda m: m.mention_id))
    own_assertions = tuple(assertions[key] for key in sorted(assertions))
    fingerprint = graph_fingerprint(plan["collection_manifest_sha256"], own_entities, own_mentions, own_assertions)
    overlay = CollectionGraphOverlayV1(tenant_id="demo", collection_version=json.loads(Path(pipeline["collection_path"]).read_bytes())["collection_version"],
        collection_manifest_sha256=plan["collection_manifest_sha256"], graph_sha256=fingerprint,
        curation_sha256=sha256_bytes(curation_raw), upstream_review_queue_sha256=sha256_bytes(queue_raw),
        entities=own_entities, mentions=own_mentions, assertions=own_assertions)
    from app.collections import load_collection

    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5",
        trusted_manifest_sha256=plan["collection_manifest_sha256"])
    validate_graph_sources(overlay, collection)
    graph_raw = canonical_json(overlay.model_dump(mode="json")) + b"\n"
    graph_path = ROOT / "data/collection-graph" / fingerprint / "reviewed-overlay-v1.json"
    write_immutable(graph_path, graph_raw)
    report = {"status": "AI_REVIEWED_EXPERIMENTAL_SUBSET_SOURCE_VERIFIED", "graph_path": str(graph_path),
        "graph_file_sha256": sha256_bytes(graph_raw), "graph_sha256": fingerprint, "curation_path": str(curation_path),
        "curation_sha256": sha256_bytes(curation_raw), "collection_manifest_sha256": plan["collection_manifest_sha256"],
        "documents": len(docs), "chunks": len(chunks), "service_groups": len(pipeline["service_counts"]),
        "entities": len(own_entities), "mentions": len(own_mentions), "assertions": len(own_assertions),
        "relation_source_documents": len({a.document_version_id for a in own_assertions}),
        "relation_source_chunks": len({a.chunk_id for a in own_assertions}),
        "mention_covered_chunks": len({m.chunk_id for m in own_mentions}), "human_reviewed": False,
        "full_relation_extraction": False, "production_eligible": False, "cloud_published": False, "model_calls": 0, "aws_calls": 0}
    write_immutable(ROOT / "results/collection-graph/curated-subset-v1.json", canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
