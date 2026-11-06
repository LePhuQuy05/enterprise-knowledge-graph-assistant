"""Offline source-window invariants on every corpus319 anchor, not retrieval."""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.document_recall_v1 import document_window
from app.graph_collection_migration import collection_records
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT

OUTPUT = ROOT / "results/facet-recall-v4-canary-v1/window-source-qualification.json"


def qualify():
    collection, _, _ = load_migrated()
    documents, chunks, _ = collection_records(collection)
    digest, counts, max_sizes = hashlib.sha256(), {}, {}
    for radius in (1, 2):
        count, maximum = 0, 0
        for key in sorted(chunks):
            keys = document_window(chunks, key, radius)
            own = chunks[key].document_version_id
            if (key not in keys or len(keys) > 2 * radius + 1 or len(keys) != len(set(keys))
                    or any(chunks[k].document_version_id != own for k in keys)
                    or any(documents[own].normalized_text[chunks[k].start_char:chunks[k].end_char] != chunks[k].text for k in keys)):
                raise ValueError("CORPUS_DOCUMENT_WINDOW_SOURCE_INVARIANT")
            digest.update(canonical_json({"radius": radius, "anchor": key, "chunks": keys}) + b"\n")
            count += 1
            maximum = max(maximum, len(keys))
        counts[str(radius)], max_sizes[str(radius)] = count, maximum
    source = ROOT / "results/permission-context-qualification-v1/receipts.json"
    controls = json.loads(source.read_bytes())
    dependencies = [(r["control_id"], SourceDependency.model_validate_json(canonical_json(r["dependency"]))) for r in controls["source_controls"]]
    dependencies.append(("PARENT_KMS_OPERATIONS_SCOPE", SourceDependency.model_validate_json(canonical_json(controls["parent_kms_dependency"]))))
    rows = []
    for name, dependency in dependencies:
        anchors = [c for c in chunks.values() if c.document_version_id == dependency.document_version_id
            and c.start_char < dependency.anchor.end_char and dependency.anchor.start_char < c.end_char]
        for radius in (1, 2):
            selected = tuple(dict.fromkeys(key for anchor in sorted(anchors, key=lambda c:c.chunk_id)
                for key in document_window(chunks, anchor.chunk_id, radius)))
            rows.append({"control": name, "radius": radius, "anchor_chunk_ids": sorted(c.chunk_id for c in anchors),
                "window_chunk_ids": selected, "audit": audit_source_dependencies(documents, chunks, selected, (dependency,), tenant_id="demo")
                if len(selected) <= 8 else {"status": "WINDOW_UNION_EXCEEDS_CONTEXT_CHUNK_CAP"},
                "annotation_used_for_ranking": False, "retrieval_recall_measured": False, "token_feasibility_measured": False})
    result = {"status": "SOURCE_WINDOW_INVARIANTS_CHECKED_NATIVE_FACET_RETRIEVAL_PENDING",
        "collection_manifest_sha256": collection.manifest_sha256, "counts_by_radius": counts, "max_window_sizes": max_sizes,
        "anchor_windows_stream_sha256": digest.hexdigest(), "source_controls": rows,
        "source_control_annotations_sha256": sha256_bytes(source.read_bytes()),
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "window_policy_implementation_sha256": sha256_bytes((ROOT / "app/document_recall_v1.py").read_bytes()),
        "new_model_calls": 0, "aws_calls": 0, "native_facets_tested": False, "serving_published": False}
    immutable(OUTPUT, result)
    print(json.dumps({"counts_by_radius": counts, "max_window_sizes": max_sizes,
        "controls": [{"control": r["control"], "radius": r["radius"], "checks": [c["status"] for c in r["audit"].get("checks", [])]} for r in rows],
        "native_facet_retrieval_pending": True, "new_model_calls": 0}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(name, side_effect=AssertionError("WINDOW_QUALIFICATION_NETWORK_BLOCKED")))
        qualify()
