"""Record exact corpus appearances of dictionary names; not semantic approval."""

import json
import re

from app.contracts import canonical_json, sha256_bytes
from app.extraction import AliasDictionary
from scripts.run_collection_benchmark import ROOT, load_inputs


def main():
    _, _, collection, _, _ = load_inputs()
    path = ROOT / "configs/extraction/aws-approved-aliases-v2.json"
    aliases = AliasDictionary.load(path)
    docs = {}
    for ref in collection.manifest.shards:
        docs.update(collection.get_shard(ref.shard_id).documents)
    rows = []
    for entry in aliases.entries:
        appearances = []
        for name in sorted(set((*entry.aliases, entry.canonical_name))):
            matches = []
            for doc in sorted(docs.values(), key=lambda d: d.document_version_id):
                match = re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", doc.normalized_text, re.I)
                if match:
                    matches.append({"document_version_id": doc.document_version_id, "source_uri": doc.source_uri,
                        "start_char": match.start(), "end_char": match.end(), "quote": match.group(),
                        "context": doc.normalized_text[max(0, match.start() - 70):match.end() + 120]})
            appearances.append({"alias": name, "source_documents": len(matches), "first_sources": matches[:2]})
        rows.append({"canonical_key": entry.canonical_key, "entity_type": entry.entity_type,
            "canonical_name": entry.canonical_name, "aliases": appearances,
            "status": "SOURCE_APPEARANCES_FOUND" if any(a["source_documents"] for a in appearances) else "UNOBSERVED_REQUIRES_REVIEW"})
    report = {"collection_manifest_sha256": collection.manifest_sha256, "aliases_sha256": sha256_bytes(path.read_bytes()),
        "entries": rows, "human_reviewed": False, "semantics_certified": False,
        "note": "Matching a name in a source does not prove synonymy or permission to merge resource instances."}
    output = ROOT / "results/method-correctness-v2/alias-source-audit-v1.json"
    raw = canonical_json(report) + b"\n"
    if output.exists() and output.read_bytes() != raw:
        raise ValueError("ALIAS_SOURCE_AUDIT_CHANGED")
    output.write_bytes(raw)
    print(json.dumps({"entries": len(rows), "observed_entries": sum(r["status"] == "SOURCE_APPEARANCES_FOUND" for r in rows),
        "unobserved_aliases": [a["alias"] for r in rows for a in r["aliases"] if not a["source_documents"]], "provider_calls": 0}))


if __name__ == "__main__":
    main()
