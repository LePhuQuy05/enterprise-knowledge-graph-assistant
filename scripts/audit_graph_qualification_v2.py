"""Rebuild a complete input status inventory and bound semantic objections.

No model/AWS call. Candidate source validity is replayed from durable raw cache;
semantic objections are explicit AI review, never an estimated accuracy score.
"""

import json
from collections import Counter

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes
from app.extraction import AliasDictionary
from app.extraction_cache import ExtractionCache
from scripts.run_collection_benchmark import ROOT, load_inputs
from scripts.run_full_graph_qualification_v2 import DIRECTORY, INVENTORY, context_for


def main():
    plan = json.loads((DIRECTORY / "plan.json").read_bytes())
    units = tuple(json.loads(row) for row in (INVENTORY / "source-units.jsonl").read_bytes().splitlines())
    _, _, collection, _, _ = load_inputs()
    docs, chunks = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        docs.update(shard.documents)
        chunks.update(shard.chunks)
    recipe = ExtractionRecipe.model_validate(plan["recipe"])
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json")
    cache = ExtractionCache(ROOT / "data/cache/external-models", namespace="LIVE")
    statuses, rows, assertions = [], [], {}
    selection = {u["unit_id"]: u for u in plan["selection"]}
    for unit in units:
        path = DIRECTORY / "units" / (unit["unit_id"] + ".json")
        if not path.exists():
            statuses.append({"unit_id": unit["unit_id"], "text_sha256": unit["text_sha256"], "status": "PENDING"})
            continue
        raw = path.read_bytes()
        if path.with_suffix(".sha256").read_text("ascii").strip() != sha256_bytes(raw):
            raise ValueError("GRAPH_QUALIFICATION_JOURNAL_CHECKSUM")
        row = json.loads(raw)
        if row["unit_id"] not in selection or row["occurrences"] != unit["occurrences"]:
            raise ValueError("GRAPH_QUALIFICATION_SOURCE_BINDING")
        if row["status"] in {"CANDIDATE", "EMPTY"}:
            replay = []
            for o in unit["occurrences"]:
                chunk = chunks[o["chunk_id"]]
                lookup = cache.resolve(context_for(unit), chunk=chunk, document=docs[chunk.document_version_id],
                    recipe=recipe, aliases=aliases)
                if not lookup.cache_hit:
                    raise ValueError("GRAPH_QUALIFICATION_REPLAY_NOT_CACHE_ONLY")
                replay.append({"chunk_id": chunk.chunk_id, **{name: [r.model_dump(mode="json") for r in getattr(lookup.canonical, name)]
                    for name in ("entities", "mentions", "assertions")}})
            if replay != row["canonical"]:
                raise ValueError("GRAPH_QUALIFICATION_CANONICAL_REPLAY_CHANGED")
            for canonical in replay:
                for a in canonical["assertions"]:
                    assertions[a["assertion_id"]] = a
        rows.append(row)
        statuses.append({"unit_id": unit["unit_id"], "text_sha256": unit["text_sha256"], "status": row["status"]})
    objections_path = ROOT / "configs/extraction/qualification-v2-objections.json"
    objections = json.loads(objections_path.read_bytes())
    review = []
    for prefix, decision in objections["decisions_by_assertion_prefix"].items():
        matches = [a for key, a in assertions.items() if key.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError("GRAPH_QUALIFICATION_REVIEW_ASSERTION_BINDING")
        a = matches[0]
        review.append({**decision, "assertion_id": a["assertion_id"], "status": "AI_OBJECTED_NOT_SERVING",
            "source_quote_sha256": sha256_bytes(a["evidence"]["quote"].encode()), "source_quote": a["evidence"]["quote"],
            "assertion_sha256": sha256_bytes(canonical_json(a)), "polarity": a["polarity"], "predicate": a["predicate"]})
    report = {"status": "STRUCTURAL_REPLAY_PASS_SEMANTIC_EXPANSION_GATE_FAILED", "collection_manifest_sha256": collection.manifest_sha256,
        "source_units": len(units), "processed_units": len(rows), "unit_statuses": dict(Counter(r["status"] for r in statuses)),
        "candidate_assertions": len(assertions), "ai_objected_assertions": len(review), "other_assertions_semantic_pending": len(assertions) - len(review),
        "objections_config_sha256": sha256_bytes(objections_path.read_bytes()), "objections": review,
        "human_reviewed": 0, "new_serving_assertions": 0, "provider_calls": 0, "aws_calls": 0,
        "next_gate": "NEW_VERSIONED_EXTRACTION_SEMANTICS_AND_SOURCE_REVIEW_BEFORE_BROAD_DISPATCH",
        "cost_note": "Do not spend the remaining allocation just to reach an edge count."}
    output = ROOT / "results/method-correctness-v2/qualification-audit-v1"
    output.mkdir(exist_ok=True)
    artifacts = {"report.json": canonical_json(report) + b"\n", "input-statuses.jsonl": b"".join(canonical_json(r) + b"\n" for r in statuses)}
    for name, raw in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("GRAPH_QUALIFICATION_AUDIT_CHANGED")
        path.write_bytes(raw)
    print(json.dumps({k: report[k] for k in ("status", "source_units", "processed_units", "unit_statuses", "candidate_assertions", "ai_objected_assertions")}))


if __name__ == "__main__":
    main()
