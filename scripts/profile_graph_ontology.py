"""Source-only lexical screen to plan a richer AWS fact schema, not extraction."""

import json
import re
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

CUES = {
    "LEGACY_SIX_PREDICATE_SURFACE": r"\b(?:uses?|using|used|supports?|provides?|enables?|offers?|requires?|needs?|depends? on|part of|belongs? to|integrates? with|replaces?)\b",
    "INVOKE_OR_TRIGGER": r"\b(?:invok(?:e|es|ed|ing)|triggers?|triggered)\b",
    "SEND_OR_DELIVER_OR_PUBLISH": r"\b(?:sends?|sending|sent|delivers?|delivery|publish(?:es|ed|ing)?)\b",
    "READ_OR_WRITE": r"\b(?:reads?|reading|writes?|writing|written)\b",
    "ENCRYPT_OR_DECRYPT": r"\b(?:encrypt(?:s|ed|ing)?|decrypt(?:s|ed|ing)?)\b",
    "ROLE_ASSUMPTION": r"\b(?:assume(?:s|d)? (?:an? |the )?role|AssumeRole)\b",
    "PERMISSION_POLICY_SURFACE": r"\b(?:permissions?|policies|policy)\b",
}


def main():
    collection, chunks, units = load_collection_inputs()
    results = {}
    for name, pattern in CUES.items():
        regex = re.compile(pattern, re.I)
        matching, examples = [], []
        for unit in units:
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            match = regex.search(text)
            if not match:
                continue
            matching.append(unit)
            if len(examples) < 3:
                examples.append({"unit_id": unit["unit_id"], "text_sha256": unit["text_sha256"],
                    "source_uri": unit["occurrences"][0]["source_uri"], "matched_surface": match.group(),
                    "excerpt": text[max(0, match.start() - 80):match.end() + 180]})
        results[name] = {"unique_texts_with_surface": len(matching),
            "uri_groups": dict(Counter(group for unit in matching for group in {o["service_group"] for o in unit["occurrences"]})),
            "examples_in_inventory_order_not_best_cases": examples}
    report = {"status": "LEXICAL_SOURCE_SCREEN_NOT_FACT_OR_ELIGIBLE_RELATION_COUNTS", "collection_manifest_sha256": collection.manifest_sha256,
        "inventory_unique_texts": len(units), "source_occurrences": sum(len(u["occurrences"]) for u in units),
        "regex_recipe_sha256": sha256_bytes(canonical_json(CUES)), "implementation_sha256": sha256_bytes(__import__('pathlib').Path(__file__).read_bytes()),
        "counts_overlap_do_not_sum": True, "no_actor_object_or_polarity_inferred": True, "screens": results,
        "provider_calls": 0, "aws_calls": 0, "graph_edges_added": 0}
    path = ROOT / "results/semantic-graph-v9/ontology-profile-v1.json"
    raw = canonical_json(report) + b"\n"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("ONTOLOGY_PROFILE_FROZEN_CONTENT_CHANGED")
    path.write_bytes(raw)
    print(json.dumps({"status": report["status"], "unique_texts": len(units),
        "surface_counts": {name: value["unique_texts_with_surface"] for name, value in results.items()}, "model_calls": 0}))


if __name__ == "__main__":
    main()
