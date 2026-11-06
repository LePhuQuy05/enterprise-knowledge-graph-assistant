"""Full local source inventory with parser-derived, unreviewed action proposals.

Zero model API calls. Default prepares source inputs; --analyze reads a complete
JSONL parser artifact and writes bounded proposals, never a serving graph.
"""

import argparse
import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from app.extraction import ExtractionError
from app.source_action_proposals import propose_actions
from app.source_syntax import SyntaxDocument, policy_recipe
from scripts.run_clause_graph_qualification import immutable
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

OUTPUT = ROOT / "results/source-syntax-inventory-v2"
PARSED = ROOT / "results/source-syntax-inventory-v1/parsed.jsonl"


def prepare():
    collection, chunks, units = load_collection_inputs()
    installation = json.loads((ROOT / "data/tools/source-parser/installation.json").read_bytes())
    texts = [chunks[u["occurrences"][0]["chunk_id"]].text for u in units]
    plan = {"version": "source-actions-parser-inventory-v1", "source_manifest_sha256": collection.manifest_sha256,
        "source_units": len(units), "source_occurrences": sum(len(u["occurrences"]) for u in units), "selection": list(units),
        "policy": policy_recipe(), "generator_sha256": sha256_bytes((ROOT / "app/source_action_proposals.py").read_bytes()),
        "parser_worker_sha256": sha256_bytes((ROOT / "scripts/source_parse_worker.py").read_bytes()),
        "installation": installation, "max_candidates_per_unit": 20, "model_api_calls": 0,
        "native_v10_progress_increment": 0, "semantic_review": "PENDING_NOT_SERVING"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    immutable(OUTPUT / "plan.json", canonical_json(plan) + b"\n")
    immutable(OUTPUT / "parse-request.json", canonical_json({"installation": installation, "texts": texts}) + b"\n")
    return plan


def sentence_document(document, start, end):
    tokens = [t for t in document["tokens"] if start <= t["start"] and t["end"] <= end]
    ids = {t["i"]: i for i, t in enumerate(tokens)}
    if not tokens or any(t["head"] not in ids for t in tokens):
        raise ExtractionError("INVENTORY_SENTENCE_DEPENDENCY_CROSSES_SOURCE_BOUNDARY")
    return SyntaxDocument({"text": document["text"][start:end], "tokens": [t | {"i": ids[t["i"]], "head": ids[t["head"]],
        "start": t["start"] - start, "end": t["end"] - start} for t in tokens]})


def analyze():
    plan = prepare()
    counts, errors, sources, candidates, capped = Counter(), Counter(), 0, 0, 0
    statuses, proposals = [], []
    with PARSED.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))["header"]
        if header["installation"] != plan["installation"] or header["expected_texts"] != plan["source_units"]:
            raise ValueError("INVENTORY_PARSER_INSTALLATION_OR_COUNT_CHANGED")
        complete = None
        for line in handle:
            row = json.loads(line)
            if "complete" in row:
                complete = row["complete"]
                if handle.read().strip():
                    raise ValueError("INVENTORY_TRAILING_UNBOUND_CONTENT")
                break
            document = row["document"]
            unit = plan["selection"][sources]
            if sha256_bytes(document["text"].encode()) != unit["text_sha256"]:
                raise ValueError("INVENTORY_SOURCE_ORDER_OR_CHECKSUM_CHANGED")
            retained, rejected = [], Counter()
            for sentence in document["sentences"]:
                try:
                    parsed = sentence_document(document, sentence["start"], sentence["end"])
                    proposed, failures = propose_actions(parsed, document["text"])
                    retained.extend(proposed)
                    rejected.update(f["reason"] for f in failures)
                except ExtractionError as exc:
                    rejected[str(exc)] += 1
            limit = plan["max_candidates_per_unit"]
            capped += len(retained) > limit
            for candidate in retained[:limit]:
                counts[candidate["record"]["action_family"]] += 1
                proposals.append({"source_unit_id": unit["unit_id"], "source_sha256": unit["text_sha256"],
                    "occurrences": unit["occurrences"], "candidate_sha256": sha256_bytes(canonical_json(candidate)), "candidate": candidate})
            sources += 1
            candidates += len(retained[:limit])
            errors.update(rejected)
            statuses.append({"source_unit_id": unit["unit_id"], "source_sha256": unit["text_sha256"],
                "retained_candidates": len(retained[:limit]), "cap_reached": len(retained) >= limit,
                "sentences": len(document["sentences"]), "rejections": dict(rejected),
                "status": "UNREVIEWED_DEPENDENCY_CANDIDATES" if retained else "NO_ADMITTED_ACTION_NOT_CORPUS_UNANSWERABLE"})
    if not complete or sources != plan["source_units"] or complete["parsed_texts"] != sources:
        raise ValueError("INVENTORY_INCOMPLETE_PARSER_ARTIFACT")
    report = {"status": "LOCAL_SOURCE_INVENTORY_COMPLETE_CANDIDATES_NOT_SEMANTICALLY_APPROVED", "source_units": sources,
        "source_occurrences": plan["source_occurrences"], "source_manifest_sha256": plan["source_manifest_sha256"],
        "units_with_candidates": sum(s["retained_candidates"] > 0 for s in statuses), "candidate_records": candidates,
        "families": dict(counts), "units_truncated_at_cap": capped, "rejection_diagnostics": dict(errors),
        "model_api_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0, "native_v10_progress_increment": 0,
        "serving_records_added": 0, "human_reviewed": 0, "parser_timing": complete,
        "claims": "Structural source proposals only; source review, graph adapter, evaluation and release gates remain separate"}
    for name, value in (("report.json", report), ("source-statuses.json", statuses)):
        immutable(OUTPUT / name, canonical_json(value) + b"\n")
    immutable(OUTPUT / "proposals.jsonl", b"".join(canonical_json(p) + b"\n" for p in proposals))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    if parser.parse_args().analyze:
        analyze()
    else:
        plan = prepare()
        print(json.dumps({"source_units": plan["source_units"], "status": "PREPARED_LOCAL_PARSE_ONLY", "model_api_calls": 0}))
