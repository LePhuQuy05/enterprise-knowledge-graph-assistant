"""Offline annotation oracle: minimum whole chunks for reviewed witnesses.

This is feasibility analysis, never a retriever or runtime source allocator.
It only covers declared reviewed alternatives, not all possible valid evidence.
"""

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from app.question_coverage_review import ReviewedWitness, audit_question_coverage, load_rubric
from app.source_dependency_audit import missing_intervals
from scripts import amend_question_coverage as amendment
from scripts import review_question_coverage as review
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable

OUTPUT = review.OUTPUT / "budget-feasibility-v1"
CODE = ("scripts/diagnose_question_budget.py", "app/question_coverage_review.py", "app/source_dependency_audit.py")


def nondominated(sets, *, limit=100000):
    """Dropping supersets preserves minimum cardinality, not every token tradeoff."""
    unique = set()
    for value in sets:
        unique.add(frozenset(value))
        if len(unique) > limit:
            raise ValueError("WITNESS_ORACLE_SEARCH_BOUND_NO_CONCLUSION")
    out = []
    for s in sorted(unique, key=lambda x: (len(x), sorted(x))):
        if not any(t <= s for t in out):
            out.append(s)
    return out


def minimum_contexts(requirements, witness_alternatives):
    """AND requirements, OR witnesses; each witness already is a chunk set."""
    plans = [frozenset()]
    steps = []
    for key in requirements:
        alternatives = nondominated(witness_alternatives[key])
        plans = nondominated(a | b for a in plans for b in alternatives)
        steps.append({"requirement_id": key, "witness_chunk_sets": len(alternatives),
            "joint_nondominated_sets": len(plans), "minimum_chunks_so_far": min(map(len, plans), default=None)})
    return plans, steps


def span_cover_options(bound, chunks):
    candidates = [c for c in chunks.values() if c.document_version_id == bound.document_version_id
        and c.start_char < bound.span.end_char and c.end_char > bound.span.start_char]

    @lru_cache(maxsize=10000)
    def walk(cursor):
        if cursor >= bound.span.end_char:
            return (frozenset(),)
        options = [tail | {c.chunk_id} for c in candidates if c.start_char <= cursor < c.end_char
            for tail in walk(c.end_char)]
        return tuple(nondominated(options))

    return walk(bound.span.start_char)


def catalog_alternatives(witnesses, chunks):
    result = defaultdict(list)
    for w in witnesses:
        if w.evidence_kind not in {"EXPLICIT_SUPPORT", "QUALIFIED_INFERENCE"}:
            continue
        plans = [frozenset()]
        for span in w.spans:
            plans = nondominated(a | b for a in plans for b in span_cover_options(span, chunks))
        result[w.requirement_id].extend(plans)
    return result


def freeze():
    paths = [review.ROOT / review.RUBRIC, amendment.OUTPUT / "witnesses.json", amendment.OUTPUT / "scope.json",
        amendment.OUTPUT / "receipts.json"]
    collection, _, _ = load_migrated()
    immutable(OUTPUT / "scope.json", {"version": "reviewed-witness-budget-oracle-v1",
        "input_sha256": {str(p.relative_to(review.ROOT)).replace("\\", "/"): sha256_bytes(p.read_bytes()) for p in paths},
        "code_sha256": {p: sha256_bytes((review.ROOT / p).read_bytes()) for p in CODE},
        "collection_manifest_sha256": collection.manifest_sha256, "current_whole_chunk_cap": 8,
        "annotation_oracle_not_runtime": True, "new_model_calls": 0, "new_retrievals": 0,
        "token_feasibility_measured": False, "human_reviewers": 0})


def diagnose():
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if any(sha256_bytes((review.ROOT / p).read_bytes()) != h for pins in (scope["input_sha256"], scope["code_sha256"]) for p, h in pins.items()):
        raise ValueError("WITNESS_ORACLE_INPUT_PIN")
    collection, _, _ = load_migrated()
    if collection.manifest_sha256 != scope["collection_manifest_sha256"]:
        raise ValueError("WITNESS_ORACLE_COLLECTION_PIN")
    docs, chunks, _ = collection_records(collection)
    rubric = load_rubric((review.ROOT / review.RUBRIC).read_bytes(), trusted_sha256=review.RUBRIC_SHA)
    raw = json.loads((amendment.OUTPUT / "witnesses.json").read_bytes())
    witnesses = tuple(ReviewedWitness.model_validate_json(canonical_json(w)) for w in raw["witnesses"])
    # Validate all witness identities, source hashes and literals before search.
    for intent in sorted({r.intent for r in rubric.requirements}):
        audit_question_coverage(rubric, witnesses, docs, chunks, (), intent=intent, tenant_id="demo")
    alternatives = catalog_alternatives(witnesses, chunks)
    rows = []
    for intent in sorted({r.intent for r in rubric.requirements}):
        requirements = [r.id for r in rubric.requirements if r.intent == intent and r.kind == "EXPLICIT_SUPPORT"]
        plans, steps = minimum_contexts(requirements, alternatives)
        best = min(plans, key=lambda s: (len(s), sorted(s))) if plans else frozenset()
        minimum = len(best) if plans else None
        # Independent interval check of at least one complete witness per requirement.
        intervals = defaultdict(list)
        for key in best:
            c = chunks[key]
            d = docs[c.document_version_id]
            if d.normalized_text[c.start_char:c.end_char] != c.text or sha256_bytes(c.text.encode()) != c.text_sha256:
                raise ValueError("WITNESS_ORACLE_CHUNK_BINDING")
            intervals[c.document_version_id].append((c.start_char, c.end_char))
        if plans and any(not any(w.requirement_id == key and w.evidence_kind in {"EXPLICIT_SUPPORT", "QUALIFIED_INFERENCE"}
                and all(not missing_intervals(s.span.start_char, s.span.end_char, intervals[s.document_version_id]) for s in w.spans)
                for w in witnesses) for key in requirements):
            raise ValueError("WITNESS_ORACLE_RECHECK_FAILED")
        rows.append({"intent": intent, "requirements": requirements, "steps": steps, "minimum_whole_chunks": minimum,
            "within_current_chunk_cap": None if minimum is None else minimum <= scope["current_whole_chunk_cap"],
            "nondominated_sets": len(plans), "example_annotation_oracle_sources": [{"chunk_id": key,
                "document_version_id": chunks[key].document_version_id, "source_uri": docs[chunks[key].document_version_id].source_uri,
                "start_char": chunks[key].start_char, "end_char": chunks[key].end_char} for key in sorted(best)],
            "derived_requirements": [r.id for r in rubric.requirements if r.intent == intent and r.kind == "QUALIFIED_INFERENCE"]})
    return {"version": "reviewed-witness-budget-oracle-v1", "intents": rows, "current_whole_chunk_cap": 8,
        "scope_sha256": sha256_bytes((OUTPUT / "scope.json").read_bytes()), "annotation_oracle_not_runtime": True,
        "token_feasibility_measured": False, "all_equivalent_sources_discovered": False,
        "all_possible_answer_contexts_proven_infeasible": False, "retrieval_or_answer_accuracy_measured": False,
        "new_model_calls": 0, "new_aws_calls": 0, "serving_published": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(target, side_effect=AssertionError("ORACLE_NETWORK_FORBIDDEN")))
        if args.freeze:
            freeze()
            print("Annotation-only feasibility scope frozen.")
            return
        result = diagnose()
        if args.run:
            immutable(OUTPUT / "report.json", result)
        else:
            if canonical_json(result) != canonical_json(json.loads((OUTPUT / "report.json").read_bytes())):
                raise ValueError("WITNESS_ORACLE_REPLAY_MISMATCH")
            immutable(OUTPUT / "replay.json", {"status": "EXACT_OFFLINE_REPLAY_PASS", "socket_access_blocked": True,
                "report_sha256": sha256_bytes((OUTPUT / "report.json").read_bytes()), "new_model_calls": 0, "new_aws_calls": 0})
        print(json.dumps([{k: r[k] for k in ("intent", "minimum_whole_chunks", "within_current_chunk_cap")} for r in result["intents"]]))


if __name__ == "__main__":
    main()
