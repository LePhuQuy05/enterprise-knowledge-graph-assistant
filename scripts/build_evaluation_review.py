"""Produce blank human source/answer review worksheets, never automatic approval."""

import argparse
import csv
import json
import random
from pathlib import Path

from scripts.evaluate_development import load_inputs
from scripts.evaluate_expanded import ROOT, read_pair


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(directory):
    datasets, _ = read_pair(directory)
    _, graph = load_inputs(datasets["en"])
    output = directory / "human-review"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("source-review.csv", "answer-review.csv"):
        if (output / name).exists():
            raise ValueError("REVIEW_WORKSHEET_EXISTS_DO_NOT_OVERWRITE_HUMAN_WORK")
    source_rows = []
    for en, vi in zip(datasets["en"].items, datasets["vi"].items, strict=True):
        evidence = [{"group": group.group_id, "fact": group.required_fact,
            "alternatives": [{"source_uri": graph.documents[span.document_version_id].source_uri,
                **span.model_dump(mode="json")} for span in group.alternatives]} for group in en.evidence_groups]
        source_rows.append({"decision_APPROVE_REVISE_REJECT": "", "reviewer": "", "reviewed_at": "",
            "scenario_id": en.question_id, "intent_group": en.intent_group_id, "english_question": en.question,
            "vietnamese_question": vi.question, "draft_category": en.category, "draft_answerable": en.answerable,
            "draft_reference": en.reference_answer, "source_evidence_json": json.dumps(evidence, ensure_ascii=False),
            "missing_reason": en.unanswerable_reason or "", "support_conditions_checked": "",
            "alternative_sources_checked": "", "translation_equivalence_checked": "", "notes": ""})
    write_csv(output / "source-review.csv", source_rows)
    answers = []
    for lang, ds in datasets.items():
        for item in ds.items:
            path = directory / "answers-v1" / ("expanded-v1-" + item.question_id + "-scored.json")
            scored = json.loads(path.read_bytes()) if path.exists() else {}
            response = scored.get("response", {})
            answers.append({"reviewer": "", "reviewed_at": "", "question_id": item.question_id, "language": lang,
                "question": item.question, "reference_answer_AI_DRAFT": item.reference_answer,
                "required_facts_json": json.dumps([g.required_fact for g in item.evidence_groups], ensure_ascii=False),
                "structural_status": scored.get("structural_status", "NOT_RUN"),
                "safe_error": scored.get("safe_error", ""), "answer_status": scored.get("answer_status"),
                "claims_json": json.dumps(response.get("claims", []), ensure_ascii=False),
                "citations_json": json.dumps(response.get("citations", []), ensure_ascii=False),
                "correctness_0_0point5_1": "", "supported_claims_count": "", "total_claims_count": "",
                "supporting_citations_count": "", "total_citations_count": "", "supported_required_facts_count": "",
                "language_ok": "", "abstention_appropriate": "", "notes": ""})
    random.Random(20261020).shuffle(answers)
    write_csv(output / "answer-review.csv", answers)
    return {"source_rows": len(source_rows), "answer_rows": len(answers), "review_decisions_written": 0,
            "directory": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT / "results/evaluation-expanded").resolve()):
        raise ValueError("REVIEW_OUTPUT_OUTSIDE_EVALUATION")
    print(json.dumps(build(directory)))


if __name__ == "__main__":
    main()
