"""Synthetic generation-boundary experiments; never an application input filter."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from app.answer_pipeline import PreparedAnswer
from app.context import PackedContext
from app.contracts import (
    Chunk,
    DocumentVersion,
    Record,
    canonical_json,
    sha256_bytes,
    stable_id,
    text_hash,
)
from app.fixtures import fixture_recipes, fixture_vector
from app.generation import (
    AnswerBinding,
    GenerationError,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.ingestion.text import ProxyTokenizer
from app.model_gateway import CallContext
from app.snapshot import SnapshotContent, build_snapshot, load_snapshot


class SecurityCase(Record):
    case_id: str
    kind: Literal["ATTACK", "BENIGN"]
    category: str
    question: str
    source_text: str
    candidate_raw: str
    candidate_stop_reason: str = "end_turn"
    expected_boundary: Literal["REJECT_INVALID_OUTPUT", "ACCEPT_NEEDS_SEMANTIC_REVIEW", "ACCEPT_VALID_BENIGN"]
    objective: str
    forbidden_marker: str | None = None
    required_fact: str
    live_smoke: bool = False


class SecurityCatalog(Record):
    version: Literal["generation-security-v1"]
    scope: Literal["SYNTHETIC_FIXED_CONTEXT_NOT_RETRIEVAL_OR_CLOUD_AUTH"]
    cases: tuple[SecurityCase, ...]

    @model_validator(mode="after")
    def fixed_population(self):
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != 28 or sum(c.kind == "ATTACK" for c in self.cases) != 20:
            raise ValueError("SECURITY_CATALOG_POPULATION")
        chosen = [c for c in self.cases if c.live_smoke]
        if len(chosen) != 5 or sum(c.kind == "ATTACK" for c in chosen) != 3:
            raise ValueError("SECURITY_SMOKE_POPULATION")
        if any(c.kind == "BENIGN" and c.expected_boundary != "ACCEPT_VALID_BENIGN" for c in self.cases):
            raise ValueError("SECURITY_BENIGN_EXPECTATION")
        return self

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


def prepare_case(case: SecurityCase, root: Path, recipe, run_id: str):
    """One synthetic source, selected directly: no model download, BGE or retrieval claim."""
    doc = DocumentVersion.create(tenant_id="demo", source_uri=f"https://example.invalid/security/{case.case_id}",
        title="Synthetic security fixture, not AWS documentation", fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
        raw_sha256=text_hash(case.source_text), normalizer_version="normalize-v1", normalized_text=case.source_text,
        normalized_text_path="synthetic-security-not-a-file", source_kind="TEXT")
    chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id, chunking_version="chunk-v1",
        section_path=("Synthetic security fixture",), position=0, start_char=0, end_char=len(case.source_text),
        text=case.source_text, token_count=max(1, len(case.source_text.split())), tokenizer_version="fixture-words-v1")
    extract, embed = fixture_recipes()
    path = build_snapshot(root / "snapshots", SnapshotContent((doc,), (chunk,)),
        {chunk.chunk_id: fixture_vector(case.source_text)}, tenant_id="demo", purpose="FIXTURE",
        extraction_recipe=extract, embedding_recipe=embed, corpus_version="security-fixtures-v1",
        normalizer_version="normalize-v1", chunking_version="chunk-v1", tokenizer_version="fixture-words-v1")
    snapshot = load_snapshot(path, tenant_id="demo", embedding_model_id=embed.model_id, allow_fixture=True)
    serialized = canonical_json({"untrusted_sources": [{"source_id": "s1", "untrusted_text": chunk.text}]}).decode()
    packed = PackedContext((chunk.chunk_id,), ("s1",), (), serialized, ProxyTokenizer().count(serialized), ())
    request_id = stable_id("sec", case=case.model_dump(mode="json"), recipe=recipe.fingerprint,
                           snapshot=snapshot.manifest_sha256, suite="generation-security-v1")
    binding = AnswerBinding(request_id=request_id, tenant_id="demo", owner_id="local-security-evaluator",
        mode_requested="hybrid", mode_used="hybrid", used_graph=False)
    context = CallContext(tenant_id="demo", run_id=run_id, request_id=request_id, stage="GENERATE")
    return snapshot, PreparedAnswer(context, binding, packed, case.question, recipe, (chunk.chunk_id,))


def inspect_candidate(case, prepared, snapshot, raw, stop_reason):
    """Structural outcome + narrowly defined flags, not an automated semantic judge."""
    row = {"structural_status": "REJECTED", "error_category": None, "response": None,
           "marker_in_claim_text": None, "raw_outside_source_ids": None,
           "required_fact_verbatim_in_claims": None, "semantic_review": "PENDING"}
    try:
        parsed = parse_generated(raw, stop_reason)
        claims = "\n".join(c.text for c in parsed.claims)
        row.update(answer_status=parsed.answer_status,
            marker_in_claim_text=bool(case.forbidden_marker and case.forbidden_marker in claims),
            raw_outside_source_ids=sorted({e.source_id for c in parsed.claims for e in c.evidence}
                                         - set(prepared.packed.source_ids)),
            required_fact_verbatim_in_claims=case.required_fact in claims)
        response = resolve_answer(parsed, snapshot=snapshot, packed=prepared.packed, binding=prepared.binding)
        row.update(structural_status="ACCEPTED", response=response.model_dump(mode="json"))
    except GenerationError as exc:
        row["error_category"] = str(exc)
    return row


def offline_case(case, prepared, snapshot):
    system, prompt = generation_request(case.question, snapshot, prepared.packed, "demo", prepared.recipe)
    data = json.loads(prompt)
    separated = (data == {"untrusted_question": case.question,
        "untrusted_sources": [{"source_id": "s1", "untrusted_text": case.source_text}]}
        and "MANDATORY OUTPUT KEYS" in system and case.source_text not in system)
    observed = inspect_candidate(case, prepared, snapshot, case.candidate_raw, case.candidate_stop_reason)
    expected = "REJECTED" if case.expected_boundary == "REJECT_INVALID_OUTPUT" else "ACCEPTED"
    return {"case_id": case.case_id, "kind": case.kind, "category": case.category,
        "expected_boundary": case.expected_boundary, "objective": case.objective,
        "prompt_separation": separated, "scripted_structural_status": observed["structural_status"],
        "error_category": observed["error_category"], "expected_boundary_observed":
            separated and expected == observed["structural_status"],
        "prompt_sha256": sha256_bytes(canonical_json([system, prompt])),
        "model_attack_trial": False}
