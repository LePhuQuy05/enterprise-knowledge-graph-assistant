"""Display qualification over source-bound native answers; never edits claims.

NO_MECHANICAL_FLAGS is deliberately not a semantic accuracy verdict. Detected
issues withhold the entire response; raw model receipts remain replayable.
"""

import json
import re
from dataclasses import dataclass

from app.citation_antecedents import TRIGGER, audit_claim_antecedents
from app.contracts import Citation, QueryResponse, canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.profiled_source_context import validate_profiled_context

POLICY_VERSION = "source-bound-display-qualification-v1"
LAMBDA = re.compile(r"\blambda\b", re.I)
FUNCTION = re.compile(r"\bfunctions?\b", re.I)
CLAIM_INVOKE = re.compile(r"\binvocations?\b|lambda:InvokeFunction|\binvok(?:e|es|ed|ing)\s+"
    r"(?:(?:the|a|this|that|your)\s+)?(?:AWS\s+)?(?:Lambda(?:\s+functions?)?|functions?)\b|\bgọi\s+(?:hàm\s+)?Lambda\b", re.I)
SOURCE_INVOKE = re.compile(r"\binvok(?:e|es|ed|ing)\b|\binvocations?\b|lambda:InvokeFunction", re.I)


def policy_fingerprint():
    return sha256_bytes(canonical_json({"version": POLICY_VERSION,
        "antecedent_policy": "LITERAL_ANAPHORIC_PREFIX_DIAGNOSTIC_V1_NOT_ENTAILMENT",
        "antecedent_trigger": TRIGGER.pattern, "lambda": LAMBDA.pattern, "function": FUNCTION.pattern,
        "claim_invoke": CLAIM_INVOKE.pattern, "source_invoke": SOURCE_INVOKE.pattern,
        "action_policy": "CONSERVATIVE_LITERAL_INVOCATION_WITNESS_REVIEW_NOT_ENTAILMENT"}))


@dataclass(frozen=True)
class QualificationFlag:
    claim_id: str
    rule_id: str
    citation_ids: tuple[str, ...]
    detail: dict

    def public_record(self):
        return {"claim_id": self.claim_id, "rule_id": self.rule_id, "citation_ids": self.citation_ids}


@dataclass(frozen=True)
class QualifiedAnswer:
    native_response: QueryResponse
    flags: tuple[QualificationFlag, ...]
    source_context_sha256: str
    policy_sha256: str

    @property
    def status(self):
        return "REVIEW_REQUIRED" if self.flags else "NO_MECHANICAL_FLAGS"

    @property
    def display_response(self):
        return None if self.flags else self.native_response

    def public_payload(self):
        # REVIEW_REQUIRED must not leak unsafe native answer text through an
        # alternate top-level field; review storage is a separate owner-bound API.
        payload = {"schema_version": 1, "qualification": self.status,
            "request_id": self.native_response.request_id,
            "mode_requested": self.native_response.mode_requested, "mode_used": self.native_response.mode_used,
            "used_graph": self.native_response.used_graph, "answer_language": self.native_response.answer_language,
            "native_answer_status": self.native_response.answer_status, "semantic_verified": False,
            "qualification_policy_sha256": self.policy_sha256, "source_context_sha256": self.source_context_sha256,
            "response": None if self.display_response is None else self.display_response.model_dump(mode="json"),
            "review_flags": [f.public_record() for f in self.flags]}
        if len(canonical_json(payload)) > 96 * 1024:
            raise GenerationError("QUALIFICATION_PUBLIC_PAYLOAD_BOUND")
        return payload

    def review_record(self):
        return self.public_payload() | {"native_response_for_review": self.native_response.model_dump(mode="json"),
            "diagnostics": [{"claim_id": f.claim_id, "rule_id": f.rule_id, "detail": f.detail} for f in self.flags],
            "raw_response_rewritten": False}


def qualify_answer(response, source_context, documents, chunks, *, binding):
    validate_profiled_context(source_context, documents, chunks)
    if type(response) is not QueryResponse:
        raise GenerationError("QUALIFICATION_TYPED_RESPONSE_REQUIRED")
    response = QueryResponse.model_validate_json(response.model_dump_json())
    if (response.request_id != binding.request_id or response.mode_requested != binding.mode_requested
            or response.mode_used != binding.mode_used or response.used_graph != binding.used_graph
            or response.answer_language != source_context.language
            or response.evidence_bundle_sha256 != source_context.fingerprint
            or source_context.tenant_id != binding.tenant_id):
        raise GenerationError("QUALIFICATION_TASK_METHOD_OR_CONTEXT_BINDING")
    if response.mode_used.startswith("graph_") and not response.used_graph:
        raise GenerationError("QUALIFICATION_GRAPH_CONTRIBUTION_MISSING_NO_FALLBACK")
    allowed = {}
    for span in json.loads(source_context.span_lookup_json):
        chunk, doc = chunks[span["chunk_id"]], documents[span["document_version_id"]]
        cid = stable_id("profiled-cite-v12", tenant=binding.tenant_id, owner=binding.owner_id,
            request=binding.request_id, context=source_context.fingerprint, chunk=chunk.chunk_id,
            start=chunk.start_char+span["start_char"], end=chunk.start_char+span["end_char"])
        allowed[cid] = Citation(citation_id=cid, chunk_id=chunk.chunk_id, document_version_id=doc.document_version_id,
            title=doc.title, source_uri=doc.source_uri, section_path=chunk.section_path, quote=span["quote"])
    for citation in response.citations:
        if allowed.get(citation.citation_id) != citation:
            raise GenerationError("QUALIFICATION_CITATION_OWNER_OR_ORIGINAL_SPAN")
    by_id = {c.citation_id: c for c in response.citations}
    flags = []
    for claim in response.claims:
        citations = [by_id[key] for key in claim.citation_ids]
        for row in audit_claim_antecedents(citations, documents, chunks, source_context.chunk_ids):
            if row["status"] != "PARAGRAPH_PREFIX_PRESENT_AND_CITED":
                flags.append(QualificationFlag(claim.claim_id, row["status"], (row["citation_id"],), row))
        if (LAMBDA.search(claim.text + " " + source_context.question) and CLAIM_INVOKE.search(claim.text)
                and not any(SOURCE_INVOKE.search(c.quote) and (LAMBDA.search(c.quote) or FUNCTION.search(c.quote))
                    for c in citations)):
            flags.append(QualificationFlag(claim.claim_id, "INVOCATION_PREMISE_NOT_EXPLICIT_IN_CITATIONS",
                tuple(claim.citation_ids), {"policy": "CONSERVATIVE_LITERAL_INVOCATION_WITNESS",
                    "meaning": "Needs source review; absence of a literal operation term is not proof the claim is false."}))
    return QualifiedAnswer(response, tuple(flags), source_context.fingerprint, policy_fingerprint())
