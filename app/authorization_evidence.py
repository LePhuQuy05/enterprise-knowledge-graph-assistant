"""Source-scoped authorization facts; requirements are not successful actions.

This is a separate local schema. It does not relax frozen V9/V10 validators,
resolve pronouns, infer permission grants, or mutate the corpus's chunk IDs.
Structural checks require a separate, explicitly labelled source review.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.contracts import Record, Sha256, canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.reviewed_source_graph import pinned_json
from app.semantic_extraction_v7 import CATALOG, catalog_entries

POLICY = "SOURCE_SCOPED_AUTHORIZATION_NARY_V1"
ORIGIN = "AI_AUTHORED_SOURCE_REVIEWED_NOT_NATIVE_EXTRACTION"
RESOURCE_HEAD = re.compile(r"\b(?:function|role|policy|key|account|resources)\b", re.I)
REFERENCE = re.compile(r"^(?:you|we|it|they|this|that|these|those|their|our|the following)\b", re.I)
CONCLUSIONS = {
    "CONDITIONAL_ACTION": "DOCUMENTED_CONDITIONAL_BEHAVIOR_NOT_OBSERVED_EXECUTION",
    "PERMISSION_REQUIREMENT": "NECESSARY_CONDITION_NOT_SUFFICIENT_OR_CONFIGURED",
    "TRUST_REQUIREMENT": "NECESSARY_CONDITION_NOT_SUFFICIENT_OR_CONFIGURED",
}


class LiteralSpan(Record):
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: Annotated[str, Field(min_length=1, max_length=6000)]

    @model_validator(mode="after")
    def exact_length(self):
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("AUTH_SPAN_LENGTH")
        return self


class AuthorizationParticipant(Record):
    role: Literal["ACTOR", "PRINCIPAL", "RESOURCE", "POLICY", "CONTEXT_SERVICE"]
    span: LiteralSpan
    identity_scope: Literal["CATALOG_TYPE", "SOURCE_LOCAL_RESOURCE", "SOURCE_LOCAL_PRINCIPAL", "UNRESOLVED_REFERENCE"]


class AuthorizationCandidate(Record):
    version: Literal["authorization-candidate-v1"] = "authorization-candidate-v1"
    tenant_id: str
    collection_manifest_sha256: Sha256
    document_version_id: str
    document_normalized_sha256: Sha256
    chunk_id: str
    chunk_sha256: Sha256
    source_uri: str
    kind: Literal["CONDITIONAL_ACTION", "PERMISSION_REQUIREMENT", "TRUST_REQUIREMENT"]
    clause: LiteralSpan
    trigger: LiteralSpan
    operation: LiteralSpan
    participants: Annotated[tuple[AuthorizationParticipant, ...], Field(min_length=2, max_length=10)]
    conditions: Annotated[tuple[LiteralSpan, ...], Field(max_length=4)]
    requirement_combination: Literal["NOT_A_REQUIREMENT", "ALL_STATED_REQUIREMENTS_NECESSARY"]
    conclusion: Literal["DOCUMENTED_CONDITIONAL_BEHAVIOR_NOT_OBSERVED_EXECUTION", "NECESSARY_CONDITION_NOT_SUFFICIENT_OR_CONFIGURED"]
    reference_resolution: Literal["NO_IMPLICIT_COREFERENCE_OR_CROSS_STATEMENT_RESOURCE_MERGE"] = "NO_IMPLICIT_COREFERENCE_OR_CROSS_STATEMENT_RESOURCE_MERGE"

    @model_validator(mode="after")
    def statement_kind(self):
        required = self.kind != "CONDITIONAL_ACTION"
        if self.conclusion != CONCLUSIONS[self.kind] or required != (
                self.requirement_combination == "ALL_STATED_REQUIREMENTS_NECESSARY"):
            raise ValueError("AUTH_REQUIREMENT_IS_NOT_ACTION_OR_GRANT")
        roles = {p.role for p in self.participants}
        if not ("RESOURCE" in roles and ("ACTOR" in roles if not required else "POLICY" in roles or "PRINCIPAL" in roles)):
            raise ValueError("AUTH_PARTICIPANT_ROLES")
        if self.kind in {"CONDITIONAL_ACTION", "TRUST_REQUIREMENT"} and not self.conditions:
            raise ValueError("AUTH_EXPLICIT_CONDITION_REQUIRED")
        obligation = bool(re.search(r"\b(?:needs?|requires?|must)\b", self.trigger.quote, re.I))
        if required != obligation:
            raise ValueError("AUTH_OBLIGATION_TRIGGER_IS_NOT_EXECUTED_ACTION")
        if len(set((p.role, p.span.start_char, p.span.end_char) for p in self.participants)) != len(self.participants):
            raise ValueError("AUTH_DUPLICATE_PARTICIPANT")
        return self


def source_span(document, quote, *, within=None, occurrence=1):
    """Authoring helper only: no semantic extraction or inferred antecedent."""
    start, end = (0, len(document.normalized_text)) if within is None else (within.start_char, within.end_char)
    matches = list(re.finditer(re.escape(quote), document.normalized_text[start:end]))
    if type(occurrence) is not int or not 1 <= occurrence <= len(matches):
        raise ValueError("AUTH_LITERAL_OCCURRENCE_REQUIRED")
    match = matches[occurrence - 1]
    return LiteralSpan(start_char=start + match.start(), end_char=start + match.end(), quote=quote)


def validate_candidate(candidate, collection, chunks, documents):
    """Check provenance and explicit identity scope, never certify entailment."""
    if type(candidate) is not AuthorizationCandidate:
        raise GenerationError("AUTH_VALIDATED_CANDIDATE_REQUIRED")
    chunk = chunks.get(candidate.chunk_id)
    document = documents.get(candidate.document_version_id)
    if (chunk is None or document is None or chunk.document_version_id != document.document_version_id
            or candidate.tenant_id != collection.manifest.tenant_id or chunk.tenant_id != candidate.tenant_id
            or document.tenant_id != candidate.tenant_id or candidate.collection_manifest_sha256 != collection.manifest_sha256
            or candidate.document_normalized_sha256 != document.normalized_sha256
            or sha256_bytes(document.normalized_text.encode()) != document.normalized_sha256
            or candidate.chunk_sha256 != chunk.text_sha256 or sha256_bytes(chunk.text.encode()) != chunk.text_sha256
            or candidate.source_uri != document.source_uri):
        raise GenerationError("AUTH_COLLECTION_DOCUMENT_CHUNK_BINDING")
    if document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text:
        raise GenerationError("AUTH_FULL_CHUNK_BINDING")
    spans = [candidate.clause, candidate.trigger, candidate.operation, *candidate.conditions]
    spans.extend(p.span for p in candidate.participants)
    for span in spans:
        if (not chunk.start_char <= span.start_char < span.end_char <= chunk.end_char
                or document.normalized_text[span.start_char:span.end_char] != span.quote):
            raise GenerationError("AUTH_LITERAL_SPAN_OR_PRIMARY_CHUNK_BOUND")
    for span in [candidate.trigger, candidate.operation, *candidate.conditions,
                 *(p.span for p in candidate.participants if p.role != "CONTEXT_SERVICE")]:
        if not candidate.clause.start_char <= span.start_char < span.end_char <= candidate.clause.end_char:
            raise GenerationError("AUTH_PARTICIPANT_OR_CONDITION_OUTSIDE_CLAUSE")
    nodes = []
    known_names = catalog_entries()
    for p in candidate.participants:
        known = known_names.get(p.span.quote.casefold())
        if p.identity_scope == "CATALOG_TYPE":
            if known is None or known["semantic_type"] != "SERVICE_PRODUCT" or p.role not in {"ACTOR", "PRINCIPAL", "CONTEXT_SERVICE"}:
                raise GenerationError("AUTH_EXACT_CATALOG_SERVICE_REQUIRED")
            # A substring of a larger word isn't a catalogue occurrence.
            if (p.span.start_char and document.normalized_text[p.span.start_char - 1].isalnum()
                    or p.span.end_char < len(document.normalized_text) and document.normalized_text[p.span.end_char].isalnum()):
                raise GenerationError("AUTH_CATALOG_WORD_BOUNDARY")
            key = stable_id("evidence-catalog", tenant=candidate.tenant_id, key=known["canonical_key"])
            name, aliases = known["canonical_name"], known["aliases"]
        elif p.identity_scope == "UNRESOLVED_REFERENCE":
            if p.role == "CONTEXT_SERVICE" or not REFERENCE.match(p.span.quote):
                raise GenerationError("AUTH_UNRESOLVED_REFERENCE_REQUIRED")
            key, name, aliases = None, p.span.quote, ()
        else:
            if known or p.role == "CONTEXT_SERVICE" or REFERENCE.match(p.span.quote):
                raise GenerationError("AUTH_SOURCE_LOCAL_CANNOT_MERGE_OR_RESOLVE")
            if p.identity_scope == "SOURCE_LOCAL_RESOURCE" and not RESOURCE_HEAD.search(p.span.quote):
                raise GenerationError("AUTH_RESOURCE_HEAD_REQUIRED")
            if p.identity_scope == "SOURCE_LOCAL_PRINCIPAL" and (p.role != "PRINCIPAL" or not re.fullmatch(
                    r"[a-z0-9-]+\.amazonaws\.com", p.span.quote)):
                raise GenerationError("AUTH_EXACT_SERVICE_PRINCIPAL_REQUIRED")
            # Role and full span prevent collapsing even same-word references
            # in two statements. A future coreference proof needs its own schema.
            key = stable_id("evidence-local", tenant=candidate.tenant_id, document=candidate.document_version_id,
                clause=candidate.clause.start_char, role=p.role, start=p.span.start_char, end=p.span.end_char,
                text=p.span.quote, scope=p.identity_scope)
            name, aliases = p.span.quote, (p.span.quote,)
        nodes.append({"node_id": key, "name": name, "aliases": tuple(aliases), "role": p.role,
            "identity_scope": p.identity_scope, "source_span": p.span.model_dump(mode="json")})
    if len({n["node_id"] for n in nodes if n["node_id"]}) < 2:
        raise GenerationError("AUTH_TWO_EXPLICIT_NODES_REQUIRED")
    return tuple(nodes)


@dataclass(frozen=True)
class ValidatedAuthorizationFacts:
    release_sha256: str
    collection_manifest_sha256: str
    records_json: tuple[str, ...]

    def records(self):
        from app.external_http import strict_json

        return tuple(strict_json(row.encode()) for row in self.records_json)


def policy_hashes():
    return {"implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "typed_catalog_sha256": sha256_bytes(CATALOG.read_bytes())}


def materialize(candidate, review, collection, chunks, documents):
    nodes = validate_candidate(candidate, collection, chunks, documents)
    digest = sha256_bytes(canonical_json(candidate.model_dump(mode="json")))
    if (set(review) != {"status", "candidate_sha256", "chunk_sha256", "reason_vi", "human_reviewed"}
            or review["status"] != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED" or review["human_reviewed"] is not False
            or review["candidate_sha256"] != digest or review["chunk_sha256"] != candidate.chunk_sha256
            or type(review["reason_vi"]) is not str or not review["reason_vi"].strip()):
        raise GenerationError("AUTH_BOUND_AI_SOURCE_REVIEW_REQUIRED")
    chunk = chunks[candidate.chunk_id]
    body = {"version": "authorization-evidence-fact-v1", "policy": POLICY, "origin": ORIGIN,
        "candidate": candidate.model_dump(mode="json"), "candidate_sha256": digest, "nodes": nodes,
        "review": review, "evidence": {"start_char": chunk.start_char, "end_char": chunk.end_char, "quote": chunk.text},
        "policy_sha256": policy_hashes(), "human_reviewed": False, "serving_published": False,
        "configuration_verified": False, "permission_granted": False, "native_extraction_claimed": False}
    return {"assertion_id": stable_id("authorization-fact-v1", **body)} | body


def load_authorization_facts(path, collection, chunks, documents, *, trusted_sha256, allow_experimental=False):
    if allow_experimental is not True:
        raise GenerationError("AUTH_LOCAL_EXPERIMENTAL_OPT_IN_REQUIRED")
    release = pinned_json(path, trusted_sha256)
    required = {"version", "policy", "collection_manifest_sha256", "records_sha256", "policy_sha256",
        "human_reviewed", "serving_published", "records_count"}
    if (set(release) != required or release["version"] != "authorization-evidence-release-v1"
            or release["policy"] != POLICY or release["collection_manifest_sha256"] != collection.manifest_sha256
            or release["policy_sha256"] != policy_hashes() or release["human_reviewed"] != 0
            or release["serving_published"] is not False or type(release["records_count"]) is not int
            or not 1 <= release["records_count"] <= 10000):
        raise GenerationError("AUTH_RELEASE_BINDING")
    payload = pinned_json(path.parent / "facts.json", release["records_sha256"])
    if set(payload) != {"records"} or type(payload["records"]) is not list or len(payload["records"]) != release["records_count"]:
        raise GenerationError("AUTH_RECORDS_ENVELOPE")
    rows, seen = [], set()
    for row in payload["records"]:
        candidate = AuthorizationCandidate.model_validate_json(canonical_json(row["candidate"]))
        expected = materialize(candidate, row["review"], collection, chunks, documents)
        if canonical_json(row) != canonical_json(expected) or row["assertion_id"] in seen:
            raise GenerationError("AUTH_RECORD_REVALIDATION_OR_DUPLICATE")
        seen.add(row["assertion_id"])
        rows.append(canonical_json(row).decode())
    return ValidatedAuthorizationFacts(trusted_sha256, collection.manifest_sha256, tuple(rows))
