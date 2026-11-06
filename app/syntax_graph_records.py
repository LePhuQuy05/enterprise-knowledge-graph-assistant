"""Source-reviewed materialization of composite syntax candidates, not native V10."""

from pathlib import Path

from app.contracts import (
    Entity,
    EvidenceSpan,
    Mention,
    Qualifier,
    RelationAssertion,
    canonical_json,
    sha256_bytes,
    stable_id,
)
from app.extraction import ExtractionError, check_source
from app.semantic_extraction_v7 import CORE_TYPES, catalog_entries
from app.semantic_extraction_v10 import RelationV10, SourceActionV1, semantic_raw_recipe
from app.source_normalization_v10 import normalize_record
from app.source_syntax import policy_recipe, validate_syntax_record

MODEL_SHA = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"


def canonical_recipe(native_recipe, installation):
    semantic_raw_recipe(native_recipe)
    if (installation["model_wheel_sha256"] != MODEL_SHA or installation["model_version"] != "3.8.0"
            or installation["spacy_version"] != "3.8.18"):
        raise ExtractionError("SYNTAX_GRAPH_PINNED_PARSER_REQUIRED")
    return native_recipe.model_copy(update={"version": "source-syntax-composite-canonical-v3",
        "canonicalization_version": "source-syntax-context-v3:" + native_recipe.parameters["typed_catalog_sha256"],
        "parameters": native_recipe.parameters | {"native_recipe_sha256": native_recipe.fingerprint, "syntax_policy": policy_recipe(),
            "parser_installation": installation, "canonicalizer_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "model_api_calls_for_postprocessing": 0}})


def canonicalize(candidate, judgment, *, chunk, document, parsed, native_recipe, installation):
    check_source(chunk, document)
    if (judgment.get("status") != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
            or judgment.get("candidate_sha256") != sha256_bytes(canonical_json(candidate))
            or candidate["source_sha256"] != chunk.text_sha256):
        raise ExtractionError("SYNTAX_GRAPH_BOUND_SOURCE_REVIEW_REQUIRED")
    record = (RelationV10 if candidate["kind"] == "RELATION" else SourceActionV1).model_validate_json(canonical_json(candidate["record"]))
    expected = validate_syntax_record(record, chunk.text, parsed, kind=candidate["kind"])
    if "normalization" in candidate:
        normalized, audit = normalize_record(candidate["normalization"]["raw_record"], chunk.text, kind=candidate["kind"])
        if canonical_json(normalized) != canonical_json(record.model_dump(mode="json")):
            raise ExtractionError("SYNTAX_GRAPH_NORMALIZATION_CHANGED")
        expected["normalization"] = audit
    if candidate.get("origin") == "DEPENDENCY_SOURCE_PROPOSAL_NOT_NATIVE_LLM_EXTRACTION":
        expected["origin"] = candidate["origin"]
    if canonical_json(expected) != canonical_json(candidate):
        raise ExtractionError("SYNTAX_GRAPH_SOURCE_PARSE_OR_CANDIDATE_CHANGED")
    recipe = canonical_recipe(native_recipe, installation)
    endpoints = (record.subject, record.object) if candidate["kind"] == "RELATION" else (record.actor, record.target, record.enabler)
    witnesses = (candidate["actor"], candidate["target"]) if candidate["kind"] == "RELATION" else (candidate["actor"], candidate["target"], candidate["enabler"])
    entities, mentions, nodes = [], [], []
    for endpoint, witness in zip(endpoints, witnesses, strict=True):
        if endpoint is None:
            continue
        identity = witness["identity"]
        known = catalog_entries().get(endpoint.name.casefold())
        if identity["scope"] == "CATALOG_TYPE":
            if not known or known["canonical_key"] != identity["canonical_key"]:
                raise ExtractionError("SYNTAX_GRAPH_CATALOG_BINDING")
            key, name, aliases = known["canonical_key"], known["canonical_name"], tuple(known["aliases"])
        else:
            key = stable_id("syntax-source-local-occurrence", key=identity["canonical_key"], document=document.document_version_id,
                chunk=chunk.chunk_id, clause_start=chunk.text.index(record.clause_quote))
            name, aliases = endpoint.name, (endpoint.name,)
        entity = Entity.create(tenant_id=chunk.tenant_id, canonical_key=key, canonical_name=name, aliases=aliases,
            entity_type=CORE_TYPES[endpoint.semantic_type], canonicalization_version=recipe.canonicalization_version)
        start, end = chunk.start_char + identity["start_char"], chunk.start_char + identity["end_char"]
        if document.normalized_text[start:end] != endpoint.name:
            raise ExtractionError("SYNTAX_GRAPH_MENTION_SOURCE")
        mention = Mention.create(tenant_id=chunk.tenant_id, entity_id=entity.entity_id, chunk_id=chunk.chunk_id,
            document_version_id=document.document_version_id, start_char=start, end_char=end, quote=endpoint.name,
            canonicalization_version=recipe.canonicalization_version)
        entities.append(entity)
        mentions.append(mention)
        nodes.append({"node_id": entity.entity_id, "name": endpoint.name, "semantic_type": endpoint.semantic_type,
            "identity_scope": identity["scope"], "source_span": {"start_char": start, "end_char": end, "quote": endpoint.name},
            "source_witness": witness, "witness_coordinate_system": "CLAUSE_UNICODE_OFFSETS_EXCEPT_IDENTITY_CHUNK_OFFSETS"})
    clause_start = chunk.start_char + chunk.text.index(record.clause_quote)
    metadata = {"kind": candidate["kind"], "native_recipe_sha256": native_recipe.fingerprint, "composite_recipe_sha256": recipe.fingerprint,
        "candidate_sha256": judgment["candidate_sha256"], "source_review": judgment, "nodes": nodes,
        "clause_span": {"start_char": clause_start, "end_char": clause_start + len(record.clause_quote), "quote": record.clause_quote},
        "record": record.model_dump(mode="json"), "syntax_witness": candidate, "serving_published": False, "human_reviewed": False,
        "native_v10_admission_claimed": False,
        "origin": candidate.get("origin", "V10_RECEIPT_COMPOSITE_POSTPROCESS"),
        "native_recipe_role": "BASELINE_SCHEMA_COMPATIBILITY_ONLY_NO_MODEL_CALL" if "origin" in candidate else "ORIGINAL_RECEIPT_RECIPE"}
    evidence = EvidenceSpan(start_char=chunk.start_char, end_char=chunk.end_char, quote=chunk.text)
    if candidate["kind"] == "RELATION":
        assertion = RelationAssertion.create(recipe=recipe, tenant_id=chunk.tenant_id, subject_id=entities[0].entity_id,
            object_id=entities[1].entity_id, predicate=record.predicate, polarity=record.polarity, chunk_id=chunk.chunk_id,
            document_version_id=document.document_version_id, evidence=evidence, qualifiers=(Qualifier(kind="LIMITATION", text=chunk.text),))
        return tuple(entities), tuple(mentions), assertion, metadata | {"assertion_id": assertion.assertion_id}
    body = {"version": "source-action-assertion-v1", "native_recipe_sha256": native_recipe.fingerprint,
        "adapter_sha256": sha256_bytes(Path(__file__).read_bytes()), "document_version_id": document.document_version_id,
        "chunk_id": chunk.chunk_id, "source_uri": document.source_uri, "record": record.model_dump(mode="json"), "nodes": nodes,
        "evidence": evidence.model_dump(mode="json"), "clause_span": metadata["clause_span"], "source_review": judgment,
        "candidate_sha256": judgment["candidate_sha256"], "human_reviewed": False, "serving_published": False,
        "context_policy": "FULL_EXACT_CHUNK_COMPOSITE_SYNTAX_V3", "syntax_metadata": metadata}
    assertion = {"assertion_id": stable_id("source-action", **body)} | body
    return tuple(entities), tuple(mentions), assertion, metadata | {"assertion_id": assertion["assertion_id"]}
