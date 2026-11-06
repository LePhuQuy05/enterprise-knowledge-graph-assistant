"""Canonical candidate records for native V9, never implicit serving approval."""

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
from app.semantic_extraction_v9 import RelationV9, semantic_raw_recipe, validate_relation
from app.source_entity_boundary import check_reference, policy_recipe


def canonical_recipe(native_recipe):
    semantic_raw_recipe(native_recipe)
    return native_recipe.model_copy(update={"version": "native-v9-clause-canonical-v1",
        "canonicalization_version": "source-occurrence-head-v9:" + native_recipe.parameters["typed_catalog_sha256"],
        "parameters": native_recipe.parameters | {"native_recipe_sha256": native_recipe.fingerprint,
            "entity_reference_promotion_policy": policy_recipe(),
            "canonicalizer_implementation_sha256": sha256_bytes(Path(__file__).read_bytes())}})


def canonicalize_relation(candidate, judgment, *, chunk, document, native_recipe):
    check_source(chunk, document)
    if (judgment.get("status") != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
            or judgment.get("candidate_sha256") != sha256_bytes(canonical_json(candidate))):
        raise ExtractionError("V9_CANONICAL_BOUND_SOURCE_REVIEW_REQUIRED")
    record = RelationV9.model_validate_json(canonical_json(candidate["record"]))
    for endpoint in (record.subject, record.object):
        allowed, reason = check_reference(endpoint, record.clause_quote)
        if not allowed:
            raise ExtractionError(reason)
    expected = validate_relation(record, chunk.text) | {"index": candidate["index"]}
    if canonical_json(expected) != canonical_json(candidate):
        raise ExtractionError("V9_CANONICAL_VALIDATION_OR_SOURCE_CHANGED")
    recipe = canonical_recipe(native_recipe)
    entities, mentions, semantic = [], [], []
    for endpoint, identity in zip((record.subject, record.object), candidate["endpoint_identities"], strict=True):
        known = catalog_entries().get(endpoint.name.casefold())
        if identity["scope"] == "CATALOG_TYPE":
            if known is None or known["canonical_key"] != identity["canonical_key"]:
                raise ExtractionError("V9_CANONICAL_CATALOG_BINDING")
            key, name, aliases = known["canonical_key"], known["canonical_name"], tuple(known["aliases"])
        else:
            key = stable_id("source-local-occurrence-v9", source_identity=identity["canonical_key"],
                document_version_id=document.document_version_id, chunk_id=chunk.chunk_id,
                clause_start=candidate["clause_span"]["start_char"])
            name, aliases = endpoint.name, (endpoint.name,)
        entity = Entity.create(tenant_id=chunk.tenant_id, canonical_key=key, canonical_name=name,
            entity_type=CORE_TYPES[endpoint.semantic_type], aliases=aliases, canonicalization_version=recipe.canonicalization_version)
        mention = Mention.create(tenant_id=chunk.tenant_id, entity_id=entity.entity_id, chunk_id=chunk.chunk_id,
            document_version_id=document.document_version_id, start_char=chunk.start_char + identity["start_char"],
            end_char=chunk.start_char + identity["end_char"], quote=endpoint.name,
            canonicalization_version=recipe.canonicalization_version)
        if document.normalized_text[mention.start_char:mention.end_char] != mention.quote:
            raise ExtractionError("V9_CANONICAL_MENTION_SOURCE")
        entities.append(entity)
        mentions.append(mention)
        semantic.append({"entity_id": entity.entity_id, "semantic_type": endpoint.semantic_type,
            "head_quote": endpoint.head_quote, "restrictions": endpoint.restrictions, "identity_scope": identity["scope"]})
    evidence = EvidenceSpan(start_char=chunk.start_char, end_char=chunk.end_char, quote=chunk.text)
    assertion = RelationAssertion.create(recipe=recipe, tenant_id=chunk.tenant_id, subject_id=entities[0].entity_id,
        object_id=entities[1].entity_id, predicate=record.predicate, polarity=record.polarity,
        chunk_id=chunk.chunk_id, document_version_id=document.document_version_id, evidence=evidence,
        qualifiers=(Qualifier(kind="LIMITATION", text=chunk.text),))
    metadata = {"assertion_id": assertion.assertion_id, "semantic_types": semantic, "modality": record.modality,
        "voice": record.voice, "trigger_quote": record.trigger_quote, "clause_span": {
            "start_char": chunk.start_char + candidate["clause_span"]["start_char"],
            "end_char": chunk.start_char + candidate["clause_span"]["end_char"], "quote": record.clause_quote},
        "native_recipe_sha256": native_recipe.fingerprint, "canonical_recipe_sha256": recipe.fingerprint,
        "candidate_sha256": judgment["candidate_sha256"], "source_review": judgment,
        "human_reviewed": False, "serving_published": False, "context_policy": candidate["context_policy"]}
    return tuple(entities), tuple(mentions), assertion, metadata
