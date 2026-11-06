"""Project reviewed typed frames into source-exact graph records, without publish.

Local resource references are further bound to document/chunk occurrence so a
raw receipt reused for identical text cannot merge roles from different sources.
Permissions remain separate triads until an explicitly designed graph adapter.
"""

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
from app.semantic_extraction import endpoint_position
from app.semantic_extraction_v7 import (
    CORE_TYPES,
    TypedFrameV7,
    catalog_entries,
    semantic_raw_recipe,
    validate_frame_v7,
)
from app.source_agent_policy import check_agent
from app.source_agent_policy import policy_recipe as agent_policy_recipe
from app.source_candidate_policy import retained


def canonical_recipe(native_recipe, composite):
    semantic_raw_recipe(native_recipe)
    if composite["native_recipe_sha256"] != native_recipe.fingerprint:
        raise ExtractionError("TYPED_CANONICAL_NATIVE_RECIPE_BINDING")
    return native_recipe.model_copy(update={
        "version": "typed-source-scoped-canonical-v1",
        "canonicalization_version": "typed-source-occurrence-v1:" + native_recipe.parameters["typed_catalog_sha256"],
        "parameters": native_recipe.parameters | {"native_recipe_sha256": native_recipe.fingerprint,
            "composite_recipe_sha256": sha256_bytes(canonical_json(composite)),
            "agent_promotion_policy": agent_policy_recipe(),
            "canonicalizer_implementation_sha256": sha256_bytes(Path(__file__).read_bytes())}})


def canonicalize_reviewed(decision, judgment, *, chunk, document, native_recipe, composite):
    check_source(chunk, document)
    if (judgment.get("status") != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
            or judgment.get("candidate_sha256") != sha256_bytes(canonical_json(decision))
            or not retained(decision)[0]):
        raise ExtractionError("TYPED_CANONICAL_BOUND_SOURCE_REVIEW_REQUIRED")
    if decision["kind"] != "RELATION":
        raise ExtractionError("TYPED_CANONICAL_PERMISSION_TRIAD_ADAPTER_NOT_IMPLEMENTED")
    frame = TypedFrameV7.model_validate_json(canonical_json(decision["normalized_record"]))
    allowed, reason = check_agent(frame)
    if not allowed:
        raise ExtractionError(reason)
    validation = validate_frame_v7(frame, chunk.text)
    if canonical_json(validation) != canonical_json(decision["validation"]):
        raise ExtractionError("TYPED_CANONICAL_SOURCE_VALIDATION_CHANGED")
    recipe = canonical_recipe(native_recipe, composite)
    evidence_start = chunk.start_char + chunk.text.index(frame.evidence_quote)
    evidence = EvidenceSpan(start_char=evidence_start, end_char=evidence_start + len(frame.evidence_quote), quote=frame.evidence_quote)
    entities, mentions, identities = [], [], []
    for endpoint, identity in zip((frame.subject, frame.object), validation["endpoint_identities"], strict=True):
        known = catalog_entries().get(endpoint.name.casefold().strip())
        if identity["scope"] == "CATALOG_TYPE":
            if known is None or known["canonical_key"] != identity["canonical_key"]:
                raise ExtractionError("TYPED_CANONICAL_CATALOG_BINDING")
            key, name, aliases = known["canonical_key"], known["canonical_name"], tuple(known["aliases"])
        else:
            key = stable_id("source-local-occurrence-v1", source_identity=identity["canonical_key"],
                document_version_id=document.document_version_id, chunk_id=chunk.chunk_id, evidence_start=evidence_start)
            name, aliases = endpoint.name, (endpoint.name,)
        entity = Entity.create(tenant_id=chunk.tenant_id, canonical_key=key, canonical_name=name,
            entity_type=CORE_TYPES[endpoint.semantic_type], aliases=aliases, canonicalization_version=recipe.canonicalization_version)
        start, end = endpoint_position(endpoint, frame.evidence_quote)
        mention = Mention.create(tenant_id=chunk.tenant_id, entity_id=entity.entity_id,
            chunk_id=chunk.chunk_id, document_version_id=chunk.document_version_id,
            start_char=evidence_start + start, end_char=evidence_start + end, quote=endpoint.name,
            canonicalization_version=recipe.canonicalization_version)
        if document.normalized_text[mention.start_char:mention.end_char] != mention.quote:
            raise ExtractionError("TYPED_CANONICAL_MENTION_SOURCE")
        entities.append(entity)
        mentions.append(mention)
        identities.append({"entity_id": entity.entity_id, "semantic_type": endpoint.semantic_type,
            "scope": identity["scope"], "canonical_key": key})
    assertion = RelationAssertion.create(recipe=recipe, tenant_id=chunk.tenant_id,
        subject_id=entities[0].entity_id, object_id=entities[1].entity_id,
        predicate=frame.predicate, polarity=frame.polarity, chunk_id=chunk.chunk_id,
        document_version_id=chunk.document_version_id, evidence=evidence,
        qualifiers=tuple(Qualifier.model_validate(q) for q in frame.model_dump(mode="json")["qualifiers"]))
    metadata = {"assertion_id": assertion.assertion_id, "semantic_types": identities,
        "modality": frame.modality, "voice": frame.voice, "trigger_quote": frame.trigger_quote,
        "candidate_sha256": judgment["candidate_sha256"], "source_review": judgment["status"],
        "review_reason_vi": judgment["reason_vi"], "native_recipe_sha256": native_recipe.fingerprint,
        "composite_recipe_sha256": sha256_bytes(canonical_json(composite)),
        "canonical_recipe_sha256": recipe.fingerprint, "human_reviewed": False,
        "serving_published": False, "model_calls": 0}
    return tuple(entities), tuple(mentions), assertion, metadata
