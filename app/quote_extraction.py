"""Quote-only model DTO v2; source offsets/local IDs are computed deterministically."""

from typing import Annotated, Literal

from pydantic import Field

from app.contracts import EntityType, Predicate, Qualifier, Record, canonical_json


class QuoteEntity(Record):
    local_id: Annotated[str, Field(min_length=1, max_length=300)]
    name: Annotated[str, Field(min_length=1, max_length=300)]
    entity_type: EntityType
    mention_contexts: Annotated[tuple[str, ...], Field(max_length=5)] = ()


class QuoteRelationship(Record):
    subject: Annotated[str, Field(min_length=1, max_length=300)]
    predicate: Predicate
    object: Annotated[str, Field(min_length=1, max_length=300)]
    evidence_quote: Annotated[str, Field(min_length=1, max_length=20000)]
    polarity: Literal["POSITIVE", "NEGATIVE"]
    qualifiers: Annotated[tuple[Qualifier, ...], Field(max_length=8)]


class QuoteExtractionOutput(Record):
    entities: Annotated[tuple[QuoteEntity, ...], Field(max_length=20)]
    relationships: Annotated[tuple[QuoteRelationship, ...], Field(max_length=30)]


def resolve_quotes(output: QuoteExtractionOutput, text: str) -> str:
    """Never choose an ambiguous occurrence or infer a relation from co-occurrence."""
    from app.extraction import ExtractionError

    def unique(quote, scope=text, base=0):
        first = scope.find(quote)
        if not quote or first < 0 or scope.find(quote, first + 1) >= 0:
            raise ExtractionError("EXTRACTION_QUOTE_CONTEXT_MISSING_OR_AMBIGUOUS")
        return {"start_char": base + first, "end_char": base + first + len(quote), "quote": quote}

    ids = [e.local_id for e in output.entities]
    if len(ids) != len(set(ids)):
        raise ExtractionError("EXTRACTION_DUPLICATE_LOCAL_ID")
    mapping = {key: "e" + str(index) for index, key in enumerate(ids)}
    evidence = []
    for rel in output.relationships:
        if rel.subject not in mapping or rel.object not in mapping:
            raise ExtractionError("EXTRACTION_UNKNOWN_ENDPOINT")
        evidence.append(unique(rel.evidence_quote))
    entities = []
    for entity in output.entities:
        spans = []
        for context in entity.mention_contexts:
            location = unique(context)
            spans.append(unique(entity.name, context, location["start_char"]))
        for rel, location in zip(output.relationships, evidence, strict=True):
            if entity.local_id in (rel.subject, rel.object):
                spans.append(unique(entity.name, rel.evidence_quote, location["start_char"]))
        if not spans:
            spans.append(unique(entity.name))
        dedup = {span["start_char"]: span for span in spans}
        if len(dedup) > 5:
            raise ExtractionError("EXTRACTION_TOO_MANY_RESOLVED_MENTIONS")
        entities.append({"local_id": mapping[entity.local_id], "name": entity.name,
                         "entity_type": entity.entity_type,
                         "mentions": [dedup[key] for key in sorted(dedup)]})
    relationships = [{"subject": mapping[rel.subject], "predicate": rel.predicate,
                      "object": mapping[rel.object], "polarity": rel.polarity,
                      "qualifiers": [q.model_dump(mode="json") for q in rel.qualifiers], "evidence": location}
                     for rel, location in zip(output.relationships, evidence, strict=True)]
    return canonical_json({"entities": entities, "relationships": relationships}).decode()
