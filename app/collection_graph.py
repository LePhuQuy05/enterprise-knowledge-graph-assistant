"""Source-bound sparse graph overlay over a complete, immutable collection.

Literal mentions and semantic assertions are separate records. Re-anchoring an
existing assertion requires a unique identical quote in the current document;
it is not a new extraction or semantic approval.
"""

import re
import time
from collections import defaultdict, deque

from app.contracts import (
    Entity,
    EvidenceSpan,
    Mention,
    RelationAssertion,
    canonical_json,
    sha256_bytes,
)


def reanchor_assertions(previous_documents, assertions, documents, chunks):
    by_uri, by_document = defaultdict(list), defaultdict(list)
    for d in documents:
        by_uri[d.source_uri].append(d)
    for c in chunks:
        by_document[c.document_version_id].append(c)
    accepted, rejected, provenance = [], [], []
    for assertion in assertions:
        original = previous_documents[assertion.document_version_id]
        matches = []
        quote = assertion.evidence.quote
        for d in by_uri[original.source_uri]:
            if d.tenant_id != assertion.tenant_id or d.tenant_id != original.tenant_id:
                raise ValueError("COLLECTION_GRAPH_TENANT_BINDING")
            positions = [m.start() for m in re.finditer(re.escape(quote), d.normalized_text)]
            for start in positions:
                containers = [c for c in by_document[d.document_version_id]
                    if c.tenant_id == d.tenant_id and c.start_char <= start and start + len(quote) <= c.end_char]
                if containers:
                    c = min(containers, key=lambda c: (c.end_char - c.start_char, c.chunk_id))
                    matches.append((d, c, start))
        if len(matches) != 1:
            rejected.append({"previous_assertion_id": assertion.assertion_id, "reason": "SOURCE_QUOTE_MISSING_OR_AMBIGUOUS", "matches": len(matches)})
            continue
        d, c, start = matches[0]
        record = assertion.model_dump()
        record.update(document_version_id=d.document_version_id, chunk_id=c.chunk_id,
            evidence=EvidenceSpan(start_char=start, end_char=start + len(quote), quote=quote))
        record["assertion_id"] = RelationAssertion.expected_id(record)
        updated = RelationAssertion.model_validate(record)
        accepted.append(updated)
        provenance.append({"assertion_id": updated.assertion_id, "previous_assertion_id": assertion.assertion_id,
            "source_uri": d.source_uri, "document_version_id": d.document_version_id,
            "chunk_id": c.chunk_id, "current_text_sha256": c.text_sha256})
    return tuple(accepted), rejected, provenance


def literal_mentions(entities, chunks, *, canonicalization_version):
    patterns = [(e, re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE))
        for e in entities for alias in sorted(set((*e.aliases, e.canonical_name))) if len(alias) >= 3]
    mentions = {}
    for c in chunks:
        for e, pattern in patterns:
            if c.tenant_id != e.tenant_id:
                raise ValueError("COLLECTION_GRAPH_TENANT_BINDING")
            for match in pattern.finditer(c.text):
                m = Mention.create(tenant_id=c.tenant_id, entity_id=e.entity_id, chunk_id=c.chunk_id,
                    document_version_id=c.document_version_id, start_char=c.start_char + match.start(),
                    end_char=c.start_char + match.end(), quote=match.group(), canonicalization_version=canonicalization_version)
                mentions[m.mention_id] = m
                if len(mentions) > 100_000:
                    raise ValueError("COLLECTION_GRAPH_MENTION_CAP")
    return tuple(sorted(mentions.values(), key=lambda m: m.mention_id))


def reanchor_mentions(previous_documents, mentions, documents, chunks):
    """Transfer a mention only when its surrounding source context matches once."""
    by_uri, by_document = defaultdict(list), defaultdict(list)
    for d in documents:
        by_uri[d.source_uri].append(d)
    for c in chunks:
        by_document[c.document_version_id].append(c)
    result = {}
    for m in mentions:
        old = previous_documents[m.document_version_id]
        left, right = max(0, m.start_char - 64), min(len(old.normalized_text), m.end_char + 64)
        context = old.normalized_text[left:right]
        matches = []
        for d in by_uri[old.source_uri]:
            if d.tenant_id != m.tenant_id:
                raise ValueError("COLLECTION_GRAPH_TENANT_BINDING")
            for hit in re.finditer(re.escape(context), d.normalized_text):
                start = hit.start() + m.start_char - left
                for c in by_document[d.document_version_id]:
                    if c.tenant_id == d.tenant_id and c.start_char <= start and start + len(m.quote) <= c.end_char:
                        matches.append((d, c, start))
        if len(matches) != 1:
            continue
        d, c, start = matches[0]
        updated = Mention.create(tenant_id=m.tenant_id, entity_id=m.entity_id, chunk_id=c.chunk_id,
            document_version_id=d.document_version_id, start_char=start, end_char=start + len(m.quote),
            quote=m.quote, canonicalization_version=m.canonicalization_version)
        result[updated.mention_id] = updated
    return tuple(sorted(result.values(), key=lambda m: m.mention_id))


def service_entities(service_names, previous):
    entities = {e.entity_id: e for e in previous}
    known_aliases = {a.casefold() for e in previous for a in (*e.aliases, e.canonical_name)}
    for name in sorted(service_names):
        if name.casefold() in known_aliases:
            continue
        key = "aws/collection-service/" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        e = Entity.create(tenant_id="demo", canonical_key=key, canonical_name=name, entity_type="AWS_SERVICE",
            aliases=(name,), canonicalization_version="collection-service-literal-v1")
        entities[e.entity_id] = e
    return tuple(sorted(entities.values(), key=lambda e: e.entity_id))


def graph_fingerprint(collection_sha, entities, mentions, assertions):
    return sha256_bytes(canonical_json({"collection_manifest_sha256": collection_sha,
        "entities": [e.model_dump(mode="json") for e in entities],
        "mentions": [m.model_dump(mode="json") for m in mentions],
        "assertions": [a.model_dump(mode="json") for a in assertions]}))


def expand_graph(entities, mentions, assertions, baseline, scores, *, hops=1, time_budget_ms=None, clock=time.monotonic):
    """Bounded educational/experimental traversal; no generation or cloud promotion."""
    if type(hops) is not int or hops not in (1, 2) or len(baseline) > 40 or not set(baseline) <= scores.keys() or (
        len(entities) > 1000 or len(mentions) > 100_000 or len(assertions) > 5000 or len(scores) > 5000
    ):
        raise ValueError("COLLECTION_GRAPH_RETRIEVAL_INPUT")
    if time_budget_ms is not None and (type(time_budget_ms) is not int or not 1 <= time_budget_ms <= 300):
        raise ValueError("COLLECTION_GRAPH_TIME_BOUND")
    started, timed_out = clock() if time_budget_ms is not None else None, False
    entity_ids = {e.entity_id for e in entities}
    mention_chunks, chunks_by_entity, adjacent = defaultdict(set), defaultdict(set), defaultdict(list)
    for m in mentions:
        if m.entity_id not in entity_ids or m.chunk_id not in scores:
            raise ValueError("COLLECTION_GRAPH_MENTION_BINDING")
        mention_chunks[m.chunk_id].add(m.entity_id)
        chunks_by_entity[m.entity_id].add(m.chunk_id)
    for a in assertions:
        if {a.subject_id, a.object_id} - entity_ids or a.chunk_id not in scores:
            raise ValueError("COLLECTION_GRAPH_ASSERTION_BINDING")
        if a.polarity == "POSITIVE":
            adjacent[a.subject_id].append(a)
            adjacent[a.object_id].append(a)
    ranks = defaultdict(list)
    for rank, key in enumerate(baseline[:5]):
        for entity in mention_chunks[key]:
            ranks[entity].append(rank)
    seeds = sorted(ranks, key=lambda key: (-len(ranks[key]), min(ranks[key]), key))[:8]
    queue = deque((key, 0, (key,), ()) for key in seeds)
    visited, visited_assertions, candidates, truncated = set(seeds), set(), {}, False
    by_assertion = {a.assertion_id: a for a in assertions}

    def add(chunk, depth, entity_path, assertion_path):
        nonlocal truncated
        if chunk in candidates or len(candidates) < 40:
            value = {"chunk_id": chunk, "hop": depth, "entity_path": entity_path,
                "assertion_path": assertion_path, "supporting_chunk_ids": tuple(dict.fromkeys(by_assertion[a].chunk_id for a in assertion_path))}
            # Preserve the explanatory relation path when the same passage is
            # also a seed-entity mention. The latter must not erase provenance.
            if chunk not in candidates or (not bool(assertion_path), depth, assertion_path) < (
                not bool(candidates[chunk]["assertion_path"]), candidates[chunk]["hop"], candidates[chunk]["assertion_path"]
            ):
                candidates[chunk] = value
        else:
            truncated = True

    while queue:
        if time_budget_ms is not None and (clock() - started) * 1000 >= time_budget_ms:
            timed_out = True
            break
        entity, depth, path, rels = queue.popleft()
        for chunk in sorted(chunks_by_entity[entity], key=lambda k: (-scores[k], k))[:5]:
            add(chunk, depth, path, rels)
        if depth == hops:
            continue
        for a in sorted(adjacent[entity], key=lambda a: (-scores[a.chunk_id], a.assertion_id))[:20]:
            other = a.object_id if a.subject_id == entity else a.subject_id
            if other in path or a.assertion_id in visited_assertions:
                continue
            if len(visited_assertions) >= 200 or other not in visited and len(visited) >= 80:
                truncated = True
                continue
            visited_assertions.add(a.assertion_id)
            add(a.chunk_id, depth + 1, (*path, other), (*rels, a.assertion_id))
            if other not in visited:
                visited.add(other)
                queue.append((other, depth + 1, (*path, other), (*rels, a.assertion_id)))
    return {"seed_entity_ids": seeds, "graph_candidates": sorted(candidates.values(), key=lambda c: (c["hop"], -scores[c["chunk_id"]], c["chunk_id"])),
        "assertion_visits": len(visited_assertions), "visited_entities": len(visited), "truncated": truncated,
        "policy": "COLLECTION_SPARSE_GRAPH_V2_PRESERVE_RELATION_PROVENANCE" if time_budget_ms is None else "COLLECTION_REVIEWED_GRAPH_V3_BOUNDED_TIME",
        "requested_hops": hops, **({"timed_out": timed_out, "visited_assertion_ids": sorted(visited_assertions)}
            if time_budget_ms is not None else {})}
