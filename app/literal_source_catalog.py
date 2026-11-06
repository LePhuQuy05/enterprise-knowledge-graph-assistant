"""Bounded source-reading catalog from literal query probes, never annotations.

Document discovery uses actual vector/BM25 hits. The catalog retains whole
original chunks and is search data, not a graph fact or an answer context.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.collection_answering import validate_collection_lexical_binding
from app.contracts import QueryRequest, canonical_json, embedding_input, sha256_bytes
from app.generation import GenerationError
from app.graph_collection_migration import collection_records
from app.retrieval import lexical_tokens, reciprocal_rank_fusion
from app.scoped_method_flow import _vector
from app.scoped_query_v6 import scoped_queries
from app.strict_collection_methods import STOPWORDS

POLICY = "LITERAL_VECTOR_DOCUMENT_READING_CATALOG_V1"
MAX_PROBES, MAX_DOCS, MAX_CHUNKS, MAX_BYTES = 12, 16, 80, 72000


def literal_probes(search, native=()):
    if QueryRequest(question=search).question != search:
        raise GenerationError("SOURCE_CATALOG_CANONICAL_SEARCH")
    spans = []
    for match in re.finditer(r"[^,;?]+", search):
        part = match.group()
        # Search-only slices: never interpreted as independent propositions.
        spans.extend(re.split(r"\b(?:and|or|for|when|while|versus)\b", part, flags=re.I))
    conditions = [m.group() for m in re.finditer(r"\b(?:across|between)\s+(?:AWS\s+)?accounts\b|\bcross-account\b", search, re.I)]
    identifiers = re.findall(r"\b[a-z][a-z0-9-]*:[A-Za-z][A-Za-z0-9*]*\b", search)
    result = [search]
    # Conditions are kept as literal probes, never attached to another actor.
    for raw in (*conditions, *spans, *identifiers, *native):
        text = raw.strip(" .:;?\t\n")
        text = re.sub(r"^(?:How do|How does|Which|What|the)\s+", "", text, flags=re.I)
        if (5 <= len(text) <= 240 and text in search and text not in result
                and set(lexical_tokens(text)) - STOPWORDS):
            result.append(text)
        if len(result) == MAX_PROBES:
            break
    return tuple(result)


@dataclass(frozen=True)
class SourceCatalog:
    collection_sha256: str
    question: str
    search: str
    language: str
    source_ids: tuple[str, ...]
    message: str
    trace: dict

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json({"policy": POLICY, "collection": self.collection_sha256,
            "question": self.question, "search": self.search, "language": self.language,
            "ids": self.source_ids, "message": self.message, "trace": self.trace}))


def build_source_catalog(collection, lexical, runtime, *, question, search, language, plan=None):
    if (QueryRequest(question=question).question != question or language not in {"en", "vi"}
            or language == "en" and question != search):
        raise GenerationError("SOURCE_CATALOG_TASK_BINDING")
    validate_collection_lexical_binding(collection, lexical)
    native = () if plan is None else scoped_queries(plan, collection=collection, question=question,
        search=search, language=language)[1:]
    probes = literal_probes(search, native)
    docs, chunks, _ = collection_records(collection)
    rankings, vector_docs, lexical_docs = [], [], []
    for probe in probes:
        vector = [hit.chunk_id for hit in collection.search_vector(_vector(collection, runtime, probe), k=20)]
        sparse = [key for key, _ in lexical.search(probe, k=20)]
        rankings.extend((vector, sparse))
        vector_docs.append(tuple(dict.fromkeys(chunks[k].document_version_id for k in vector)))
        lexical_docs.append(tuple(dict.fromkeys(chunks[k].document_version_id for k in sparse)))
    # Explicit recipe change: document breadth is protected before reading.
    ordered_docs = tuple(dict.fromkeys(key for channel in (vector_docs, lexical_docs)
        for depth in range(3) for row in channel for key in row[depth:depth+1]))[:MAX_DOCS]
    ids = sorted(k for k, c in chunks.items() if c.document_version_id in ordered_docs)
    if not ids:
        body = canonical_json({"untrusted_question": question, "untrusted_english_search": search,
            "language": language, "untrusted_sources": []}).decode()
        return SourceCatalog(collection.manifest_sha256, question, search, language, (), body,
            {"policy": POLICY, "probes": probes, "document_ids": (), "rankings": rankings,
             "source_annotations_used": False, "whole_original_chunks": True, "omitted_by_bounds": 0})
    by_doc = {key: [] for key in ordered_docs}
    bm25 = BM25Okapi([lexical_tokens(embedding_input(docs[chunks[k].document_version_id], chunks[k])) for k in ids])
    sparse_rankings = []
    for probe in probes:
        tokens = set(lexical_tokens(probe))
        scores = bm25.get_scores(list(lexical_tokens(probe)))
        eligible = [i for i, k in enumerate(ids) if tokens.intersection(lexical_tokens(chunks[k].text))]
        sparse_rankings.append([ids[i] for i in sorted(eligible, key=lambda i: (-float(scores[i]), ids[i]))])
    fused = reciprocal_rank_fusion([*rankings, *sparse_rankings])
    rank = {key: i for i, key in enumerate(fused)}
    for key in ids:
        by_doc[chunks[key].document_version_id].append(key)
    for keys in by_doc.values():
        sequential = sorted(keys, key=lambda k: (chunks[k].start_char, k))
        keys.sort(key=lambda k: (rank.get(k, len(rank)), k))
        heads = keys[:2]
        neighbors = [key for head in heads for key in sequential[max(0, sequential.index(head)-1):sequential.index(head)+2]]
        keys[:] = tuple(dict.fromkeys((*heads, *neighbors, *keys)))
    body = {"untrusted_question": question, "untrusted_english_search": search,
        "language": language, "untrusted_sources": []}
    selected, omitted = [], 0
    for depth in range(max(map(len, by_doc.values()), default=0)):
        for document_id in ordered_docs:
            for key in by_doc[document_id][depth:depth+1]:
                c, doc = chunks[key], docs[document_id]
                if (c.tenant_id != collection.manifest.tenant_id or doc.tenant_id != c.tenant_id
                        or doc.normalized_text[c.start_char:c.end_char] != c.text
                        or sha256_bytes(c.text.encode()) != c.text_sha256):
                    raise GenerationError("SOURCE_CATALOG_ORIGINAL_SOURCE_BINDING")
                source = {"source_id": f"r{len(selected)+1}", "untrusted_title": doc.title,
                    "untrusted_section": c.section_path, "untrusted_text": c.text}
                candidate = body | {"untrusted_sources": [*body["untrusted_sources"], source]}
                if len(selected) >= MAX_CHUNKS or len(canonical_json(candidate)) > MAX_BYTES:
                    omitted += 1
                    continue
                selected.append(key)
                body = candidate
    trace = {"policy": POLICY, "probes": probes, "document_ids": ordered_docs, "rankings": rankings,
        "source_annotations_used": False, "whole_original_chunks": True, "omitted_by_bounds": omitted,
        "max_sources": MAX_CHUNKS, "max_bytes": MAX_BYTES, "max_documents": MAX_DOCS,
        "scope": "CANDIDATE_READING_ONLY_NOT_GENERATION_CONTEXT_OR_GRAPH_EDGE"}
    return SourceCatalog(collection.manifest_sha256, question, search, language, tuple(selected),
        canonical_json(body).decode(), trace)
