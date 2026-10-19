"""Common baseline/graph context packer; a path's supporting chunks are atomic."""

from dataclasses import dataclass

from app.contracts import canonical_json
from app.graph_retrieval import GraphRetrievalResult
from app.ingestion.text import ProxyTokenizer
from app.snapshot import LoadedSnapshot


@dataclass(frozen=True)
class PackedContext:
    chunk_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]
    serialized: str
    proxy_tokens: int
    warnings: tuple[str, ...]
    tokenizer: str = "cl100k_base"
    provider_exact_token_count: bool = False


def pack_context(snapshot: LoadedSnapshot, result: GraphRetrievalResult, *, max_chunks: int = 8,
                 proxy_token_budget: int = 4096) -> PackedContext:
    if type(max_chunks) is not int or not 1 <= max_chunks <= 8 or type(proxy_token_budget) is not int or not 1 <= proxy_token_budget <= 4096:
        raise ValueError("CONTEXT_BOUND_OUT_OF_RANGE")
    tokenizer = ProxyTokenizer()  # Hash-verified local asset; never an implicit download.
    graph = {c.chunk_id: c for c in result.graph_candidates}
    assertions = {a.assertion_id: a for a in snapshot.content.assertions}
    selected, used_assertions = [], []
    warnings = set()

    def serialize(chunk_ids, assertion_ids):
        sources = []
        ids = {key: "s" + str(i + 1) for i, key in enumerate(chunk_ids)}
        for key in chunk_ids:
            chunk = snapshot.chunks[key]
            document = snapshot.documents[chunk.document_version_id]
            sources.append({"source_id": ids[key], "chunk_id": key, "document_version_id": chunk.document_version_id,
                "title": document.title, "source_uri": document.source_uri,
                "start_char": chunk.start_char, "end_char": chunk.end_char, "untrusted_text": chunk.text})
        paths = []
        for key in assertion_ids:
            assertion = assertions[key]
            if assertion.chunk_id not in ids:
                raise ValueError("CONTEXT_PATH_EVIDENCE_MISSING")
            paths.append({"assertion_id": key, "source_id": ids[assertion.chunk_id],
                "subject": snapshot.entities[assertion.subject_id].canonical_name,
                "predicate": assertion.predicate, "object": snapshot.entities[assertion.object_id].canonical_name,
                "polarity": assertion.polarity, "qualifiers": [q.model_dump(mode="json") for q in assertion.qualifiers],
                "untrusted_evidence_quote": assertion.evidence.quote})
        return canonical_json({"untrusted_sources": sources, "untrusted_graph_assertions": paths}).decode()

    for key in result.ranked_chunk_ids:
        if key not in snapshot.chunks:
            raise ValueError("CONTEXT_UNKNOWN_CHUNK")
        candidate = graph.get(key)
        path = () if candidate is None else candidate.assertion_path
        for assertion_id in path:
            if assertion_id not in assertions:
                raise ValueError("CONTEXT_UNKNOWN_ASSERTION")
        support = tuple(dict.fromkeys(assertions[a].chunk_id for a in path))
        if candidate is not None and support != candidate.supporting_chunk_ids:
            raise ValueError("CONTEXT_PATH_SUPPORT_MISMATCH")
        proposal = list(dict.fromkeys([*selected, key, *support]))
        proposed_assertions = list(dict.fromkeys([*used_assertions, *path]))
        if len(proposal) > max_chunks:
            warnings.add("CONTEXT_BUNDLE_SKIPPED")
            continue
        serialized = serialize(proposal, proposed_assertions)
        if tokenizer.count(serialized) > proxy_token_budget:
            warnings.add("CONTEXT_BUNDLE_SKIPPED")
            continue
        selected, used_assertions = proposal, proposed_assertions
    serialized = serialize(selected, used_assertions)
    count = tokenizer.count(serialized)
    if count > proxy_token_budget:
        # Even the empty envelope may exceed an intentionally tiny test/config limit.
        raise ValueError("CONTEXT_ENVELOPE_EXCEEDS_BUDGET")
    return PackedContext(tuple(selected), tuple("s" + str(i + 1) for i in range(len(selected))),
                         tuple(used_assertions), serialized, count, tuple(sorted(warnings)))
