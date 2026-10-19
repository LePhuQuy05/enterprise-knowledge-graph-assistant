"""Exact vector and lexical baseline primitives, independent of any model/API."""

import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from app.contracts import Chunk, DocumentVersion, embedding_input
from app.snapshot import LoadedSnapshot


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class LexicalIndex:
    """One authorized tenant per index; no-match queries return no lexical ranks."""

    def __init__(self, documents: Sequence[DocumentVersion], chunks: Sequence[Chunk], *, tenant_id: str):
        if any(r.tenant_id != tenant_id for r in (*documents, *chunks)):
            raise ValueError("LEXICAL_TENANT_MISMATCH")
        docs = {d.document_version_id: d for d in documents}
        self.chunks = sorted(chunks, key=lambda c: c.chunk_id)
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("LEXICAL_DUPLICATE_CHUNK")
        corpus = []
        for chunk in self.chunks:
            doc = docs.get(chunk.document_version_id)
            if doc is None or chunk.text != doc.normalized_text[chunk.start_char:chunk.end_char]:
                raise ValueError("LEXICAL_SOURCE_MISMATCH")
            corpus.append(lexical_tokens(embedding_input(doc, chunk)))
        self.tokens = tuple(frozenset(tokens) for tokens in corpus)
        self.index = BM25Okapi(corpus, k1=1.5, b=0.75) if any(corpus) else None

    def search(self, question: str, *, k: int = 20) -> list[tuple[str, float]]:
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("LEXICAL_CANDIDATE_CAP")
        tokens = lexical_tokens(question)
        if self.index is None or not tokens:
            return []
        scores = self.index.get_scores(tokens)
        # Matching terms may legitimately have zero/negative IDF in a tiny corpus.
        eligible = [i for i, own in enumerate(self.tokens) if own.intersection(tokens)]
        ranked = sorted(eligible, key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in ranked]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], constant: int = 60) -> list[str]:
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        unique = list(dict.fromkeys(ranking))
        for rank, chunk_id in enumerate(unique, 1):
            scores[chunk_id] += 1 / (constant + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


def retrieve_baseline(
    snapshot: LoadedSnapshot, question: str, query_vector: np.ndarray, *, mode: str = "hybrid",
    vector_k: int = 20, lexical_k: int = 20,
) -> list[str]:
    if mode not in {"vector", "hybrid"}:
        raise ValueError("Only vector/hybrid baselines are implemented in this slice")
    if not 1 <= vector_k <= 20 or not 1 <= lexical_k <= 20:
        raise ValueError("Baseline candidate caps must be 1..20")
    if query_vector.shape != (snapshot.manifest.embedding_recipe.dimension,) or query_vector.dtype != np.dtype("float32"):
        raise ValueError("Query vector must be float32 and match the snapshot dimension")
    if not np.isfinite(query_vector).all() or not np.isclose(
        np.linalg.norm(query_vector), 1, atol=1e-5, rtol=0,
    ):
        raise ValueError("Query vector must be finite and normalized")
    scores = snapshot.vectors @ query_vector
    ids = snapshot.chunk_ids
    vector_ranking = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:vector_k]
    vector_ids = [ids[i] for i in vector_ranking]
    if mode == "vector":
        return vector_ids
    lexical = LexicalIndex(snapshot.content.documents, snapshot.content.chunks,
                           tenant_id=snapshot.manifest.tenant_id)
    return reciprocal_rank_fusion([vector_ids, [key for key, _ in lexical.search(question, k=lexical_k)]])
