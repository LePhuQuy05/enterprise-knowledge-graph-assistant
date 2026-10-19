"""Bounded logical-hop graph candidates with source paths; no transitive fact creation."""

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from app.extraction import normalize_alias
from app.retrieval import LexicalIndex, reciprocal_rank_fusion, retrieve_baseline
from app.snapshot import LoadedSnapshot


@dataclass(frozen=True)
class GraphLimits:
    seed_chunks: int = 5
    max_seed_entities: int = 8
    max_assertions_per_entity: int = 20
    max_assertion_visits: int = 200
    max_visited_entities: int = 80
    max_graph_candidate_chunks: int = 80
    max_mention_chunks_per_entity: int = 5
    time_budget_ms: int = 300

    def __post_init__(self):
        ceilings = {"seed_chunks": 5, "max_seed_entities": 8, "max_assertions_per_entity": 20,
                    "max_assertion_visits": 200, "max_visited_entities": 80,
                    "max_graph_candidate_chunks": 80, "max_mention_chunks_per_entity": 5,
                    "time_budget_ms": 300}
        if any(type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= cap
               for key, cap in ceilings.items()):
            raise ValueError("GRAPH_LIMIT_OUT_OF_RANGE")


@dataclass(frozen=True)
class GraphCandidate:
    chunk_id: str
    hop: int
    entity_path: tuple[str, ...]
    assertion_path: tuple[str, ...]
    supporting_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphRetrievalResult:
    ranked_chunk_ids: tuple[str, ...]
    baseline_chunk_ids: tuple[str, ...]
    graph_candidates: tuple[GraphCandidate, ...]
    seed_entity_ids: tuple[str, ...]
    mode_used: str
    used_graph: bool
    warnings: tuple[str, ...]
    assertion_visits: int
    visited_entities: int


class GraphIndex:
    def __init__(self, snapshot: LoadedSnapshot):
        self.snapshot = snapshot
        self.mentions_by_chunk = defaultdict(set)
        self.chunks_by_entity = defaultdict(set)
        self.adjacency = defaultdict(list)
        self.assertions = {r.assertion_id: r for r in snapshot.content.assertions}
        # Mention-only entities are retained independently of relation endpoints.
        for mention in snapshot.content.mentions:
            self.mentions_by_chunk[mention.chunk_id].add(mention.entity_id)
            self.chunks_by_entity[mention.entity_id].add(mention.chunk_id)
        for assertion in snapshot.content.assertions:
            if assertion.polarity == "POSITIVE":
                self.adjacency[assertion.subject_id].append(assertion)
                self.adjacency[assertion.object_id].append(assertion)

    def seeds(self, question, baseline_ids, limits):
        ranks = defaultdict(list)
        for rank, chunk in enumerate(baseline_ids[:limits.seed_chunks]):
            for entity in self.mentions_by_chunk[chunk]:
                ranks[entity].append(rank)
        question = normalize_alias(question)

        def priority(entity_id):
            entity = self.snapshot.entities[entity_id]
            # Only dictionary-approved identities get the service-alias preference.
            approved = entity.canonical_key.startswith("aws/") and entity.entity_type == "AWS_SERVICE"
            match = approved and any(re.search(r"(?<!\w)" + re.escape(normalize_alias(alias)) + r"(?!\w)", question)
                                     for alias in entity.aliases)
            return (not match, -len(ranks[entity_id]), min(ranks[entity_id]), entity_id)
        return tuple(sorted(ranks, key=priority)[:limits.max_seed_entities])

    def retrieve(self, question: str, query_vector: np.ndarray, *, hops: int = 1,
                 limits: GraphLimits | None = None, clock=time.monotonic) -> GraphRetrievalResult:
        if type(hops) is not int or hops not in (0, 1, 2):
            raise ValueError("GRAPH_HOPS_MUST_BE_0_1_2")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("GRAPH_QUESTION_LIMIT")
        limits = limits or GraphLimits()
        vector_ids = retrieve_baseline(self.snapshot, question, query_vector, mode="vector")
        lexical = LexicalIndex(self.snapshot.content.documents, self.snapshot.content.chunks,
                               tenant_id=self.snapshot.manifest.tenant_id)
        lexical_ids = [key for key, _ in lexical.search(question)]
        baseline = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids]))
        if hops == 0:
            return GraphRetrievalResult(baseline, baseline, (), (), "hybrid", False, (), 0, 0)
        started = clock()
        seed_candidates = self.seeds(question, baseline, limits)
        seeds = seed_candidates[:limits.max_visited_entities]
        scores = {key: float(score) for key, score in zip(self.snapshot.chunk_ids,
                  self.snapshot.vectors @ query_vector, strict=True)}
        queue = deque((entity, 0, (entity,), ()) for entity in seeds)
        visited = set(seeds)
        assertions_seen = set()
        candidates = {}
        warnings = {"GRAPH_TRUNCATED"} if len(seeds) < len(seed_candidates) else set()

        def expired():
            if (clock() - started) * 1000 >= limits.time_budget_ms:
                warnings.add("GRAPH_TIMEOUT")
                return True
            return False

        def add_candidate(chunk, depth, entity_path, assertion_path):
            bundle = tuple(dict.fromkeys(self.assertions[key].chunk_id for key in assertion_path))
            candidate = GraphCandidate(chunk, depth, entity_path, assertion_path, bundle)
            previous = candidates.get(chunk)
            if previous is not None:
                if (candidate.hop, candidate.assertion_path, candidate.entity_path) < (
                    previous.hop, previous.assertion_path, previous.entity_path
                ):
                    candidates[chunk] = candidate
            elif len(candidates) < limits.max_graph_candidate_chunks:
                candidates[chunk] = candidate
            else:
                warnings.add("GRAPH_TRUNCATED")

        while queue:
            if expired():
                break
            entity, depth, entity_path, path = queue.popleft()
            mentioning = sorted(self.chunks_by_entity[entity], key=lambda c: (-scores[c], c))
            if len(mentioning) > limits.max_mention_chunks_per_entity:
                warnings.add("GRAPH_TRUNCATED")
            for chunk in mentioning[:limits.max_mention_chunks_per_entity]:
                add_candidate(chunk, depth, entity_path, path)
            if depth == hops:
                continue
            adjacent = sorted(self.adjacency[entity], key=lambda a: (-scores[a.chunk_id], a.assertion_id))
            if len(adjacent) > limits.max_assertions_per_entity:
                warnings.add("GRAPH_TRUNCATED")
            for assertion in adjacent[:limits.max_assertions_per_entity]:
                if expired():
                    break
                if assertion.assertion_id in assertions_seen:
                    continue
                if len(assertions_seen) >= limits.max_assertion_visits:
                    warnings.add("GRAPH_TRUNCATED")
                    queue.clear()
                    break
                other = assertion.object_id if assertion.subject_id == entity else assertion.subject_id
                if other in entity_path:
                    continue
                if other not in visited and len(visited) >= limits.max_visited_entities:
                    warnings.add("GRAPH_TRUNCATED")
                    continue
                assertions_seen.add(assertion.assertion_id)
                next_path = (*path, assertion.assertion_id)
                next_entities = (*entity_path, other)
                add_candidate(assertion.chunk_id, depth + 1, next_entities, next_path)
                if other not in visited:
                    visited.add(other)
                    queue.append((other, depth + 1, next_entities, next_path))
        # Timeout returns the complete hybrid ranking, never a disguised graph success.
        if "GRAPH_TIMEOUT" in warnings:
            return GraphRetrievalResult(baseline, baseline, (), seeds, "hybrid", False,
                                        tuple(sorted(warnings)), len(assertions_seen), len(visited))
        graph = tuple(sorted(candidates.values(), key=lambda c: (c.hop, -scores[c.chunk_id], c.chunk_id)))
        if not graph:
            return GraphRetrievalResult(baseline, baseline, (), seeds, "hybrid", False,
                                        ("GRAPH_EMPTY",), len(assertions_seen), len(visited))
        ranked = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids, [c.chunk_id for c in graph]]))
        return GraphRetrievalResult(ranked, baseline, graph, seeds, f"graph_{hops}hop", True,
                                    tuple(sorted(warnings)), len(assertions_seen), len(visited))
