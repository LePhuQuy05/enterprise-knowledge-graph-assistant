from app.fixtures import fixture_vector
from scripts.evaluate_graph_components import ablation_indexes


def test_ablation_preserves_seeds_but_isolates_relations_and_mentions(loaded):
    mentions, relations = ablation_indexes(loaded)
    assert mentions.mentions_by_chunk == relations.mentions_by_chunk
    assert not mentions.adjacency and relations.adjacency
    assert mentions.chunks_by_entity and not relations.chunks_by_entity
    start = next(e.entity_id for e in loaded.content.entities if e.canonical_name == "Service A")
    mentions.seeds = relations.seeds = lambda *args: (start,)
    mention_result = mentions.retrieve("Service A", fixture_vector("Service A"), hops=2)
    relation_result = relations.retrieve("Service A", fixture_vector("Service A"), hops=2)
    assert mention_result.used_graph and all(not c.assertion_path for c in mention_result.graph_candidates)
    assert relation_result.used_graph and all(c.assertion_path for c in relation_result.graph_candidates)
    assert any(len(c.assertion_path) == 2 for c in relation_result.graph_candidates)
    assert mention_result.baseline_chunk_ids == relation_result.baseline_chunk_ids
