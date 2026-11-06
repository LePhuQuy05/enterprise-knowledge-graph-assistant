"""Portable structural fixtures, not AWS answer-quality or human labels."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.authorization_evidence import (
    CONCLUSIONS,
    POLICY,
    AuthorizationCandidate,
    AuthorizationParticipant,
    LiteralSpan,
    load_authorization_facts,
    materialize,
    policy_hashes,
    source_span,
    validate_candidate,
)
from app.authorization_retrieval import combine_evidence_graph, prepare_authorization_answer
from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2
from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
from app.collections import build_collection, load_collection
from app.contracts import canonical_json, sha256_bytes
from app.fixtures import FIXTURE_EMBEDDING_MODEL, fixture_recipes, fixture_vector
from app.generation import GenerationRecipeV9, authorized_sources, generation_request
from app.reviewed_source_graph import load_reviewed_source_graph
from app.snapshot import SnapshotContent, build_snapshot
from app.syntax_graph_records import canonical_recipe, canonicalize
from tests.test_extraction import source
from tests.test_reviewed_source_runtime import write_release
from tests.test_source_syntax import FIXTURE
from tests.test_syntax_graph_records import fixture

ASSUME = "Lambda automatically assumes your execution role when you invoke your function."
INVOKE = "To invoke your function, Amazon S3 needs permission from the function's resource-based policy."
SDK = "If your function uses the AWS SDK to manage Amazon S3 resources, it also needs Amazon S3 permissions in its execution role."
DECRYPT = "To download an object encrypted with an AWS KMS key, you need kms:Decrypt permissions for the key."


def author(collection, doc, chunk, clause, kind, trigger, operation, condition, participants):
    span = source_span(doc, clause)
    full = LiteralSpan(start_char=chunk.start_char, end_char=chunk.end_char, quote=chunk.text)
    return AuthorizationCandidate(tenant_id="demo", collection_manifest_sha256=collection.manifest_sha256,
        document_version_id=doc.document_version_id, document_normalized_sha256=doc.normalized_sha256,
        chunk_id=chunk.chunk_id, chunk_sha256=chunk.text_sha256, source_uri=doc.source_uri, kind=kind, clause=span,
        trigger=source_span(doc, trigger, within=span), operation=source_span(doc, operation, within=span),
        conditions=(source_span(doc, condition, within=span),),
        participants=tuple(AuthorizationParticipant(role=role, identity_scope=scope,
            span=source_span(doc, text, within=full if role == "CONTEXT_SERVICE" and text not in clause else span))
            for role, text, scope in participants),
        requirement_combination="NOT_A_REQUIREMENT" if kind == "CONDITIONAL_ACTION" else "ALL_STATED_REQUIREMENTS_NECESSARY",
        conclusion=CONCLUSIONS[kind])


def review(candidate):
    return {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(candidate.model_dump(mode="json"))),
        "chunk_sha256": candidate.chunk_sha256, "reason_vi": "Synthetic structural test only; no independent source-quality label.",
        "human_reviewed": False}


def write_auth(root, collection, records):
    root.mkdir(exist_ok=True)
    raw = canonical_json({"records": records}) + b"\n"
    (root / "facts.json").write_bytes(raw)
    release = {"version": "authorization-evidence-release-v1", "policy": POLICY,
        "collection_manifest_sha256": collection.manifest_sha256, "records_sha256": sha256_bytes(raw),
        "policy_sha256": policy_hashes(), "human_reviewed": 0, "serving_published": False, "records_count": len(records)}
    path = root / "release.json"
    path.write_bytes(canonical_json(release) + b"\n")
    return path, sha256_bytes(path.read_bytes())


@pytest.fixture(scope="module")
def auth_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("authorization-fixture")
    candidate, judgment, _, _, native, parsed = fixture("ACTION")
    old_doc, old_chunk = source(candidate["evidence"]["quote"], suffix="old-action")
    judgment = judgment | {"source_sha256": old_chunk.text_sha256}
    es, ms, old_action, metadata = canonicalize(candidate, judgment, chunk=old_chunk, document=old_doc,
        parsed=parsed, native_recipe=native, installation=FIXTURE["installation"])
    recipe = canonical_recipe(native, FIXTURE["installation"])
    # Appended 'Lambda' is an explicit synthetic context anchor, never an AWS fact.
    doc0, c0 = source(ASSUME, suffix="auth-role")
    doc1, c1 = source(INVOKE + "\n\n" + SDK + "\n\nLambda", suffix="auth-permissions")
    doc2, c2 = source(DECRYPT, suffix="auth-kms")
    documents = {d.document_version_id: d for d in (old_doc, doc0, doc1, doc2)}
    chunks = {c.chunk_id: c for c in (old_chunk, c0, c1, c2)}
    _, embedding = fixture_recipes()
    base = build_snapshot(root / "snapshots", SnapshotContent(tuple(documents.values()), tuple(chunks.values())),
        {k: fixture_vector(c.text) for k, c in chunks.items()}, tenant_id="demo", purpose="FIXTURE",
        extraction_recipe=recipe, embedding_recipe=embedding, corpus_version="auth-test-v1", normalizer_version="fixture-v1",
        chunking_version="fixture-v1", tokenizer_version="fixture-v1")
    collection_path = build_collection(base.parent, [base], tenant_id="demo", embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    collection = load_collection(collection_path, tenant_id="demo", embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    overlay = CollectionGraphOverlayV2(tenant_id="demo", collection_version=collection.manifest.collection_version,
        collection_manifest_sha256=collection.manifest_sha256, graph_sha256=graph_fingerprint(collection.manifest_sha256, es, ms, ()),
        source_inventory_sha256="1" * 64, extraction_recipe_sha256=recipe.fingerprint, aliases_sha256="2" * 64,
        curation_sha256="3" * 64, upstream_review_queue_sha256="4" * 64, inventory_units=10, processed_units=1,
        entities=es, mentions=ms, assertions=())
    payloads = {"relations-v2.json": overlay.model_dump(mode="json"), "actions.json": {
        "version": "source-actions-reviewed-composite-v1", "recipe": recipe.model_dump(mode="json"),
        "source_manifest_sha256": collection.manifest_sha256, "assertions": [old_action], "serving_published": False,
        "human_reviewed": 0}, "sidecar.json": [metadata]}
    old_path = root / "reviewed"
    old_path.mkdir()
    path, digest = write_release(old_path, (collection, None, payloads))
    reviewed = load_reviewed_source_graph(path, collection, trusted_sha256=digest, allow_experimental=True)
    candidates = (
        author(collection, doc0, c0, ASSUME, "CONDITIONAL_ACTION", "assumes", "assumes", "when you invoke your function", (
            ("ACTOR", "Lambda", "CATALOG_TYPE"), ("RESOURCE", "your execution role", "SOURCE_LOCAL_RESOURCE"))),
        author(collection, doc1, c1, INVOKE, "PERMISSION_REQUIREMENT", "needs permission", "invoke", "To invoke your function", (
            ("PRINCIPAL", "Amazon S3", "CATALOG_TYPE"), ("RESOURCE", "your function", "SOURCE_LOCAL_RESOURCE"),
            ("POLICY", "the function's resource-based policy", "SOURCE_LOCAL_RESOURCE"), ("CONTEXT_SERVICE", "Lambda", "CATALOG_TYPE"))),
        author(collection, doc1, c1, SDK, "PERMISSION_REQUIREMENT", "needs", "manage", "If your function uses the AWS SDK to manage Amazon S3 resources", (
            ("PRINCIPAL", "it", "UNRESOLVED_REFERENCE"), ("RESOURCE", "Amazon S3 resources", "SOURCE_LOCAL_RESOURCE"),
            ("POLICY", "its execution role", "SOURCE_LOCAL_RESOURCE"), ("CONTEXT_SERVICE", "Amazon S3", "CATALOG_TYPE"))),
        author(collection, doc2, c2, DECRYPT, "PERMISSION_REQUIREMENT", "need", "kms:Decrypt", "To download an object encrypted with an AWS KMS key", (
            ("PRINCIPAL", "you", "UNRESOLVED_REFERENCE"), ("RESOURCE", "an AWS KMS key", "SOURCE_LOCAL_RESOURCE"),
            ("CONTEXT_SERVICE", "AWS KMS", "CATALOG_TYPE"))),
    )
    records = [materialize(c, review(c), collection, chunks, documents) for c in candidates]
    path, digest = write_auth(root / "authorization", collection, records)
    auth = load_authorization_facts(path, collection, chunks, documents, trusted_sha256=digest, allow_experimental=True)
    graph = combine_evidence_graph(collection, reviewed, auth)
    lexical_path = build_global_lexical_index(collection, root / "lexical")
    with load_global_lexical_index(lexical_path, collection) as lexical:
        yield SimpleNamespace(collection=collection, chunks=chunks, documents=documents, candidates=candidates,
            records=records, graph=graph, lexical=lexical, reviewed=reviewed, authorization=auth)


def prepare(case, question="Which execution role and resource-based policy permissions does Lambda need for Amazon S3?", **overrides):
    params = dict(tenant_id="demo", owner_id="fixture", run_id="auth-fixture", recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64),
        hops=2, allow_experimental=True, clock=lambda: 0.0)
    return prepare_authorization_answer(case.collection, case.lexical, case.graph, question, fixture_vector(question), **(params | overrides))


def test_possessive_resource_keeps_words_and_does_not_become_catalog_or_resolved_role(auth_case):
    candidate = auth_case.candidates[0]
    nodes = validate_candidate(candidate, auth_case.collection, auth_case.chunks, auth_case.documents)
    assert nodes[1]["name"] == "your execution role"
    assert nodes[1]["identity_scope"] == "SOURCE_LOCAL_RESOURCE"
    assert candidate.conditions[0].quote == "when you invoke your function"


@pytest.mark.parametrize("field,value", [("collection_manifest_sha256", "f" * 64), ("tenant_id", "other"),
    ("document_normalized_sha256", "e" * 64), ("chunk_sha256", "c" * 64), ("source_uri", "https://example.org/other"),
    ("chunk_id", "missing")])
def test_candidate_source_binding_fails_closed(auth_case, field, value):
    candidate = auth_case.candidates[0].model_copy(update={field: value})
    with pytest.raises(ValueError, match="AUTH_COLLECTION_DOCUMENT_CHUNK_BINDING"):
        validate_candidate(candidate, auth_case.collection, auth_case.chunks, auth_case.documents)


def test_permission_is_not_actual_action_or_grant(auth_case):
    raw = auth_case.candidates[1].model_dump(mode="json")
    raw.update(kind="CONDITIONAL_ACTION", conclusion=CONCLUSIONS["CONDITIONAL_ACTION"], requirement_combination="NOT_A_REQUIREMENT")
    raw["participants"][0]["role"] = "ACTOR"
    with pytest.raises(ValidationError, match="AUTH_OBLIGATION_TRIGGER"):
        AuthorizationCandidate.model_validate_json(canonical_json(raw))
    raw = auth_case.candidates[1].model_dump(mode="json") | {"permission_granted": True}
    with pytest.raises(ValidationError):
        AuthorizationCandidate.model_validate_json(canonical_json(raw))


def test_unresolved_reference_is_inspectable_but_not_a_traversal_node(auth_case):
    nodes = auth_case.records[2]["nodes"]
    principal = next(n for n in nodes if n["role"] == "PRINCIPAL")
    assert principal["name"] == "it" and principal["node_id"] is None
    fact = next(f for f in auth_case.graph.facts if f.assertion_id == auth_case.records[2]["assertion_id"])
    assert all(None not in pair[:2] for pair in fact.links)


def test_reference_cannot_be_silently_promoted_to_resource(auth_case):
    raw = auth_case.candidates[2].model_dump(mode="json")
    raw["participants"][0]["identity_scope"] = "SOURCE_LOCAL_RESOURCE"
    candidate = AuthorizationCandidate.model_validate_json(canonical_json(raw))
    with pytest.raises(ValueError, match="AUTH_SOURCE_LOCAL_CANNOT_MERGE_OR_RESOLVE"):
        validate_candidate(candidate, auth_case.collection, auth_case.chunks, auth_case.documents)


def test_catalog_service_cannot_replace_qualified_resource(auth_case):
    raw = auth_case.candidates[2].model_dump(mode="json")
    raw["participants"][1]["identity_scope"] = "CATALOG_TYPE"
    candidate = AuthorizationCandidate.model_validate_json(canonical_json(raw))
    with pytest.raises(ValueError, match="AUTH_EXACT_CATALOG_SERVICE_REQUIRED"):
        validate_candidate(candidate, auth_case.collection, auth_case.chunks, auth_case.documents)


def test_altered_exact_quote_and_out_of_primary_chunk_fail(auth_case):
    original = auth_case.candidates[0]
    for span in (original.clause.model_copy(update={"quote": "X" + original.clause.quote[1:]}),
                 original.clause.model_copy(update={"start_char": original.clause.start_char + 1, "end_char": original.clause.end_char + 1})):
        with pytest.raises(ValueError, match="AUTH_LITERAL_SPAN_OR_PRIMARY_CHUNK_BOUND"):
            validate_candidate(original.model_copy(update={"clause": span}), auth_case.collection, auth_case.chunks, auth_case.documents)


@pytest.mark.parametrize("change", [{"candidate_sha256": "a" * 64}, {"human_reviewed": True},
    {"chunk_sha256": "b" * 64}, {"status": "AI_APPROVED"}, {"reason_vi": ""}])
def test_review_must_bind_candidate_and_source_without_human_claim(auth_case, change):
    candidate = auth_case.candidates[1]
    with pytest.raises(ValueError, match="AUTH_BOUND_AI_SOURCE_REVIEW_REQUIRED"):
        materialize(candidate, review(candidate) | change, auth_case.collection, auth_case.chunks, auth_case.documents)


@pytest.mark.parametrize("mutation", ["grant", "source", "node", "duplicate"])
def test_even_rehashed_release_revalidates_every_record(auth_case, tmp_path, mutation):
    records = deepcopy(auth_case.records)
    if mutation == "grant":
        records[1]["permission_granted"] = True
    elif mutation == "source":
        records[1]["evidence"]["quote"] += "Grant everything."
    elif mutation == "node":
        records[1]["nodes"][1]["node_id"] = records[0]["nodes"][1]["node_id"]
    else:
        records.append(records[0])
    path, digest = write_auth(tmp_path, auth_case.collection, records)
    with pytest.raises(ValueError, match="AUTH_RECORD_REVALIDATION_OR_DUPLICATE"):
        load_authorization_facts(path, auth_case.collection, auth_case.chunks, auth_case.documents, trusted_sha256=digest, allow_experimental=True)


def test_release_wrong_checksum_missing_file_and_opt_in(auth_case, tmp_path):
    path, digest = write_auth(tmp_path, auth_case.collection, auth_case.records)
    for expected, kwargs in (("AUTH_LOCAL_EXPERIMENTAL_OPT_IN_REQUIRED", {}),
                             ("REVIEWED_GRAPH_FILE_CHECKSUM", {"allow_experimental": True, "trusted_sha256": "f" * 64})):
        with pytest.raises(ValueError, match=expected):
            load_authorization_facts(path, auth_case.collection, auth_case.chunks, auth_case.documents, **({"trusted_sha256": digest} | kwargs))
    (tmp_path / "facts.json").unlink()
    with pytest.raises(ValueError, match="REVIEWED_GRAPH_FILE_BOUND"):
        load_authorization_facts(path, auth_case.collection, auth_case.chunks, auth_case.documents, trusted_sha256=digest, allow_experimental=True)


@pytest.mark.parametrize("hops", [1, 2])
def test_typed_requirements_reach_common_generation_context_with_source(auth_case, hops):
    result = prepare(auth_case, hops=hops)
    assert result.outcome.mechanism_status == "USED_GRAPH" and result.prepared.binding.mode_used == f"graph_{hops}hop"
    packed = json.loads(result.prepared.packed.serialized)
    requirements = [r for r in packed["untrusted_graph_assertions"] if r["kind"] == "PERMISSION_REQUIREMENT"]
    assert requirements and all(r["permission_granted"] is False for r in requirements)
    assert all(r["record"]["requirement_combination"] == "ALL_STATED_REQUIREMENTS_NECESSARY" for r in requirements)
    for path in result.trace["packed_paths"]:
        assert path["hop"] <= hops and set(path["supporting_chunk_ids"]) <= set(result.outcome.context_chunk_ids)
    assert result.prepared.packed.proxy_tokens <= 4096
    _, prompt = generation_request(result.prepared.question, result.view, result.prepared.packed, "demo", result.prepared.recipe)
    # Generation deliberately rebuilds pinned source spans rather than trusting
    # graph annotations as authority. Facts affect retrieval, not truth rules.
    sources = authorized_sources(result.view, result.prepared.packed, "demo")
    assert all(r["untrusted_evidence_quote"] in [s["untrusted_text"] for s in sources] for r in requirements)
    assert "Amazon S3 needs permission" in prompt
    assert "NECESSARY_CONDITION_NOT_SUFFICIENT_OR_CONFIGURED" not in prompt
    assert result.trace["hidden_fallbacks"] == 0 and not result.trace["answer_generation_executed"]


def test_g2_crosses_two_distinct_facts_not_two_incidence_arcs(auth_case):
    result = prepare(auth_case)
    paths = [p for p in result.trace["packed_paths"] if p["hop"] == 2]
    assert paths and all(len(set(p["assertion_ids"])) == 2 for p in paths)
    assert all(p["condition_compatibility"] == "NOT_INFERRED" for p in paths)
    assert any("CONTEXT_SERVICE_EVIDENCE_ASSOCIATION" in p["projection_labels"] for p in paths)


def test_no_catalog_link_returns_explicit_graph_zero_without_fake_graph(auth_case):
    result = prepare(auth_case, question="How does Neptune Analytics manage snapshots?")
    assert result.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION"
    assert result.outcome.zero_expansion_reason == "NO_LINKED_ENTITY"
    assert result.outcome.effective_method == "graph_2hop" and not result.prepared.binding.used_graph
    assert not result.prepared.packed.assertion_ids and result.trace["hidden_fallbacks"] == 0


def test_timeout_never_prepares_a_hybrid_answer(auth_case):
    result = prepare(auth_case, traversal_clock=lambda: 10.0)
    assert result.outcome.mechanism_status == "BUDGET_EXCEEDED" and result.prepared is None
    assert result.outcome.effective_method is None


@pytest.mark.parametrize("change", [{"hops": 0}, {"hops": True}, {"time_budget_ms": 301}, {"tenant_id": "other"},
    {"allow_experimental": False}, {"answer_language": "ja"}])
def test_runtime_limits_and_tenant_fail_closed(auth_case, change):
    with pytest.raises(ValueError):
        prepare(auth_case, **change)


def test_changed_release_changes_request_identity_and_preserves_old_adapter(auth_case):
    original = prepare(auth_case)
    graph = replace(auth_case.graph, release_sha256="a" * 64)
    altered = SimpleNamespace(**(vars(auth_case) | {"graph": graph}))
    assert prepare(altered).prepared.binding.request_id != original.prepared.binding.request_id
    assert len(auth_case.reviewed.edges) == 1 and auth_case.reviewed.edges[0].kind == "ACTION"


def test_identical_possessive_words_in_another_document_do_not_merge(auth_case):
    original = auth_case.candidates[0]
    doc, chunk = source(ASSUME, suffix="another-auth-role")
    candidate = author(auth_case.collection, doc, chunk, ASSUME, "CONDITIONAL_ACTION", "assumes", "assumes",
        "when you invoke your function", (("ACTOR", "Lambda", "CATALOG_TYPE"), ("RESOURCE", "your execution role", "SOURCE_LOCAL_RESOURCE")))
    nodes = validate_candidate(candidate, auth_case.collection, auth_case.chunks | {chunk.chunk_id: chunk},
        auth_case.documents | {doc.document_version_id: doc})
    old = validate_candidate(original, auth_case.collection, auth_case.chunks, auth_case.documents)
    assert nodes[0]["node_id"] == old[0]["node_id"] and nodes[1]["node_id"] != old[1]["node_id"]


def test_cross_account_keeps_both_policy_requirements_and_full_scope(auth_case):
    text = ("Cross-account access requires permission in the key policy of the KMS key and in an IAM policy in the external user's account."
        "\n\nCross-account permission is effective only for the following operations:\nCryptographic operations\n\nAWS KMS")
    doc, chunk = source(text, suffix="cross-account-structural-fixture")
    clause = text.split("\n\n")[0]
    candidate = author(auth_case.collection, doc, chunk, clause, "PERMISSION_REQUIREMENT", "requires permission",
        "Cross-account access", "Cross-account access", (("POLICY", "the key policy of the KMS key", "SOURCE_LOCAL_RESOURCE"),
        ("POLICY", "an IAM policy in the external user's account", "SOURCE_LOCAL_RESOURCE"),
        ("RESOURCE", "the KMS key", "SOURCE_LOCAL_RESOURCE"), ("CONTEXT_SERVICE", "AWS KMS", "CATALOG_TYPE")))
    record = materialize(candidate, review(candidate), auth_case.collection, auth_case.chunks | {chunk.chunk_id: chunk},
        auth_case.documents | {doc.document_version_id: doc})
    assert len([n for n in record["nodes"] if n["role"] == "POLICY"]) == 2
    assert record["candidate"]["requirement_combination"] == "ALL_STATED_REQUIREMENTS_NECESSARY"
    assert "effective only" in record["evidence"]["quote"] and record["permission_granted"] is False
    with pytest.raises(ValidationError):
        AuthorizationCandidate.model_validate_json(canonical_json(candidate.model_dump(mode="json") | {"requirement_combination": "EITHER_POLICY"}))
