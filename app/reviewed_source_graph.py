"""Checksum-pinned relation/action evidence, separate from legacy graph releases.

This adapter validates stored source/review bindings without a parser, model or
network dependency. A path locates passages; it never proves an AWS workflow.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.collection_graph_v2 import load_collection_graph_v2
from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.external_http import strict_json
from app.generation import GenerationError
from app.semantic_extraction_v10 import RelationV10, SourceActionV1

RELEASE_VERSION = "reviewed-source-runtime-release-v1"
INTERPRETATION = "RETRIEVAL_EVIDENCE_NOT_TRANSITIVE_WORKFLOW_PROOF"
ARTIFACT_NAMES = frozenset({"relations-v2.json", "actions.json", "sidecar.json"})
MAX_FILE_BYTES = 64 * 1024 * 1024


def pinned_json(path, digest, *, array=False):
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise GenerationError("REVIEWED_GRAPH_FILE_BOUND")
    with path.open("rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or sha256_bytes(raw) != digest:
        raise GenerationError("REVIEWED_GRAPH_FILE_CHECKSUM")
    if array:
        rows = strict_json(b'{"rows":' + raw + b'}')["rows"]
        if type(rows) is not list:
            raise GenerationError("REVIEWED_GRAPH_SIDECAR_ARRAY_REQUIRED")
        return rows
    return strict_json(raw)


@dataclass(frozen=True)
class ReviewedSourceEdge:
    assertion_id: str
    subject_id: str
    object_id: str
    chunk_id: str
    kind: str
    family: str
    polarity: str
    source_record_json: str
    metadata_json: str

    def source_record(self):
        return strict_json(self.source_record_json.encode())

    def metadata(self):
        return strict_json(self.metadata_json.encode())

    def context_record(self):
        source, metadata = self.source_record(), self.metadata()
        return {"assertion_id": self.assertion_id, "chunk_id": self.chunk_id, "kind": self.kind,
            "record": metadata["record"], "nodes": metadata["nodes"], "clause_span": metadata["clause_span"],
            "origin": metadata["origin"], "human_reviewed": False,
            "untrusted_evidence_quote": source["evidence"]["quote"], "interpretation": INTERPRETATION}


@dataclass(frozen=True)
class ValidatedReviewedSourceGraph:
    relation_graph: object
    edges: tuple[ReviewedSourceEdge, ...]
    release_sha256: str
    release_json: str

    @property
    def overlay(self):
        return self.relation_graph.overlay

    @property
    def file_sha256(self):
        return self.release_sha256


def _validate_metadata(metadata, source, *, kind, recipe, chunks, documents, entities, mentions, scopes):
    chunk = chunks.get(source["chunk_id"])
    if chunk is None or chunk.document_version_id != source["document_version_id"]:
        raise GenerationError("REVIEWED_GRAPH_SOURCE_BINDING")
    document = documents[chunk.document_version_id]
    evidence = source["evidence"]
    if (evidence != {"start_char": chunk.start_char, "end_char": chunk.end_char, "quote": chunk.text}
            or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
        raise GenerationError("REVIEWED_GRAPH_FULL_SOURCE_REQUIRED")
    review, candidate = metadata["source_review"], metadata["syntax_witness"]
    digest = sha256_bytes(canonical_json(candidate))
    source_digest = sha256_bytes(chunk.text.encode())
    if (metadata["kind"] != kind or metadata["composite_recipe_sha256"] != recipe.fingerprint
            or metadata["native_recipe_sha256"] != recipe.parameters["native_recipe_sha256"]
            or metadata["native_v10_admission_claimed"] is not False
            or metadata["human_reviewed"] is not False or metadata["serving_published"] is not False
            or review["status"] != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
            or metadata["candidate_sha256"] != digest or review["candidate_sha256"] != digest
            or review["source_sha256"] != source_digest or candidate["source_sha256"] != source_digest
            or candidate["kind"] != kind or candidate["record"] != metadata["record"]
            or metadata["origin"] != candidate.get("origin", "V10_RECEIPT_COMPOSITE_POSTPROCESS")
            or metadata["origin"] not in {"V10_RECEIPT_COMPOSITE_POSTPROCESS", "DEPENDENCY_SOURCE_PROPOSAL_NOT_NATIVE_LLM_EXTRACTION"}):
        raise GenerationError("REVIEWED_GRAPH_REVIEW_RECIPE_OR_CANDIDATE_BINDING")
    record = (RelationV10 if kind == "RELATION" else SourceActionV1).model_validate_json(canonical_json(metadata["record"]))
    clause = metadata["clause_span"]
    if (not chunk.start_char <= clause["start_char"] < clause["end_char"] <= chunk.end_char
            or document.normalized_text[clause["start_char"]:clause["end_char"]] != record.clause_quote
            or clause["quote"] != record.clause_quote):
        raise GenerationError("REVIEWED_GRAPH_CLAUSE_SOURCE")
    endpoints = (record.subject, record.object) if kind == "RELATION" else (
        (record.actor, record.target) if record.enabler is None else (record.actor, record.target, record.enabler))
    if len(metadata["nodes"]) != len(endpoints):
        raise GenerationError("REVIEWED_GRAPH_ENDPOINT_COUNT")
    witness_roles = ("actor", "target") if len(endpoints) == 2 else ("actor", "target", "enabler")
    for node, endpoint, role in zip(metadata["nodes"], endpoints, witness_roles, strict=True):
        key, span = node["node_id"], node["source_span"]
        scope = node["identity_scope"]
        witness = candidate[role]
        identity_key = witness["identity"]["canonical_key"]
        expected_key = identity_key if scope == "CATALOG_TYPE" else stable_id("syntax-source-local-occurrence",
            key=identity_key, document=document.document_version_id, chunk=chunk.chunk_id,
            clause_start=chunk.text.index(record.clause_quote))
        if (key not in entities or node["name"] != endpoint.name or node["semantic_type"] != endpoint.semantic_type
                or entities[key].canonical_key != expected_key or node["source_witness"] != witness
                or span["quote"] != endpoint.name
                or not clause["start_char"] <= span["start_char"] < span["end_char"] <= clause["end_char"]
                or document.normalized_text[span["start_char"]:span["end_char"]] != endpoint.name
                or (chunk.chunk_id, key, span["start_char"], span["end_char"], endpoint.name) not in mentions
                or scope not in {"CATALOG_TYPE", "SOURCE_LOCAL"}
                or scope != witness["identity"]["scope"]
                or span["start_char"] != chunk.start_char + witness["identity"]["start_char"]
                or span["end_char"] != chunk.start_char + witness["identity"]["end_char"]):
            raise GenerationError("REVIEWED_GRAPH_ENDPOINT_SOURCE_OR_SCOPE")
        identity = (scope, chunk.document_version_id if scope == "SOURCE_LOCAL" else None)
        if key in scopes and scopes[key] != identity:
            raise GenerationError("REVIEWED_GRAPH_FALSE_ENTITY_MERGE")
        scopes[key] = identity
    return record


def load_reviewed_source_graph(path: Path, collection, *, trusted_sha256, allow_experimental=False):
    if allow_experimental is not True:
        raise GenerationError("REVIEWED_GRAPH_EXPERIMENTAL_OPT_IN_REQUIRED")
    release = pinned_json(path, trusted_sha256)
    if (release.get("version") != RELEASE_VERSION or release.get("source_manifest_sha256") != collection.manifest_sha256
            or release.get("collection_version") != collection.manifest.collection_version
            or set(release.get("artifacts", {})) != ARTIFACT_NAMES
            or release.get("human_reviewed") != 0 or release.get("serving_published") is not False):
        raise GenerationError("REVIEWED_GRAPH_RELEASE_BINDING")
    artifacts = {name: pinned_json(path.parent / name, digest, array=name == "sidecar.json")
        for name, digest in release["artifacts"].items()}
    graph = load_collection_graph_v2(path.parent / "relations-v2.json", collection,
        trusted_sha256=release["artifacts"]["relations-v2.json"], allow_experimental=True)
    actions, sidecar = artifacts["actions.json"], artifacts["sidecar.json"]
    recipe = ExtractionRecipe.model_validate(actions["recipe"])
    if (actions.get("version") != "source-actions-reviewed-composite-v1"
            or actions.get("source_manifest_sha256") != collection.manifest_sha256
            or actions.get("human_reviewed") != 0 or actions.get("serving_published") is not False
            or recipe.fingerprint != graph.overlay.extraction_recipe_sha256
            or len(actions["assertions"]) + len(graph.overlay.assertions) > 20000
            or any(e.canonicalization_version != recipe.canonicalization_version for e in graph.overlay.entities)):
        raise GenerationError("REVIEWED_GRAPH_ACTION_ENVELOPE_OR_RECIPE")
    rows = [a.model_dump(mode="json") for a in graph.overlay.assertions] + actions["assertions"]
    ids = {r["assertion_id"] for r in rows}
    meta = {m["assertion_id"]: m for m in sidecar}
    if len(ids) != len(rows) or len(meta) != len(sidecar) or ids != meta.keys():
        raise GenerationError("REVIEWED_GRAPH_MISSING_OR_DUPLICATE_SIDECAR")
    chunks, documents = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        chunks.update(shard.chunks)
        documents.update(shard.documents)
    entities = {e.entity_id: e for e in graph.overlay.entities}
    mentions = {(m.chunk_id, m.entity_id, m.start_char, m.end_char, m.quote) for m in graph.overlay.mentions}
    edges, scopes = [], {}
    relation_ids = {a.assertion_id for a in graph.overlay.assertions}
    for row in rows:
        key = row["assertion_id"]
        kind = "RELATION" if key in relation_ids else "ACTION"
        metadata = meta[key]
        record = _validate_metadata(metadata, row, kind=kind, recipe=recipe, chunks=chunks,
            documents=documents, entities=entities, mentions=mentions, scopes=scopes)
        left, right = (n["node_id"] for n in metadata["nodes"][:2])
        if kind == "ACTION":
            body = {k: v for k, v in row.items() if k != "assertion_id"}
            if (key != stable_id("source-action", **body) or row.get("version") != "source-action-assertion-v1"
                    or row["record"] != metadata["record"] or row["syntax_metadata"] != {k: v for k, v in metadata.items() if k != "assertion_id"}
                    or row["nodes"] != metadata["nodes"] or row["source_review"] != metadata["source_review"]
                    or row["clause_span"] != metadata["clause_span"] or row["candidate_sha256"] != metadata["candidate_sha256"]
                    or row["source_uri"] != documents[row["document_version_id"]].source_uri
                    or row["adapter_sha256"] != recipe.parameters["canonicalizer_sha256"]
                    or row["native_recipe_sha256"] != metadata["native_recipe_sha256"]
                    or row["human_reviewed"] is not False or row["serving_published"] is not False
                    or row["context_policy"] != "FULL_EXACT_CHUNK_COMPOSITE_SYNTAX_V3"):
                raise GenerationError("REVIEWED_GRAPH_ACTION_RECORD_BINDING")
            family = record.action_family
        else:
            if ((left, right) != (row["subject_id"], row["object_id"])
                    or row["predicate"] != record.predicate or row["polarity"] != record.polarity
                    or row["extraction_recipe_sha256"] != recipe.fingerprint
                    or row["qualifiers"] != [{"kind": "LIMITATION", "text": chunks[row["chunk_id"]].text}]):
                raise GenerationError("REVIEWED_GRAPH_RELATION_RECORD_BINDING")
            family = record.predicate
        edges.append(ReviewedSourceEdge(key, left, right, row["chunk_id"], kind, family, record.polarity,
            canonical_json(row).decode(), canonical_json(metadata).decode()))
    return ValidatedReviewedSourceGraph(graph, tuple(edges), trusted_sha256, canonical_json(release).decode())


def evidence_adjacency(graph):
    """Relations permit source lookup in either direction; actions follow actor→target.

    Even inverse relation traversal retains the stored semantic direction.
    Negative records are retained for inspection, never positive expansion.
    """
    adjacent = defaultdict(list)
    for edge in graph.edges:
        if edge.polarity != "POSITIVE":
            continue
        adjacent[edge.subject_id].append((edge, edge.object_id, "FORWARD"))
        if edge.kind == "RELATION":
            adjacent[edge.object_id].append((edge, edge.subject_id, "REVERSE_SOURCE_LOOKUP"))
    return adjacent
