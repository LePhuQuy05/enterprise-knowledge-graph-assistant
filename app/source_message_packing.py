"""Budget the generation user message; keep graph audit records out of that budget.

Versioned alternative to context.py, whose historical receipts stay unchanged.
Generation still rebuilds all spans from the authorized snapshot. PackedContext
serialized is the exact generation user message for this policy, not an audit
envelope. Complete path support and original graph records live in a separate,
checksum-bound sidecar, never in the model's trusted instructions.
"""

from dataclasses import dataclass
from pathlib import Path

from app.context import PackedContext
from app.contracts import canonical_json, sha256_bytes
from app.generation import GenerationError, GenerationRecipeV9, generation_request
from app.ingestion.text import ProxyTokenizer

POLICY = "GENERATION_USER_MESSAGE_BUDGET_ATOMIC_SOURCE_PATHS_V1"
MAX_AUDIT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SourceMessagePacking:
    packed: PackedContext
    audit_json: str
    decisions_json: str
    recipe_sha256: str
    system_sha256: str
    user_message_sha256: str
    system_proxy_tokens: int

    @property
    def audit_sha256(self):
        return sha256_bytes(self.audit_json.encode())


def source_packing_recipe(recipe, tokenizer, *, max_chunks=8, proxy_token_budget=4096):
    return {"version": POLICY, "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "generation_recipe_sha256": recipe.fingerprint, "tokenizer": tokenizer.version,
        "budget_basis": "ENTIRE_GENERATION_USER_MESSAGE_INCLUDING_QUESTION_TITLES_SPAN_IDS",
        "system_prompt_budgeted_separately": True, "max_chunks": max_chunks,
        "user_message_proxy_token_budget": proxy_token_budget, "audit_in_model_input": False,
        "provider_exact_token_count": False}


def pack_source_message(snapshot, result, *, question, recipe, tenant_id, answer_language="en",
    max_chunks=8, proxy_token_budget=4096, graph_records=None):
    if (type(max_chunks) is not int or not 1 <= max_chunks <= 8 or type(proxy_token_budget) is not int
            or not 1 <= proxy_token_budget <= 4096 or not isinstance(recipe, GenerationRecipeV9)):
        raise GenerationError("SOURCE_MESSAGE_RECIPE_OR_BUDGET")
    if answer_language not in {"en", "vi"} or snapshot.manifest.tenant_id != tenant_id:
        raise GenerationError("SOURCE_MESSAGE_TENANT_OR_LANGUAGE")
    ranking = result.ranked_chunk_ids
    if len(ranking) > 40 or len(set(ranking)) != len(ranking):
        raise GenerationError("SOURCE_MESSAGE_RANKING_BOUND_OR_DUPLICATE")
    graph = {c.chunk_id: c for c in result.graph_candidates}
    if len(graph) != len(result.graph_candidates) or not set(graph) <= set(ranking):
        raise GenerationError("SOURCE_MESSAGE_GRAPH_CANDIDATE_BINDING")
    records = {} if graph_records is None else graph_records
    if len(records) > 20000 or len(result.graph_candidates) > 40:
        raise GenerationError("SOURCE_MESSAGE_GRAPH_RECORD_BOUND")
    tokenizer = ProxyTokenizer()
    specification = source_packing_recipe(recipe, tokenizer, max_chunks=max_chunks, proxy_token_budget=proxy_token_budget)
    fingerprint = sha256_bytes(canonical_json(specification))
    chosen, selected_facts, decisions, warnings = [], [], [], set()

    def request(keys, ids):
        provisional = PackedContext(tuple(keys), tuple("s" + str(i + 1) for i in range(len(keys))),
            tuple(ids), "", 0, ())
        system, prompt = generation_request(question, snapshot, provisional, tenant_id, recipe,
            answer_language=answer_language)
        return system, prompt, tokenizer.count(prompt)

    # Even an empty envelope/question has a cost; don't silently violate a tiny cap.
    _, _, empty_tokens = request((), ())
    if empty_tokens > proxy_token_budget:
        raise GenerationError("SOURCE_MESSAGE_EMPTY_ENVELOPE_EXCEEDS_BUDGET")
    for key in ranking:
        if key not in snapshot.chunks:
            raise GenerationError("SOURCE_MESSAGE_UNKNOWN_CHUNK")
        candidate = graph.get(key)
        path = () if candidate is None else candidate.assertion_path
        if len(path) > 2 or len(set(path)) != len(path) or candidate is not None and len(path) != candidate.hop:
            raise GenerationError("SOURCE_MESSAGE_DISTINCT_FACT_HOPS_REQUIRED")
        support = []
        for assertion_id in path:
            record = records.get(assertion_id)
            if not isinstance(record, dict) or record.get("assertion_id") != assertion_id:
                raise GenerationError("SOURCE_MESSAGE_UNKNOWN_OR_CHANGED_FACT")
            source_key = record.get("chunk_id")
            chunk = snapshot.chunks.get(source_key)
            if (chunk is None or source_key not in ranking or record.get("untrusted_evidence_quote") != chunk.text):
                raise GenerationError("SOURCE_MESSAGE_FACT_SOURCE_BINDING")
            support.append(source_key)
        support = tuple(dict.fromkeys(support))
        if candidate is not None and support != candidate.supporting_chunk_ids:
            raise GenerationError("SOURCE_MESSAGE_PATH_SUPPORT_MISMATCH")
        proposal = tuple(dict.fromkeys((*chosen, key, *support)))
        facts = tuple(dict.fromkeys((*selected_facts, *path)))
        decision = {"candidate_chunk_id": key, "path_assertion_ids": path, "supporting_chunk_ids": support,
            "proposed_chunk_ids": proposal, "proposed_source_count": len(proposal),
            "user_message_proxy_tokens": None, "selected": False, "reason": "MAX_CHUNKS"}
        if len(proposal) <= max_chunks:
            _, _, count = request(proposal, facts)
            decision.update(user_message_proxy_tokens=count, reason="MODEL_CONTEXT_BUDGET")
            if count <= proxy_token_budget:
                chosen, selected_facts = list(proposal), list(facts)
                decision.update(selected=True, reason="WITHIN_SOURCE_MESSAGE_BUDGET")
        if not decision["selected"]:
            warnings.add("SOURCE_MESSAGE_BUNDLE_SKIPPED")
        decisions.append(decision)
    system, prompt, count = request(chosen, selected_facts)
    if count > proxy_token_budget:
        raise GenerationError("SOURCE_MESSAGE_FINAL_BUDGET")
    source_ids = {key: "s" + str(i + 1) for i, key in enumerate(chosen)}
    audit = {"version": "source-message-evidence-audit-v1", "policy": POLICY, "packing_recipe": specification,
        "packing_recipe_sha256": fingerprint, "manifest_sha256": snapshot.manifest_sha256,
        "tenant_id": tenant_id, "answer_language": answer_language,
        "user_message_sha256": sha256_bytes(prompt.encode()), "system_sha256": sha256_bytes(system.encode()),
        "sources": [{"source_id": source_ids[key], "chunk_id": key,
            "document_version_id": snapshot.chunks[key].document_version_id,
            "start_char": snapshot.chunks[key].start_char, "end_char": snapshot.chunks[key].end_char,
            "text_sha256": snapshot.chunks[key].text_sha256,
            "source_uri": snapshot.documents[snapshot.chunks[key].document_version_id].source_uri} for key in chosen],
        "untrusted_graph_assertions": [records[k] | {"source_id": source_ids[records[k]["chunk_id"]]} for k in selected_facts],
        "audit_is_model_input": False, "provider_exact_token_count": False}
    raw_audit = canonical_json(audit)
    if len(raw_audit) > MAX_AUDIT_BYTES:
        raise GenerationError("SOURCE_MESSAGE_AUDIT_BOUND")
    packed = PackedContext(tuple(chosen), tuple(source_ids.values()), tuple(selected_facts), prompt, count,
        tuple(sorted(warnings)), tokenizer=tokenizer.version, provider_exact_token_count=False)
    return SourceMessagePacking(packed, raw_audit.decode(), canonical_json(decisions).decode(), fingerprint,
        sha256_bytes(system.encode()), sha256_bytes(prompt.encode()), tokenizer.count(system))
