"""Versioned opt-in whole-source context profiles; no evaluator annotations.

One source is still one original chunk. Both profiles use the same full-span
renderer and 4096 user-message proxy-token cap. Nothing is silently truncated.
Historical context/generation modules are deliberately unchanged.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from app.contracts import Chunk, DocumentVersion, Record, canonical_json, sha256_bytes
from app.generation import GenerationError
from app.ingestion.text import ProxyTokenizer


class SourceContextProfile(Record):
    version: Literal["whole-source-profile-v1"] = "whole-source-profile-v1"
    max_chunks: Literal[8, 12] = 8
    user_message_proxy_tokens: Literal[4096] = 4096
    span_characters: Literal[480] = 480
    span_boundary_min: Literal[160] = 160
    partial_source_policy: Literal["REJECT_WHOLE_BUNDLE_NEVER_TRUNCATE"] = "REJECT_WHOLE_BUNDLE_NEVER_TRUNCATE"

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True)
class ProfiledSourceContext:
    profile: SourceContextProfile
    tenant_id: str
    collection_sha256: str
    question: str
    language: str
    chunk_ids: tuple[str, ...]
    user_message: str
    proxy_tokens: int
    span_lookup_json: str
    source_audit_json: str

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json({"profile": self.profile.fingerprint, "tenant": self.tenant_id,
            "collection": self.collection_sha256, "question": self.question, "language": self.language,
            "chunks": self.chunk_ids, "message": self.user_message, "lookup": self.span_lookup_json,
            "sources": self.source_audit_json, "tokens": self.proxy_tokens}))


def source_spans(text, source_id):
    """Identical 480-char boundary policy; retain every non-whitespace byte."""
    start, spans, lookup = 0, [], []
    while start < len(text):
        end = min(start + 480, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start + 160, end), text.rfind("\n", start + 160, end))
            if boundary > start:
                end = boundary + 1
        quote = text[start:end]
        if quote.strip():
            if len(spans) >= 99:
                raise GenerationError("PROFILE_SOURCE_SPAN_BOUND")
            span_id = f"{source_id}p{len(spans)+1}"
            spans.append({"span_id": span_id, "text": quote})
            lookup.append({"source_id": source_id, "span_id": span_id,
                "start_char": start, "end_char": end, "quote": quote})
        start = end
    return spans, lookup


def render_profiled_context(documents, chunks, keys, *, question, language, tenant_id, collection_sha256,
    profile, enforce_budget=True):
    if type(profile) is not SourceContextProfile:
        raise GenerationError("PROFILE_TYPED_PROFILE_REQUIRED")
    SourceContextProfile.model_validate_json(profile.model_dump_json())
    if (not isinstance(question, str) or not question.strip() or len(question) > 2000 or language not in {"en", "vi"}
            or len(keys) > profile.max_chunks or len(set(keys)) != len(keys)
            or len(collection_sha256) != 64 or any(c not in "0123456789abcdef" for c in collection_sha256)):
        raise GenerationError("PROFILE_QUESTION_LANGUAGE_OR_CONTEXT_BOUND")
    sources, lookup, audit, validated = [], [], [], set()
    for i, key in enumerate(keys, 1):
        chunk = chunks.get(key)
        doc = None if chunk is None else documents.get(chunk.document_version_id)
        if (chunk is None or chunk.chunk_id != key or doc is None or doc.document_version_id != chunk.document_version_id
                or chunk.tenant_id != tenant_id or doc.tenant_id != tenant_id):
            raise GenerationError("PROFILE_SOURCE_TENANT_OR_ID")
        Chunk.model_validate_json(chunk.model_dump_json())
        if doc.document_version_id not in validated:
            DocumentVersion.model_validate_json(doc.model_dump_json())
            if sha256_bytes(doc.normalized_text.encode()) != doc.normalized_sha256:
                raise GenerationError("PROFILE_DOCUMENT_HASH")
            validated.add(doc.document_version_id)
        if doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text or sha256_bytes(chunk.text.encode()) != chunk.text_sha256:
            raise GenerationError("PROFILE_ORIGINAL_CHUNK_BINDING")
        sid = f"s{i}"
        spans, entries = source_spans(chunk.text, sid)
        if not spans:
            raise GenerationError("PROFILE_EMPTY_SOURCE")
        sources.append({"source_id": sid, "untrusted_document_title": doc.title, "spans": spans})
        lookup.extend(entry | {"chunk_id": key, "document_version_id": doc.document_version_id} for entry in entries)
        audit.append({"source_id": sid, "chunk_id": key, "document_version_id": doc.document_version_id,
            "start_char": chunk.start_char, "end_char": chunk.end_char, "text_sha256": chunk.text_sha256,
            "source_uri": doc.source_uri, "document_normalized_sha256": doc.normalized_sha256})
    prompt = canonical_json({"untrusted_question": question, "untrusted_sources": sources}).decode()
    count = ProxyTokenizer().count(prompt)
    if len(prompt.encode()) > 80000:
        raise GenerationError("PROFILE_INPUT_BYTE_BOUND")
    if enforce_budget and count > profile.user_message_proxy_tokens:
        raise GenerationError("PROFILE_USER_MESSAGE_TOKEN_BUDGET")
    return ProfiledSourceContext(profile, tenant_id, collection_sha256, question, language, tuple(keys), prompt, count,
        canonical_json(lookup).decode(), canonical_json(audit).decode())


def validate_profiled_context(context, documents, chunks, *, expected_question=None, expected_language=None,
    expected_collection_sha256=None):
    if type(context) is not ProfiledSourceContext:
        raise GenerationError("PROFILE_TYPED_CONTEXT_REQUIRED")
    if ((expected_question is not None and context.question != expected_question)
            or (expected_language is not None and context.language != expected_language)
            or (expected_collection_sha256 is not None and context.collection_sha256 != expected_collection_sha256)):
        raise GenerationError("PROFILE_TRUSTED_TASK_BINDING")
    rebuilt = render_profiled_context(documents, chunks, context.chunk_ids, question=context.question,
        language=context.language, tenant_id=context.tenant_id, collection_sha256=context.collection_sha256,
        profile=context.profile)
    if rebuilt != context:
        raise GenerationError("PROFILE_CONTEXT_REBUILD_MISMATCH")
    return rebuilt


def round_robin(bundles, order):
    groups = defaultdict(list)
    if len(order) != len(set(order)):
        raise GenerationError("PROFILE_QUERY_ORDER_DUPLICATE")
    for b in bundles:
        if b["query_index"] not in order:
            raise GenerationError("PROFILE_QUERY_ORDER_MISSING_GROUP")
        groups[b["query_index"]].append(b)
    return [groups[q][i] for i in range(max(map(len, groups.values()), default=0))
        for q in order if i < len(groups[q])]


def pack_profiled_context(documents, chunks, candidate_ids, bundles, *, question, language, tenant_id,
    collection_sha256, profile, balanced=False, query_order=(), graph_records=None):
    """Only packing changes across profiles. All support bundles remain atomic."""
    records = {} if graph_records is None else graph_records
    if (len(candidate_ids) > 40 or len(set(candidate_ids)) != len(candidate_ids) or len(bundles) > 40
            or len({b["bundle_id"] for b in bundles}) != len(bundles) or type(balanced) is not bool
            or len(records) > 20000):
        raise GenerationError("PROFILE_POOL_BUNDLE_OR_RECORD_BOUND")
    # Reject a corrupt/foreign candidate even if a bundle would later exceed a cap.
    for key in candidate_ids:
        render_profiled_context(documents, chunks, (key,), question=question, language=language,
            tenant_id=tenant_id, collection_sha256=collection_sha256, profile=profile, enforce_budget=False)
    for b in bundles:
        ids, facts = b["chunk_ids"], b.get("assertion_ids", ())
        if (type(b.get("query_index")) is not int or not 0 <= b["query_index"] <= 3
                or not ids or len(ids) != len(set(ids)) or not set(ids) <= set(candidate_ids)
                or len(facts) > 2 or len(set(facts)) != len(facts)):
            raise GenerationError("PROFILE_BUNDLE_BINDING")
        if facts:
            support = []
            for key in facts:
                r = records.get(key)
                c = None if r is None else chunks.get(r.get("chunk_id"))
                if (r is None or r.get("assertion_id") != key or c is None
                        or r.get("untrusted_evidence_quote") != c.text or c.tenant_id != tenant_id):
                    raise GenerationError("PROFILE_GRAPH_FACT_SOURCE_BINDING")
                support.append(c.chunk_id)
            if set(support) != set(ids):
                raise GenerationError("PROFILE_GRAPH_ATOMIC_SUPPORT")
    ordered = round_robin(bundles, query_order) if balanced else list(bundles)
    kwargs = dict(question=question, language=language, tenant_id=tenant_id, collection_sha256=collection_sha256, profile=profile)
    current = render_profiled_context(documents, chunks, (), **kwargs)
    decisions, selected_facts = [], []
    for b in ordered:
        proposal = tuple(dict.fromkeys((*current.chunk_ids, *b["chunk_ids"])))
        reason, count = "MAX_CHUNKS", None
        if len(proposal) <= profile.max_chunks:
            rendered = render_profiled_context(documents, chunks, proposal, **kwargs, enforce_budget=False)
            count = rendered.proxy_tokens
            reason = "TOKEN_BUDGET" if count > profile.user_message_proxy_tokens else "SELECTED"
            if reason == "SELECTED":
                current = rendered
                selected_facts.extend(b.get("assertion_ids", ()))
        decisions.append({"bundle_id": b["bundle_id"], "chunk_ids": list(b["chunk_ids"]),
            "assertion_ids": list(b.get("assertion_ids", ())), "reason": reason, "proposal_proxy_tokens": count})
    return current, {"profile": profile.model_dump(mode="json"), "decisions": decisions,
        "bundle_order": [b["bundle_id"] for b in ordered], "context_assertion_ids": list(dict.fromkeys(selected_facts)),
        "used_graph": bool(selected_facts), "hidden_fallbacks": 0, "partial_source_truncations": 0,
        "provider_exact_token_count": False, "source_message_sha256": sha256_bytes(current.user_message.encode())}
