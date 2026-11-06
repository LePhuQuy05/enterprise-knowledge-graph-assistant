"""Conservative anaphoric-prefix diagnostics over immutable source bytes.

This is a literal dependency heuristic, not semantic entailment or permission
analysis. A complete paragraph prefix can still be irrelevant to a claim.
"""

import re

from app.contracts import sha256_bytes
from app.generation import GenerationError
from app.source_dependency_audit import missing_intervals

TRIGGER = re.compile(r"^\s*(Regardless\b|However\b|Therefore\b|This (?:policy|permission|role|key)\b|These (?:permissions|policies)\b)", re.I)


def _location(citation, documents, chunks):
    chunk = chunks.get(citation.chunk_id)
    doc = None if chunk is None else documents.get(chunk.document_version_id)
    if (chunk is None or doc is None or citation.document_version_id != doc.document_version_id
            or citation.source_uri != doc.source_uri or chunk.tenant_id != doc.tenant_id
            or sha256_bytes(doc.normalized_text.encode()) != doc.normalized_sha256
            or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text
            or sha256_bytes(chunk.text.encode()) != chunk.text_sha256):
        raise GenerationError("ANTECEDENT_CITATION_SOURCE_BINDING")
    if chunk.text.count(citation.quote) != 1:
        return doc, chunk, None
    start = chunk.start_char + chunk.text.index(citation.quote)
    return doc, chunk, (start, start + len(citation.quote))


def audit_claim_antecedents(citations, documents, chunks, context_ids):
    """Expose uncited/missing prefixes; never repair or rewrite native claims."""
    locations = [(_location(c, documents, chunks), c) for c in citations]
    rows = []
    for (doc, anchor, bounds), citation in locations:
        trigger = TRIGGER.match(citation.quote)
        if trigger is None:
            continue
        if bounds is None:
            rows.append({"citation_id": citation.citation_id, "status": "AMBIGUOUS_QUOTE_LOCATION_REQUIRES_REVIEW"})
            continue
        start, _ = bounds
        paragraph_start = doc.normalized_text.rfind("\n\n", 0, start) + 2
        if paragraph_start == 1:  # rfind == -1
            paragraph_start = 0
        if not doc.normalized_text[paragraph_start:start].strip():
            end = paragraph_start
            while end > 0 and doc.normalized_text[end-1].isspace():
                end -= 1
            paragraph_start = doc.normalized_text.rfind("\n\n", 0, end) + 2
            if paragraph_start == 1:
                paragraph_start = 0
        interval = (paragraph_start, start)
        context_intervals, cited_intervals, support = [], [], []
        for key in context_ids:
            c = chunks[key]
            if c.document_version_id == doc.document_version_id:
                if c.tenant_id != anchor.tenant_id or doc.normalized_text[c.start_char:c.end_char] != c.text:
                    raise GenerationError("ANTECEDENT_CONTEXT_SOURCE_BINDING")
                context_intervals.append((c.start_char, c.end_char))
        for (other_doc, _, own_bounds), _ in locations:
            if other_doc.document_version_id == doc.document_version_id and own_bounds:
                cited_intervals.append(own_bounds)
        for key, c in chunks.items():
            if c.document_version_id == doc.document_version_id and c.start_char < start and c.end_char > paragraph_start:
                support.append(key)

        def nonempty_gaps(intervals):
            return [(a, b) for a, b in missing_intervals(*interval, intervals) if doc.normalized_text[a:b].strip()]

        context_gaps, citation_gaps = nonempty_gaps(context_intervals), nonempty_gaps(cited_intervals)
        rows.append({"citation_id": citation.citation_id, "anchor_chunk_id": anchor.chunk_id,
            "document_version_id": doc.document_version_id, "trigger": trigger.group().strip(),
            "status": "MISSING_ANTECEDENT_IN_CONTEXT" if context_gaps else
                "ANTECEDENT_AVAILABLE_BUT_NOT_CITED" if citation_gaps else "PARAGRAPH_PREFIX_PRESENT_AND_CITED",
            "prefix_start_char": paragraph_start, "prefix_end_char": start,
            "prefix_sha256": sha256_bytes(doc.normalized_text[paragraph_start:start].encode()),
            "context_missing_ranges": context_gaps, "citation_missing_ranges": citation_gaps,
            "candidate_support_chunk_ids": tuple(sorted(support, key=lambda k: (chunks[k].start_char, k))),
            "policy": "LITERAL_ANAPHORIC_PREFIX_DIAGNOSTIC_V1_NOT_ENTAILMENT"})
    return rows
