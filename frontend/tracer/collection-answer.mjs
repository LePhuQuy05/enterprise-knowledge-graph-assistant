// Owned collection answers use run source routes, never legacy /v1/sources.
export function validateCollectionAnswer(result) {
  const a = result?.response, fail = () => { throw new Error("INVALID_COLLECTION_ANSWER"); };
  const text = (v, max = 16000) => typeof v === "string" && v.length > 0 && v.length <= max;
  if (!a || !/^astage_[a-f0-9]{64}$/.test(a.request_id) || a.snapshot_version !== result.collection_version ||
      !["ANSWERED", "PARTIAL", "ABSTAINED"].includes(a.answer_status) || !text(a.answer) ||
      !["hybrid", "vector"].includes(a.mode_used) || a.mode_used !== result.method_id || a.used_graph !== false ||
      !["en", "vi"].includes(a.answer_language) || !Array.isArray(a.claims) || a.claims.length > 4 ||
      !Array.isArray(a.citations) || a.citations.length > 8) fail();
  const citations = new Map(), claims = new Set(), used = new Set();
  for (const c of a.citations) {
    const source = result.context_sources.find(s => s.chunk_id === c.chunk_id);
    if (!text(c.citation_id, 160) || citations.has(c.citation_id) || !text(c.quote, 600) || !source ||
        c.document_version_id !== source.document_version_id || c.source_uri !== source.source_uri || c.title !== source.title) fail();
    citations.set(c.citation_id, c);
  }
  for (const c of a.claims) {
    if (!text(c.claim_id, 160) || claims.has(c.claim_id) || !text(c.text) || !Array.isArray(c.citation_ids) ||
        !c.citation_ids.length || c.citation_ids.length > 8 || new Set(c.citation_ids).size !== c.citation_ids.length ||
        c.citation_ids.some(id => !citations.has(id))) fail();
    claims.add(c.claim_id); c.citation_ids.forEach(id => used.add(id));
  }
  if (a.answer_status === "ABSTAINED" ? a.claims.length || a.citations.length :
      !a.claims.length || a.answer !== a.claims.map(c => c.text).join("\n")) fail();
  if (used.size !== citations.size) fail();
  return a;
}
