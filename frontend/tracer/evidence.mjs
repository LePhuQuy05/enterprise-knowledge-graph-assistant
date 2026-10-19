import { queryPath, sourcePath } from "./client.mjs";

export class EvidenceFailure extends Error {
  constructor() { super("This saved answer could not be safely displayed. Refresh your history."); }
}
const requireValue = condition => { if (!condition) throw new EvidenceFailure(); };
const text = (value, max = 16000) => typeof value === "string" && value.length > 0 && value.length <= max;

export function projectEvidence(record) {
  requireValue(record?.status === "COMPLETED" && text(record.request_id) && text(record.question, 2000));
  queryPath(record.request_id);
  requireValue(text(record.snapshot_version, 160) && text(record.created_at, 80) && Number.isFinite(Date.parse(record.created_at)));
  const answer = record.response;
  requireValue(answer && answer.request_id === record.request_id && answer.snapshot_version === record.snapshot_version);
  requireValue(["ANSWERED", "PARTIAL", "ABSTAINED"].includes(answer.answer_status) && text(answer.answer));
  requireValue(Array.isArray(answer.claims) && answer.claims.length <= 4 && Array.isArray(answer.citations) && answer.citations.length <= 8);
  requireValue(["vector", "hybrid", "graph_1hop", "graph_2hop"].includes(answer.mode_used) && typeof answer.used_graph === "boolean");
  requireValue(!answer.used_graph || ["graph_1hop", "graph_2hop"].includes(answer.mode_used));
  const citationIds = new Set(), claimIds = new Set(), referenced = new Set();
  const citations = answer.citations.map(citation => {
    requireValue(citation && text(citation.citation_id, 160) && !citationIds.has(citation.citation_id));
    sourcePath(record.request_id, citation.chunk_id);
    requireValue(text(citation.document_version_id, 160) && text(citation.title, 2000) && text(citation.quote, 600) && text(citation.source_uri, 8000));
    requireValue(Array.isArray(citation.section_path) && citation.section_path.every(part => text(part, 2000)));
    citationIds.add(citation.citation_id);
    return { citation_id: citation.citation_id, chunk_id: citation.chunk_id, document_version_id: citation.document_version_id,
      title: citation.title, quote: citation.quote, source_uri: citation.source_uri, section_path: [...citation.section_path] };
  });
  const claims = answer.claims.map(claim => {
    requireValue(claim && text(claim.claim_id, 160) && !claimIds.has(claim.claim_id) && text(claim.text));
    requireValue(Array.isArray(claim.citation_ids) && claim.citation_ids.length > 0 && claim.citation_ids.length <= 8);
    requireValue(new Set(claim.citation_ids).size === claim.citation_ids.length && claim.citation_ids.every(id => citationIds.has(id)));
    claimIds.add(claim.claim_id); claim.citation_ids.forEach(id => referenced.add(id));
    return { claim_id: claim.claim_id, text: claim.text, citation_ids: [...claim.citation_ids] };
  });
  requireValue(referenced.size === citationIds.size);
  if (answer.answer_status === "ABSTAINED") requireValue(!claims.length && !citations.length);
  else requireValue(claims.length > 0 && answer.answer === claims.map(claim => claim.text).join("\n"));
  requireValue(Array.isArray(answer.warnings) && answer.warnings.length <= 32 && answer.warnings.every(warning => text(warning, 2000)));
  requireValue(answer.retrieval_backend === undefined || ["snapshot_numpy_adjacency", "aws_knowledge_mcp"].includes(answer.retrieval_backend));
  requireValue(answer.answer_language === undefined || ["en", "vi"].includes(answer.answer_language));
  requireValue(answer.evidence_bundle_sha256 == null || /^[0-9a-f]{64}$/.test(answer.evidence_bundle_sha256));
  return { request_id: record.request_id, question: record.question, created_at: record.created_at, snapshot_version: record.snapshot_version,
    answer_status: answer.answer_status, answer: answer.answer, mode_used: answer.mode_used, used_graph: answer.used_graph,
    retrieval_backend: answer.retrieval_backend || "snapshot_numpy_adjacency",
    answer_language: answer.answer_language || "en",
    evidence_bundle_sha256: answer.evidence_bundle_sha256 || null,
    warnings: [...answer.warnings], claims, citations };
}

export function splitExactQuote(source, quote) {
  if (typeof source !== "string" || typeof quote !== "string" || !quote) return { matched: false, text: source || "" };
  const at = source.indexOf(quote);
  if (at < 0 || source.indexOf(quote, at + 1) !== -1) return { matched: false, text: source };
  return { matched: true, before: source.slice(0, at), quote, after: source.slice(at + quote.length) };
}

export function createEvidenceController({ loadSource, onChange = () => {}, onError = () => {} }) {
  let epoch = 0, revision = 0, sequence = 0;
  let state = { record: null, claimId: null, citationId: null, phase: "idle", source: null };
  const emit = () => onChange({ ...state });
  function clear() {
    epoch++; revision++; sequence++;
    state = { record: null, claimId: null, citationId: null, phase: "idle", source: null }; emit();
  }
  function setRecord(record) {
    clear();
    state.record = projectEvidence(record); revision++; emit();
    return state.record;
  }
  function selectClaim(id) {
    if (!state.record?.claims.some(claim => claim.claim_id === id)) throw new EvidenceFailure();
    sequence++; state = { ...state, claimId: id, citationId: null, phase: "idle", source: null }; emit();
  }
  async function selectCitation(id) {
    const record = state.record, citation = record?.citations.find(item => item.citation_id === id);
    const claim = record?.claims.find(item => item.claim_id === state.claimId) || record?.claims.find(item => item.citation_ids.includes(id));
    if (!citation || !claim?.citation_ids.includes(id)) throw new EvidenceFailure();
    const captured = { epoch, revision, sequence: ++sequence };
    const current = () => captured.epoch === epoch && captured.revision === revision && captured.sequence === sequence;
    state = { ...state, claimId: claim.claim_id, citationId: id, phase: "loading", source: null }; emit();
    try {
      const source = await loadSource(record, citation);
      if (!current()) return false;
      requireValue(source?.request_id === record.request_id && source.chunk_id === citation.chunk_id && source.snapshot_version === record.snapshot_version);
      requireValue(source.document_version_id === citation.document_version_id && text(source.text, 200000) && text(source.title, 2000));
      requireValue(Array.isArray(source.section_path) && source.section_path.every(part => text(part, 2000)));
      requireValue(source.retrieved_at === undefined || Number.isFinite(Date.parse(source.retrieved_at)));
      requireValue(source.evidence_bundle_sha256 === undefined || source.evidence_bundle_sha256 === record.evidence_bundle_sha256);
      state = { ...state, phase: "ready", source: { title: source.title, section_path: [...source.section_path], text: source.text,
        snapshot_version: source.snapshot_version, retrieved_at: source.retrieved_at || null,
        normalizer_version: typeof source.normalizer_version === "string" ? source.normalizer_version : null } }; emit(); return true;
    } catch (error) {
      if (!current()) return false;
      state = { ...state, phase: "error", source: null }; emit(); onError(error); return false;
    }
  }
  return { clear, setRecord, selectClaim, selectCitation, getState: () => ({ ...state }) };
}
