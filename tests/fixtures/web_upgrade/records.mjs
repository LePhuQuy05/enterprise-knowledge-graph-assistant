// Synthetic fixtures only. None of these passages are quotations from AWS.
export function fixtureRecord(status = "ANSWERED", digit = "a") {
  const request_id = `q_${digit.repeat(64)}`, snapshot_version = `snap_${"f".repeat(64)}`;
  const citations = [1, 2, 3].map(n => ({ citation_id: `citation-${n}`, chunk_id: `chk_${String(n).repeat(64)}`,
    document_version_id: `doc_${String(n).repeat(64)}`, title: `Synthetic source ${n} — not an AWS quotation`,
    source_uri: "https://docs.aws.amazon.com/lambda/latest/dg/lambda-permissions.html", section_path: ["Synthetic example", `Section ${n}`],
    quote: `Synthetic evidence passage ${n}.` }));
  const claims = [{ claim_id: "claim-one", text: "Synthetic statement one illustrates separate permission layers.", citation_ids: ["citation-1", "citation-2"] },
    { claim_id: "claim-two", text: "Synthetic statement two illustrates a condition that needs checking.", citation_ids: ["citation-2", "citation-3"] }];
  const record = { request_id, snapshot_version, question: "SYNTHETIC EXAMPLE — investigate an AWS workflow", created_at: "2026-10-18T04:00:00+00:00", mode: "hybrid", status: "COMPLETED", idempotent_replay: false };
  if (["PROCESSING", "FAILED"].includes(status)) return { ...record, status, error: status === "FAILED" ? { code: "GENERATION_INVALID" } : undefined };
  const abstain = status === "ABSTAINED";
  return { ...record, response: { request_id, snapshot_version, answer_status: status, mode_requested: "hybrid", mode_used: "hybrid",
    used_graph: false, retrieval_backend: "snapshot_numpy_adjacency", warnings: [], claims: abstain ? [] : claims,
    citations: abstain ? [] : citations, answer: abstain ? "The synthetic corpus does not establish this fact." : claims.map(claim => claim.text).join("\n") } };
}

export function fixtureSource(record, citation) {
  return { request_id: record.request_id, chunk_id: citation.chunk_id, document_version_id: citation.document_version_id,
    snapshot_version: record.snapshot_version, title: citation.title, source_uri: citation.source_uri, section_path: [...citation.section_path],
    text: `SYNTHETIC PREVIEW — this is not an AWS quotation.\n\nContext before the quote. ${citation.quote} Context after the quote.\n\nRead the actual AWS page for real guidance.` };
}

export function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}
