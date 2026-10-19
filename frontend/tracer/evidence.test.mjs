import test from "node:test";
import assert from "node:assert/strict";
import { fixtureRecord, fixtureSource, deferred } from "../../tests/fixtures/web_upgrade/records.mjs";
import { projectEvidence, splitExactQuote, createEvidenceController } from "./evidence.mjs";

test("projection joins shared sources by ID, not array position", () => {
  const record = fixtureRecord(); record.response.citations.reverse();
  const data = projectEvidence(record);
  assert.equal(data.claims[0].citation_ids[0], "citation-1");
  assert.equal(data.citations.find(c => c.citation_id === "citation-1").quote, "Synthetic evidence passage 1.");
});

test("live source metadata is owner-record bound and a changed evidence hash is rejected", async () => {
  const record = fixtureRecord();
  record.response.retrieval_backend = "aws_knowledge_mcp";
  record.response.answer_language = "vi";
  record.response.evidence_bundle_sha256 = "a".repeat(64);
  assert.equal(projectEvidence(record).evidence_bundle_sha256, "a".repeat(64));
  const c = createEvidenceController({ loadSource: async () => ({
    ...fixtureSource(record, record.response.citations[0]), evidence_bundle_sha256: "b".repeat(64),
    retrieved_at: "2026-10-18T12:00:00Z" }) });
  c.setRecord(record);
  assert.equal(await c.selectCitation("citation-1"), false);
  assert.equal(c.getState().source, null);
});
for (const [name, mutate] of [
  ["unknown citation", r => r.response.claims[0].citation_ids.push("missing")],
  ["duplicate citation", r => r.response.citations.push(r.response.citations[0])],
  ["duplicate claim", r => r.response.claims.push(r.response.claims[0])],
  ["unused citation", r => r.response.claims[1].citation_ids.pop()],
  ["mismatched request", r => r.response.request_id = `q_${"b".repeat(64)}`],
  ["mismatched snapshot", r => r.response.snapshot_version = "other"],
  ["fabricated abstention", r => r.response.answer_status = "ABSTAINED"],
  ["invalid source ID", r => r.response.citations[0].chunk_id = "forged"],
]) test(`invalid evidence fails closed: ${name}`, () => { const r = fixtureRecord(); mutate(r); assert.throws(() => projectEvidence(r)); });

test("abstention and partial status are preserved; processing is not evidence", () => {
  assert.equal(projectEvidence(fixtureRecord("ABSTAINED")).claims.length, 0);
  assert.equal(projectEvidence(fixtureRecord("PARTIAL")).answer_status, "PARTIAL");
  assert.throws(() => projectEvidence(fixtureRecord("PROCESSING")));
});
test("exact quote segmentation handles Unicode, boundaries, missing and overlapping duplicates", () => {
  assert.deepEqual(splitExactQuote("xin chào\nλ", "chào\n"), { matched: true, before: "xin ", quote: "chào\n", after: "λ" });
  assert.equal(splitExactQuote("quote end", "quote").before, "");
  assert.equal(splitExactQuote("start quote", "quote").after, "");
  assert.equal(splitExactQuote("nothing", "missing").matched, false);
  assert.equal(splitExactQuote("aaa", "aa").matched, false);
});
test("a late source A never replaces source B", async () => {
  const r = fixtureRecord(), a = deferred(), b = deferred(), events = [];
  const c = createEvidenceController({ loadSource: (_, citation) => citation.citation_id === "citation-1" ? a.promise : b.promise, onChange: state => events.push(state) });
  c.setRecord(r); c.selectClaim("claim-one");
  const first = c.selectCitation("citation-1"), second = c.selectCitation("citation-2");
  b.resolve(fixtureSource(r, r.response.citations[1])); assert.equal(await second, true);
  a.resolve(fixtureSource(r, r.response.citations[0])); assert.equal(await first, false);
  assert.equal(c.getState().citationId, "citation-2"); assert.equal(events.at(-1).source.title, r.response.citations[1].title);
});
for (const action of ["clear", "record", "claim"]) test(`pending source cannot survive ${action}`, async () => {
  const r = fixtureRecord(), pending = deferred(); let errors = 0;
  const c = createEvidenceController({ loadSource: () => pending.promise, onError: () => errors++ });
  c.setRecord(r); const call = c.selectCitation("citation-1");
  if (action === "clear") c.clear();
  if (action === "record") c.setRecord(fixtureRecord("ANSWERED", "b"));
  if (action === "claim") c.selectClaim("claim-two");
  pending.reject(new Error("old session failure")); assert.equal(await call, false);
  assert.equal(c.getState().source, null); assert.equal(errors, 0);
});
test("source identity mismatch fails; a failure does not retain the previous passage", async () => {
  const r = fixtureRecord(); let mismatch = false;
  const c = createEvidenceController({ loadSource: (_, citation) => ({ ...fixtureSource(r, citation), ...(mismatch ? { document_version_id: "wrong" } : {}) }) });
  c.setRecord(r); await c.selectCitation("citation-1"); assert.equal(c.getState().phase, "ready");
  mismatch = true; await c.selectCitation("citation-2"); assert.equal(c.getState().phase, "error"); assert.equal(c.getState().source, null);
});
test("selecting a source outside the selected claim is rejected without I/O", async () => {
  let calls = 0; const c = createEvidenceController({ loadSource: () => calls++ });
  c.setRecord(fixtureRecord()); c.selectClaim("claim-one");
  await assert.rejects(c.selectCitation("citation-3")); assert.equal(calls, 0);
});
