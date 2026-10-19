import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { validateDashboard, loadDashboardData } from "./dashboard.mjs";

const data = JSON.parse(await readFile(new URL("./dashboard-data.json", import.meta.url), "utf8"));
test("recorded data retains null approved score and known failure without graph serving", () => {
  assert.equal(validateDashboard(data), data);
  const c = data.cases.find(c => c.id === "dev-11");
  assert.ok(c.methods.B2_HYBRID.coverage.candidates.every(Boolean));
  assert.ok(!c.methods.B2_HYBRID.coverage.context.every(Boolean));
  assert.equal(data.summaries.B2_HYBRID.approved.complete_evidence_at_k, null);
});
for (const [name, mutate] of [
  ["unsafe URL", d => { d.documents[0].url = "javascript:alert(1)"; }],
  ["duplicate document", d => { d.documents[1].id = d.documents[0].id; }],
  ["foreign chunk reference", d => { d.cases[0].methods.B2_HYBRID.ranked[0] = 9000; }],
  ["false approved score", d => { d.summaries.B2_HYBRID.approved.complete_evidence_at_k = 1; }],
  ["misleading coverage", d => { d.cases[0].methods.B2_HYBRID.coverage.context = ["true"]; }],
  ["metric mismatch", d => { d.summaries.B2_HYBRID.draft.complete_evidence_at_k = 0; }],
  ["coverage/score disagreement", d => { d.cases[0].methods.B2_HYBRID.complete_at_k = 0; }],
  ["enabled graph", d => { d.run.graph_serving = true; }],
]) test(`projection rejects ${name}`, () => { const d = structuredClone(data); mutate(d); assert.throws(() => validateDashboard(d), /DATA_INVALID/); });
test("recorded loader uses a public GET without auth and refuses redirect/error/oversize", async () => {
  await loadDashboardData(async (url, options) => {
    assert.equal(url, "/assets/dashboard-data.json"); assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error"); assert.equal(options.body, undefined); assert.equal(options.headers, undefined);
    return { ok: true, text: async () => JSON.stringify(data) };
  });
  await assert.rejects(loadDashboardData(async () => ({ ok: false })), /UNAVAILABLE/);
  await assert.rejects(loadDashboardData(async () => ({ ok: true, text: async () => " ".repeat(400001) })), /DATA_INVALID/);
});
