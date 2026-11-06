import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { validateBenchmark, loadBenchmark, createBenchmark } from "./benchmark.mjs";
const data = JSON.parse(await readFile(new URL("./benchmark-data.json", import.meta.url), "utf8"));

test("all eight methods stay visible without inventing collection scores", () => {
  assert.equal(validateBenchmark(data).methods.length, 8);
  assert.equal(data.collection.documents, 316);
  assert.equal(data.collection.assertions, 0);
  assert.equal(data.methods.filter(r => r.id.startsWith("graph_")).length, 2);
});
test("pilot scores cannot masquerade as collection quality or released graph", () => {
  for (const mutate of [d => d.collection_quality = .925, d => d.methods[6].cloud_retrieval = "RECORDED_DEPLOYED",
    d => d.methods[0].collection_quality = .85, d => d.methods.reverse(), d => d.pilot.results[0].en.complete = 999]) {
    const bad = structuredClone(data); mutate(bad); assert.throws(() => validateBenchmark(bad));
  }
});
test("report loading is a single bounded public GET, no model route or credentials", async () => {
  let calls = 0;
  const result = await loadBenchmark(async (url, options) => {
    calls++; assert.equal(url, "/assets/benchmark-data.json"); assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error"); return { ok: true, text: async () => JSON.stringify(data) };
  });
  assert.equal(result.pilot.review, "AI_DRAFT"); assert.equal(calls, 1);
  await assert.rejects(loadBenchmark(async () => ({ ok: false })));
  await assert.rejects(loadBenchmark(async () => ({ ok: true, text: async () => "x".repeat(100001) })));
});
test("logout fences a pending report and clears rendered results", async () => {
  const nodes = new Map(); const document = { getElementById(id) { if (!nodes.has(id)) nodes.set(id, { textContent: "", children: [], replaceChildren(...n) { this.children = n; } }); return nodes.get(id); } };
  let resolve; let calls = 0;
  const view = createBenchmark({ document, loadData: () => { calls++; return new Promise(r => resolve = r); } });
  view.activate(); await view.show("overview"); assert.equal(calls, 0);
  const pending = view.show("benchmark"); assert.equal(calls, 1); view.clear(); resolve(data); await pending;
  assert.equal(document.getElementById("benchmark-status").textContent, "");
  assert.deepEqual(document.getElementById("benchmark-methods").children, []);
});
