import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { validateBenchmark316, validateBenchmarkCase, loadBenchmark316, loadBenchmark316Case } from "./benchmark316.mjs";

const bytes = await readFile(new URL("./benchmark316-summary.json", import.meta.url), "utf8");
const data = JSON.parse(bytes);
test("all 100 cases retain their pinned bytes, source links, controls and real method receipts", async () => {
  assert.equal(validateBenchmark316(data), data);
  for (const c of data.cases) {
    const raw = await readFile(new URL(`./benchmark316-cases/${c.id}.json`, import.meta.url), "utf8");
    assert.equal(createHash("sha256").update(raw).digest("hex"), c.sha256);
    assert.ok(Buffer.byteLength(raw) <= 256 * 1024);
    validateBenchmarkCase(JSON.parse(raw), c);
  }
  assert.ok(Buffer.byteLength(bytes) <= 128 * 1024);
});
for (const [name, mutate] of [
  ["false holdout", d => { d.independent_holdout = true; }],
  ["false human review", d => { d.human_approved = 100; }],
  ["duplicate scenario", d => { d.cases[1].id = d.cases[0].id; }],
  ["wrong denominator", d => { d.methods[0].expected_scenarios++; }],
  ["unrun method score", d => { const m = d.methods[0]; Object.assign(m, { not_run: m.expected_scenarios, attempted: 0, failed: 0, answerable_denominator: 0, controls: 0, complete_matrix: false, recall_at_k: 1, complete_evidence_at_k: null, context_complete: null }); }],
  ["out-of-range score", d => { d.methods[0].recall_at_k = 1.5; }],
]) test(`summary rejects ${name}`, () => { const d = structuredClone(data); mutate(d); assert.throws(() => validateBenchmark316(d)); });
test("unsafe source URL and foreign chunk reference are rejected", async () => {
  const c = data.cases[0], raw = JSON.parse(await readFile(new URL(`./benchmark316-cases/${c.id}.json`, import.meta.url), "utf8"));
  const bad = structuredClone(raw); Object.values(bad.chunks)[0].uri = "javascript:alert(1)";
  assert.throws(() => validateBenchmarkCase(bad, c));
  (raw.methods_by_language?.en || raw.methods).hybrid.context_chunk_ids.push("unbound-chunk");
  assert.throws(() => validateBenchmarkCase(raw, c));
});
test("public reads are GET only with no auth, no redirects and a size bound", async () => {
  await loadBenchmark316(async (url, options) => {
    assert.equal(url, "/assets/benchmark316-summary.json"); assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error"); assert.equal(options.body, undefined); assert.equal(options.headers, undefined);
    return { ok: true, text: async () => bytes };
  });
  await assert.rejects(loadBenchmark316(async () => ({ ok: true, text: async () => "x".repeat(128 * 1024 + 1) })));
  const c = data.cases[0];
  await assert.rejects(loadBenchmark316Case(c, async () => ({ ok: true, text: async () => "{}" })));
  assert.throws(() => loadBenchmark316Case({ id: "../private", sha256: "a".repeat(64) }));
});
