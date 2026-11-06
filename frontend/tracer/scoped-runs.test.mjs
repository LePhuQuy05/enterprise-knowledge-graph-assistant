import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SCOPED_METHODS, scopedRunPath, validateScopedCapabilities, validateScopedRun, validateScopedResult, qualificationMessage } from "./scoped-runs.mjs";

const fixture = JSON.parse(readFileSync(new URL("../../tests/fixtures/scoped_runs/wire.json", import.meta.url)));
const records = fixture.cases.flatMap(c => Object.entries(c.results).map(([method, result]) => ({ method, result, run: c.run })));
const representative = method => structuredClone(records.find(r => r.method === method));
const check = x => validateScopedResult(x.result, x.method, x.run.run_id);

test("capabilities separate actual319 inventory from the sparse experimental graph", () => {
  const c = validateScopedCapabilities(fixture.capabilities);
  assert.equal(c.corpus.documents, 319); assert.equal(c.corpus.chunks, 3024);
  assert.equal(c.graph.facts, 27); assert.equal(c.graph.source_documents, 21);
  assert.equal(c.graph.human_reviewed, false); assert.equal(Object.keys(SCOPED_METHODS).length, 8);
  assert.equal(fixture.provider_calls, 0); assert.match(fixture.notice, /SYNTHETIC/);
});
for (const row of records) test(`actual source projection validates ${row.method}/${row.result.answer_language}/${row.result.qualification}`, () => {
  assert.equal(validateScopedRun(row.run), row.run); assert.equal(check(row), row.result);
  assert.equal(row.result.config.fixture_only, true); // This is not native answer-quality evidence.
});
test("V2 paths do not permit cross-protocol traversal or foreign identifiers", () => {
  const id = records[0].run.run_id;
  assert.equal(scopedRunPath(id), `/v2/scoped-runs/${id}`);
  for (const bad of ["../v1/runs", "run_x", "q_"+"a".repeat(64), "run_"+"a".repeat(65)]) assert.throws(() => scopedRunPath(bad));
  assert.throws(() => validateScopedRun({ ...records[0].run, schema_version: 1 }));
});
for (const [label, mutate] of [
  ["silently changed method", r => { r.effective_method = "hybrid"; }],
  ["different run", r => { r.run_id = "run_"+"0".repeat(64); }],
  ["false semantic verification", r => { r.semantic_verified = true; }],
  ["invented accuracy", r => { r.quality_metrics = { accuracy: 1 }; }],
  ["fallback", r => { r.retrieval.hidden_fallbacks = 1; }],
  ["truncated source", r => { r.retrieval.partial_source_truncations = 1; }],
  ["wrong source order", r => { r.context_sources.reverse(); }],
  ["raw source embedded in public projection", r => { r.context_sources[0].text = "must be read through source endpoint"; }],
  ["unsafe official source URL", r => { r.context_sources[0].source_uri = "javascript:alert(1)"; }],
  ["raw receipt field", r => { r.raw_output = "withheld private raw output"; }],
  ["unexpected root field", r => { r.debug = "private"; }],
  ["unexpected config field", r => { r.config.debug = "private"; }],
  ["unexpected planner field", r => { r.retrieval.raw_response = "private"; }],
  ["missing fixture classification", r => { delete r.config.fixture_only; }],
  ["wrong source profile", r => { r.config.source_profile.max_chunks = 8; }],
  ["wrong output bound", r => { r.config.generation_recipe.output_tokens = 800; }],
  ["invalid graph role", r => { r.graph_evidence[0].nodes[0].role = {}; }],
  ["private owner", r => { r.owner_id = "some-owner"; }],
  ["invalid context budget", r => { r.context_proxy_tokens = 4097; }],
  ["negative cost", r => { r.new_cost_micro_usd = -1; }],
  ["charge on cached replay", r => { r.cache_hit = true; r.new_cost_micro_usd = 1; }],
  ["graph grant claim", r => { r.graph_evidence[0].permission_granted = true; }],
  ["unselected graph fact", r => { r.graph_evidence[0].assertion_id = "invented"; }],
  ["wrong graph endpoint", r => { r.graph_evidence[0].links[0][0] = "not-a-node"; }],
  ["graph path without source support", r => { r.retrieval.bundles.find(b => b.selection === "SELECTED").chunk_ids = ["chk_"+"0".repeat(64)]; }],
  ["citation from another source", r => { r.response.citations[0].document_version_id = "foreign"; }],
  ["unknown stage", r => { r.trace[0].stage = "imaginary_model_thinking"; }],
]) test(`V2 rejects ${label}`, () => { const x = representative("graph_2hop"); mutate(x.result); assert.throws(() => check(x)); });
test("Q and QB retain actual planner receipt identity and distinct allocator semantics", () => {
  const q = representative("hybrid_multi_query"), qb = representative("hybrid_multi_query_balanced");
  assert.deepEqual(q.result.retrieval.queries, qb.result.retrieval.queries);
  assert.equal(q.result.retrieval.pool_fingerprint, qb.result.retrieval.pool_fingerprint);
  assert.equal(q.result.retrieval.balanced_allocator_used, false); assert.equal(qb.result.retrieval.balanced_allocator_used, true);
  q.result.retrieval.planner_receipt_sha256 = null; assert.throws(() => check(q));
  qb.result.retrieval.balanced_allocator_used = false; assert.throws(() => check(qb));
});
test("original question identity must match the owned run before displaying or exporting", () => {
  const x = representative("hybrid");
  assert.equal(validateScopedResult(x.result, x.method, x.run.run_id, x.result.original_question_sha256), x.result);
  assert.throws(() => validateScopedResult(x.result, x.method, x.run.run_id, "0".repeat(64)));
});
test("withheld response stays null and does not publish synthetic invalid claim text", () => {
  const c = fixture.cases.find(c => c.id === "held"), r = c.results.hybrid;
  assert.equal(r.response, null); assert.equal(r.qualification, "REVIEW_REQUIRED"); assert.ok(r.review_flags.length);
  assert.ok(!JSON.stringify(r).includes("Lambda can invoke Lambda."));
  const copy = structuredClone(r); copy.response = representative("hybrid").result.response;
  assert.throws(() => validateScopedResult(copy, "hybrid", c.run.run_id));
  const flag = structuredClone(r); flag.review_flags[0].raw_output = "private";
  assert.throws(() => validateScopedResult(flag, "hybrid", c.run.run_id));
  assert.match(qualificationMessage("REVIEW_REQUIRED"), /withheld/);
  assert.match(qualificationMessage("NO_MECHANICAL_FLAGS"), /not been independently verified/);
});
test("NOT_READY is distinct from failed work and requires no generation/cost", () => {
  const x = representative("graph_1hop"), r = x.result;
  Object.assign(r, { qualification: "NOT_READY", response: null, native_answer_status: null, review_flags: [], answer_origin: "NOT_READY",
    new_cost_micro_usd: 0, original_receipt_cost_micro_usd: null, usage: null, used_graph: false, graph_evidence: [], cache_hit: false });
  r.retrieval.mechanism_status = "NOT_READY_GRAPH_ZERO_EXPANSION"; r.retrieval.bundles = [];
  r.trace = r.trace.filter(e => e.stage !== "generation_and_display_qualification");
  r.trace.push({ stage: "mechanism_not_ready_no_dispatch", duration_ms: 0, status: "COMPLETED", input_sha256: r.source_context_sha256, output_count: 0 });
  assert.equal(check(x), r);
  r.new_cost_micro_usd = 1; assert.throws(() => check(x));
});
for (const [label, mutate] of [
  ["false graph review", c => { c.graph.human_reviewed = true; }],
  ["unknown method", c => { c.methods.push("unknown"); }],
  ["second allowance", c => { c.monthly_child_limit = 600; }],
  ["too many answer methods", c => { c.maximum_answer_methods = 4; }],
  ["empty collection", c => { c.corpus.documents = 0; }],
]) test(`capability rejects ${label}`, () => { const c = structuredClone(fixture.capabilities); mutate(c); assert.throws(() => validateScopedCapabilities(c)); });
