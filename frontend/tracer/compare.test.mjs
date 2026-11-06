import test from "node:test";
import assert from "node:assert/strict";
import { runPath, validateRun, validateRunResult, contextOverlap, isAdmissionRejection } from "./compare.mjs";

test("only explicit pre-admission rejection on the first attempt releases an intent", () => {
  for (const [status, code] of [[400, "INVALID_REQUEST"], [400, "MODE_UNAVAILABLE"], [429, "BUDGET_LIMIT"],
    [429, "RUN_CONCURRENCY_LIMIT"], [429, "QUERY_QUOTA_EXCEEDED"]]) {
    assert.equal(isAdmissionRejection({ status, code }, false), true);
    assert.equal(isAdmissionRejection({ status, code }, true), false, "recovery cannot prove that an earlier attempt was not admitted");
  }
  for (const error of [{ status: 503, code: "STATE_BUSY" }, { status: 0, code: "REQUEST_OUTCOME_UNKNOWN" },
    { status: 429, code: "SERVICE_UNAVAILABLE" }, { status: 409, code: "IDEMPOTENCY_CONFLICT" }]) {
    assert.equal(isAdmissionRejection(error, false), false);
  }
});

const chunk = "chk_" + "b".repeat(64);
const result = { method_id: "hybrid", status: "RETRIEVED", quality_metrics: null, new_cost_micro_usd: 0,
  config_sha256: "c".repeat(64), semantic_review: "NOT_REVIEWED", ranked_chunk_ids: [chunk], context_chunk_ids: [chunk],
  context_sources: [{ chunk_id: chunk, title: "AWS KMS key policies", source_uri: "https://docs.aws.amazon.com/kms/latest/developerguide/key-policies.html" }] };
const run = { run_id: "run_" + "a".repeat(64), collection_version: "collection_" + "d".repeat(64), quality_metrics: null,
  status: "COMPLETED", request: { original_question: "Which KMS policies?", run_kind: "retrieval" },
  methods: [{ method_id: "hybrid", status: "COMPLETED", result_available: true }] };
test("paths reject traversal or foreign identifiers before a private request", () => {
  assert.equal(runPath(run.run_id), "/v1/runs/" + run.run_id);
  for (const id of ["../other", "q_" + "a".repeat(64), "run_" + "a".repeat(63)]) assert.throws(() => runPath(id));
});
test("retrieval projections retain null quality rather than treating similarity as accuracy", () => {
  assert.equal(validateRun(run), run); assert.equal(validateRunResult(result, "hybrid"), result);
  assert.throws(() => validateRun({ ...run, quality_metrics: { accuracy: 1 } }));
  assert.throws(() => validateRunResult({ ...result, quality_metrics: { accuracy: 1 } }, "hybrid"));
});
for (const [name, mutate] of [
  ["unsafe source", r => { r.context_sources[0].source_uri = "javascript:alert(1)"; }],
  ["foreign method", r => { r.method_id = "graph_2hop"; }],
  ["duplicate ranks", r => { r.ranked_chunk_ids.push(chunk); }],
  ["context outside candidate pool", r => { r.ranked_chunk_ids = []; }],
  ["source binding", r => { r.context_sources[0].chunk_id = "chk_" + "f".repeat(64); }],
  ["false human review", r => { r.semantic_review = "APPROVED"; }],
]) test(`result validation rejects ${name}`, () => { const r = structuredClone(result); mutate(r); assert.throws(() => validateRunResult(r, "hybrid")); });
test("overlap compares sets of context, including empty results, without a quality claim", () => {
  assert.equal(contextOverlap([result]), null);
  assert.deepEqual(contextOverlap([result, { context_chunk_ids: [] }]), { shared: 0, union: 1 });
  assert.deepEqual(contextOverlap([result, result]), { shared: 1, union: 1 });
});
