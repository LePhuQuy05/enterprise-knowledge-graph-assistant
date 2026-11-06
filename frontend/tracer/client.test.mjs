import test from "node:test";
import assert from "node:assert/strict";
import { ApiFailure, apiRequest, errorMessage, officialSourceLink, queryPath, sourcePath } from "./client.mjs";

test("production transport admits scoped V2 capabilities, history, detail, result, source and mutations", async () => {
  const run = "run_" + "a".repeat(64), chunk = "chk_" + "b".repeat(64);
  const paths = ["/v2/scoped-run-capabilities", "/v2/scoped-runs?limit=10", `/v2/scoped-runs/${run}`,
    `/v2/scoped-runs/${run}/methods/graph_2hop`, `/v2/scoped-runs/${run}/methods/graph_2hop/sources/${chunk}`,
    `/v2/scoped-runs/${run}/trace`];
  let calls = 0;
  const fetcher = async (path, options) => { calls++; assert.equal(options.headers.Authorization, "Bearer synthetic");
    assert.equal(options.credentials, "omit"); assert.equal(options.redirect, "error"); return { ok: true, text: async () => '{}' }; };
  for (const path of paths) await apiRequest(path, "synthetic", { fetcher });
  const key = "11111111-1111-4111-8111-111111111111";
  await apiRequest("/v2/scoped-runs", "synthetic", { fetcher, key, body: { method_ids: ["graph_2hop"] } });
  await apiRequest(`/v2/scoped-runs/${run}/cancel`, "synthetic", { fetcher, key, body: {} });
  assert.equal(calls, 8);
});

test("V2 transport remains restricted to the approved endpoint family and same-origin paths", async () => {
  let calls = 0;
  for (const path of ["/v2/admin", "/v2/scoped-runs-other", "https://other.example/v2/scoped-runs", "/v2/scoped-runs/../admin"]) {
    await assert.rejects(apiRequest(path, "synthetic", { fetcher: async () => { calls++; } }), e => e.code === "INVALID_CLIENT_REQUEST");
  }
  assert.equal(calls, 0);
});

test("one submit carries its idempotency key, excludes cookies and forbids redirects", async () => {
  let calls = 0;
  const key = "11111111-1111-4111-8111-111111111111";
  const data = await apiRequest("/v1/queries", "synthetic-access-token", { key, body: { question: "Question?" },
    fetcher: async (path, options) => {
      calls += 1;
      assert.equal(options.headers["Idempotency-Key"], key);
      assert.equal(options.credentials, "omit"); assert.equal(options.redirect, "error");
      assert.equal(options.method, "POST");
      return { ok: true, text: async () => '{"status":"COMPLETED"}' };
    } });
  assert.equal(calls, 1); assert.equal(data.status, "COMPLETED");
});

test("ambiguous network failure does not resubmit", async () => {
  let calls = 0;
  await assert.rejects(apiRequest("/v1/queries", "fixture", { fetcher: async () => {
    calls += 1; throw new Error("private transport details");
  } }), error => error.code === "REQUEST_OUTCOME_UNKNOWN" && !error.message.includes("private"));
  assert.equal(calls, 1);
});

test("typed failure keeps the request ID but drops raw server details", async () => {
  await assert.rejects(apiRequest("/v1/queries", "fixture", { fetcher: async () => ({ ok: false, status: 502,
    text: async () => JSON.stringify({ error: { code: "GENERATION_INVALID", message: "private details" }, request_id: "q_example" }) }) }),
  error => error instanceof ApiFailure && error.requestId === "q_example" && error.message === "GENERATION_INVALID");
});

test("only approved official HTTPS links are clickable", () => {
  assert.equal(officialSourceLink("https://docs.aws.amazon.com/kms/latest/developerguide/key-policies.html"),
    "https://docs.aws.amazon.com/kms/latest/developerguide/key-policies.html");
  for (const value of ["javascript:alert(1)", "https://evil.test", "https://docs.aws.amazon.com.evil.test/",
    "https://user:password@docs.aws.amazon.com/", "https://docs.aws.amazon.com:444/", "http://docs.aws.amazon.com/"]) {
    assert.equal(officialSourceLink(value), null);
  }
});

test("source URLs require the associated request and valid IDs", () => {
  const rid = "q_" + "a".repeat(64), cid = "chk_" + "b".repeat(64);
  assert.equal(sourcePath(rid, cid), `/v1/sources/${cid}?request_id=${rid}`);
  assert.throws(() => queryPath("../../private"));
  assert.throws(() => sourcePath(rid, "forged?tenant=other"));
});

test("expired sessions and uncertain requests have actionable safe messages", () => {
  assert.match(errorMessage(new ApiFailure("ACCESS_TOKEN_REQUIRED", 401)), /Sign in again/);
  assert.match(errorMessage(new ApiFailure("REQUEST_OUTCOME_UNKNOWN")), /not been submitted again/);
});
