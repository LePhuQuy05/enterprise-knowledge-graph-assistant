const recorded = await (await fetch("/answer-fixture.json")).json();
await import("/retrieval-preview.mjs");
const { ApiFailure } = await import("/assets/client.mjs");
const original = __preview.overrides.runs, answers = new Map();
const question = "Which policies must permit cross-account use of an AWS KMS key?";
function detail(run, advance = false) {
  if (advance) run.reads++;
  const done = run.reads > 1;
  return { ...structuredClone(__ownedFixture.run), run_id: recorded.run_id, request: structuredClone(run.body),
    status: done ? "COMPLETED" : "RUNNING", quality_metrics: null,
    methods: [{ method_id: "hybrid", status: done ? "COMPLETED" : "QUEUED", result_available: done, error_category: null }],
    budget: { settled_micro_usd: done ? recorded.new_cost_micro_usd : 0, reserved_micro_usd: done ? 0 : 315533,
      max_cost_micro_usd: 315533, cost_complete: done, stages: [] } };
}
__preview.overrides.runs = async (path, options = {}) => {
  if (path === "/v1/run-capabilities") return { ...(await original(path, options)), answering_enabled: true,
    answer_methods: ["hybrid"], maximum_answer_methods: 1 };
  if (path === "/v1/runs" && options.body?.run_kind === "answer") {
    if (options.body.original_question !== question || options.body.method_ids.join() !== "hybrid") throw new ApiFailure("RECORDED_FIXTURE_KMS_HYBRID_ONLY", 400);
    __ownedPreview.posts.push(structuredClone(options));
    const existing = answers.get(options.key);
    if (existing && JSON.stringify(existing.body) !== JSON.stringify(options.body)) throw new ApiFailure("IDEMPOTENCY_CONFLICT", 409);
    if (!existing) answers.set(options.key, { body: structuredClone(options.body), reads: 0 });
    return detail(answers.get(options.key));
  }
  const run = [...answers.values()][0];
  if (path.startsWith("/v1/runs?")) {
    const old = await original(path, options); return { ...old, items: run ? [detail(run), ...old.items] : old.items };
  }
  if (run && path.startsWith("/v1/runs/" + recorded.run_id)) {
    if (path.endsWith("/trace")) return { run_id: recorded.run_id, methods: [], budget: detail(run).budget };
    const source = path.match(/\/sources\/(chk_[a-f0-9]{64})$/);
    if (source) return structuredClone(__ownedFixture.sources["hybrid:" + source[1]]);
    if (path.endsWith("/methods/hybrid")) return structuredClone(recorded);
    return detail(run, true);
  }
  return original(path, options);
};
document.querySelector(".preview-banner").textContent = "LOCAL PREVIEW — recorded public KMS answer and sources. Account, queue, model use and cost display are simulated; no provider/AWS calls.";
__preview.ui.clear(); await __preview.ui.activate();
if (document.getElementById('workspace-tour').open) document.getElementById('tour-close').click();
globalThis.__answerPreview = { answers, recorded, question };
