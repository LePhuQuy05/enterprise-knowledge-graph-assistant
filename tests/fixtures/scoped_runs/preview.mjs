// Source/map fixtures are from actual319; account/queue/provider are simulated.
const fixture = await (await fetch("/scoped-fixture.json")).json();
await import("/base-preview.mjs");
const { ApiFailure } = await import("/assets/client.mjs");
const copy = value => structuredClone(value);
const state = { runs: [], posts: [], requests: [], held: false, notReady: false, paused: false, ambiguous: false,
  authError: false, unavailable: false, badSource: false, delaySource: false, sourceResolvers: [] };
const keys = new Map();
function result(run, method) {
  const r = copy(run.fixture.results[method]);
  if (run.notReady) {
    Object.assign(r, { qualification: "NOT_READY", response: null, native_answer_status: null, review_flags: [], answer_origin: "NOT_READY",
      new_cost_micro_usd: 0, original_receipt_cost_micro_usd: null, usage: null, used_graph: false, graph_evidence: [], cache_hit: false });
    r.retrieval.mechanism_status = method.startsWith("graph_") ? "NOT_READY_GRAPH_ZERO_EXPANSION" : "NOT_READY_NO_DISTINCT_NATIVE_FACETS";
    r.retrieval.bundles = [];
    r.trace = r.trace.filter(e => e.stage !== "generation_and_display_qualification");
    r.trace.push({ stage: "mechanism_not_ready_no_dispatch", duration_ms: 0, status: "COMPLETED", input_sha256: r.source_context_sha256, output_count: 0 });
  }
  return r;
}
function detail(run) {
  const done = run.cancelled || !state.paused && run.reads++ >= 1;
  const status = run.cancelled ? "CANCELLED" : done ? "COMPLETED" : "RUNNING";
  return { ...copy(run.fixture.run), status, request: { ...copy(run.fixture.run.request), ...copy(run.body), schema_version: 2 },
    methods: run.body.method_ids.map(method_id => ({ method_id, status: run.cancelled ? "CANCELLED" : done ? "COMPLETED" : "QUEUED", result_available: !run.cancelled && done, error_category: null })) };
}
__preview.overrides.scopedRuns = async (path, options = {}) => {
  state.requests.push(path);
  if (state.authError) throw new ApiFailure("ACCESS_TOKEN_REQUIRED", 401);
  if (state.unavailable) throw new ApiFailure("SCOPED_UNAVAILABLE", 503);
  if (path === "/v2/scoped-run-capabilities") return copy(fixture.capabilities);
  if (path === "/v2/scoped-runs" && options.body) {
    state.posts.push(copy(options));
    let run = keys.get(options.key);
    if (!run) {
      const wanted = state.held ? "held" : options.body.language === "vi" ? "vietnamese" : null;
      const entry = fixture.cases.find(c => (!wanted || c.id === wanted) && c.id !== (state.held ? "" : "held") &&
        c.run.request.original_question === options.body.original_question && options.body.method_ids.every(m => c.results[m]));
      if (!entry) throw new ApiFailure("FIXTURE_SUPPORTS_ONLY_RECORDED_QUESTION_AND_METHOD_GROUPS", 400);
      run = { fixture: entry, body: copy(options.body), reads: 0, cancelled: false, notReady: state.notReady };
      keys.set(options.key, run); state.runs.unshift(run);
    } else if (JSON.stringify(run.body) !== JSON.stringify(options.body)) throw new ApiFailure("IDEMPOTENCY_CONFLICT", 409);
    if (state.ambiguous) { state.ambiguous = false; throw new ApiFailure("REQUEST_OUTCOME_UNKNOWN"); }
    return detail(run);
  }
  if (path.startsWith("/v2/scoped-runs?")) return { items: state.runs.slice(0, 10).map(detail), next_cursor: null };
  const match = path.match(/^\/v2\/scoped-runs\/(run_[a-f0-9]{64})(?:\/(.*))?$/);
  const run = state.runs.find(r => r.fixture.run.run_id === match?.[1]);
  if (!run) throw new ApiFailure("RUN_NOT_FOUND", 404);
  if (!match[2]) return detail(run);
  if (match[2] === "cancel") { run.cancelled = true; return detail(run); }
  if (match[2] === "trace") return { schema_version: 2, run_id: run.fixture.run.run_id, quality_metrics: null,
    methods: run.body.method_ids.map(m => ({ method_id: m, events: result(run, m).trace, retrieval: result(run, m).retrieval })) };
  const method = match[2].split("/")[1], chunk = match[2].split("/")[3];
  if (!run.body.method_ids.includes(method)) throw new ApiFailure("RUN_NOT_FOUND", 404);
  if (chunk) {
    const source = copy(run.fixture.sources[method][chunk]);
    if (!source) throw new ApiFailure("SOURCE_NOT_FOUND", 404);
    if (state.badSource) source.text += " changed";
    if (state.delaySource) return new Promise(resolve => state.sourceResolvers.push(() => resolve(source)));
    return source;
  }
  return result(run, method);
};
__preview.overrides.runs = async path => {
  if (path === "/v1/run-capabilities") return { schema_version: 1, available: true, answering_enabled: false,
    methods: ["bm25", "vector", "hybrid", "hybrid_reranked"], quality_metrics: null, corpus: { documents: 316, chunks: 2978, service_groups: 24 } };
  if (path.startsWith("/v1/runs?")) return { items: [], next_cursor: null };
  throw new ApiFailure("LEGACY_FIXTURE_READ_ONLY", 404);
};
globalThis.__scopedPreviewEnabled = true;
globalThis.__scopedPreview = state;
globalThis.__scopedFixture = fixture;
__preview.realDownload = false;
document.querySelector(".preview-banner").textContent = fixture.notice + " " + fixture.planner;
await __preview.ui.compare.activate();
__preview.ui.dashboard.show("compare", true);
