const fixture = await (await fetch("/owned-fixture.json")).json();
await import("/base-preview.mjs");
const { ApiFailure } = await import("/assets/client.mjs");
const state = { runs: [], posts: [], reads: 0, ambiguous: false, hold: false, authError: false, delaySource: false, sourceResolvers: [] };
const requests = new Map();
const methods = fixture.results.map(r => r.method_id);
const copy = value => structuredClone(value);
function detail(run) {
  const done = run.cancelled || (!state.hold && run.reads++ >= 1);
  const status = run.cancelled ? "CANCELLED" : done ? "COMPLETED" : "RUNNING";
  return { ...copy(fixture.run), run_id: run.id, status, request: copy(run.body),
    methods: run.body.method_ids.map(method_id => ({ method_id, status: run.cancelled ? "CANCELLED" : done ? "COMPLETED" : "QUEUED", result_available: !run.cancelled && done, error_category: null })) };
}
__preview.overrides.runs = async (path, options = {}) => {
  if (state.authError) throw new ApiFailure("ACCESS_TOKEN_REQUIRED", 401);
  if (path === "/v1/run-capabilities") return { schema_version: 1, available: true, answering_enabled: false, methods, quality_metrics: null, corpus: fixture.corpus };
  if (path === "/v1/runs" && options.body) {
    state.posts.push(copy(options));
    let run = requests.get(options.key);
    if (!run) {
      if (options.body.original_question !== fixture.run.request.original_question) throw new ApiFailure("FIXTURE_SUPPORTS_ONLY_KMS_EXAMPLE", 400);
      run = { id: "run_" + String(requests.size + 1).padStart(64, "0"), body: copy(options.body), reads: 0, cancelled: false };
      requests.set(options.key, run); state.runs.unshift(run);
    } else if (JSON.stringify(run.body) !== JSON.stringify(options.body)) throw new ApiFailure("IDEMPOTENCY_CONFLICT", 409);
    if (state.ambiguous) { state.ambiguous = false; throw new ApiFailure("REQUEST_OUTCOME_UNKNOWN"); }
    return { ...detail(run), idempotent_replay: state.posts.filter(p => p.key === options.key).length > 1 };
  }
  if (path.startsWith("/v1/runs?")) return { items: state.runs.map(detail), next_cursor: null };
  const match = path.match(/^\/v1\/runs\/(run_[a-f0-9]{64})(?:\/(.*))?$/);
  const run = state.runs.find(r => r.id === match?.[1]);
  if (!run) throw new ApiFailure("RUN_NOT_FOUND", 404);
  state.reads++;
  const suffix = match[2];
  if (!suffix) return detail(run);
  if (suffix === "cancel") { run.cancelled = true; return detail(run); }
  if (suffix === "trace") return { run_id: run.id, methods: run.body.method_ids.map(method_id => ({ method_id, events: [], preview_note: "Trace stages are synthetic in this preview." })), quality_metrics: null };
  const method = suffix.split("/")[1], chunk = suffix.split("/")[3];
  if (!run.body.method_ids.includes(method)) throw new ApiFailure("RUN_NOT_FOUND", 404);
  if (chunk) {
    const source = copy(fixture.sources[method + ":" + chunk]);
    if (state.delaySource) return new Promise(resolve => state.sourceResolvers.push(() => resolve(source)));
    return source;
  }
  return copy(fixture.results.find(r => r.method_id === method));
};
document.querySelector(".preview-banner").textContent = "LOCAL PREVIEW — recorded public AWS retrieval. Queue, account, status and trace interactions are simulated. No AWS or model calls.";
globalThis.__ownedPreview = state;
globalThis.__ownedFixture = fixture;
await __preview.ui.compare.activate();
