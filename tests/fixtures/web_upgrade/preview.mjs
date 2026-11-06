import { createWorkspace } from "/assets/ui.mjs";
import { ApiFailure, apiRequest } from "/assets/client.mjs";
import { createBriefDownloader } from "/assets/brief.mjs";
import { fixtureRecord, fixtureSource } from "/fixtures.mjs";
import { loadBenchmark316, loadBenchmark316Case } from "/assets/benchmark316.mjs";

// Isolated test support: never packaged or imported by the production entry point.
const exampleQuestions = [
  "Why can S3 invoke Lambda while the function cannot read an SSE-KMS object?",
  "Which policies permit cross-account use of a KMS key?",
  "What is the current KMS quota in my account?",
  "How do S3 event notifications invoke a Lambda function?",
  "What changes when S3 object ACLs are disabled?",
];
const records = ["ANSWERED", "PARTIAL", "ABSTAINED", "PROCESSING", "FAILED"].map((status, index) => ({
  ...fixtureRecord(status, "abcde"[index]), question: `Synthetic example — ${exampleQuestions[index]}`,
}));
const calls = [], downloads = [], overrides = {};
const fetchStatic = globalThis.fetch.bind(globalThis);
const realDownloader = createBriefDownloader({ document });
const harness = { records, calls, downloads, overrides, realDownload: true, ui: null };
async function request(path, options = {}) {
  calls.push({ path, method: options.body === undefined ? "GET" : "POST", body: options.body, key: options.key });
  if (overrides.scopedRuns && (path.startsWith("/v2/scoped-runs") || path === "/v2/scoped-run-capabilities")) return overrides.scopedRuns(path, options);
  if (path.startsWith("/v2/scoped-runs") || path === "/v2/scoped-run-capabilities") throw new ApiFailure("NOT_FOUND", 404);
  if (overrides.runs && (path.startsWith("/v1/runs") || path === "/v1/run-capabilities")) return overrides.runs(path, options);
  if (path.startsWith("/v1/runs") || path === "/v1/run-capabilities") throw new ApiFailure("NOT_FOUND", 404);
  const kind = path.startsWith("/v1/sources/") ? "source" : options.body ? "submit" : path.startsWith("/v1/queries/") ? "detail" : "history";
  if (overrides[kind]) return overrides[kind](path, options);
  if (kind === "history") return { items: records.map(({ request_id, question, created_at, status }) => ({ request_id, question, created_at, status })), next_cursor: null };
  if (kind === "submit") return structuredClone(records[0]);
  if (kind === "detail") {
    const record = records.find(record => path.endsWith(record.request_id)); if (!record) throw new ApiFailure("NOT_FOUND", 404);
    return structuredClone(record);
  }
  const url = new URL(path, location.origin), record = records.find(record => record.request_id === url.searchParams.get("request_id"));
  const citation = record?.response?.citations.find(c => url.pathname.endsWith(c.chunk_id));
  if (!citation) throw new ApiFailure("NOT_FOUND", 404);
  return fixtureSource(record, citation);
}
const setSignedIn = signedIn => { document.getElementById("login").hidden = signedIn; document.getElementById("logout").hidden = !signedIn; };
// Keep the actual production transport in the browser test path. Only its
// fetch boundary is mocked, so endpoint-family guards cannot be bypassed.
const transportRequest = (path, options = {}) => apiRequest(path, "synthetic-local-access", { ...options,
  fetcher: async (url, init) => {
    try {
      const data = await request(url, { ...options, body: init.body === undefined ? undefined : JSON.parse(init.body),
        key: init.headers["Idempotency-Key"] });
      return { ok: true, status: 200, text: async () => JSON.stringify(data) };
    } catch (error) {
      if (!Number.isInteger(error?.status) || error.status < 400) throw error;
      return { ok: false, status: error.status, text: async () => JSON.stringify({ error: { code: error.code || "FIXTURE_HTTP_ERROR" } }) };
    }
  } });
const ui = createWorkspace({ document, request: transportRequest, confirmReplace: () => true, scopedEnabled: () => globalThis.__scopedPreviewEnabled === true,
  loadBenchmarkResults: () => loadBenchmark316(fetchStatic), loadBenchmarkCase: c => loadBenchmark316Case(c, fetchStatic),
  onSessionExpired: () => { setSignedIn(false); document.getElementById("status").textContent = "SYNTHETIC session ended"; },
  loadDashboard: async () => {
    if (overrides.dashboard) return overrides.dashboard();
    const response = await fetchStatic("/assets/dashboard-data.json"); if (!response.ok) throw new Error("NO_DASHBOARD");
    harness.dashboardData = await response.json(); return harness.dashboardData;
  },
  loadMethodExamples: async () => {
    if (overrides.methods) return overrides.methods();
    const response = await fetchStatic("/assets/methods-data.json"); if (!response.ok) throw new Error("NO_BENCHMARK");
    return response.json();
  },
  downloader: { clear: () => realDownloader.clear(), download: (text, filename, contentType) => { downloads.push({ text, filename, contentType }); if (harness.realDownload) realDownloader.download(text, filename, contentType); } } });
harness.ui = ui; globalThis.__preview = harness;
globalThis.fetch = async () => { throw new Error("Network fetch is disabled in the synthetic preview."); };
const banner = document.createElement("div"); banner.className = "preview-banner";
banner.textContent = "LOCAL PREVIEW — mocked private answers/login; no provider or AWS calls. Knowledge and Lab use recorded public AWS development data, not live inference.";
document.querySelector("main").prepend(banner);
const controls = document.createElement("div"); controls.className = "preview-controls";
const label = document.createElement("label"); label.textContent = "Example record"; label.htmlFor = "preview-state";
const select = document.createElement("select"); select.id = "preview-state";
for (const record of records) { const option = document.createElement("option"); option.value = record.request_id; option.textContent = record.response?.answer_status || record.status; select.append(option); }
const open = document.createElement("button"); open.type = "button"; open.textContent = "Open synthetic answer";
open.addEventListener("click", () => ui.openRecord(select.value)); controls.append(label, select, open);
const tools = document.createElement("details"); tools.className = "preview-tools";
const toolsTitle = document.createElement("summary"); toolsTitle.textContent = "Local preview · synthetic state controls";
tools.append(toolsTitle, controls); document.getElementById("workspace-content").prepend(tools);
document.getElementById("login").disabled = false;
document.getElementById("login").addEventListener("click", async () => { await ui.activate(); setSignedIn(true); });
document.getElementById("logout").hidden = false;
document.getElementById("logout").addEventListener("click", () => { ui.clear(); setSignedIn(false); });
document.getElementById("status").textContent = "Synthetic local session — no credentials loaded.";
await ui.activate();
setSignedIn(true);
