import { officialSourceLink } from "./client.mjs";

export const BENCHMARK_METHODS = Object.freeze({ bm25: "BM25", vector: "Vector", hybrid: "Hybrid", hybrid_reranked: "Hybrid + Reranker", graph_1hop: "Graph · 1 hop", graph_2hop: "Graph · 2 hops", hybrid_multi_query: "Multi-query", hybrid_multi_query_balanced: "Query-balanced" });
const fail = () => { throw new Error("BENCHMARK316_DATA_INVALID"); };
const hash = v => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const id = v => typeof v === "string" && /^aws100-\d{3}$/.test(v);
const text = (v, max = 10000) => typeof v === "string" && v.length > 0 && v.length <= max;
const score = v => v === null || (Number.isFinite(v) && v >= 0 && v <= 1);
const splits = ["ALL", "DEVELOPMENT", "EVALUATION"];
export function validateBenchmark316(data) {
  const paired = data?.schema_version === 3;
  if (![2, 3].includes(data?.schema_version) || data.scope !== (paired ? "PUBLIC_RECORDED_COLLECTION316_PAIRED_RETRIEVAL_AI_DRAFT" : "PUBLIC_RECORDED_COLLECTION316_RETRIEVAL_AI_DRAFT") || data.documents !== 316 || data.chunks !== 2978 || data.service_groups !== 24 || data.independent_holdout !== false || data.human_approved !== 0 || data.answer_quality_measured !== false || data.model_calls !== 0 || data.aws_calls !== 0 || !hash(data.source_manifest_sha256) || !hash(data.protocol_sha256) || !hash(data.receipt_sha256) || !Array.isArray(data.cases) || data.cases.length !== 100 || !Array.isArray(data.methods) || data.methods.length !== (paired ? 48 : 24)) fail();
  if (paired && (!Number.isInteger(data.shared_preparation_calls) || data.shared_preparation_calls < 1 || data.shared_preparation_calls > 300 || !Number.isInteger(data.shared_preparation_cost_micro_usd) || data.shared_preparation_cost_micro_usd < 0 || data.shared_preparation_cost_micro_usd > 500000 || data.baseline_receipts_reused !== 600 || data.query_tracks?.en !== "EN_ORIGINAL" || data.query_tracks?.vi !== "VI_SHARED_REWRITE")) fail();
  if (new Set(data.cases.map(c => c.id)).size !== 100) fail();
  for (const c of data.cases) if (!id(c.id) || !hash(c.sha256) || !splits.slice(1).includes(c.split) || !text(c.question_en, 2000) || !text(c.question_vi, 2000) || typeof c.answerable !== "boolean" || !Array.isArray(c.services) || c.services.some(s => !text(s, 100))) fail();
  const pairs = new Set();
  for (const m of data.methods) {
    const key = `${m.language}/${m.method}/${m.split}`;
    if (!Object.hasOwn(BENCHMARK_METHODS, m.method) || !splits.includes(m.split) || pairs.has(key) || !(paired ? ["en", "vi"] : ["en"]).includes(m.language) || ![m.recall_at_k, m.complete_evidence_at_k, m.context_complete].every(score)) fail();
    pairs.add(key);
    const cases = data.cases.filter(c => m.split === "ALL" || c.split === m.split);
    for (const k of ["expected_scenarios", "attempted", "failed", "not_run", "answerable_denominator", "controls", "graph_used", "graph_fallback"]) if (!Number.isInteger(m[k]) || m[k] < 0 || m[k] > cases.length) fail();
    if (m.expected_scenarios !== cases.length || m.attempted + m.not_run !== cases.length || m.answerable_denominator + m.controls !== m.attempted || m.failed > m.attempted || m.complete_matrix !== (m.not_run === 0)) fail();
    if (!m.complete_matrix && [m.recall_at_k, m.complete_evidence_at_k, m.context_complete].some(v => v !== null)) fail();
  }
  return data;
}
export function validateBenchmarkCase(data, expected) {
  const paired = data?.schema_version === 2;
  if (![1, 2].includes(data?.schema_version) || data.scenario_id !== expected.id || data.question_en !== expected.question_en || data.question_vi !== expected.question_vi || data.split !== expected.split || data.answerable !== expected.answerable || data.review_status !== "AI_DRAFT" || !Array.isArray(data.groups) || data.groups.length > 12 || data.answerable !== Boolean(data.groups.length) || !(paired ? data.methods_by_language : data.methods) || !data.chunks) fail();
  for (const g of data.groups) {
    if (!text(g.fact_en) || !text(g.fact_vi) || !Array.isArray(g.alternatives) || !g.alternatives.length) fail();
    for (const a of g.alternatives) if (!officialSourceLink(a.uri) || !text(a.title, 1000) || !text(a.quote, 20000)) fail();
  }
  for (const c of Object.values(data.chunks)) if (!officialSourceLink(c.uri) || !text(c.title, 1000) || !text(c.preview, 800) || typeof c.preview_truncated !== "boolean") fail();
  const sets = paired ? [data.methods_by_language.en, data.methods_by_language.vi] : [data.methods];
  if (paired) for (const language of ["en", "vi"]) { const q = data.queries?.[language]; if (!q || !text(q.search_query, 2000) || !Array.isArray(q.subqueries) || q.subqueries.length > 2 || q.subqueries.some(s => !text(s, 240)) || !["READY", "PLAN_FAILED"].includes(q.preparation_status)) fail(); }
  for (const methods of sets) {
  if (!methods || (paired && Object.keys(methods).length !== 8)) fail();
  for (const [method, m] of Object.entries(methods)) {
    if (!Object.hasOwn(BENCHMARK_METHODS, method) || !["COMPLETED", "FAILED"].includes(m.status) || !Array.isArray(m.ranked_chunk_ids) || !Array.isArray(m.context_chunk_ids) || m.context_chunk_ids.length > 8 || [...m.ranked_chunk_ids.slice(0, 5), ...m.context_chunk_ids].some(key => !data.chunks[key]) || typeof m.used_graph !== "boolean") fail();
    if (data.answerable ? !m.metrics || !Array.isArray(m.metrics.covered_groups_at_k) || m.metrics.covered_groups_at_k.length !== data.groups.length || m.metrics.covered_groups_at_k.some(v => typeof v !== "boolean") : m.metrics?.complete_evidence_at_k !== null) fail();
  }
  }
  return data;
}
async function loadAsset(path, maxBytes, expectedHash, fetcher = fetch) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetcher(path, { credentials: "omit", cache: "no-store", redirect: "error", signal: controller.signal });
    if (!response.ok) fail();
    const raw = await response.text(), bytes = new TextEncoder().encode(raw);
    if (bytes.byteLength > maxBytes) fail();
    if (expectedHash) {
      const actual = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(b => b.toString(16).padStart(2, "0")).join("");
      if (actual !== expectedHash) fail();
    }
    return JSON.parse(raw);
  } finally { clearTimeout(timer); }
}
export const loadBenchmark316 = (fetcher = fetch) => loadAsset("/assets/benchmark316-summary.json", 128 * 1024, null, fetcher);
export const loadBenchmark316Case = (c, fetcher = fetch) => { if (!id(c.id) || !hash(c.sha256)) fail(); return loadAsset(`/assets/benchmark316-cases/${c.id}.json`, 256 * 1024, c.sha256, fetcher); };
const percent = v => v === null ? "Not run" : `${(v * 100).toFixed(2)}%`;
export function createBenchmark316({ document, loadData = loadBenchmark316, loadCase = loadBenchmark316Case, download }) {
  const get = key => document.getElementById("benchmark316-" + key);
  const node = (tag, content) => { const el = document.createElement(tag); if (content !== undefined) el.textContent = content; return el; };
  let active = false, epoch = 0, detailEpoch = 0, data = null, detail = null, loading = false;
  function cell(row, content, tag = "td") { const n = node(tag, content); if (tag === "th") n.scope = "row"; row.append(n); }
  function setTab(tab) {
    for (const name of ["results", "cases", "methodology"]) { get(name).hidden = name !== tab; get("tab-" + name).setAttribute("aria-pressed", String(name === tab)); }
  }
  function renderResults() {
    const split = get("split").value, language = get("run-language").value, body = get("rows"); body.replaceChildren();
    for (const method of Object.keys(BENCHMARK_METHODS)) {
      const m = data.methods.find(r => r.method === method && r.split === split && r.language === language), row = node("tr");
      cell(row, BENCHMARK_METHODS[method], "th"); cell(row, `${m.attempted}/${m.expected_scenarios}${m.failed ? ` · ${m.failed} failed` : ""}`);
      cell(row, m.answerable_denominator ? `${Math.round(m.complete_evidence_at_k * m.answerable_denominator)}/${m.answerable_denominator} · ${percent(m.complete_evidence_at_k)}` : "Not run");
      if (m.complete_evidence_at_k !== null) { const meter = node("meter"); meter.min = 0; meter.max = 1; meter.value = m.complete_evidence_at_k; meter.setAttribute("aria-label", `${BENCHMARK_METHODS[method]} provisional complete evidence at 5`); row.lastElementChild.append(meter); }
      cell(row, percent(m.recall_at_k)); cell(row, percent(m.context_complete));
      cell(row, method.startsWith("graph_") ? `${m.graph_used} used / ${m.graph_fallback} fallback` : "—"); body.append(row);
    }
    const cases = data.cases.filter(c => split === "ALL" || c.split === split);
    get("caption").textContent = `${language === "vi" ? "Vietnamese with shared English rewrite" : "English original queries"} · same pinned corpus and packer`;
    get("denominator").textContent = `${cases.length} scenarios · ${cases.filter(c => c.answerable).length} answerable · ${cases.filter(c => !c.answerable).length} controls. Retrieval metrics exclude controls. Evaluation is reserved and nonblind.`;
    get("track-note").textContent = data.schema_version === 3 ? "Eight methods were evaluated for both query languages. Vietnamese uses one saved shared English rewrite per input; Q and QB reuse the same plan. Failed attempts remain in the denominator. No generated answers were evaluated." : "Multi-query and Query-balanced are not run in this batch. Vietnamese text is a display translation only; product-Vietnamese retrieval is not measured here.";
  }
  function renderCaseList() {
    const query = get("search").value.toLocaleLowerCase(), select = get("case"), previous = select.value;
    const cases = data.cases.filter(c => (get("split").value === "ALL" || c.split === get("split").value) && `${c.id} ${c.question_en} ${c.question_vi} ${c.services.join(" ")}`.toLocaleLowerCase().includes(query));
    select.replaceChildren();
    for (const c of cases) { const option = node("option", `${c.id} · ${c.services.join(" / ")} · ${c.question_en}`); option.value = c.id; select.append(option); }
    if (cases.some(c => c.id === previous)) select.value = previous;
    get("case-count").textContent = `${cases.length} matching scenarios. Measured query language is selected above; the question-text language only changes the displayed translation.`;
    void openCase();
  }
  function sourceList(parent, ids, chunks, title) {
    parent.append(node("h4", title));
    if (!ids.length) { parent.append(node("p", "No passages were produced for this recorded method attempt.")); return; }
    const list = node("ol");
    for (const key of ids) {
      const c = chunks[key], item = node("li"), disclosure = node("details"), summary = node("summary", c.title);
      disclosure.append(summary, node("p", `${key} · source offsets ${c.start_char}–${c.end_char}`), node("blockquote", c.preview));
      if (c.preview_truncated) disclosure.append(node("p", "Excerpt shown. Open the official source for the complete page."));
      const link = node("a", "Open official AWS page"); link.href = officialSourceLink(c.uri); link.target = "_blank"; link.rel = "noopener noreferrer"; disclosure.append(link); item.append(disclosure); list.append(item);
    }
    parent.append(list);
  }
  function renderDetail() {
    if (!detail) return;
    const language = get("language").value, queryLanguage = get("run-language").value, method = get("method").value, methods = detail.methods_by_language?.[queryLanguage] || detail.methods, chosen = methods[method], baseline = methods.hybrid, parent = get("detail");
    parent.replaceChildren(node("h3", detail["question_" + language]), node("p", `${detail.scenario_id} · ${detail.split} · AI_DRAFT · ${detail.answerable ? "Answerable" : "Control"}`));
    if (!chosen) { parent.append(node("p", "Not run. This method needs a separately pinned shared planner stage; no model call is started by this page.")); return; }
    const queries = detail.queries?.[queryLanguage];
    if (queries) { parent.append(node("p", `Measured query: ${queryLanguage === "vi" ? "Vietnamese → shared English rewrite" : "English original"}. Search text: ${queries.search_query}`)); if (method.includes("multi_query")) parent.append(node("p", queries.preparation_status === "PLAN_FAILED" ? "Shared planner failed validation. This attempt scores zero evidence coverage; it was not repaired, retried or replaced with Hybrid." : `Shared search angles: ${queries.subqueries.join(" | ") || "No extra queries"}. ${chosen.context_selection_used ? "Context order uses a fixed round-robin within the same Q candidate pool." : "Candidate lists use reciprocal rank fusion."}`)); }
    parent.append(node("p", `Requested ${BENCHMARK_METHODS[method]}; effective ${BENCHMARK_METHODS[chosen.effective_method] || chosen.status}. Graph relations packed: ${chosen.context_assertion_ids.length}.`));
    if (!detail.answerable) parent.append(node("p", `Expected answer behavior: ${detail.expected_behavior}. Retrieval results cannot score this control as correctly answered. No generation was evaluated.`));
    else {
      parent.append(node("p", `Context complete: Hybrid ${baseline.metrics.context_complete ? "yes" : "no"}; selected ${chosen.metrics.context_complete ? "yes" : "no"}. This is source-span coverage, not answer correctness.`));
      const table = node("table"); table.className = "benchmark-table"; const head = node("tr");
      for (const title of ["Required fact", "Hybrid top 5", "Selected top 5"]) cell(head, title, "th");
      const thead = node("thead"); thead.append(head); const body = node("tbody");
      detail.groups.forEach((group, n) => { const row = node("tr"); cell(row, group["fact_" + language], "th"); cell(row, baseline.metrics.covered_groups_at_k[n] ? "Covered" : "Missing"); cell(row, chosen.metrics.covered_groups_at_k[n] ? "Covered" : "Missing"); body.append(row); });
      for (const th of head.children) th.scope = "col";
      table.append(thead, body); const hint = node("p", "Scroll the evidence table horizontally to compare both methods."); hint.className = "benchmark-scroll-hint"; parent.append(hint);
      const wrap = node("div"); wrap.className = "benchmark-table-wrap"; wrap.tabIndex = 0; wrap.setAttribute("role", "region"); wrap.setAttribute("aria-label", "Evidence group comparison"); wrap.append(table); parent.append(wrap);
      for (const group of detail.groups) { const facts = node("details"); facts.append(node("summary", group["fact_" + language]));
        for (const source of group.alternatives) { const link = node("a", source.title); link.href = officialSourceLink(source.uri); link.target = "_blank"; link.rel = "noopener noreferrer"; facts.append(link, node("blockquote", source.quote), node("p", `Original-source offsets ${source.start_char}–${source.end_char}; AI-draft supporting span.`)); } parent.append(facts); }
    }
    sourceList(parent, chosen.ranked_chunk_ids.slice(0, 5), detail.chunks, "Selected method · top 5 ranked passages");
    sourceList(parent, chosen.context_chunk_ids, detail.chunks, "Selected method · packed context");
    parent.append(node("p", `Recorded warnings: ${chosen.warnings.join(", ") || "None"}.`));
  }
  async function openCase() {
    const c = data?.cases.find(c => c.id === get("case").value), captured = epoch, revision = ++detailEpoch;
    detail = null; get("detail").replaceChildren(node("p", c ? "Loading saved evidence…" : "No matching scenarios."));
    if (!c) return;
    try { const result = validateBenchmarkCase(await loadCase(c), c); if (active && captured === epoch && revision === detailEpoch) { detail = result; renderDetail(); } }
    catch { if (active && captured === epoch && revision === detailEpoch) get("detail").replaceChildren(node("p", "Saved case evidence is unavailable or failed integrity validation. Select a case to retry. No model request was made.")); }
  }
  async function show(page) {
    if (page !== "benchmark316" || !active || data || loading) return;
    const captured = epoch; loading = true; get("status").textContent = "Loading recorded collection results…";
    try {
      const result = validateBenchmark316(await loadData()); if (!active || captured !== epoch) return; data = result;
      get("status").textContent = data.schema_version === 3 ? `1,600 recorded retrieval results · 100 paired EN/VI scenarios · ${data.shared_preparation_calls} shared preparation calls, $${(data.shared_preparation_cost_micro_usd / 1000000).toFixed(6)} published-rate upper bound. No model call is made when viewing or exporting.` : "600 recorded English method runs · 100 paired-language scenarios · 0 paid model calls. Reading or exporting this page starts no experiment.";
      get("run-language").querySelector('[value="vi"]').disabled = data.schema_version !== 3;
      get("run-language").value = "en";
      for (const option of get("method").options) { const label = BENCHMARK_METHODS[option.value]; option.textContent = label + (data.schema_version === 2 && option.value.includes("multi_query") ? " · not run" : ""); }
      get("identity").textContent = `${data.documents} documents / ${data.chunks.toLocaleString("en")} chunks / ${data.service_groups} service groups`;
      get("hashes").textContent = `Source manifest: ${data.source_manifest_sha256}\nProtocol: ${data.protocol_sha256}\nReceipts: ${data.receipt_sha256}`;
      renderResults(); renderCaseList();
      get("export").disabled = false;
    } catch { if (active && captured === epoch) get("status").textContent = "Results are unavailable or invalid. Reopen Benchmarks to retry. No model request was made."; }
    finally { if (captured === epoch) loading = false; }
  }
  get("split").addEventListener("change", () => { if (data) { renderResults(); renderCaseList(); } });
  get("search").addEventListener("input", () => { if (data) renderCaseList(); });
  get("case").addEventListener("change", openCase);
  get("method").addEventListener("change", renderDetail); get("language").addEventListener("change", renderDetail);
  get("run-language").addEventListener("change", () => { if (data) { get("language").value = get("run-language").value; renderResults(); renderDetail(); } });
  for (const tab of ["results", "cases", "methodology"]) get("tab-" + tab).addEventListener("click", () => setTab(tab));
  get("export").addEventListener("click", () => { if (data) download(JSON.stringify({ ...data, selected_case: detail }, null, 2), "aws-evidence-benchmark316.json", "application/json;charset=utf-8"); });
  function clear() { active = false; epoch++; detailEpoch++; data = null; detail = null; loading = false; for (const key of ["rows", "case", "detail"]) get(key).replaceChildren(); get("status").textContent = ""; get("search").value = ""; get("export").disabled = true; setTab("results"); }
  return { show, clear, activate() { clear(); active = true; } };
}
