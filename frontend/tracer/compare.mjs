import { officialSourceLink } from "./client.mjs";
import { validateCollectionAnswer } from "./collection-answer.mjs";
import { SCOPED_METHODS, scopedRunPath, validateScopedCapabilities, validateScopedRun, validateScopedResult,
  renderScopedResult, renderScopedGraph, renderScopedSteps } from "./scoped-runs.mjs";

export const RUN_ID = /^run_[a-f0-9]{64}$/;
export const RUN_METHODS = Object.freeze({ bm25: "BM25", vector: "Vector", hybrid: "Hybrid", hybrid_reranked: "Hybrid + reranker" });
const terminal = new Set(["COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED", "UNKNOWN"]);
export function isAdmissionRejection(error, recovering) {
  if (recovering) return false; // A previous ambiguous attempt may already exist.
  return error?.status === 400 && ["INVALID_REQUEST", "MODE_UNAVAILABLE"].includes(error.code) ||
    error?.status === 429 && ["BUDGET_LIMIT", "QUERY_QUOTA_EXCEEDED", "RUN_CONCURRENCY_LIMIT"].includes(error.code);
}
export function runPath(id) { if (!RUN_ID.test(id)) throw new Error("INVALID_RUN_ID"); return `/v1/runs/${id}`; }
export function validateRun(row) {
  runPath(row?.run_id);
  if (!row.request || typeof row.request.original_question !== "string" || row.request.original_question.length > 2000 ||
      !["retrieval", "answer"].includes(row.request.run_kind) || !/^collection_[a-f0-9]{64}$/.test(row.collection_version) || !["QUEUED", "RUNNING", ...terminal].includes(row.status) ||
      !Array.isArray(row.methods) || row.methods.length < 1 || row.methods.length > 3 ||
      new Set(row.methods.map(m => m.method_id)).size !== row.methods.length ||
      row.methods.some(m => !Object.hasOwn(RUN_METHODS, m.method_id) || !["QUEUED", "CLAIMED", "EXECUTING", ...terminal].includes(m.status) ||
        typeof m.result_available !== "boolean" || m.result_available !== (m.status === "COMPLETED")) || row.quality_metrics !== null) throw new Error("INVALID_RUN_RESPONSE");
  return row;
}
export function validateRunResult(result, method, runKind = "retrieval", runId = null) {
  const answering = runKind === "answer";
  if (!["retrieval", "answer"].includes(runKind) || result?.method_id !== method || !Object.hasOwn(RUN_METHODS, method) || result.status !== (answering ? "COMPLETED" : "RETRIEVED") || result.quality_metrics !== null ||
      !Number.isSafeInteger(result.new_cost_micro_usd) || result.new_cost_micro_usd < 0 || !answering && result.new_cost_micro_usd !== 0 ||
      !/^[a-f0-9]{64}$/.test(result.config_sha256) || result.semantic_review !== "NOT_REVIEWED" ||
      !Array.isArray(result.context_chunk_ids) || result.context_chunk_ids.length > 8 || !Array.isArray(result.context_sources) ||
      result.context_sources.length !== result.context_chunk_ids.length || !Array.isArray(result.ranked_chunk_ids) || result.ranked_chunk_ids.length > 40 ||
      new Set(result.ranked_chunk_ids).size !== result.ranked_chunk_ids.length || new Set(result.context_chunk_ids).size !== result.context_chunk_ids.length ||
      result.ranked_chunk_ids.some(id => !/^chk_[a-f0-9]{64}$/.test(id)) || result.context_chunk_ids.some(id => !result.ranked_chunk_ids.includes(id)) ||
      result.context_sources.some((s, i) => s.chunk_id !== result.context_chunk_ids[i] || typeof s.title !== "string" || !officialSourceLink(s.source_uri))) throw new Error("INVALID_RUN_RESULT");
  if (answering) { if (!RUN_ID.test(result.run_id) || runId !== null && result.run_id !== runId) throw new Error("INVALID_RUN_RESULT"); validateCollectionAnswer(result); }
  else if (result.response !== undefined) throw new Error("INVALID_RUN_RESULT");
  return result;
}
export function contextOverlap(results) {
  if (results.length < 2) return null;
  const first = new Set(results[0].context_chunk_ids), union = new Set(results.flatMap(r => r.context_chunk_ids));
  const shared = [...first].filter(id => results.every(r => r.context_chunk_ids.includes(id))).length;
  return { shared, union: union.size }; // Evidence overlap, never answer accuracy.
}

export function createCompare({ document, request, uuid = () => crypto.randomUUID(), onSessionExpired, download,
  schedule = setTimeout, unschedule = clearTimeout, scopedEnabled = () => false, onCapabilities = () => {}, onRunChanged = () => {}, onInspectMethod = () => {}, onSourceOpened = () => {} }) {
  const get = id => document.getElementById(id);
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  const button = (text, action) => { const n = el("button", text, "secondary small"); n.type = "button"; n.addEventListener("click", action); return n; };
  let active = false, epoch = 0, version = 0, capability = null, row = null, timer = null, busy = false, pending = null;
  let results = new Map(), selected = "hybrid", pollCount = 0, refreshing = false, sourceVersion = 0;
  let scoped = false;
  const labels = () => scoped ? SCOPED_METHODS : RUN_METHODS;
  const rootPath = () => scoped ? "/v2/scoped-runs" : "/v1/runs";
  const pathFor = id => scoped ? scopedRunPath(id) : runPath(id);
  const checkRun = value => scoped ? validateScopedRun(value) : validateRun(value);
  const checkResult = (value, method, kind, id, questionSha) => scoped ? validateScopedResult(value, method, id, questionSha) : validateRunResult(value, method, kind, id);
  const current = (e, v = version) => active && epoch === e && version === v;
  function stop() { if (timer !== null) unschedule(timer); timer = null; }
  function controls() {
    const chosen = [...document.querySelectorAll("#compare-methods input:checked")].map(n => n.value);
    const answering = get("compare-kind").value === "answer";
    const maximum = answering ? capability?.maximum_answer_methods || 1 : 3;
    get("compare-submit").disabled = !active || !capability || busy || !!pending || !chosen.length || chosen.length > maximum;
    get("compare-new").disabled = !active || !capability || busy || !!pending;
    get("compare-question").disabled = busy || !!pending;
    get("compare-language").disabled = busy || !!pending;
    get("compare-kind").disabled = busy || !!pending;
    get("compare-selection-status").textContent = `${chosen.length} / ${maximum} methods selected. Uncheck a method before choosing another. / Bỏ chọn để đổi phương pháp.`;
    get("compare-recover").hidden = !pending;
    get("compare-refresh").disabled = !row || busy;
    get("compare-cancel").disabled = !row || terminal.has(row.status) || busy;
    get("compare-export").disabled = !row || !results.size || busy;
    get("compare-protocol").disabled = busy || !!pending;
    get("compare-kind").querySelector('option[value="answer"]').disabled = capability?.answering_enabled !== true;
    get("compare-kind").querySelector('option[value="retrieval"]').disabled = scoped;
    get("compare-submit").textContent = answering ? "Ask collection" : "Run comparison";
    get("compare-scope-note").textContent = answering
      ? "Answer generation is budgeted. Automated source checks may withhold the answer for review; they do not verify semantic correctness. Opening saved results does not call a model."
      : "Compare retrieved evidence without a model call. Sources and similarity scores do not establish answer correctness.";
    get("compare-language").querySelector('option[value="vi"]').textContent = answering ? "Tiếng Việt · shared English search rewrite" : "Tiếng Việt · diagnostic";
    get("compare-help").textContent = scoped
      ? "Up to three budgeted methods on the pinned collection. Vietnamese uses a shared English search rewrite. Graph requires supported paths; Q/QB use literal task facets. No silent method substitution."
      : answering
      ? "One Hybrid or Vector answer per run. Vietnamese questions share an English search rewrite. Answers and sources are saved in your runs; model use is budgeted."
      : "Each method uses one of the shared 300 demo attempts per month. Retrieval does not call a model; raw Vietnamese queries use the English encoder directly.";
    for (const input of document.querySelectorAll("#compare-methods input")) {
      input.closest("label").hidden = !scoped && !Object.hasOwn(RUN_METHODS, input.value);
      const allowed = (answering ? capability?.answer_methods : capability?.methods)?.includes(input.value);
      input.disabled = !allowed || busy || !!pending || !input.checked && chosen.length >= maximum;
      if (capability && !allowed) input.checked = false;
    }
    for (const b of get("compare-history").querySelectorAll("button")) b.disabled = busy || !!pending;
  }
  function newQuestion({ keepText = false, focus = true } = {}) {
    if (!active || !capability || busy || pending) return false;
    const continues = row && !terminal.has(row.status);
    stop(); version++; sourceVersion++; row = null; results.clear(); refreshing = false; pollCount = 0;
    for (const id of ["compare-run-meta", "compare-progress", "compare-results", "compare-source", "compare-trace", "compare-overlap"]) get(id).replaceChildren();
    if (!keepText) get("compare-question").value = "";
    get("compare-status").textContent = continues
      ? "New draft. The previous run continues in Saved runs; it has not been cancelled. / Lượt cũ tiếp tục trong lịch sử."
      : "Ready for a new question. Previous results remain in Saved runs. Nothing has been submitted. / Sẵn sàng cho câu hỏi mới.";
    controls();
    if (focus) { get("compare-question").focus({ preventScroll: true }); get("compare-question").scrollIntoView({ block: "center", behavior: "auto" }); }
    return true;
  }
  function failure(error) {
    if (error?.status === 401 || error?.status === 403) { clear(); onSessionExpired?.(error); return; }
    get("compare-status").textContent = error?.status === 429 ? "Usage or concurrency limit reached. Existing runs remain in history."
      : "The request could not be confirmed. Check runs; a new job has not been submitted automatically.";
    stop();
  }
  function clear() {
    stop(); active = false; epoch++; version++; sourceVersion++; row = null; capability = null; pending = null; results.clear(); busy = false; refreshing = false;
    for (const id of ["compare-corpus", "compare-status", "compare-run-meta", "compare-progress", "compare-results", "compare-source", "compare-trace", "compare-history", "compare-overlap", "overview-live-collection"]) get(id)?.replaceChildren();
    if (get("compare-question")) get("compare-question").value = "";
    get("compare-kind").value = scoped ? "answer" : "retrieval"; onCapabilities(null);
    controls();
  }
  function render() {
    if (!row) return;
    const oldFocus = document.activeElement;
    get("compare-run-meta").textContent = `${row.status} · ${row.request.run_kind === "answer" ? "Answer" : "Retrieval"} · Run ${row.run_id.slice(4, 16)} · Collection ${row.collection_version.slice(11, 23)}`;
    get("compare-run-meta").title = `${row.run_id} · ${row.collection_version}`;
    get("compare-progress").replaceChildren();
    for (const method of row.methods) {
      const n = button(`${labels()[method.method_id]} · ${method.status}`, () => { selected = method.method_id; render(); });
      n.setAttribute("aria-pressed", String(selected === method.method_id));
      n.dataset.method = method.method_id; get("compare-progress").append(n);
    }
    const overlap = contextOverlap([...results.values()]);
    get("compare-overlap").textContent = overlap ? `${overlap.shared} context chunks shared by all completed methods, ${overlap.union} in their union. Evidence overlap is not an accuracy score.` : "Choose two or three methods to compare their retrieved evidence.";
    const area = get("compare-results"); area.dataset.kind = row.request.run_kind; area.replaceChildren();
    for (const method of row.methods) {
      const card = el("article", undefined, "dashboard-card compare-result"); card.dataset.selected = String(selected === method.method_id);
      card.append(el("p", row.request.run_kind === "answer" ? "Collection answer · saved evidence" : "Live run · retrieval only", "eyebrow"), el("h3", labels()[method.method_id]), el("p", method.status, "badge"));
      const result = results.get(method.method_id);
      if (result && scoped) {
        const capturedEpoch = epoch, capturedVersion = version, capturedRun = row;
        card.append(button("Inspect this method", () => {
          if (!current(capturedEpoch, capturedVersion)) return;
          onInspectMethod(capturedRun, result, (chunk, quote) => {
            if (current(capturedEpoch, capturedVersion)) void openSource(method.method_id, chunk, quote ? { quote } : null);
          });
        }));
        renderScopedResult(document, card, result, (chunk, citation) => openSource(method.method_id, chunk, citation));
        const steps = el("details"); steps.append(el("summary", "Recorded execution · search and evidence packing"));
        const host = el("div"); renderScopedSteps(document, host, result); steps.append(host); card.append(steps);
        if (method.method_id.startsWith("graph_")) { const graph = el("div", undefined, "scoped-graph"); renderScopedGraph(document, graph, result, chunk => openSource(method.method_id, chunk)); card.append(graph); }
      } else if (result) {
        card.append(el("p", `${result.ranked_chunk_ids.length} candidates · ${result.context_chunk_ids.length} context chunks · ${result.context_proxy_tokens} proxy tokens`, "muted"));
        const list = el("ol");
        for (const source of result.context_sources) {
          const item = el("li"), action = button(source.title, () => openSource(method.method_id, source.chunk_id));
          const rank = result.ranked_chunk_ids.indexOf(source.chunk_id) + 1;
          item.append(action, el("p", `Candidate rank ${rank} · ${source.section_path?.join(" / ") || "Document passage"}`, "muted")); list.append(item);
        }
        if (result.response) {
          const answer = result.response;
          card.append(el("p", answer.answer_status, "badge"));
          for (const claim of answer.claims) {
            const block = el("div", undefined, "collection-claim"); block.append(el("p", claim.text));
            for (const id of claim.citation_ids) {
              const citation = answer.citations.find(c => c.citation_id === id);
              const index = answer.citations.findIndex(c => c.citation_id === id) + 1;
              block.append(button(`[${index}] Read source`, () => openSource(method.method_id, citation.chunk_id, citation)));
            }
            card.append(block);
          }
          if (answer.answer_status === "ABSTAINED") card.append(el("p", answer.answer));
          if (answer.answer_status === "PARTIAL") card.append(el("p", "The supplied evidence supports only part of the question. Inspect the citations and remaining conditions.", "muted"));
          card.append(el("p", "Quality is not independently graded. A citation is a source to inspect, not a guarantee of correctness.", "muted"));
          if (row.budget) card.append(el("p", `Run model usage: $${(row.budget.settled_micro_usd / 1000000).toFixed(6)} · ${row.budget.cost_complete ? "accounted" : "accounting pending"} · published-rate upper bound, not invoice`, "muted"));
          const details = el("details"), summary = el("summary", "Retrieved evidence passages"); details.append(summary, list); card.append(details);
        } else card.append(list, el("p", "Answer quality: not graded. No answer was generated.", "muted"));
      } else card.append(el("p", method.error_category || "Waiting for a worker. Completed methods stay available if another fails.", "muted"));
      area.append(card);
    }
    if (oldFocus?.dataset.method) get("compare-progress").querySelector(`[data-method="${oldFocus.dataset.method}"]`)?.focus({ preventScroll: true });
    controls();
    const shownRow = row, shownEpoch = epoch, shownVersion = version;
    onRunChanged(shownRow, results, (key, quote) => {
      if (!current(shownEpoch, shownVersion) || row?.run_id !== shownRow.run_id) return false;
      void openSource(shownRow.methods[0].method_id, key, quote ? { quote } : null); return true;
    });
  }
  async function openSource(method, chunk, citation = null) {
    const e = epoch, v = version, sourceTicket = ++sourceVersion;
    try {
      if (!row || !Object.hasOwn(labels(), method) || !/^chk_[a-f0-9]{64}$/.test(chunk)) return;
      get("compare-source").replaceChildren(el("p", "Loading the selected source passage…", "muted"));
      const source = await request(`${pathFor(row.run_id)}/methods/${method}/sources/${chunk}`);
      if (!current(e, v) || sourceTicket !== sourceVersion) return;
      if (source.chunk_id !== chunk || typeof source.text !== "string" || source.text.length > 120000 || !officialSourceLink(source.source_uri)) throw new Error("INVALID_SOURCE");
      const expected = results.get(method)?.context_sources.find(s => s.chunk_id === chunk);
      if (!expected || ["document_version_id", "shard_id", "shard_manifest_sha256", "start_char", "end_char", "source_uri", "title", "text_sha256"].some(k => source[k] !== expected[k])) throw new Error("INVALID_SOURCE");
      const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source.text)))].map(n => n.toString(16).padStart(2, "0")).join("");
      if (!current(e, v) || sourceTicket !== sourceVersion) return;
      if (digest !== expected.text_sha256 || citation && (!source.text.includes(citation.quote) || source.text.indexOf(citation.quote) !== source.text.lastIndexOf(citation.quote))) throw new Error("INVALID_SOURCE");
      const panel = get("compare-source"); panel.replaceChildren(el("p", `${labels()[method]} · selected source`, "eyebrow"), el("h3", source.title), el("p", `Source characters ${source.start_char}–${source.end_char}`, "muted"), el("blockquote", source.text));
      const link = el("a", "Read official AWS page"); link.href = officialSourceLink(source.source_uri); link.target = "_blank"; link.rel = "noopener noreferrer";
      if (citation) panel.prepend(el("p", "Selected citation quote", "eyebrow"), el("blockquote", citation.quote));
      onSourceOpened(source, citation?.quote, row);
      panel.append(link); panel.tabIndex = -1; panel.focus({ preventScroll: true }); panel.scrollIntoView({ block: "nearest", behavior: "auto" });
    } catch (error) { if (current(e, v) && sourceTicket === sourceVersion) {
      get("compare-source").replaceChildren(el("p", "This source is unavailable. No previous passage is shown.", "muted"));
      onSourceOpened(null, null, row);
      failure(error);
    } }
  }
  async function refresh() {
    if (!active || !row || busy || refreshing) return;
    stop(); const e = epoch, v = version, id = row.run_id; refreshing = true;
    try {
      const next = checkRun(await request(pathFor(id)));
      if (!current(e, v)) return;
      if (next.run_id !== id) throw new Error("INVALID_RUN_RESPONSE");
      row = next;
      let questionSha;
      if (scoped) {
        questionSha = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(row.request.original_question)))].map(n => n.toString(16).padStart(2, "0")).join("");
        if (!current(e, v)) return;
      }
      for (const method of row.methods.filter(m => m.result_available && !results.has(m.method_id))) {
        const data = checkResult(await request(`${pathFor(id)}/methods/${method.method_id}`), method.method_id, row.request.run_kind, id, questionSha);
        if (data.collection_version !== row.collection_version) throw new Error("RUN_COLLECTION_MISMATCH");
        if (!current(e, v)) return;
        results.set(method.method_id, data);
      }
      render(); get("compare-status").textContent = terminal.has(row.status) ? "Run finished. Sources, trace and export reuse saved results." : "Run in progress. Status checks do not submit a new run.";
      if (!terminal.has(row.status) && pollCount++ < 100 && !document.hidden && visible()) timer = schedule(refresh, Math.min(10000, 2000 + pollCount * 250));
    } catch (error) { if (current(e, v)) failure(error); }
    finally { if (current(e, v)) refreshing = false; }
  }
  async function open(id) {
    if (!active || busy || pending) return false;
    pathFor(id); stop(); const e = epoch, v = ++version;
    results.clear(); row = null; refreshing = false; get("compare-source").replaceChildren(); get("compare-trace").replaceChildren();
    try {
      const next = checkRun(await request(pathFor(id)));
      if (!current(e, v)) return;
      row = next; pending = null; pollCount = 0; get("compare-question").value = row.request.original_question;
      get("compare-language").value = row.request.language;
      get("compare-kind").value = row.request.run_kind;
      for (const input of document.querySelectorAll("#compare-methods input")) input.checked = row.request.method_ids.includes(input.value);
      selected = row.methods[0].method_id; render(); await refresh();
    } catch (error) { if (current(e, v)) failure(error); }
  }
  async function history() {
    const e = epoch;
    try {
      const data = await request(`${rootPath()}?limit=10`);
      if (!active || e !== epoch) return;
      if (!Array.isArray(data.items) || data.items.length > 10) throw new Error("INVALID_HISTORY");
      get("compare-history").replaceChildren();
      for (const item of data.items) { checkRun(item); get("compare-history").append(button(`${item.status} · ${item.request.original_question}`, () => open(item.run_id))); }
      if (!data.items.length) get("compare-history").append(el("p", "No runs yet. Your saved runs will appear here.", "muted"));
      controls();
    } catch (error) { if (active && e === epoch) failure(error); }
  }
  async function submit() {
    if (!active || busy || !capability) return;
    const e = epoch, recovering = !!pending;
    if (!pending) {
      const methods = [...document.querySelectorAll("#compare-methods input:checked")].map(n => n.value);
      const question = get("compare-question").value.trim();
      const kind = get("compare-kind").value;
      if (!question || question.length > 2000 || methods.length < 1 || methods.length > (kind === "answer" ? capability.maximum_answer_methods || 1 : 3) || scoped && kind !== "answer" ||
          kind === "answer" && (capability.answering_enabled !== true || methods.some(m => !capability.answer_methods.includes(m)))) {
        get("compare-status").textContent = scoped ? "Enter a question and choose one to three enabled answer methods." : kind === "answer" ? "Choose one enabled answer method." : "Enter a question and select one to three methods."; return;
      }
      newQuestion({ keepText: true, focus: false });
      pending = { key: uuid(), body: { original_question: question, method_ids: methods, language: get("compare-language").value, source_scope: "local_collection", run_kind: kind } };
    }
    busy = true; controls(); get("compare-status").textContent = "Submitting this run once…";
    try {
      const next = checkRun(await request(rootPath(), pending));
      if (!active || e !== epoch) return;
      pending = null; row = next; version++; refreshing = false; results.clear(); selected = row.methods[0].method_id; pollCount = 0;
      get("compare-source").replaceChildren(); get("compare-trace").replaceChildren(); render();
    } catch (error) { if (active && e === epoch) {
      if (isAdmissionRejection(error, recovering)) {
        pending = null; stop();
        get("compare-status").textContent = error.status === 429
          ? "Usage or concurrency limit reached. This submission was not admitted. Choose fewer methods or try later; you can edit your draft. / Chưa tạo lượt chạy mới; bạn có thể sửa câu hỏi hoặc giảm phương pháp."
          : "This question or method selection was rejected before admission. Edit the draft and submit when ready. / Chưa tạo lượt chạy; hãy sửa câu hỏi hoặc phương pháp.";
      } else failure(error);
    } }
    finally { if (active && e === epoch) { busy = false; controls(); if (!pending) { await refresh(); await history(); } } }
  }
  get("compare-form").addEventListener("submit", event => { event.preventDefault(); submit(); });
  get("compare-new").addEventListener("click", () => newQuestion());
  get("compare-question").addEventListener("input", () => { if (!row || get("compare-question").value.trim() !== row.request.original_question) newQuestion({ keepText: true, focus: false }); });
  for (const input of document.querySelectorAll("#compare-methods input")) input.addEventListener("change", controls);
  get("compare-kind").addEventListener("change", () => {
    if (get("compare-kind").value === "answer") for (const input of document.querySelectorAll("#compare-methods input")) input.checked = input.value === "hybrid";
    controls();
  });
  get("compare-recover").addEventListener("click", submit); // Same frozen body/key after ambiguous transport, never automatic POST.
  get("compare-refresh").addEventListener("click", refresh);
  get("compare-history-refresh").addEventListener("click", history);
  get("compare-cancel").addEventListener("click", async () => {
    if (!row || busy) return;
    const e = epoch, v = version, id = row.run_id;
    try { const data = checkRun(await request(`${pathFor(id)}/cancel`, { body: {}, key: uuid() })); if (current(e, v)) { row = data; render(); await refresh(); } }
    catch (error) { if (current(e, v)) failure(error); }
  });
  get("compare-trace-open").addEventListener("click", async () => {
    if (!row) return; const e = epoch, v = version, id = row.run_id;
    try {
      const trace = await request(`${pathFor(id)}/trace`); if (!current(e, v)) return;
      if (trace.run_id !== id || !Array.isArray(trace.methods) || trace.methods.length > 3) throw new Error("INVALID_TRACE");
      get("compare-trace").replaceChildren(el("pre", JSON.stringify(trace, null, 2)));
    } catch (error) { if (current(e, v)) failure(error); }
  });
  get("compare-export").addEventListener("click", async () => {
    if (!row || !results.size || busy) return;
    const e = epoch, v = version, id = row.run_id, saved = [...results.values()];
    busy = true; controls();
    try {
      const detail = checkRun(await request(pathFor(id)));
      if (!current(e, v)) return;
      if (detail.run_id !== id || detail.collection_version !== row.collection_version ||
          saved.some(result => !detail.methods.some(m => m.method_id === result.method_id && m.result_available))) throw new Error("INVALID_EXPORT_STATE");
      download(JSON.stringify({ schema_version: scoped ? 2 : 1, run: detail, results: saved, quality_metrics: null }, null, 2), `aws-${detail.request.run_kind}-${id.slice(4, 16)}.json`);
    } catch (error) { if (current(e, v)) failure(error); }
    finally { if (current(e, v)) { busy = false; controls(); } }
  });
  const visible = () => !get("page-compare").hidden || [...document.querySelectorAll('[data-view^="method-"]')].some(n => !n.hidden);
  const visibility = () => { if (document.hidden || !visible()) stop(); else if (active && row && !terminal.has(row.status)) refresh(); };
  document.addEventListener("visibilitychange", visibility);
  get("compare-protocol").addEventListener("change", () => { if (active && !busy && !pending) void activate(); });
  async function activate() {
    scoped = scopedEnabled() === true && get("compare-protocol").value === "scoped-v2";
    get("compare-protocol-control").hidden = scopedEnabled() !== true;
    clear(); active = true; const e = epoch;
    try {
      const data = await request(scoped ? "/v2/scoped-run-capabilities" : "/v1/run-capabilities");
      if (!active || e !== epoch) return;
      if (scoped) validateScopedCapabilities(data);
      else {
        if (data.available !== true || typeof data.answering_enabled !== "boolean" || !Array.isArray(data.methods) || !data.methods.length ||
            data.methods.some(m => !Object.hasOwn(RUN_METHODS, m)) || data.quality_metrics !== null || !data.corpus) throw new Error("RUNS_NOT_ENABLED");
        if (data.answering_enabled && (data.maximum_answer_methods !== 1 || !Array.isArray(data.answer_methods) || !data.answer_methods.length || data.answer_methods.some(m => !["hybrid", "vector"].includes(m)))) throw new Error("INVALID_ANSWER_CAPABILITY");
      }
      capability = data; onCapabilities(data);
      get("compare-corpus").textContent = scoped
        ? `${data.corpus.documents} documents · ${data.corpus.chunks} chunks · ${data.graph.facts} experimental graph facts from ${data.graph.source_documents} documents · human review pending`
        : `${data.corpus.documents} documents · ${data.corpus.chunks} chunks · ${data.corpus.service_groups} service groups · versioned collection`;
      get("overview-live-collection").textContent = get("compare-corpus").textContent + ".";
      get("compare-status").textContent = scoped ? "Ready. Each method uses one shared demo attempt. Answer generation is budgeted; source checks are not a correctness score."
        : "Ready. Each selected method uses one demo attempt; no model call is made. Vietnamese queries currently use the English encoder directly.";
      controls(); await history();
    } catch (error) { if (active && e === epoch) {
      if (error?.status === 401 || error?.status === 403) { failure(error); return; }
      capability = null; controls();
      get("compare-status").textContent = "This collection workflow is unavailable. No alternative workflow or method was submitted.";
      get("overview-live-collection").textContent = "Not enabled in this deployment.";
    } }
  }
  return { clear, open, refresh, history, stop, newQuestion, pageChanged: visibility,
    startMethod: async (question, language, method, kind = "retrieval") => {
      if (!active || busy || pending || !(kind === "answer" ? capability?.answering_enabled && capability.answer_methods.includes(method) : capability?.methods.includes(method))) return false;
      get("compare-question").value = question; get("compare-language").value = language; get("compare-kind").value = kind;
      for (const input of document.querySelectorAll("#compare-methods input")) input.checked = input.value === method;
      controls(); await submit(); return !pending;
    },
    startAnswer: async (question, language, method) => {
      if (!active || busy || pending || capability?.answering_enabled !== true || !capability.answer_methods.includes(method)) return false;
      get("compare-question").value = question; get("compare-language").value = language; get("compare-kind").value = "answer";
      for (const input of document.querySelectorAll("#compare-methods input")) input.checked = input.value === method;
      controls(); await submit(); return true;
    },
    activate };
}
