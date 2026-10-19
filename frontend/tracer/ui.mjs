import { beginLogin, finishLogin, validateConfig, createRegistrationDialog, registerWithCognito } from "./auth.mjs";
import { ApiFailure, apiRequest, errorMessage, officialSourceLink, queryPath, sourcePath } from "./client.mjs";
import { SCENARIOS, INVESTIGATION_STEPS, buildQuestion } from "./investigation.mjs";
import { EvidenceFailure, createEvidenceController, splitExactQuote, projectEvidence } from "./evidence.mjs";
import { createBriefDownloader, serializeBrief, briefFilename } from "./brief.mjs";
import { createDashboard, loadDashboardData } from "./dashboard.mjs";
import { createOnboarding } from "./onboarding.mjs";

export function createWorkspace({ document, request, downloader = createBriefDownloader({ document }),
  uuid = () => crypto.randomUUID(), onSessionExpired = () => {}, confirmReplace = message => globalThis.confirm(message), loadDashboard = loadDashboardData, tourStorage }) {
  const get = id => document.getElementById(id);
  function syncSourceOptions() {
    const live = get("corpus-scope").value === "aws_official";
    get("pilot-method").hidden = live;
    if (live) get("mode").value = "hybrid";
    get("source-scope-help").textContent = live
      ? "Search official AWS documentation beyond the pilot. Retrieved excerpts are saved with each answer; sources and coverage may be incomplete."
      : "Search only the frozen 20-document pilot. Knowledge Explorer and Retrieval Lab describe this recorded corpus.";
  }
  get("corpus-scope").addEventListener("change", syncSourceOptions);
  syncSourceOptions();
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  let active = false, session = 0, view = 0, historySequence = 0, exportSequence = 0;
  let record = null, pending = null, cursor = null, busy = false, exporting = false, scenarioId = null;
  const reads = new Set();
  const current = captured => active && session === captured;
  const mayReplace = () => !get("brief-notes").value.trim() || confirmReplace("Changing investigations will clear your session-only notes. Continue?");
  const dashboard = createDashboard({ document, loadData: loadDashboard, onOpenRecord: id => openRecord(id),
    onScenario: id => chooseScenario(id), onQuestion: question => {
      if (!active || busy || !mayReplace()) return;
      pending = null; resetAnswer(); get("question").value = question; dashboard.show("investigate"); get("question").focus();
      get("query-status").textContent = "Draft ready from the catalog. Edit it before choosing Ask a question. Nothing has been submitted.";
    } });
  const onboarding = createOnboarding({ document, navigate: page => dashboard.show(page), getPage: dashboard.getPage,
    canStart: () => active && !busy && !exporting && !reads.size, storage: tourStorage });
  async function readRequest(path) {
    const ticket = Symbol(); reads.add(ticket); onboarding.syncAvailability();
    try { return await request(path); }
    finally { reads.delete(ticket); onboarding.syncAvailability(); }
  }
  function setBusy(value) {
    busy = value;
    for (const element of document.querySelectorAll("#question-form button, #question-form textarea, #question-form select, #scenario-cards button, #scenario-fields input, #scenario-fields select, #scenario-fields textarea, #build-question, #history button")) element.disabled = value;
    get("export-brief").disabled = value || exporting || !record || record.status !== "COMPLETED";
    onboarding.syncAvailability();
  }
  function clearInspector() {
    for (const id of ["source-title", "source-meta", "source-text", "source-link", "quote-status"]) get(id).replaceChildren();
  }
  function sourceLink(raw, label = "Official AWS page ↗") {
    const url = officialSourceLink(raw);
    if (!url) return null;
    const link = node("a", label); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; return link;
  }
  function showError(error) {
    const message = error instanceof EvidenceFailure ? error.message : errorMessage(error);
    if (error?.status === 401 || error?.status === 403) { clear(); onSessionExpired(error); }
    get("query-status").textContent = message;
  }
  function renderEvidence(state) {
    get("source-back").hidden = state.phase !== "ready";
    clearInspector(); get("citations").replaceChildren();
    for (const claim of document.querySelectorAll("[data-claim-id]")) {
      const selected = claim.dataset.claimId === state.claimId;
      claim.classList.toggle("selected", selected);
      const button = claim.querySelector(".claim-select"); if (button) button.setAttribute("aria-pressed", String(selected));
    }
    const data = state.record, selectedClaim = data?.claims.find(claim => claim.claim_id === state.claimId);
    if (data?.citations.length && !selectedClaim) get("citations").append(node("p", "Select a statement above to inspect its cited sources.", "muted"));
    for (const id of selectedClaim?.citation_ids || []) {
      const citation = data.citations.find(item => item.citation_id === id), index = data.citations.indexOf(citation);
      const card = node("article", undefined, `citation${id === state.citationId ? " selected" : ""}`);
      card.append(node("strong", `[${index + 1}] ${citation.title}`), node("blockquote", citation.quote));
      const button = node("button", "Read source passage", "secondary small"); button.type = "button";
      button.addEventListener("click", () => selectCitation(id)); card.append(button);
      const link = sourceLink(citation.source_uri); if (link) card.append(link);
      get("citations").append(card);
    }
    for (const button of document.querySelectorAll("[data-citation-id]")) button.setAttribute("aria-pressed", String(button.dataset.citationId === state.citationId));
    if (!data || !state.citationId) {
      get("source-status").textContent = selectedClaim ? "Choose a cited source to read the original passage." : "Select a statement, then a cited source to inspect the original passage.";
      return;
    }
    const citation = data.citations.find(item => item.citation_id === state.citationId);
    get("source-title").textContent = citation.title;
    if (state.phase === "loading") { get("source-status").textContent = "Loading this source passage…"; return; }
    if (state.phase === "error") { get("source-status").textContent = "This source is unavailable. No previous passage is shown. Choose a source to try reading it again."; return; }
    if (state.phase !== "ready") return;
    const source = state.source, split = splitExactQuote(source.text, citation.quote);
    get("source-status").textContent = "Source passage loaded. Review its conditions and context.";
    get("source-title").textContent = source.title;
    get("source-meta").textContent = state.record?.retrieval_backend === "aws_knowledge_mcp"
      ? `Saved official AWS search excerpt · pinned to this answer${source.retrieved_at ? ` · Retrieved ${new Date(source.retrieved_at).toLocaleString()}` : ""}${source.normalizer_version?.endsWith("-v2") ? " · Prose line wraps normalized; original excerpt retained" : ""} · Page update date unknown`
      : `${source.section_path.join(" › ")} · Snapshot ${source.snapshot_version.slice(0, 17)}…`;
    if (split.matched) get("source-text").append(document.createTextNode(split.before), node("mark", split.quote), document.createTextNode(split.after));
    else get("source-text").textContent = source.text;
    get("quote-status").textContent = split.matched ? "The cited quote is highlighted below." : "The quote could not be located uniquely. The passage is shown without a highlight.";
    const link = sourceLink(citation.source_uri); if (link) get("source-link").append(link);
  }
  const evidence = createEvidenceController({ loadSource: (data, citation) => readRequest(sourcePath(data.request_id, citation.chunk_id)), onChange: renderEvidence, onError: showError });
  function resetChecks() { for (const input of document.querySelectorAll("#investigation-checklist input")) input.checked = false; }
  function resetAnswer() {
    view++; exportSequence++; record = null; exporting = false; downloader.clear(); evidence.clear(); resetChecks();
    for (const id of ["answer", "citations", "answer-title", "answer-meta", "answer-badge", "evidence-links", "export-status"]) get(id).replaceChildren();
    get("answer-panel").hidden = true; get("brief-panel").hidden = true; get("evidence-map").hidden = true;
    get("brief-notes").value = ""; get("export-brief").disabled = true;
    onboarding.syncAvailability();
  }
  function clear() {
    active = false; session++; historySequence++; cursor = null; pending = null; scenarioId = null;
    reads.clear();
    onboarding.deactivate();
    document.body.dataset.workspaceActive = "false";
    dashboard.clear();
    resetAnswer(); get("workspace").hidden = true; get("history").replaceChildren(); get("question").value = "";
    get("scenario-fields").replaceChildren(); get("scenario-title").textContent = ""; get("scenario-detail").hidden = true;
    get("scenario-library").open = false;
    get("scenario-status").textContent = ""; get("query-status").textContent = ""; get("more-history").hidden = true;
    for (const card of document.querySelectorAll("[data-scenario-id]")) card.setAttribute("aria-pressed", "false");
    setBusy(false);
  }
  async function selectCitation(id) {
    if (!active) return;
    const captured = session, revision = view;
    let applied = false;
    try { applied = await evidence.selectCitation(id); }
    catch (error) { showError(error); }
    if (applied && current(captured) && view === revision) { get("source-heading").focus({ preventScroll: true }); get("source-panel").scrollIntoView({ behavior: "auto", block: "nearest" }); }
  }
  function renderRecord(next) {
    if (!active) return;
    dashboard.show("investigate");
    queryPath(next.request_id);
    const data = next.status === "COMPLETED" ? projectEvidence(next) : null;
    resetAnswer(); record = next; get("answer-panel").hidden = false; get("answer-title").textContent = next.question;
    if (!data) {
      get("answer-badge").textContent = next.status === "PROCESSING" ? "Processing" : "Could not complete";
      get("answer").append(node("p", next.status === "PROCESSING" ? "Refresh history to check the outcome. This request is not submitted again automatically." : errorMessage(new ApiFailure(next.error?.code))));
      return;
    }
    evidence.setRecord(next);
    get("answer-badge").textContent = { ANSWERED: "Answered", PARTIAL: "Partial answer", ABSTAINED: "Not enough evidence" }[data.answer_status];
    get("answer-badge").dataset.status = data.answer_status;
    const method = data.retrieval_backend === "aws_knowledge_mcp" ? "AWS official live search · saved excerpts" : data.mode_used === "hybrid" ? "Pilot hybrid search" : "Pilot semantic search";
    get("answer-meta").textContent = `${method} · ${data.answer_language === "vi" ? "Tiếng Việt" : "English"} · Saved ${new Date(data.created_at).toLocaleString()}`;
    if (data.answer_status === "PARTIAL") get("answer").append(node("p", data.answer_language === "vi" ? "Các nguồn tìm được chưa đủ để trả lời đầy đủ mọi phần của câu hỏi." : "Some parts of this question are not fully supported by the available sources.", "notice"));
    if (data.answer_status === "ABSTAINED") get("answer").append(node("p", data.answer, "notice"));
    for (const [index, claim] of data.claims.entries()) {
      const item = node("article", undefined, "claim"); item.dataset.claimId = claim.claim_id;
      const button = node("button", undefined, "claim-select"); button.type = "button"; button.setAttribute("aria-pressed", "false");
      button.append(node("span", `Statement ${index + 1} · ${claim.citation_ids.length} cited source${claim.citation_ids.length > 1 ? "s" : ""}`, "claim-label"), node("span", claim.text));
      button.addEventListener("click", () => evidence.selectClaim(claim.claim_id)); item.append(button);
      for (const id of claim.citation_ids) {
        const citation = data.citations.find(item => item.citation_id === id), reference = node("button", `[${data.citations.indexOf(citation) + 1}] Read source`, "reference");
        reference.type = "button"; reference.dataset.citationId = id; reference.title = citation.title;
        reference.addEventListener("click", () => { evidence.selectClaim(claim.claim_id); selectCitation(id); }); item.append(reference);
      }
      get("answer").append(item);
      const mapRow = node("li"), mapButton = node("button", `Statement ${index + 1}`, "map-claim"); mapButton.type = "button";
      mapButton.addEventListener("click", () => evidence.selectClaim(claim.claim_id));
      mapRow.append(mapButton, document.createTextNode(` → cites ${claim.citation_ids.map(id => `[${data.citations.findIndex(c => c.citation_id === id) + 1}]`).join(", ")}`)); get("evidence-links").append(mapRow);
    }
    const explanations = data.answer_language === "vi" ? {
      LIVE_SEARCH_NOT_PILOT_BENCHMARK: "Đây là tìm kiếm trực tiếp, tách biệt với benchmark của corpus pilot.",
      AWS_OFFICIAL_LIVE_SEARCH_EXCERPTS: "Bằng chứng là các đoạn trích từ tìm kiếm tài liệu AWS chính thức, đã lưu theo câu trả lời này.",
      FULL_AWS_CORPUS_NOT_DOWNLOADED: "Hệ thống chưa tải hoặc lập chỉ mục toàn bộ tài liệu AWS."
    } : {
      LIVE_SEARCH_NOT_PILOT_BENCHMARK: "Live search is separate from the recorded pilot benchmark.",
      AWS_OFFICIAL_LIVE_SEARCH_EXCERPTS: "Evidence uses official AWS search excerpts saved with this answer.",
      FULL_AWS_CORPUS_NOT_DOWNLOADED: "The full AWS documentation library has not been downloaded or indexed."
    };
    for (const warning of data.warnings) {
      if (!warning.startsWith("GENERATION_REASON_")) get("answer").append(node("p", explanations[warning] || warning, "muted"));
    }
    get("evidence-map").hidden = !data.claims.length; get("brief-panel").hidden = false; get("export-brief").disabled = busy;
    renderEvidence(evidence.getState());
  }
  async function openRecord(id) {
    if (!active || busy || !mayReplace()) return;
    const captured = session; resetAnswer(); const revision = view;
    try { const data = await readRequest(queryPath(id)); if (current(captured) && view === revision) renderRecord(data); }
    catch (error) { if (current(captured) && view === revision) showError(error); }
  }
  async function loadHistory(append = false) {
    if (!active || busy) return;
    const captured = session, sequence = ++historySequence;
    get("refresh-history").disabled = true; get("more-history").disabled = true;
    try {
      const data = await readRequest("/v1/queries?limit=10" + (append && cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""));
      if (!current(captured) || sequence !== historySequence) return;
      if (!Array.isArray(data.items)) throw new EvidenceFailure();
      if (!append) dashboard.setHistory(data.items);
      if (!append) get("history").replaceChildren();
      if (!append && !data.items.length) get("history").append(node("p", "Your investigations will appear here.", "muted"));
      for (const item of data.items) {
        queryPath(item.request_id);
        const button = node("button", undefined, "history-item"); button.type = "button";
        button.append(node("span", item.question), node("small", `${new Date(item.created_at).toLocaleString()} · ${String(item.status).toLowerCase()}`));
        button.addEventListener("click", () => openRecord(item.request_id)); get("history").append(button);
      }
      cursor = data.next_cursor || null; get("more-history").hidden = !cursor;
    } catch (error) { if (current(captured) && sequence === historySequence) showError(error); }
    finally { if (current(captured) && sequence === historySequence) { get("refresh-history").disabled = false; get("more-history").disabled = false; } }
  }
  function chooseScenario(id) {
    if (!active || busy || !mayReplace()) return;
    const scenario = SCENARIOS.find(item => item.id === id); if (!scenario) return;
    dashboard.show("investigate");
    scenarioId = id; resetChecks(); get("brief-notes").value = "";
    get("scenario-library").open = true;
    for (const card of document.querySelectorAll("[data-scenario-id]")) card.setAttribute("aria-pressed", String(card.dataset.scenarioId === id));
    get("scenario-fields").replaceChildren(); get("scenario-title").textContent = scenario.title; get("scenario-status").textContent = "";
    for (const field of scenario.fields) {
      const wrapper = node("div", undefined, field.options ? "" : "text-field"), label = node("label", field.label); label.htmlFor = `scenario-${field.id}`;
      const input = node(field.options ? "select" : "textarea"); input.id = `scenario-${field.id}`; input.dataset.fieldId = field.id;
      if (field.options) for (const [value, title] of field.options) { const option = node("option", title); option.value = value; input.append(option); }
      else { input.rows = 2; input.maxLength = field.maxLength; input.required = !!field.required; }
      wrapper.append(label, input); get("scenario-fields").append(wrapper);
    }
    get("scenario-detail").hidden = false;
  }
  for (const [index, scenario] of SCENARIOS.entries()) {
    const button = node("button", undefined, "scenario-card"); button.type = "button"; button.dataset.scenarioId = scenario.id; button.setAttribute("aria-pressed", "false");
    button.append(node("span", `0${index + 1}`, "card-number"), node("strong", scenario.title), node("span", scenario.description, "card-description"), node("span", scenario.tag, "card-tag"));
    button.addEventListener("click", () => chooseScenario(scenario.id)); get("scenario-cards").append(button);
  }
  for (const step of INVESTIGATION_STEPS) {
    const item = node("div", undefined, "check-step"), label = node("label"), input = node("input"); input.type = "checkbox";
    label.append(input, node("span", `${step.title} · reviewed by me`)); item.append(label, node("p", step.question));
    const link = sourceLink(step.url, "Read the AWS guidance ↗"); if (link) item.append(link); get("investigation-checklist").append(item);
  }
  get("build-question").addEventListener("click", () => {
    if (!active || busy) return;
    const values = Object.fromEntries([...document.querySelectorAll("[data-field-id]")].map(input => [input.dataset.fieldId, input.value]));
    try {
      const question = buildQuestion(scenarioId, values);
      if (get("question").value.trim() && !confirmReplace("Replace the text in the question editor with this scenario draft?")) return;
      get("question").value = question; get("question").focus(); get("scenario-status").textContent = "Draft ready. Edit it, then choose Ask a question. Nothing has been submitted.";
    } catch (error) { get("scenario-status").textContent = error.message; }
  });
  get("question-form").addEventListener("submit", async event => {
    event.preventDefault(); if (!active || busy || !mayReplace()) return;
    const body = { question: get("question").value.trim(), mode: get("mode").value,
      corpus_scope: get("corpus-scope").value, answer_language: get("answer-language").value };
    if (!body.question || body.question.length > 2000 || !["hybrid", "vector"].includes(body.mode)) return;
    const fingerprint = JSON.stringify(body);
    if (pending?.fingerprint === fingerprint) { get("query-status").textContent = "This question was already submitted. Check history, or choose New question to start another request."; return; }
    const captured = session; pending = { fingerprint, key: uuid() }; resetAnswer(); const revision = view; setBusy(true);
    get("query-status").textContent = "Searching the documentation and checking citations… The first question may take longer.";
    try {
      const data = await request("/v1/queries", { body, key: pending.key });
      if (!current(captured) || view !== revision) return;
      renderRecord(data); get("query-status").textContent = data.status === "COMPLETED" ? "Answer saved to your history. Select a statement to investigate its evidence." : "Check your request's final status in history.";
    } catch (error) { if (current(captured) && view === revision) showError(error); }
    finally { if (current(captured)) { setBusy(false); await loadHistory(); } }
  });
  get("new-question").addEventListener("click", () => {
    if (!active || busy || !mayReplace()) return;
    pending = null; resetAnswer(); get("question").value = ""; get("question").focus(); get("query-status").textContent = "Ready for a new question.";
  });
  get("export-brief").addEventListener("click", async () => {
    if (!active || busy || exporting || record?.status !== "COMPLETED") return;
    const captured = session, revision = view, sequence = ++exportSequence, id = record.request_id, snapshot = record.snapshot_version, notes = get("brief-notes").value;
    const valid = () => current(captured) && view === revision && sequence === exportSequence;
    exporting = true; onboarding.syncAvailability(); get("export-brief").disabled = true; get("export-status").textContent = "Checking current access to this saved answer…";
    try {
      const fresh = await request(queryPath(id));
      if (!valid()) return;
      if (fresh.request_id !== id || fresh.snapshot_version !== snapshot) throw new EvidenceFailure();
      const markdown = serializeBrief(fresh, notes);
      if (!valid()) return;
      downloader.download(markdown, briefFilename(fresh)); get("export-status").textContent = "Brief downloaded. Personal notes are separate from model evidence.";
    } catch (error) {
      if (!valid()) return;
      if (error?.status === 401 || error?.status === 403) showError(error);
      else get("export-status").textContent = "The saved answer could not be verified for export. No cached file was downloaded.";
    } finally { if (valid()) { exporting = false; get("export-brief").disabled = false; onboarding.syncAvailability(); } }
  });
  get("refresh-history").addEventListener("click", () => loadHistory());
  get("more-history").addEventListener("click", () => loadHistory(true));
  clear();
  return { clear, renderRecord, openRecord, loadHistory, chooseScenario, evidence, dashboard, onboarding,
    activate: async ({ load = true } = {}) => { clear(); active = true; get("workspace").hidden = false; document.body.dataset.workspaceActive = "true";
      const captured = session, ready = dashboard.activate(); if (load) await loadHistory(); await ready;
      if (current(captured)) onboarding.activate(); } };
}

export async function startApp({ document = globalThis.document, location = globalThis.location,
  history = globalThis.history, storage = globalThis.sessionStorage, fetcher = globalThis.fetch,
  request = apiRequest, makeWorkspace = createWorkspace, registrationRequest = registerWithCognito } = {}) {
  const get = id => document.getElementById(id);
  let accessToken = null, config = null, checking = false, departing = false, revision = 0;
  function controls({ signedIn = false, pending = false, busy = false } = {}) {
    get("login").hidden = signedIn; get("login").disabled = busy || !config;
    get("signup").hidden = signedIn || config?.registration_enabled !== true; get("signup").disabled = busy || !config;
    get("forgot-password").hidden = signedIn || !config; get("forgot-password").disabled = busy || !config;
    get("logout").hidden = !signedIn;
    get("check-access").hidden = !signedIn || !pending; get("check-access").disabled = busy;
    get("account-security").hidden = signedIn;
  }
  const workspace = makeWorkspace({ document, request: (path, options) => request(path, accessToken, options),
    onSessionExpired: error => { accessToken = null; revision++; controls(); get("status").textContent = errorMessage(error); } });
  async function loadAccess() {
    if (!accessToken || checking || departing) return;
    checking = true; const captured = revision;
    controls({ signedIn: true, pending: true, busy: true });
    get("status").textContent = "Checking your account and workspace… / Đang kiểm tra tài khoản…";
    try {
      const state = await request("/v1/status", accessToken);
      if (departing || revision !== captured) return;
      controls({ signedIn: true });
      get("status").textContent = `${state.message} ${state.quota_remaining} demo requests remain this month.`;
      if (state.model_enabled) await workspace.activate();
    } catch (error) {
      if (departing || revision !== captured) return;
      workspace.clear();
      if (error?.status === 401) { accessToken = null; revision++; controls(); }
      else controls({ signedIn: true, pending: true });
      get("status").textContent = errorMessage(error);
    } finally { checking = false; }
  }
  try {
    const callbackURL = location.pathname === "/auth/callback" && location.search ? location.href : null;
    if (callbackURL) history.replaceState(null, "", "/auth/callback");
    const response = await fetcher("/config.json", { cache: "no-store", credentials: "omit", redirect: "error" });
    if (!response.ok) throw new Error("CONFIG_UNAVAILABLE");
    config = validateConfig(await response.json(), location.origin); controls();
    get("account-help").textContent = "Đăng ký gồm username, email/Gmail và mật khẩu riêng của website → xác minh email → đăng nhập bằng username. Tài khoản đã tạo trước đây giữ tên đăng nhập cũ (kể cả email nếu đã dùng email làm username).";
    const registration = createRegistrationDialog({ document, config, onSignIn: () => get("login").click(), register: registrationRequest });
    get("signup").addEventListener("click", () => { if (!departing && !checking) registration.open(); });
    if (!config.registration_enabled) get("account-help").textContent = "Registration is not enabled in this deployment. Existing reviewer accounts can still sign in. / Bản này chưa mở đăng ký.";
    for (const [id, intent] of [["login", "signin"], ["forgot-password", "recovery"]]) {
      get(id).addEventListener("click", async () => {
        if (departing || checking) return;
        departing = true; controls({ busy: true });
        get("status").textContent = "Opening secure account steps… / Đang mở trang tài khoản bảo mật…";
        try { location.assign(await beginLogin(config, storage, globalThis.crypto, Date.now(), intent)); }
        catch { departing = false; controls(); get("status").textContent = "Could not open account steps. Please try again. / Chưa thể mở trang tài khoản; hãy thử lại."; }
      });
    }
    get("check-access").addEventListener("click", loadAccess);
    get("logout").addEventListener("click", () => {
      departing = true; revision++; accessToken = null; workspace.clear(); storage.removeItem("kg.oauth");
      const url = new URL("/logout", config.auth_domain);
      url.search = new URLSearchParams({ client_id: config.client_id, logout_uri: config.logout_url }).toString(); location.assign(url.href);
    });
    if (!callbackURL) { get("status").textContent = config.registration_enabled
      ? "Ready to sign in. New here? Create an account. / Đăng nhập hoặc tạo tài khoản mới."
      : "Ready. Sign in with your existing account. / Đăng nhập bằng tài khoản hiện có."; return; }
    accessToken = await finishLogin(config, callbackURL, storage, fetcher);
    await loadAccess();
  } catch (error) {
    accessToken = null; workspace.clear(); controls();
    get("status").textContent = error instanceof ApiFailure ? errorMessage(error) : "Sign-in could not be completed. Please try again.";
  }
}
