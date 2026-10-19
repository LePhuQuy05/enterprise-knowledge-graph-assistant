import { officialSourceLink } from "./client.mjs";

export const METHODS = Object.freeze({ B1_VECTOR: "Vector", B2_HYBRID: "Hybrid", G1_DRAFT: "Graph 1-hop · experimental", G2_DRAFT: "Graph 2-hop · experimental" });
const stages = ["corpus", "candidates", "top_k", "context"];
const fail = () => { throw new Error("DASHBOARD_DATA_INVALID"); };
const text = (value, max = 5000) => typeof value === "string" && value.length > 0 && value.length <= max;
const unique = values => new Set(values).size === values.length;

export function validateDashboard(value) {
  if (value?.schema_version !== 1 || value.scope !== "PUBLIC_RECORDED_DEVELOPMENT" ||
      !Array.isArray(value.documents) || !value.documents.length || value.documents.length > 100 ||
      !Array.isArray(value.chunks) || value.chunks.length > 5000 || !Array.isArray(value.cases) || !value.cases.length || value.cases.length > 50 ||
      value.snapshot?.documents !== value.documents.length || value.snapshot?.chunks !== value.chunks.length ||
      !/^snap_[a-f0-9]{64}$/.test(value.snapshot?.id) || !text(value.run?.id, 100) || value.run.k !== 5 ||
      value.run.graph_serving !== false || value.run.label_status !== "AI_DRAFT" || value.run.generation_calls !== 0) fail();
  const docs = new Set();
  for (const d of value.documents) {
    if (!/^docv_[a-f0-9]{64}$/.test(d.id) || docs.has(d.id) || !text(d.title, 1000) ||
        !text(d.service, 50) || !officialSourceLink(d.url) || !Number.isInteger(d.chunks) || d.chunks < 0) fail();
    docs.add(d.id);
  }
  if (value.documents.reduce((n, d) => n + d.chunks, 0) !== value.chunks.length) fail();
  const ids = [];
  for (const c of value.chunks) {
    if (!/^chk_[a-f0-9]{64}$/.test(c.id) || !docs.has(c.document) || !text(c.preview, 280) ||
        !Array.isArray(c.section) || c.section.some(s => !text(s, 1000)) || typeof c.truncated !== "boolean") fail();
    ids.push(c.id);
  }
  if (!unique(ids) || !unique(value.cases.map(c => c.id))) fail();
  for (const c of value.cases) {
    if (!text(c.id, 100) || !text(c.question, 2000) || c.review_status !== "AI_DRAFT" || typeof c.answerable !== "boolean" ||
        !Array.isArray(c.groups) || c.groups.length > 12 || c.answerable !== !!c.groups.length) fail();
    for (const g of c.groups) {
      if (!text(g.id, 100) || !text(g.fact) || !Array.isArray(g.alternatives) || !g.alternatives.length ||
          g.alternatives.some(a => !docs.has(a.document) || !text(a.quote, 4000))) fail();
    }
    for (const method of Object.keys(METHODS)) {
      const m = c.methods?.[method];
      if (!m || !Array.isArray(m.ranked) || !Array.isArray(m.context) || m.ranked.length > 100 || m.context.length > 8 ||
          !unique(m.ranked) || !unique(m.context) || [...m.ranked, ...m.context].some(n => !Number.isInteger(n) || !value.chunks[n]) ||
          !Number.isInteger(m.context_proxy_tokens) || m.context_proxy_tokens < 0 || typeof m.used_graph !== "boolean" ||
          !Array.isArray(m.warnings) || m.warnings.some(s => !text(s, 100))) fail();
      for (const stage of stages) {
        const cells = m.coverage?.[stage];
        if (c.answerable ? !Array.isArray(cells) || cells.length !== c.groups.length || cells.some(v => typeof v !== "boolean") : cells !== null) fail();
      }
      if (c.answerable ? m.complete_at_k !== Number(m.coverage.top_k.every(Boolean)) || m.context_complete !== Number(m.coverage.context.every(Boolean)) : m.complete_at_k !== null || m.context_complete !== null) fail();
    }
  }
  for (const method of Object.keys(METHODS)) {
    const summary = value.summaries?.[method];
    const answerable = value.cases.filter(c => c.answerable);
    const wins = answerable.filter(c => c.methods[method].complete_at_k === 1).length;
    if (!summary || summary.draft.answerable_denominator !== answerable.length ||
        summary.draft.complete_evidence_at_k !== (answerable.length ? wins / answerable.length : null) ||
        summary.approved.answerable_denominator !== 0 || summary.approved.complete_evidence_at_k !== null) fail();
  }
  return value;
}

export async function loadDashboardData(fetcher = fetch) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetcher("/assets/dashboard-data.json", { cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal });
    if (!response.ok) throw new Error("DASHBOARD_UNAVAILABLE");
    const raw = await response.text(); if (raw.length > 400000) fail();
    return validateDashboard(JSON.parse(raw));
  } finally { clearTimeout(timer); }
}

export function createDashboard({ document, loadData = loadDashboardData, onQuestion, onOpenRecord, onScenario }) {
  const get = id => document.getElementById(id);
  const el = (tag, value, className) => { const n = document.createElement(tag); if (value !== undefined) n.textContent = value; if (className) n.className = className; return n; };
  const button = (label, action, style = "secondary small") => { const n = el("button", label, style); n.type = "button"; n.addEventListener("click", action); return n; };
  const link = (url, title = "Read official AWS page ↗") => { const n = el("a", title); n.href = officialSourceLink(url); n.target = "_blank"; n.rel = "noopener noreferrer"; return n; };
  let active = false, epoch = 0, data = null, page = "overview", questionId = "dev-11", selectedMethod = "B1_VECTOR";
  function show(next, focus = false) {
    if (!active || !["overview", "investigate", "knowledge", "lab"].includes(next)) return;
    page = next;
    get("current-view").textContent = ({ overview: "Overview", investigate: "Investigate", knowledge: "Knowledge Explorer", lab: "Retrieval Lab" })[page];
    for (const view of document.querySelectorAll("[data-view]")) view.hidden = view.dataset.view !== page;
    for (const nav of document.querySelectorAll("[data-page]")) { if (nav.dataset.page === page) nav.setAttribute("aria-current", "page"); else nav.removeAttribute("aria-current"); }
    if (focus) {
      const heading = get(({ overview: "overview-title", investigate: "investigation-title", knowledge: "knowledge-title", lab: "lab-title" })[page]);
      heading.tabIndex = -1; heading.focus({ preventScroll: true }); heading.scrollIntoView({ behavior: "auto", block: "start" });
    }
  }
  for (const nav of document.querySelectorAll("[data-page]")) nav.addEventListener("click", () => show(nav.dataset.page, true));
  get("overview-investigate").addEventListener("click", () => { if (active) { show("investigate"); onScenario("s3-lambda-read"); } });
  get("overview-knowledge").addEventListener("click", () => show("knowledge", true));
  for (const b of document.querySelectorAll("[data-lab-case]")) b.addEventListener("click", () => { if (active && data) { questionId = b.dataset.labCase; get("lab-question").value = questionId; renderLab(); show("lab", true); } });
  function metric(title, value, note) { const n = el("article", undefined, "metric-card"); n.append(el("p", title, "eyebrow"), el("strong", value), el("p", note, "muted")); return n; }
  function renderOverview() {
    get("overview-metrics").replaceChildren(
      metric("Indexed corpus", `${data.snapshot.documents} documents`, `${data.snapshot.chunks} chunks · recorded snapshot`),
      metric("Serving retrieval", "Hybrid + Vector", "Graph answering is not enabled"),
      metric("Development dataset", `${data.cases.length} questions`, "AI_DRAFT · not an independent holdout"),
      metric("Human-approved quality", "Not measured", "0 approved labels · no accuracy claim"));
    const release = get("overview-release"); release.replaceChildren();
    release.append(el("p", `${data.snapshot.embedding_model} · ${data.snapshot.dimension} dimensions`, "muted"),
      el("p", `Corpus snapshot created ${new Date(data.snapshot.created_at).toLocaleDateString()}. Fetch time is not the AWS page's update date.`, "muted"),
      el("code", data.snapshot.id, "identity"), el("p", "DeepSeek generates answers. This dashboard's lab reads a frozen retrieval run; it does not generate answers.", "muted"));
  }
  function renderCatalog() {
    if (!data || !active) return;
    const needle = get("catalog-search").value.toLowerCase().trim(), service = get("catalog-service").value;
    const docs = data.documents.filter(d => (service === "all" || service === d.service) && `${d.title} ${d.service} ${d.url}`.toLowerCase().includes(needle));
    get("catalog-count").textContent = `${docs.length} of ${data.documents.length} indexed documents · catalog filter only`;
    const list = get("catalog-list"); list.replaceChildren();
    if (!docs.length) list.append(el("p", "No indexed documents match. Try a broader term.", "muted"));
    for (const d of docs) {
      const b = button("", () => selectDocument(d), "document-card"); b.dataset.documentId = d.id; b.setAttribute("aria-pressed", "false");
      b.append(el("span", d.service, "eyebrow"), el("strong", d.title), el("span", `${d.chunks} chunks · open details →`, "muted")); list.append(b);
    }
    get("catalog-detail").replaceChildren(el("h3", "Select a document."), el("p", "Choose a result to see its version and indexed sections.", "muted"));
  }
  function selectDocument(d) {
    if (!active || !data) return;
    for (const row of document.querySelectorAll("[data-document-id]")) row.setAttribute("aria-pressed", String(row.dataset.documentId === d.id));
    const n = get("catalog-detail"); n.replaceChildren(el("p", d.service, "eyebrow"), el("h3", d.title),
      el("p", `${d.chunks} chunks · fetched ${new Date(d.fetched_at).toLocaleDateString()}. Fetch date is not last-modified date.`, "muted"), link(d.url), el("code", d.id, "identity"));
    n.append(el("h4", "Indexed sections")); const list = el("ul", undefined, "section-list");
    const sections = [...new Set(data.chunks.filter(c => c.document === d.id).map(c => c.section.join(" › ")))];
    for (const section of sections.slice(0, 18)) list.append(el("li", section));
    if (sections.length > 18) list.append(el("li", `${sections.length - 18} more indexed sections`));
    n.append(list, button("Draft a question about this document →", () => onQuestion(`What does the indexed AWS documentation explain about ${d.title}?`)));
    n.tabIndex = -1; n.focus({ preventScroll: true });
    if (document.defaultView.matchMedia("(max-width: 700px)").matches) n.scrollIntoView({ behavior: "auto", block: "start" });
  }
  function selectGroup(c, index) {
    if (!active || !data) return;
    const g = c.groups[index], n = get("radar-detail"); n.replaceChildren(el("p", "Annotated required fact · AI_DRAFT", "eyebrow"), el("h4", g.fact));
    for (const a of g.alternatives) {
      const d = data.documents.find(d => d.id === a.document);
      n.append(el("blockquote", a.quote), el("p", `Source characters ${a.start_char}–${a.end_char} · ${d.title}`, "muted"), link(d.url));
    }
    n.append(el("p", "A stored annotation is a review target, not a human-approved fact. Coverage tests exact source ranges and gap-free unions, not keywords.", "muted"));
  }
  function inspectChunk(index) {
    const c = data.chunks[index], d = data.documents.find(d => d.id === c.document), n = get("lab-source");
    n.replaceChildren(el("h4", d.title), el("p", c.section.join(" › "), "muted"), el("blockquote", c.preview),
      el("p", c.truncated ? "Truncated preview · open the official page for full context." : "Complete stored chunk text · recorded experiment.", "muted"),
      link(d.url), el("code", c.id, "identity"));
  }
  function renderLab() {
    if (!active || !data) return;
    const c = data.cases.find(c => c.id === questionId) || data.cases[0]; questionId = c.id; get("lab-question").value = c.id;
    get("lab-case-meta").textContent = `${c.id} · ${c.category.replaceAll("_", " ")} · ${c.review_status} · ${c.groups.length} required evidence groups`;
    get("lab-question-text").textContent = c.question;
    get("lab-source").replaceChildren(el("p", "Choose a ranked chunk to inspect a recorded preview. No source API or model call is made.", "muted"));
    const radar = get("radar"); radar.replaceChildren(); get("radar-detail").replaceChildren();
    if (!c.answerable) radar.append(el("p", `Corpus-unanswerable: ${c.unanswerable_reason} Retrieval coverage is not scored for this item.`, "notice"));
    else {
      const table = el("table"); table.append(el("caption", "Evidence coverage for Hybrid · AI_DRAFT source annotations"));
      const head = el("thead"), h = el("tr");
      for (const title of ["Required evidence", "In corpus", "In candidates", "In top 5", "In context"]) { const cell = el("th", title); cell.scope = "col"; h.append(cell); } head.append(h); table.append(head);
      const body = el("tbody"), coverage = c.methods.B2_HYBRID.coverage;
      for (const [i, g] of c.groups.entries()) {
        const row = el("tr"), name = el("th"); name.scope = "row"; name.append(button(g.id, () => selectGroup(c, i), "radar-group")); row.append(name);
        for (const stage of stages) { const present = coverage[stage][i], cell = el("td"); cell.append(el("span", present ? "✓ Present" : "− Missing", `coverage ${present ? "present" : "missing"}`)); row.append(cell); }
        body.append(row);
      }
      table.append(body); radar.append(table); selectGroup(c, 0);
    }
    const columns = get("lab-rankings"); columns.replaceChildren();
    for (const method of ["B2_HYBRID", selectedMethod]) {
      const m = c.methods[method], card = el("article", undefined, "dashboard-card ranking-card");
      card.append(el("p", method.startsWith("G") ? "Recorded · unapproved graph" : "Recorded · same frozen run", "eyebrow"), el("h3", METHODS[method]),
        el("p", `${m.context.length} context chunks · ${m.context_proxy_tokens} proxy tokens · graph contributed: ${m.used_graph ? "yes" : "no"}`, "muted"));
      if (c.answerable) card.append(el("p", `Top 5 complete: ${m.complete_at_k ? "yes" : "no"} · context complete: ${m.context_complete ? "yes" : "no"}`, "method-result"));
      if (m.warnings.length) card.append(el("p", m.warnings.join(" · "), "notice"));
      const list = el("ol", undefined, "ranking-list");
      for (const [rank, index] of m.ranked.slice(0, data.run.k).entries()) {
        const chunk = data.chunks[index], d = data.documents.find(d => d.id === chunk.document), item = el("li");
        const b = button("", () => inspectChunk(index), "ranked-chunk"); b.append(el("span", `#${rank + 1} · ${m.context.includes(index) ? "Included in context" : "Not in context"}`, "rank-label"), el("strong", d.title), el("span", chunk.section.at(-1) || d.service, "muted")); item.append(b); list.append(item);
      }
      const detail = el("details"), summary = el("summary", `Packed context · ${m.context.length} chunks`); detail.append(summary);
      for (const index of m.context) { const ch = data.chunks[index], d = data.documents.find(d => d.id === ch.document); detail.append(button(`${d.title} · ${ch.section.at(-1) || "section"}`, () => inspectChunk(index), "context-chunk")); }
      card.append(list, detail, el("p", "Ranks are recorded. A chunk's exclusion reason was not recorded; no cause is inferred from its absence.", "muted")); columns.append(card);
    }
  }
  function setHistory(items) {
    if (!active) return;
    const target = get("overview-history"); target.replaceChildren();
    if (!items.length) target.append(el("p", "No saved investigations on this page yet.", "muted"));
    for (const item of items.slice(0, 3)) target.append(button(item.question, () => onOpenRecord(item.request_id), "overview-history-item"));
  }
  get("catalog-search").addEventListener("input", renderCatalog); get("catalog-service").addEventListener("change", renderCatalog);
  get("lab-question").addEventListener("change", () => { questionId = get("lab-question").value; renderLab(); });
  get("lab-method").addEventListener("change", () => { selectedMethod = get("lab-method").value; renderLab(); });
  function clear() {
    active = false; epoch++; data = null; page = "overview"; questionId = "dev-11"; selectedMethod = "B1_VECTOR";
    for (const id of ["overview-history", "overview-metrics", "overview-release", "catalog-list", "catalog-detail", "lab-summary", "lab-question", "radar", "radar-detail", "lab-rankings", "lab-source"]) get(id).replaceChildren();
    for (const id of ["catalog-count", "lab-run-meta", "lab-case-meta", "lab-question-text", "dashboard-status"]) get(id).textContent = "";
    get("catalog-search").value = ""; get("catalog-service").replaceChildren(); get("lab-method").value = "B1_VECTOR";
  }
  async function activate() {
    clear(); active = true; const captured = epoch; show("overview");
    get("dashboard-status").textContent = "Loading recorded corpus and experiment…";
    try {
      const loaded = validateDashboard(await loadData()); if (!active || captured !== epoch) return;
      data = loaded; renderOverview();
      const all = el("option", "All services"); all.value = "all"; get("catalog-service").append(all);
      for (const service of [...new Set(data.documents.map(d => d.service))].sort()) { const o = el("option", service); o.value = service; get("catalog-service").append(o); }
      for (const c of data.cases) { const o = el("option", `${c.id} · ${c.question}`); o.value = c.id; get("lab-question").append(o); }
      get("lab-run-meta").textContent = `Run ${data.run.id} · ${new Date(data.run.completed_at).toLocaleString()} · top ${data.run.k} · same source vectors/backend · RRF ${data.run.rrf_constant} · context cap ${data.run.context_max_chunks} chunks / ${data.run.context_proxy_tokens} proxy tokens`;
      const answerable = data.cases.filter(c => c.answerable).length;
      for (const method of ["B1_VECTOR", "B2_HYBRID"]) {
        const count = data.cases.filter(c => c.answerable && c.methods[method].complete_at_k === 1).length;
        get("lab-summary").append(metric(`${METHODS[method]} · complete evidence @5`, `${count}/${answerable}`, "AI_DRAFT · retrieval coverage, not accuracy"));
      }
      get("lab-summary").append(metric("Human-approved metric", "Not measured", "0 approved cases · null score"));
      renderCatalog(); renderLab(); get("dashboard-status").textContent = "Recorded corpus and development run loaded. New answers require an explicit Ask.";
    } catch { if (active && captured === epoch) get("dashboard-status").textContent = "Recorded dashboard data is unavailable. Investigate and your own history remain available; no model request was made."; }
  }
  return { activate, clear, show, setHistory, getPage: () => page };
}
