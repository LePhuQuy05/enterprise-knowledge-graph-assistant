// V2 wire checks and evidence views. No inference, token storage or raw receipt reads.
import { officialSourceLink } from "./client.mjs";

export const SCOPED_METHODS = Object.freeze({ bm25: "BM25", vector: "Vector", hybrid: "Hybrid", hybrid_reranked: "Hybrid + reranker",
  graph_1hop: "Graph · 1 hop", graph_2hop: "Graph · 2 hops", hybrid_multi_query: "Multi-query", hybrid_multi_query_balanced: "Query-balanced" });
const runID = /^run_[a-f0-9]{64}$/, chunkID = /^chk_[a-f0-9]{64}$/, sha = /^[a-f0-9]{64}$/;
const terminals = ["COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED", "UNKNOWN"];
const text = (v, max = 20000) => typeof v === "string" && v.length > 0 && v.length <= max;
const count = (v, max = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(v) && v >= 0 && v <= max;
const method = v => Object.hasOwn(SCOPED_METHODS, v);
const unique = xs => new Set(xs).size === xs.length;
let sceneSerial = 0;
const fail = () => { throw new Error("INVALID_SCOPED_RESPONSE"); };
const closed = (value, keys) => value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).every(k => keys.split(" ").includes(k));
export function scopedRunPath(id) { if (!runID.test(id)) fail(); return `/v2/scoped-runs/${id}`; }

export function validateScopedCapabilities(c) {
  if (c?.schema_version !== 2 || c.available !== true || c.answering_enabled !== true || c.semantic_verified !== false || c.quality_metrics !== null ||
      c.maximum_methods !== 3 || c.maximum_answer_methods !== 3 || c.monthly_child_limit !== 300 ||
      !Array.isArray(c.methods) || !c.methods.length || c.methods.length > 8 || !unique(c.methods) || c.methods.some(m => !method(m)) ||
      !Array.isArray(c.answer_methods) || c.answer_methods.length !== c.methods.length || c.methods.some(m => !c.answer_methods.includes(m)) ||
      !Array.isArray(c.run_kinds) || c.run_kinds.length !== 1 || c.run_kinds[0] !== "answer" ||
      !count(c.corpus?.documents) || !c.corpus.documents || !count(c.corpus?.chunks) || !c.corpus.chunks ||
      !count(c.graph?.facts) || !count(c.graph?.source_documents, c.corpus.documents) || c.graph.experimental !== true || c.graph.human_reviewed !== false ||
      !/^collection_[a-f0-9]{64}$/.test(c.collection_version)) fail();
  return c;
}
export function validateScopedRun(r) {
  scopedRunPath(r?.run_id);
  if (r.schema_version !== 2 || r.workflow !== "SCOPED_V14_QUALIFIED_ANSWER" || r.quality_metrics !== null ||
      !/^collection_[a-f0-9]{64}$/.test(r.collection_version) || !["QUEUED", "RUNNING", ...terminals].includes(r.status) ||
      r.request?.run_kind !== "answer" || !text(r.request.original_question, 2000) ||
      !Array.isArray(r.methods) || !r.methods.length || r.methods.length > 3 || !unique(r.methods.map(m => m.method_id)) ||
      r.methods.some(m => !method(m.method_id) || !["QUEUED", "CLAIMED", "EXECUTING", ...terminals].includes(m.status) ||
        m.result_available !== (m.status === "COMPLETED")) ||
      !count(r.corpus?.documents) || !r.corpus.documents || !count(r.corpus?.chunks) || !r.corpus.chunks ||
      !count(r.corpus.graph_facts) || !count(r.corpus.graph_source_documents, r.corpus.documents)) fail();
  return r;
}

function sourceRef(s) {
  return s && chunkID.test(s.chunk_id) && text(s.source_id, 3) && /^s(?:[1-9]|1[0-2])$/.test(s.source_id) && text(s.title, 2000) &&
    text(s.document_version_id, 256) && text(s.shard_id, 256) && sha.test(s.shard_manifest_sha256) && sha.test(s.text_sha256) &&
    count(s.start_char) && count(s.end_char) && s.end_char > s.start_char &&
    Array.isArray(s.section_path) && s.section_path.length <= 32 && s.section_path.every(x => text(x, 2000)) && !!officialSourceLink(s.source_uri);
}
function validateAnswer(r) {
  const a = r.response;
  if (!a || a.request_id !== r.request_id || a.mode_requested !== r.method_id || a.mode_used !== r.method_id || a.used_graph !== r.used_graph ||
      a.answer_language !== r.answer_language || a.evidence_bundle_sha256 !== r.source_context_sha256 ||
      a.answer_status !== r.native_answer_status || !["ANSWERED", "PARTIAL", "ABSTAINED"].includes(a.answer_status) || !text(a.answer) ||
      !Array.isArray(a.claims) || a.claims.length > 8 || !unique(a.claims.map(c => c.claim_id)) ||
      !Array.isArray(a.citations) || a.citations.length > 64 || !unique(a.citations.map(c => c.citation_id))) fail();
  const used = new Set();
  for (const c of a.citations) {
    const s = r.context_sources.find(s => s.chunk_id === c.chunk_id);
    if (!/^profiled-cite-v12_[a-f0-9]{64}$/.test(c.citation_id) || !text(c.quote, 600) || !s ||
        ["document_version_id", "title", "source_uri"].some(k => c[k] !== s[k]) || JSON.stringify(c.section_path) !== JSON.stringify(s.section_path)) fail();
  }
  for (const c of a.claims) {
    if (!/^c[1-8]$/.test(c.claim_id) || !text(c.text) || !Array.isArray(c.citation_ids) || !c.citation_ids.length || c.citation_ids.length > 8 ||
        !unique(c.citation_ids) || c.citation_ids.some(id => !a.citations.some(x => x.citation_id === id))) fail();
    c.citation_ids.forEach(id => used.add(id));
  }
  if (used.size !== a.citations.length || (a.answer_status === "ABSTAINED" ? a.claims.length || a.citations.length :
      !a.claims.length || a.answer !== a.claims.map(c => c.text).join("\n"))) fail();
}

export function validateScopedResult(r, expectedMethod, expectedRun, expectedQuestionSha = null) {
  if (!closed(r, "schema_version status run_id request_id original_question_sha256 answer_language requested_method effective_method config config_sha256 collection_version ranked_chunk_ids context_chunk_ids context_proxy_tokens context_sources source_context_sha256 qualification_policy_sha256 qualification response native_answer_status review_flags used_graph answer_origin semantic_verified cache_hit new_cost_micro_usd original_receipt_cost_micro_usd usage trace retrieval graph_evidence method_id quality_metrics semantic_review") ||
      !closed(r.config, "schema_version implementation method_id collection_manifest_sha256 lexical_index_version document_recipe_sha256 query_recipe_sha256 reranker_recipe_sha256 graph_release_sha256 planner_recipe_sha256 generation_recipe source_profile fixture_only") ||
      !closed(r.config.generation_recipe, "model_metadata_sha256 output_tokens rules_sha256 source_protocol user_message_proxy_tokens version") ||
      !closed(r.config.source_profile, "max_chunks partial_source_policy span_boundary_min span_characters user_message_proxy_tokens version") ||
      !closed(r.retrieval, "mechanism_status queries query_rankings pool_fingerprint bundles balanced_allocator_used planner_origin planner_operation_id planner_receipt_sha256 hidden_fallbacks partial_source_truncations")) fail();
  if (!r || r.schema_version !== 2 || r.status !== "COMPLETED" || r.run_id !== expectedRun || !runID.test(r.run_id) || !method(expectedMethod) ||
      [r.method_id, r.requested_method, r.effective_method, r.config?.method_id].some(m => m !== expectedMethod) ||
      r.semantic_verified !== false || r.semantic_review !== "NOT_REVIEWED" || r.quality_metrics !== null ||
      !sha.test(r.config_sha256) || !sha.test(r.source_context_sha256) || !sha.test(r.qualification_policy_sha256) || !sha.test(r.original_question_sha256) ||
      expectedQuestionSha !== null && r.original_question_sha256 !== expectedQuestionSha ||
      r.config.schema_version !== 2 || typeof r.config.fixture_only !== "boolean" || r.config.generation_recipe.output_tokens !== 1600 || r.config.source_profile.max_chunks !== 12 ||
      !text(r.request_id, 256) || !["en", "vi"].includes(r.answer_language) || !count(r.new_cost_micro_usd) ||
      !count(r.context_proxy_tokens, 4096) || !Array.isArray(r.ranked_chunk_ids) || r.ranked_chunk_ids.length > 40 || !unique(r.ranked_chunk_ids) ||
      r.ranked_chunk_ids.some(id => !chunkID.test(id)) || !Array.isArray(r.context_chunk_ids) || r.context_chunk_ids.length > 12 || !unique(r.context_chunk_ids) ||
      !Array.isArray(r.context_sources) || r.context_sources.length !== r.context_chunk_ids.length ||
      r.context_sources.some((s, i) => !sourceRef(s) || !closed(s, "source_id chunk_id document_version_id shard_id shard_manifest_sha256 source_uri title section_path start_char end_char text_sha256") || s.source_id !== `s${i + 1}` || s.chunk_id !== r.context_chunk_ids[i] || !r.ranked_chunk_ids.includes(s.chunk_id)) ||
      ["owner_id", "tenant_id", "raw_output", "payload_json", "raw_receipt"].some(k => Object.hasOwn(r, k))) fail();
  const t = r.retrieval;
  const ready = t?.mechanism_status?.startsWith("READY") === true;
  if (!["NO_MECHANICAL_FLAGS", "REVIEW_REQUIRED", "NOT_READY"].includes(r.qualification) ||
      !Array.isArray(r.review_flags) || r.review_flags.length > 64 || (r.qualification === "REVIEW_REQUIRED") !== !!r.review_flags.length ||
      (r.qualification === "NOT_READY") === ready || (r.response !== null) !== (r.qualification === "NO_MECHANICAL_FLAGS") ||
      !["READY", "READY_NO_EVIDENCE_ABSTENTION", "NOT_READY_NO_DISTINCT_NATIVE_FACETS", "NOT_READY_GRAPH_ZERO_EXPANSION", "NOT_READY_GRAPH_PATH_NOT_PACKED"].includes(t?.mechanism_status) ||
      t.hidden_fallbacks !== 0 || t.partial_source_truncations !== 0 || !sha.test(t.pool_fingerprint) ||
      !Array.isArray(t.queries) || !t.queries.length || t.queries.length > 4 || t.queries.some(q => !text(q, 2000)) ||
      !Array.isArray(t.query_rankings) || t.query_rankings.length > 4 || t.query_rankings.some(xs => !Array.isArray(xs) || xs.length > 40 || !unique(xs) || xs.some(x => !chunkID.test(x))) ||
      t.balanced_allocator_used !== (expectedMethod === "hybrid_multi_query_balanced") || !Array.isArray(t.bundles) || t.bundles.length > 40 ||
      typeof r.used_graph !== "boolean" || typeof r.cache_hit !== "boolean" || r.cache_hit && r.new_cost_micro_usd !== 0) fail();
  for (const f of r.review_flags) if (!/^c[1-8]$/.test(f.claim_id) || !["INVOCATION_PREMISE_NOT_EXPLICIT_IN_CITATIONS",
    "ANTECEDENT_AVAILABLE_BUT_NOT_CITED", "MISSING_ANTECEDENT_IN_CONTEXT", "AMBIGUOUS_QUOTE_LOCATION_REQUIRES_REVIEW"].includes(f.rule_id) ||
    !Array.isArray(f.citation_ids) || f.citation_ids.length > 64 || f.citation_ids.some(id => !/^profiled-cite-v12_[a-f0-9]{64}$/.test(id)) ||
    Object.keys(f).some(k => !["claim_id", "rule_id", "citation_ids"].includes(k))) fail();
  const graph = expectedMethod.startsWith("graph_");
  const q = expectedMethod.startsWith("hybrid_multi_query");
  if (q && (!sha.test(t.planner_receipt_sha256) || !text(t.planner_operation_id, 256) ||
      !["NATIVE_V6", "FIXTURE"].includes(t.planner_origin) || !r.config.fixture_only && t.planner_origin !== "NATIVE_V6" || ready && t.queries.length < 2)) fail();
  const assertions = new Set();
  for (const b of t.bundles) {
    if (!text(b.bundle_id, 256) || !Array.isArray(b.chunk_ids) || !b.chunk_ids.length || b.chunk_ids.length > 12 || !unique(b.chunk_ids) ||
        b.chunk_ids.some(id => !r.ranked_chunk_ids.includes(id)) || !Array.isArray(b.assertion_ids) || b.assertion_ids.length > (expectedMethod === "graph_1hop" ? 1 : 2) ||
        !unique(b.assertion_ids) || !graph && b.assertion_ids.length || !["SELECTED", "MAX_CHUNKS", "TOKEN_BUDGET"].includes(b.selection)) fail();
    if (b.selection === "SELECTED") {
      if (b.chunk_ids.some(id => !r.context_chunk_ids.includes(id))) fail();
      b.assertion_ids.forEach(id => assertions.add(id));
    }
  }
  if (r.used_graph !== (assertions.size > 0) || graph && ready && !r.used_graph ||
      !Array.isArray(r.graph_evidence) || r.graph_evidence.length !== assertions.size || !unique(r.graph_evidence.map(f => f.assertion_id))) fail();
  for (const f of r.graph_evidence) {
    if (!assertions.has(f.assertion_id) || !r.context_chunk_ids.includes(f.chunk_id) || !text(f.clause) || !text(f.kind, 128) ||
        f.permission_granted !== false || f.interpretation !== "RETRIEVAL_SOURCE_ASSOCIATIONS_NOT_ACCESS_GRANTS" ||
        typeof f.unresolved_references_present !== "boolean" || !Array.isArray(f.nodes) || f.nodes.length < 2 || f.nodes.length > 12 ||
        !unique(f.nodes.map(n => n.node_id)) || f.nodes.some(n => !text(n.node_id, 256) || !text(n.name, 2000) || !text(n.identity_scope, 256) || n.role !== null && n.role !== undefined && !text(n.role, 2000)) ||
        !Array.isArray(f.links) || !f.links.length || f.links.length > 132 || f.links.some(l => !Array.isArray(l) || l.length !== 3 ||
          !f.nodes.some(n => n.node_id === l[0]) || !f.nodes.some(n => n.node_id === l[1]) || !text(l[2], 256))) fail();
  }
  if (!ready && (r.answer_origin !== "NOT_READY" || r.native_answer_status !== null || r.usage !== null || r.new_cost_micro_usd !== 0)) fail();
  if (!Array.isArray(r.trace) || r.trace.length > 5 || r.trace.some(e => !["shared_search_translation", "shared_native_task_facets", "retrieval_and_atomic_source_packing",
      "generation_and_display_qualification", "mechanism_not_ready_no_dispatch"].includes(e.stage) || e.status !== "COMPLETED" ||
      !Number.isFinite(e.duration_ms) || e.duration_ms < 0 || !sha.test(e.input_sha256) || !count(e.output_count))) fail();
  if (r.trace.at(-1)?.stage !== (ready ? "generation_and_display_qualification" : "mechanism_not_ready_no_dispatch") ||
      !ready && r.trace.some(e => e.stage === "generation_and_display_qualification") ||
      q && ready && t.query_rankings.length !== t.queries.length) fail();
  if (r.response !== null) validateAnswer(r);
  return r;
}

export const qualificationMessage = q => ({
  NO_MECHANICAL_FLAGS: "Automated checks found no flags. Answer correctness has not been independently verified.",
  REVIEW_REQUIRED: "Answer withheld for source review. No claim text is published; inspect the retrieved evidence and flags.",
  NOT_READY: "This method could not prepare supported evidence. Generation was not run; no other method was substituted."
})[q];
function dom(document) {
  const el = (tag, value, cls) => { const n = document.createElement(tag); if (value !== undefined) n.textContent = value; if (cls) n.className = cls; return n; };
  const button = (label, action) => { const n = el("button", label, "secondary small"); n.type = "button"; n.addEventListener("click", action); return n; };
  return { el, button };
}
export function renderScopedResult(document, host, result, openSource) {
  const { el, button } = dom(document), r = result;
  const status = el("div", undefined, "scoped-qualification"); status.dataset.qualification = r.qualification;
  status.append(el("p", r.qualification.replaceAll("_", " "), "eyebrow"), el("p", qualificationMessage(r.qualification)));
  if (r.config.fixture_only) status.prepend(el("p", "FIXTURE PROVIDER · no live inference", "badge"));
  host.append(status);
  if (r.response) {
    host.append(el("p", r.response.answer_status, "badge"));
    for (const claim of r.response.claims) {
      const block = el("div", undefined, "collection-claim"); block.append(el("p", claim.text));
      for (const id of claim.citation_ids) {
        const i = r.response.citations.findIndex(c => c.citation_id === id), citation = r.response.citations[i];
        block.append(button(`[${i + 1}] Read source`, () => openSource(citation.chunk_id, citation)));
      }
      host.append(block);
    }
    if (r.response.answer_status === "ABSTAINED") host.append(el("p", r.response.answer));
    if (r.response.assessment?.limitations?.length) {
      const limits = el("details"); limits.append(el("summary", "Model-reported limitations · not verified absence of evidence"),
        el("pre", JSON.stringify(r.response.assessment.limitations, null, 2))); host.append(limits);
    }
  } else if (r.review_flags.length) {
    const list = el("ul"); for (const flag of r.review_flags) list.append(el("li", `${flag.claim_id} · ${flag.rule_id}`)); host.append(list);
  } else host.append(el("p", r.retrieval.mechanism_status, "muted"));
  host.append(el("p", `${r.ranked_chunk_ids.length} candidates · ${r.context_chunk_ids.length} whole source chunks · ${r.context_proxy_tokens} proxy tokens`, "muted"));
  const evidence = el("details"); evidence.append(el("summary", "Retrieved evidence · open original source passages"));
  const list = el("ol"); for (const s of r.context_sources) { const li = el("li"); li.append(button(s.title, () => openSource(s.chunk_id)), el("p", s.section_path.join(" / "), "muted")); list.append(li); }
  evidence.append(list); host.append(evidence);
  const provenance = el("details"); provenance.append(el("summary", "Execution identity and accounting"), el("pre", JSON.stringify({
    run_id: r.run_id, method_requested: r.requested_method, method_used: r.effective_method, config_sha256: r.config_sha256,
    context_sha256: r.source_context_sha256, planner_origin: r.retrieval.planner_origin, cache_hit: r.cache_hit,
    new_cost_micro_usd: r.new_cost_micro_usd, usage: r.usage, basis: "Published-rate upper bound, not invoice", semantic_verified: false
  }, null, 2))); host.append(provenance);
}

export function renderScopedGraph(document, host, r, openSource) {
  const { el, button } = dom(document); host.replaceChildren();
  const bundles = r.retrieval.bundles.filter(b => b.selection === "SELECTED" && b.assertion_ids.length);
  host.append(el("h3", "Selected graph evidence"), el("p", "Connections come from this run's packed source bundles. Arrows describe documented associations, not actual access grants.", "muted"));
  if (!bundles.length) { host.append(el("p", "No graph path was selected. No connection has been inferred.", "notice")); return; }
  const picker = el("div", undefined, "scoped-path-picker"), scene = el("div", undefined, "scoped-graph-scene"), factsArea = el("div", undefined, "scoped-facts");
  function draw(index) {
    [...picker.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
    const bundle = bundles[index], facts = bundle.assertion_ids.map(id => r.graph_evidence.find(f => f.assertion_id === id));
    const nodes = [...new Map(facts.flatMap(f => f.nodes).map(n => [n.node_id, n])).values()];
    scene.replaceChildren(); factsArea.replaceChildren();
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 740 420"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `Selected source bundle ${index + 1}: ${facts.map(f => f.clause).join("; ")}. Accessible source list follows.`);
    const make = (tag, attrs, value) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (value) n.textContent = value; return n; };
    const arrowId = `scoped-arrow-${++sceneSerial}`;
    const defs = make("defs", {}), marker = make("marker", { id: arrowId, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto" });
    marker.append(make("path", { d: "M0 0L10 5L0 10Z", fill: "currentColor" })); defs.append(marker); svg.append(defs);
    const positions = new Map(nodes.map((n, i) => [n.node_id, { x: 370 + 250 * Math.cos(2 * Math.PI * i / nodes.length), y: 210 + 140 * Math.sin(2 * Math.PI * i / nodes.length) }]));
    const associations = new Set();
    for (const fact of facts) for (const [left, right, kind] of fact.links) {
      const a = positions.get(left), b = positions.get(right), length = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      const association = /LOOKUP|ASSOCIATION|REQUIREMENT/.test(kind);
      if (association) { const key = JSON.stringify([left, right].sort()); if (associations.has(key)) continue; associations.add(key); }
      const edge = make("line", { x1: a.x + (b.x - a.x) * 25 / length, y1: a.y + (b.y - a.y) * 25 / length,
        x2: b.x - (b.x - a.x) * 30 / length, y2: b.y - (b.y - a.y) * 30 / length, class: `scoped-graph-link${association ? " scoped-association" : ""}` });
      if (!association) edge.setAttribute("marker-end", `url(#${arrowId})`);
      edge.append(make("title", {}, kind)); svg.append(edge);
    }
    for (const n of nodes) {
      const p = positions.get(n.node_id), group = make("g", {}); group.append(make("title", {}, `${n.name} · ${n.identity_scope}${n.role ? ` · ${n.role}` : ""}`), make("circle", { cx: p.x, cy: p.y, r: 21, class: "scoped-graph-node" }));
      group.append(make("text", { x: p.x, y: p.y + 41, "text-anchor": "middle" }, n.name.length > 28 ? n.name.slice(0, 25) + "…" : n.name)); svg.append(group);
    }
    scene.append(svg, el("p", `${facts.length} source-backed fact(s) · ${nodes.length} nodes · ${bundle.chunk_ids.length} whole supporting chunk(s)`, "muted"),
      el("p", "Solid arrows: documented direction. Dashed lines: source associations, not permissions. Exact links and directions remain in the source list.", "muted"));
    for (const f of facts) {
      const section = el("article", undefined, "scoped-fact"); section.append(el("p", f.kind.replaceAll("_", " "), "eyebrow"), el("blockquote", f.clause));
      if (f.unresolved_references_present) section.append(el("p", "Source contains unresolved references; do not treat the connection as a fully resolved permission rule.", "notice"));
      const list = el("ul"); for (const [left, right, kind] of f.links) list.append(el("li", `${nodes.find(n => n.node_id === left).name} → ${nodes.find(n => n.node_id === right).name} · ${kind.replaceAll("_", " ").toLowerCase()}`));
      const nodeList = el("ul", undefined, "scoped-node-list"); f.nodes.forEach(n => nodeList.append(el("li", `${n.name}${n.role ? ` · ${n.role.replaceAll("_", " ").toLowerCase()}` : ""}`)));
      const links = el("details"); links.append(el("summary", `Exact link types and directions (${f.links.length})`), list);
      section.append(nodeList, links, button("Read supporting source", () => openSource(f.chunk_id, { quote: f.clause })));
      factsArea.append(section);
    }
  }
  bundles.forEach((b, i) => picker.append(button(`Bundle ${i + 1} · ${b.assertion_ids.length} fact(s)`, () => draw(i))));
  host.append(picker, scene, factsArea); draw(0);
}

export function renderScopedSteps(document, host, r) {
  const { el } = dom(document); host.replaceChildren();
  const names = { shared_search_translation: "Translate the search query", shared_native_task_facets: "Extract task facets from the question",
    retrieval_and_atomic_source_packing: "Retrieve and pack whole source evidence", generation_and_display_qualification: "Resolve answer receipt and check display eligibility",
    mechanism_not_ready_no_dispatch: "Stop: evidence mechanism not ready" };
  const list = el("ol", undefined, "scoped-stage-list");
  for (const e of r.trace) { const item = el("li"); item.append(el("strong", names[e.stage]), el("p", `${(e.duration_ms / 1000).toFixed(3)} s · ${e.output_count} output item(s) · recorded execution`, "muted")); list.append(item); }
  host.append(list, el("h4", "Search inputs")); const queries = el("ol"); r.retrieval.queries.forEach(q => queries.append(el("li", q))); host.append(queries);
  if (r.effective_method.startsWith("hybrid_multi_query")) host.append(el("p", `Planner: ${r.retrieval.planner_origin}. Original search is retained. ${r.retrieval.balanced_allocator_used ? "Query-balanced scheduling was used." : "Rank fusion was used."}`, "muted"));
  if (r.retrieval.query_rankings.length) {
    const details = el("details"); details.append(el("summary", "Per-query candidate ranks"));
    r.retrieval.query_rankings.forEach((ranking, index) => { details.append(el("h4", r.retrieval.queries[index])); const list = el("ol");
      for (const id of ranking) { const source = r.context_sources.find(s => s.chunk_id === id); list.append(el("li", source?.title || id, source ? undefined : "identity")); } details.append(list); });
    host.append(details);
  } else host.append(el("p", "This receipt records the final pool and source packing. Per-branch raw scores are not exposed; no token-level or reranker score is inferred.", "muted"));
  const table = el("table", undefined, "scoped-bundles");
  const caption = el("caption", "Evidence packing decisions"), header = el("tr"); ["Bundle", "Whole chunks", "Graph facts", "Decision"].forEach(t => header.append(el("th", t)));
  const head = el("thead"), body = el("tbody"); head.append(header);
  for (const [i, b] of r.retrieval.bundles.entries()) { const row = el("tr"); [i + 1, b.chunk_ids.length, b.assertion_ids.length, b.selection].forEach(x => row.append(el("td", String(x)))); body.append(row); }
  table.append(caption, head, body); const wrap = el("div", undefined, "scoped-table-wrap"); wrap.append(table); host.append(wrap);
}
