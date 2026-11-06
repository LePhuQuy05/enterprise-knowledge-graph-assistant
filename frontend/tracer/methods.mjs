import { officialSourceLink } from "./client.mjs";

import { renderScopedResult, renderScopedGraph, renderScopedSteps } from "./scoped-runs.mjs";

export const METHOD_GUIDES = Object.freeze({
  bm25: { label: "BM25 · Keywords", short: "Exact terms, explainable matches.", mechanism: "BM25 tìm từ khóa trong inverted index. Từ hiếm được tăng trọng số; lặp từ nhiều có lợi ích giảm dần và độ dài đoạn được chuẩn hóa.", formula: "score(q,d) = Σ IDF(t) × tf(t,d) × (k₁+1) / [tf(t,d) + k₁ × (1−b+b×|d|/avgdl)]", why: "Hữu ích với tên dịch vụ, action IAM và mã lỗi chính xác. Không cần gọi embedding cho truy vấn.", limit: "Dễ bỏ sót cách diễn đạt khác từ khóa; tokenizer hiện tại và corpus tiếng Anh hạn chế câu hỏi tiếng Việt trực tiếp.", stages: ["Input", "Tokenize", "Lexical ranking", "Pack context", "Generate", "Verify sources"] },
  vector: { label: "Vector · Semantic", short: "Find passages with related meaning.", mechanism: "BGE-small-en-v1.5 mã hóa câu hỏi và đoạn tài liệu thành vector 384 chiều. Vector được chuẩn hóa; tích vô hướng xếp hạng độ tương đồng.", formula: "similarity(q,d) = q · d  (||q||₂ = ||d||₂ = 1)", why: "Tìm được nhiều cách diễn đạt cùng ý, kể cả khi không trùng từ. Model chạy local và cùng recipe với vector corpus.", limit: "Độ tương đồng không chứng minh đoạn trả lời đúng. BGE ở đây là model tiếng Anh; answering tiếng Việt cần bước chuyển truy vấn được ghi nhận.", stages: ["Input", "Embed query", "Vector ranking", "Pack context", "Generate", "Verify sources"] },
  hybrid: { label: "Hybrid · Rank fusion", short: "Join lexical precision and semantic recall.", mechanism: "Chạy BM25 và Vector song song về mặt logic, lấy tối đa 20 ứng viên mỗi nhánh rồi gộp thứ hạng bằng Reciprocal Rank Fusion. Không cộng trực tiếp hai thang điểm khác nhau.", formula: "RRF(d) = Σ branches 1 / (60 + rank_branch(d))", why: "Một nguồn có thể mạnh ở từ khóa, nguồn khác mạnh ở ý nghĩa. RRF cho cả hai nhánh đóng góp mà không cần hiệu chỉnh điểm cùng thang.", limit: "Nguồn xuất hiện ở cả hai nhánh dễ lên cao nhưng vẫn có thể sai ngữ cảnh. Một fusion rank tốt chưa bảo đảm đủ mọi khía cạnh câu hỏi.", stages: ["Input", "Two search branches", "RRF fusion", "Pack context", "Generate", "Verify sources"] },
  hybrid_reranked: { label: "Hybrid + Reranker", short: "Read the query and passage together.", mechanism: "Hybrid tạo pool có giới hạn. Cross-encoder/ms-marco-MiniLM-L2-v2 đọc cặp câu hỏi–đoạn, chia cửa sổ để không bỏ phần cuối, dùng max raw logit để xếp lại.", formula: "score(d) = max_window CrossEncoder(question, window(d))", why: "Cross-encoder xét tương tác chi tiết hơn hai vector độc lập. Chỉ đọc pool nhỏ để kiểm soát CPU và độ trễ.", limit: "Không cứu được nguồn không có trong pool. Logit không phải xác suất đúng; passage dài/nhiều cửa sổ có thể tạo distractor.", stages: ["Input", "Hybrid candidates", "Cross-encoder rerank", "Pack context", "Generate", "Verify sources"] },
  hybrid_multi_query: { label: "Multi-query RAG", short: "Search the separate parts of a question.", mechanism: "Planner tạo tối đa hai truy vấn bổ sung tiếng Anh. Mỗi truy vấn chạy Hybrid, rồi các rankings được gộp bằng RRF60 vào pool có giới hạn.", formula: "question → base + ≤2 subqueries → Hybrid per query → equal RRF60", why: "Câu hỏi nhiều vế có thể cần những nguồn khác nhau. Các truy vấn bổ sung giúp tìm từng vế thay vì dựa hoàn toàn vào một embedding.", limit: "Planner có thể lệch ý. Tìm thấy nguồn ở một nhánh chưa có nghĩa nguồn đó vượt fusion và được đóng gói vào context.", stages: ["Input", "Plan subqueries", "Search each query", "Fuse & pack", "Generate", "Verify sources"] },
  hybrid_multi_query_balanced: { label: "Query-balanced RAG", short: "Keep room for different evidence needs.", mechanism: "Dùng cùng họ truy vấn với Multi-query nhưng phân bổ lượt chọn context giữa các rankings. Ghi nhận chunk trùng, quá ngân sách và chunk thực sự được giữ.", formula: "subquery rankings → bounded balanced selection → ≤8 chunks / 4096 proxy tokens", why: "Pool đa dạng không đủ nếu top context bị một khía cạnh chiếm hết. Bộ chọn cân bằng cố gắng giữ chỗ cho nhiều hướng bằng chứng.", limit: "Cân bằng truy vấn không đồng nghĩa cân bằng sự thật. Một nhánh lệch ý vẫn có thể được cấp chỗ; phải đọc lại nguồn và câu trả lời.", stages: ["Input", "Shared query plan", "Per-query rankings", "Balanced context", "Generate", "Verify sources"] },
  graph_1hop: { label: "Graph-assisted · 1 hop", short: "Follow one source-backed connection.", mechanism: "Hybrid lấy seed chunks; mentions trong các chunks tạo seed entities. BFS duyệt quan hệ POSITIVE có nguồn; ba danh sách Vector, BM25 và Graph được gộp bằng RRF60. Mũi tên biểu thị nghĩa quan hệ; traversal hiện tại đọc adjacency ở cả hai đầu.", formula: "question → Hybrid seeds → mentions → entity — assertion → evidence chunk", why: "Bằng chứng liên quan có thể nằm ở một đoạn khác nhưng nối với entity đã tìm được. Quan hệ giữ quote, chiều, polarity và điều kiện để kiểm tra.", limit: "Graph là tập con thử nghiệm đã được AI review theo nguồn trên collection316, chưa có nhãn độc lập của con người. Ví dụ có câu trả lời local và citations; chạy câu hỏi Graph mới trên cloud chưa được bật. Khi không giữ được đường quan hệ có bằng chứng, executor trả đúng Hybrid fallback.", stages: ["Input", "Hybrid & mentions", "Traverse 1 hop", "Fuse & pack", "Generate & cite", "Inspect provenance"] },
  graph_2hop: { label: "Graph-assisted · 2 hops", short: "Explore a bounded two-edge path.", mechanism: "Cùng pipeline seed và provenance của 1-hop, nhưng cho phép mở rộng tối đa hai cạnh. Giới hạn visited entities/assertions/candidates và thời gian để tránh graph bùng nổ.", formula: "seed → assertion₁ → entity → assertion₂ → evidence (depth ≤ 2)", why: "Một khía cạnh có thể cần qua entity trung gian. Cho xem đường đi và các nguồn ở mỗi cạnh giúp người xem kiểm tra điều kiện thay vì tin vào hình nối.", limit: "Độ sâu cho phép là2 không có nghĩa lần chạy đã dùng đường2cạnh. Hai trang Graph dùng hai câu hỏi khác nhau để minh họa, không phải phép so chất lượng1-hop/2-hop. Không suy fact bắc cầu hoặc tạo thêm cạnh để làm đẹp hình. Cloud execution cho câu hỏi Graph mới còn khóa.", stages: ["Input", "Hybrid & mentions", "Traverse ≤2 hops", "Fuse & pack", "Generate & cite", "Inspect provenance"] },
});
const fail = () => { throw new Error("METHOD_WALKTHROUGH_INVALID"); };
const hash = s => typeof s === "string" && /^[a-f0-9]{64}$/.test(s);
export function validateWalkthroughs(data) {
  if (data?.schema_version !== 1 || data.scope !== "METHOD_WALKTHROUGHS_NOT_QUALITY_SCORES" || !Array.isArray(data.examples) || data.examples.length !== 6 || !data.sources || data.graph?.review !== "PENDING") fail();
  const graph = data.graph, ids = new Set(graph.nodes?.map(n => n.entity_id));
  if (ids.size !== graph.nodes.length || ids.size < 1 || ids.size > 200 || !Array.isArray(graph.edges) || graph.edges.length > 1000 || graph.documents !== 316 || graph.chunks !== 2978 || !hash(graph.graph_sha256) || !hash(graph.manifest_sha256) || graph.manifest_sha256 !== data.examples[0].config.collection_manifest_sha256 || graph.full_relation_extraction !== false) fail();
  if (graph.source_review !== "AI_PROVISIONAL_NOT_HUMAN_GOLD" || graph.human_reviewed !== false || graph.legacy_graph_reused !== false || !hash(graph.curation_sha256) || graph.selection?.selected !== graph.edges.length || ![graph.selection.rejected, graph.selection.pending].every(n => Number.isSafeInteger(n) && n >= 0)) fail();
  for (const s of Object.values(data.sources)) if (!officialSourceLink(s.url) || !hash(s.text_sha256) || typeof s.text !== "string" || s.text.length > 120000 || s.end_char - s.start_char !== [...s.text].length) fail();
  for (const [i, e] of data.examples.entries()) {
    if (e.id !== Object.keys(METHOD_GUIDES)[i] || e.kind !== "RECORDED_LOCAL_EXECUTION" || e.review !== "NOT_REVIEWED" || !hash(e.artifact_sha256) || !hash(e.config_sha256) || typeof e.question !== "string" || e.question.length > 2000 || e.context_chunk_ids?.length > 8 || !Array.isArray(e.ranked_chunk_ids) || e.ranked_chunk_ids.some(k => !data.sources[k]) || e.context_chunk_ids.some(k => !e.ranked_chunk_ids.includes(k))) fail();
    for (const c of e.response.citations) if (!data.sources[c.chunk_id]?.text.includes(c.quote) || data.sources[c.chunk_id].document_version_id !== c.document_version_id) fail();
    for (const c of e.response.claims) if (c.citation_ids.some(id => !e.response.citations.some(s => s.citation_id === id))) fail();
  }
  for (const e of graph.edges) {
    const s = data.sources[e.chunk_id];
    if (!ids.has(e.subject_id) || !ids.has(e.object_id) || !s || !["POSITIVE", "NEGATIVE"].includes(e.polarity) || [...s.text].slice(e.evidence.start_char - s.start_char, e.evidence.end_char - s.start_char).join("") !== e.evidence.quote || e.review_status !== "AI_SUPPORTED_FOR_EXPERIMENT" || typeof e.review_note_vi !== "string" || e.review_note_vi.length > 1200 || e.qualifiers.some(q => !e.evidence.quote.includes(q.text))) fail();
  }
  if (graph.modes?.length !== 2 || graph.modes.some((m, i) => m.requested_hops !== i + 1 || m.seed_entity_ids.some(id => !ids.has(id)) || m.graph_candidates.some(c => c.assertion_path.some(id => !graph.edges.some(e => e.assertion_id === id))))) fail();
  for (const [i, m] of graph.modes.entries()) {
    if (!Array.isArray(m.answer_review_notes_vi) || m.answer_review_notes_vi.length < 1 || m.answer_review_notes_vi.length > 4 || m.answer_review_notes_vi.some(n => typeof n !== "string" || !n.length || n.length > 1200)) fail();
    if (typeof m.question !== "string" || m.question.length > 2000 || !hash(m.artifact_sha256) || !hash(m.execution_identity_sha256) || !Array.isArray(m.visited_assertion_ids) || m.visited_assertion_ids.some(id => !graph.edges.some(e => e.assertion_id === id && e.polarity === "POSITIVE")) || m.context_chunk_ids.length > 8 || m.final_ranking.length > 40 || m.final_ranking.some(k => !data.sources[k]) || m.context_chunk_ids.some(k => !m.final_ranking.includes(k))) fail();
    if (m.used_graph ? m.mode_used !== `graph_${i + 1}hop` || !m.context_assertion_ids.length : m.mode_used !== "hybrid" || m.context_assertion_ids.length) fail();
    if (m.context_assertion_ids.some(id => !m.visited_assertion_ids.includes(id) || !m.context_chunk_ids.includes(graph.edges.find(e => e.assertion_id === id)?.chunk_id))) fail();
    const r = m.response;
    if (!r || !["ANSWERED", "PARTIAL", "ABSTAINED"].includes(r.answer_status) || !["en", "vi"].includes(r.answer_language) || !Array.isArray(r.claims) || r.claims.length > 4 || !Array.isArray(r.citations) || r.citations.length > 8) fail();
    for (const c of r.citations) if (!m.context_chunk_ids.includes(c.chunk_id) || !data.sources[c.chunk_id]?.text.includes(c.quote) || data.sources[c.chunk_id].document_version_id !== c.document_version_id) fail();
    const used = new Set(r.claims.flatMap(c => c.citation_ids));
    if (new Set(r.citations.map(c => c.citation_id)).size !== r.citations.length || used.size !== r.citations.length || r.claims.some(c => !c.citation_ids.length || c.citation_ids.some(id => !r.citations.some(s => s.citation_id === id)))) fail();
    if (r.answer_status === "ABSTAINED" ? r.claims.length || r.citations.length : !r.claims.length) fail();
  }
  return data;
}
export async function loadWalkthroughs(fetcher = fetch) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetcher("/assets/methods-data.json", { credentials: "omit", cache: "no-store", redirect: "error", signal: controller.signal });
    if (!response.ok) throw new Error("METHOD_WALKTHROUGH_UNAVAILABLE");
    const text = await response.text(); if (text.length > 524288) fail();
    const data = validateWalkthroughs(JSON.parse(text));
    for (const source of Object.values(data.sources)) {
      const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source.text)))].map(x => x.toString(16).padStart(2, "0")).join("");
      if (digest !== source.text_sha256) fail();
    }
    return data;
  } finally { clearTimeout(timer); }
}
// Exploration only: not the production question→seed algorithm.
export function neighborhood(graph, seed, depth) {
  if (![1, 2].includes(depth) || !graph.nodes.some(n => n.entity_id === seed)) throw new Error("INVALID_GRAPH_SELECTION");
  const seen = new Set([seed]), edges = new Set(); let frontier = [seed];
  for (let hop = 0; hop < depth; hop++) {
    const next = [];
    for (const id of frontier) for (const e of graph.edges.filter(e => e.polarity === "POSITIVE" && [e.subject_id, e.object_id].includes(id))) {
      edges.add(e.assertion_id);
      for (const n of [e.subject_id, e.object_id]) if (!seen.has(n)) { seen.add(n); next.push(n); }
    }
    frontier = next;
  }
  return { nodes: graph.nodes.filter(n => seen.has(n.entity_id)), edges: graph.edges.filter(e => edges.has(e.assertion_id)) };
}

export function createMethodPages({ document, loadData = loadWalkthroughs, runMethod, refreshRun, openCompare }) {
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  const button = (text, action, cls = "secondary small") => { const n = el("button", text, cls); n.type = "button"; n.addEventListener("click", action); return n; };
  const roots = new Map();
  let active = false, epoch = 0, data = null, loading = null, page = null, capability = null;
  const scopedGuides = {
    hybrid_multi_query: { mechanism: "V6 extracts up to three literal English task facets from the search question. The full original search is always retained. Each search runs Hybrid; the shared pool is fused by rank, without using benchmark labels.", formula: "full search + ≤3 literal facets → Hybrid per query → shared ranked pool → ≤12 whole chunks / 4096 proxy tokens", limit: "No distinct validated native facet means NOT_READY and no generation. A facet is a part of the input question, not a new fact or an answer." },
    hybrid_multi_query_balanced: { mechanism: "Uses the same native facets and candidate pool as Multi-query. The allocator alternates evidence needs when selecting whole-source bundles, recording selections, duplicates and budget exclusions.", formula: "shared Q pool → query-balanced bundle scheduling → ≤12 whole chunks / 4096 proxy tokens", limit: "Balanced selection does not establish correctness or completeness. Empty validated facets stop the method; another method is never substituted." },
    graph_1hop: { mechanism: "Find seed evidence, then follow one source-backed fact in the pinned experimental graph. Direction, action roles, conditions and source-lookup associations are retained. Supporting chunks are packed atomically with the selected fact.", formula: "Hybrid seeds → typed graph lookup → one fact + whole supporting sources → V14 answer checks", limit: "The graph covers only part of this collection. No supported expansion, or a path that cannot fit into context, means NOT_READY with no generation. Connections do not prove actual account permissions." },
    graph_2hop: { mechanism: "Extend source-backed graph lookup to bundles containing up to two connected facts. Both source supports must fit in the final context. Inspect each clause and link meaning; a two-hop lookup is not proof of an executable AWS workflow.", formula: "seeds → connected fact bundle (≤2 facts) → atomic whole-source packing → V14 answer checks", limit: "Zero usable expansion or an unpacked path stops generation. Reverse source lookup is not an inverse action; graph requirements are not access grants. There is no Hybrid fallback." }
  };
  function setGuide(id, scoped) {
    const root = roots.get(id).root, guide = scoped ? { ...METHOD_GUIDES[id], ...scopedGuides[id] } : METHOD_GUIDES[id];
    root.querySelector(".method-mechanism").textContent = guide.mechanism;
    root.querySelector(".method-tradeoff").textContent = guide.limit;
    root.querySelector(".method-formula").textContent = guide.formula;
  }
  function emptyScoped(id) {
    const r = roots.get(id), root = r.root; root.dataset.protocol = "scoped-v2";
    root.querySelector(".method-record-kind").textContent = "COLLECTION V2 · no run started";
    root.querySelector(".method-record-meta").textContent = `${capability.corpus.documents} documents · ${capability.corpus.chunks} chunks · ${capability.graph.facts} experimental facts from ${capability.graph.source_documents} source documents`;
    root.querySelector(".method-example-question").textContent = "Trace your own question through this method";
    root.querySelector(".method-detail").replaceChildren(el("p", "Enter a question and run this method. The steps, sources and map will come from that run; no progress or output is simulated here.", "muted"));
    root.querySelector(".method-source").replaceChildren(el("p", "A selected source passage will appear here after a run.", "muted"));
    if (id.startsWith("graph_")) {
      root.querySelector(".method-graph-area").hidden = false;
      root.querySelector(".method-graph-area .badge").textContent = "Selected run evidence · review pending";
      root.querySelector(".method-graph-counts").textContent = `${capability.graph.facts} experimental facts from ${capability.graph.source_documents}/${capability.corpus.documents} documents. Full graph extraction is incomplete.`;
      root.querySelector(".method-graph-scene").replaceChildren(el("p", "No run selected. A map will show only facts supported by the run's packed sources.", "muted"));
    }
  }
  function renderScopedMethod(id, row, result) {
    const r = roots.get(id), root = r.root; r.live = true; r.example = { scoped: true }; root.dataset.protocol = "scoped-v2";
    setGuide(id, true);
    root.querySelector(".method-source").replaceChildren(el("p", "Select a citation or supporting source for this run.", "muted"));
    const open = (chunk, citation) => r.openSource?.(chunk, citation?.quote);
    root.querySelector(".method-record-kind").textContent = "YOUR V2 RUN · saved sources and observable execution";
    root.querySelector(".method-example-question").textContent = row.request.original_question;
    root.querySelector(".method-record-meta").textContent = `${row.corpus.documents} documents · ${row.corpus.chunks} chunks · ${result.qualification} · ${row.run_id.slice(0, 24)}…`;
    const steps = root.querySelector(".method-steps"); steps.replaceChildren();
    const labels = ["Question", "Search inputs", "Candidate evidence", "Whole-source packing", "Answer eligibility", "Sources & connections"];
    function select(index) {
      [...steps.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
      const panel = root.querySelector(".method-detail"); panel.replaceChildren(el("p", `STEP ${index + 1} / 6 · saved execution`, "eyebrow"), el("h3", labels[index]));
      if (index === 0) panel.append(el("blockquote", row.request.original_question), el("p", `${result.answer_language} answer · ${METHOD_GUIDES[id].label} · no method substitution`, "muted"));
      if (index === 1) {
        const queries = el("ol"); result.retrieval.queries.forEach(q => queries.append(el("li", q))); panel.append(queries,
          el("p", `Planner: ${result.retrieval.planner_origin || "not used"}. Full search is retained. These are retrieval inputs, not model reasoning or verified facts.`, "muted"));
      }
      if (index === 2) {
        panel.append(el("p", `${result.ranked_chunk_ids.length} candidates in the final bounded pool. Rankings are not accuracy scores.`));
        const list = el("ol"); for (const chunk of result.ranked_chunk_ids) { const li = el("li"), source = result.context_sources.find(s => s.chunk_id === chunk);
          li.append(source ? button(source.title, () => open(chunk)) : el("code", chunk, "identity"), el("span", source ? " · included in context" : " · not in final context", "muted")); list.append(li); } panel.append(list);
      }
      if (index === 3) { const content = el("div"); renderScopedSteps(document, content, result); panel.append(content); }
      if (index === 4) renderScopedResult(document, panel, result, open);
      if (index === 5) { panel.append(el("p", "Open a whole source and compare the exact claim or graph clause. An official URL or a connection does not establish correctness or actual access rights."));
        for (const s of result.context_sources) panel.append(button(s.title, () => open(s.chunk_id)));
        if (id.startsWith("graph_")) panel.append(el("p", "The selected graph bundles and their source clauses are shown in the map below.", "muted")); }
    }
    labels.forEach((label, i) => { const b = button(`${String(i + 1).padStart(2, "0")}  ${label}`, () => select(i), "method-step"); b.setAttribute("aria-controls", `method-${id}-detail`); steps.append(b); });
    select(4);
    if (id.startsWith("graph_")) {
      root.querySelector(".method-graph-area").hidden = false;
      root.querySelector(".method-graph-area .badge").textContent = "Selected V2 source bundles · review pending";
      root.querySelector(".method-graph-counts").textContent = `${row.corpus.graph_facts} runtime graph facts from ${row.corpus.graph_source_documents}/${row.corpus.documents} documents; map shows ${result.graph_evidence.length} selected fact(s).`;
      renderScopedGraph(document, root.querySelector(".method-graph-scene"), result, open);
    }
  }
  function sourcePanel(root, source, quote = null) {
    const pane = root.querySelector(".method-source"); pane.replaceChildren();
    if (!source) { pane.append(el("p", "Nguồn này chưa được tải. Dùng source inspector trong Live Compare để đọc passage của run riêng.", "muted")); return; }
    pane.append(el("p", "Source inspector · tài liệu gốc", "eyebrow"), el("h3", source.title));
    if (quote) pane.append(el("p", "Selected evidence", "eyebrow"), el("blockquote", quote));
    const details = el("details"), summary = el("summary", "Read the full retrieved passage"); details.append(summary, el("blockquote", source.text)); pane.append(details);
    const a = el("a", "Mở tài liệu AWS chính thức ↗"); a.href = officialSourceLink(source.url); a.target = "_blank"; a.rel = "noopener noreferrer";
    pane.append(a, el("p", `Normalized source characters ${source.start_char}–${source.end_char} · Unicode offsets`, "muted"));
    const pin = el("details"); pin.append(el("summary", "Document & content identity"), el("code", source.document_version_id, "identity"), el("code", source.text_sha256, "identity")); pane.append(pin);
    pane.tabIndex = -1; pane.focus({ preventScroll: true });
    if (document.defaultView.matchMedia("(max-width: 900px)").matches) pane.scrollIntoView({ block: "nearest" });
  }
  function renderExample(id, example, live = false) {
    const r = roots.get(id); r.example = example; r.live = live;
    r.root.dataset.protocol = "legacy";
    r.root.querySelector(".method-graph-area .badge").textContent = "Collection316 · recorded graph · review pending";
    const root = r.root, guide = METHOD_GUIDES[id], graphMode = id.startsWith("graph_");
    root.querySelector(".method-record-kind").textContent = live ? "YOUR RUN · private results" : "RECORDED EXECUTION · public AWS example";
    root.querySelector(".method-example-question").textContent = example.question;
    root.querySelector(".method-record-meta").textContent = `${example.documents || 316} documents · ${example.chunks || 2978} chunks · ${example.recorded_at ? new Date(example.recorded_at).toLocaleString() : live ? "Current run" : "Recorded local retrieval"} · ${example.review || "NOT_REVIEWED"}`;
    root.querySelector(".method-source").replaceChildren(el("p", "Chọn một passage, citation hoặc cạnh graph để đọc nguồn ở đây.", "muted"));
    const steps = root.querySelector(".method-steps"); steps.replaceChildren();
    guide.stages.forEach((label, index) => {
      const b = button(`${String(index + 1).padStart(2, "0")}  ${label}`, () => selectStep(id, index), "method-step");
      b.setAttribute("aria-controls", `method-${id}-detail`); steps.append(b);
    });
    selectStep(id, 0);
    if (graphMode) drawGraph(id);
  }
  function selectStep(id, index) {
    const r = roots.get(id), e = r.example, root = r.root, graph = id.startsWith("graph_");
    if (!e) return;
    r.step = index;
    [...root.querySelectorAll(".method-step")].forEach((b, i) => b.setAttribute("aria-pressed", String(index === i)));
    const panel = root.querySelector(".method-detail"); panel.replaceChildren(el("p", `STEP ${index + 1} / 6`, "eyebrow"), el("h3", METHOD_GUIDES[id].stages[index]));
    const say = text => panel.append(el("p", text));
    if (index === 0) {
      say("Câu hỏi là đầu vào của pipeline. Dữ liệu công khai chỉ giải thích cơ chế AWS; không cho biết route, quyền hoặc log thực tế của tài khoản bạn.");
      panel.append(el("blockquote", e.question));
      say(`Search query: ${e.search_query || e.question}`);
      if (graph) say(`Requested: ${id} · Effective: ${e.graph_trace.mode_used}. ${e.graph_trace.used_graph ? "Có đường quan hệ trong context đã dùng." : "Hybrid fallback; không giữ được đường quan hệ có bằng chứng."}`);
      say(r.live ? "Đây là kết quả của câu hỏi bạn vừa gửi; không thay bằng ví dụ đã lưu." : "Đây là đầu vào của lần chạy đã lưu. Sửa ô câu hỏi phía trên để tạo lượt thử riêng khi phương pháp được bật.");
    } else if (index === 1) {
      say(METHOD_GUIDES[id].mechanism);
      if (e.subqueries?.length) { say("Các truy vấn thực sự do planner tạo trong lần chạy này:"); const list = el("ol"); for (const q of e.subqueries) list.append(el("li", q)); panel.append(list); }
      else if (graph) { say(`${e.graph_trace.seed_entity_ids.length} seed entities được chọn từ mentions trong hybrid seed chunks:`); const list = el("ul"); for (const key of e.graph_trace.seed_entity_ids) list.append(el("li", data.graph.nodes.find(n => n.entity_id === key)?.canonical_name || key)); panel.append(list); }
      else say(id === "bm25" ? "Không có query embedding ở nhánh lexical." : "Encoder dùng BGE-small-en-v1.5 / 384 chiều. Không tạo vector minh họa giả để giải thích run.");
      say(e.live && !e.search_query ? "Nội dung rewrite chưa được lưu trong public result của phiên bản này; không suy đoán bản dịch." : "Search query được hiển thị từ artifact của lượt chạy, không tự viết lại khi mở trang.");
    } else if (index === 2) {
      panel.append(el("pre", METHOD_GUIDES[id].formula, "method-formula"));
      say(METHOD_GUIDES[id].why);
      if (graph) {
        say(`Observed: ${e.graph_trace.assertion_visits} assertion visits · ${e.graph_trace.visited_entities} visited entities · ${e.graph_trace.graph_candidates.length} graph candidate chunks.`);
        const visited = e.graph_trace.visited_assertion_ids || [];
        if (visited.length) {
          const details = el("details"); details.append(el("summary", `${visited.length} relations actually traversed · kiểm từng nguồn`));
          for (const key of visited) { const edge = data.graph.edges.find(a => a.assertion_id === key); details.append(button(`${edge.predicate} · ${edge.evidence.quote.slice(0, 90)}…`, () => inspectEdge(id, edge), "method-evidence-row")); }
          panel.append(details);
        }
        for (const c of e.graph_trace.graph_candidates) {
          const box = el("article", undefined, "method-path"); box.append(el("strong", `${c.hop} hop${c.assertion_path.length ? " · relation path" : " · mention expansion (no relation traversed)"} · ` + c.entity_path.map(key => data.graph.nodes.find(n => n.entity_id === key)?.canonical_name || key).join(" → ")));
          for (const aid of c.assertion_path) { const edge = data.graph.edges.find(a => a.assertion_id === aid); box.append(button(edge.predicate, () => inspectEdge(id, edge))); }
          panel.append(box);
        }
        say("Đường đi trên đây lấy từ GraphIndex đã chạy. Việc bấm một node trên bản đồ bên dưới là exploration riêng, không viết lại trace này.");
      } else rankings(panel, r, e.ranked_chunk_ids, "Candidate ranking · kết quả thực tế");
      if (e.rerank) { const details = el("details"); details.append(el("summary", "Actual cross-encoder scores & window trace"), el("pre", JSON.stringify(e.rerank, null, 2))); panel.append(details); }
      if (e.query_rankings?.length) for (const q of e.query_rankings) { const details = el("details"); details.append(el("summary", q.query)); rankings(details, r, q.ranked_chunk_ids, "Per-query candidates"); panel.append(details); }
    } else if (index === 3) {
      say(`Giữ ${e.context_chunk_ids.length} chunks · ${e.context_proxy_tokens} proxy tokens. Packer bị chặn bởi tối đa 8 chunks và 4096 proxy tokens; proxy không phải số token provider tính tiền.`);
      say(id === "hybrid_multi_query_balanced" ? "Bộ chọn phân bổ giữa các query rankings trước khi hoàn tất context. So candidate rank và context position để thấy nguồn nào thực sự còn chỗ." : "Ranked pool không được gửi toàn bộ vào model. Chỉ các chunks đóng gói dưới đây trở thành evidence context; rank cao có thể bị bỏ vì trùng hoặc vượt giới hạn.");
      rankings(panel, r, e.context_chunk_ids, "Context order · những nguồn model được nhận");
      if (e.selection) { const d = el("details"); d.append(el("summary", "Actual context-selection decisions"), el("pre", JSON.stringify(e.selection, null, 2))); panel.append(d); }
      if (graph) {
        const kept = e.graph_trace.context_assertion_ids || [];
        say(`${kept.length} relation path records được giữ trong context. Các quan hệ chung supporting chunk không tự tạo thêm nguồn hoặc fact mới.`);
        for (const key of kept) { const edge = data.graph.edges.find(a => a.assertion_id === key); panel.append(button(`Inspect packed relation · ${edge.predicate}`, () => inspectEdge(id, edge))); }
      }
    } else if (index === 4) {
      if (!e.response) say("Lần chạy này chỉ retrieval. Chưa có câu trả lời sinh từ context; không lấy câu trả lời Hybrid để gắn nhãn Graph/BM25/Reranker. Hãy kiểm evidence trước khi mở thêm generation.");
      else {
        say(`DeepSeek · ${e.response.answer_status} · ${e.response.answer_language}. Model được yêu cầu trả claims và citation IDs; server kiểm liên kết nguồn. Đây không phải bằng chứng câu trả lời đã được người chấm xác nhận.`);
        for (const note of e.graph_trace?.answer_review_notes_vi || []) panel.append(el("p", note, "notice method-answer-review"));
        for (const c of e.response.claims) {
          const block = el("article", undefined, "method-claim"); block.append(el("p", c.text));
          for (const key of c.citation_ids) { const cite = e.response.citations.find(s => s.citation_id === key); block.append(button(`[${e.response.citations.indexOf(cite) + 1}] Read evidence`, () => showSource(r, cite.chunk_id, cite.quote))); }
          panel.append(block);
        }
      }
    } else {
      say("Citation cần hỗ trợ đúng claim và các điều kiện của claim. Cùng một trang tài liệu, hoặc một quote có thật, chưa bảo đảm kết luận không bỏ sót điều kiện.");
      const cites = e.response?.citations || [];
      if (cites.length) for (const [i, c] of cites.entries()) panel.append(button(`[${i + 1}] ${c.quote.slice(0, 100)}…`, () => showSource(r, c.chunk_id, c.quote), "method-evidence-row"));
      else rankings(panel, r, e.context_chunk_ids, "Evidence available for inspection");
      say(METHOD_GUIDES[id].limit);
      const d = el("details"); d.append(el("summary", "Version pins & actual execution events"), el("code", e.collection_version || data.graph.snapshot_version, "identity"), el("code", e.config_sha256 || data.graph.trace_sha256, "identity"), el("pre", JSON.stringify(e.events || e.graph_trace, null, 2))); panel.append(d);
    }
    const actions = el("div", undefined, "actions method-step-actions");
    if (index) actions.append(button("← Previous step", () => selectStep(id, index - 1)));
    if (index < 5) actions.append(button("Next step →", () => selectStep(id, index + 1)));
    panel.append(actions);
  }
  function showSource(r, key, quote) {
    if (r.live) {
      r.root.querySelector(".method-source").replaceChildren(el("p", "Loading the selected source for your run…", "muted"));
      if (r.openSource?.(key, quote) === false) sourcePanel(r.root, null);
    }
    else sourcePanel(r.root, data.sources[key], quote);
  }
  function rankings(panel, r, ids, label) {
    panel.append(el("h4", label)); const list = el("ol", undefined, "method-ranks");
    for (const [i, key] of ids.slice(0, 40).entries()) {
      const s = r.live ? r.example.context_sources?.find(s => s.chunk_id === key) : data.sources[key];
      const row = el("li"), inContext = r.example.context_chunk_ids.indexOf(key);
      row.append(el("span", String(i + 1).padStart(2, "0"), "method-rank-number"));
      const body = el("div");
      if (s && (!r.live || inContext >= 0)) body.append(button(s.title, () => showSource(r, key), "text-button"));
      else body.append(el("code", key.slice(0, 20) + "…"));
      body.append(el("p", inContext < 0 ? "Candidate only · not included in context" : `Context position ${inContext + 1} · candidate rank ${r.example.ranked_chunk_ids.indexOf(key) + 1}`, "muted")); row.append(body); list.append(row);
    }
    panel.append(list); if (!ids.length) panel.append(el("p", "No evidence returned. Empty results remain empty.", "muted"));
  }
  function inspectEdge(id, edge) {
    const root = roots.get(id).root, panel = root.querySelector(".method-graph-inspector");
    const name = key => data.graph.nodes.find(n => n.entity_id === key).canonical_name;
    panel.replaceChildren(el("p", "Selected assertion · human review pending", "eyebrow"), el("h4", `${name(edge.subject_id)} → ${name(edge.object_id)}`), el("p", `${edge.predicate} · ${edge.polarity}`), el("p", edge.qualifiers.length ? `Conditions: ${edge.qualifiers.map(q => q.text).join(" · ")}` : "No qualifier recorded. This does not prove the relation is unconditional."), el("blockquote", edge.evidence.quote));
    panel.append(button("Read original passage", () => sourcePanel(root, data.sources[edge.chunk_id], edge.evidence.quote)));
    panel.append(el("p", "AI source review · experimental, not human gold", "muted"), el("p", edge.review_note_vi));
    const identity = el("details"); identity.append(el("summary", "Assertion identity"), el("code", edge.assertion_id, "identity")); panel.append(identity);
    for (const edgeNode of root.querySelectorAll("[data-assertion]")) edgeNode.setAttribute("data-selected", String(edgeNode.dataset.assertion === edge.assertion_id));
  }
  function drawGraph(id) {
    const r = roots.get(id), root = r.root, controls = root.querySelector(".method-graph-controls"), host = root.querySelector(".method-graph-scene");
    if (!controls.children.length) {
      const seedLabel = el("label", "Explore from entity"), select = el("select"); select.className = "method-graph-seed";
      for (const n of data.graph.nodes) { const o = el("option", n.canonical_name); o.value = n.entity_id; select.append(o); }
      const pathSeed = r.example.graph_trace.graph_candidates.find(c => c.assertion_path.length)?.entity_path[0];
      select.value = pathSeed || r.example.graph_trace.seed_entity_ids[0]; seedLabel.append(select); controls.append(seedLabel);
      select.addEventListener("change", () => drawGraph(id));
      controls.append(button("Recorded path", () => { select.value = pathSeed || r.example.graph_trace.seed_entity_ids[0]; drawGraph(id); }));
      const allLabel = el("label", "Show all source-backed connections"), all = el("input"); allLabel.className = "method-graph-toggle"; all.type = "checkbox"; all.className = "method-graph-all"; allLabel.prepend(all); controls.append(allLabel); all.addEventListener("change", () => drawGraph(id));
    }
    const seed = controls.querySelector("select").value, depth = id.endsWith("2hop") ? 2 : 1, all = controls.querySelector(".method-graph-all").checked;
    const endpoints = new Set(data.graph.edges.flatMap(e => [e.subject_id, e.object_id]));
    const scene = all ? { nodes: data.graph.nodes.filter(n => endpoints.has(n.entity_id)), edges: data.graph.edges } : neighborhood(data.graph, seed, depth);
    host.replaceChildren();
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 900 520"); svg.setAttribute("role", "group"); svg.setAttribute("aria-label", `${depth}-hop entity neighborhood. Use the relation list below for an accessible alternative.`);
    svg.classList.toggle("graph-scene-all", all);
    const make = (tag, attributes, text) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attributes)) n.setAttribute(k, v); if (text) n.textContent = text; return n; };
    const defs = make("defs", {}), marker = make("marker", { id: `arrow-${id}`, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" }); marker.append(make("path", { d: "M0 0L10 5L0 10Z", fill: "#4c6d5e" })); defs.append(marker); svg.append(defs);
    const positions = new Map(), others = scene.nodes.filter(n => n.entity_id !== seed);
    positions.set(seed, { x: 450, y: 260 });
    others.forEach((n, i) => { const angle = 2 * Math.PI * i / Math.max(others.length, 1) - Math.PI / 2; positions.set(n.entity_id, { x: 450 + 285 * Math.cos(angle), y: 260 + 183 * Math.sin(angle) }); });
    if (all) {
      const pending = new Set(scene.nodes.map(n => n.entity_id)), groups = [];
      while (pending.size) {
        const group = [], queue = [pending.values().next().value]; pending.delete(queue[0]);
        while (queue.length) { const key = queue.shift(); group.push(key); for (const e of scene.edges.filter(e => [e.subject_id, e.object_id].includes(key))) for (const n of [e.subject_id, e.object_id]) if (pending.delete(n)) queue.push(n); }
        groups.push(group);
      }
      const columns = groups.length > 4 ? 3 : 2, rows = Math.ceil(groups.length / columns), height = Math.max(520, rows * 220);
      svg.setAttribute("viewBox", `0 0 900 ${height}`);
      groups.forEach((group, groupIndex) => group.forEach((key, index) => {
        const x = (groupIndex % columns + .5) * 900 / columns, y = (Math.floor(groupIndex / columns) + .5) * height / rows, angle = 2 * Math.PI * index / group.length - Math.PI / 2;
        positions.set(key, { x: x + (group.length > 1 ? 85 : 0) * Math.cos(angle), y: y + (group.length > 1 ? 66 : 0) * Math.sin(angle) });
      }));
    }
    scene.edges.forEach((e, i) => {
      const a = positions.get(e.subject_id), b = positions.get(e.object_id), dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
      const group = make("g", { role: "button", tabindex: "0", "aria-label": `${e.polarity} ${e.predicate}: ${data.graph.nodes.find(n => n.entity_id === e.subject_id).canonical_name} to ${data.graph.nodes.find(n => n.entity_id === e.object_id).canonical_name}. Read evidence.`, "data-assertion": e.assertion_id, "data-polarity": e.polarity, class: "graph-edge" });
      const offset = (i % 3 - 1) * 18, mx = (a.x + b.x) / 2 - dy / len * offset, my = (a.y + b.y) / 2 + dx / len * offset;
      const path = `M${a.x + dx / len * 25},${a.y + dy / len * 25} Q${mx},${my} ${b.x - dx / len * 29},${b.y - dy / len * 29}`;
      group.append(make("path", { d: path, class: "graph-edge-hit" }), make("path", { d: path, class: "graph-edge-line", "marker-end": `url(#arrow-${id})` }));
      const text = make("text", { x: mx, y: my - 7, "text-anchor": "middle", class: "graph-edge-label" }, e.predicate.replaceAll("_", " ")); group.append(text);
      group.addEventListener("click", () => inspectEdge(id, e)); group.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); inspectEdge(id, e); } }); svg.append(group);
    });
    for (const n of scene.nodes) {
      const p = positions.get(n.entity_id), group = make("g", { role: "button", tabindex: "0", "aria-label": `Explore ${n.canonical_name}`, class: "graph-node", "data-type": n.entity_type });
      const label = make("text", { x: p.x, y: p.y + 39, "text-anchor": "middle" });
      const lines = []; let line = "";
      for (const word of n.canonical_name.split(" ")) { if (line && (line + " " + word).length > 19) { lines.push(line); line = word; } else line += (line ? " " : "") + word; }
      if (line) lines.push(line);
      lines.slice(0, 3).forEach((text, i) => label.append(make("tspan", { x: p.x, dy: i ? 15 : 0 }, text.length > 23 ? text.slice(0, 20) + "…" : text)));
      group.append(make("title", {}, n.canonical_name), make("circle", { cx: p.x, cy: p.y, r: n.entity_id === seed ? 26 : 20 }), label);
      const choose = () => { controls.querySelector("select").value = n.entity_id; controls.querySelector(".method-graph-all").checked = false; drawGraph(id); };
      group.addEventListener("click", choose); group.addEventListener("keydown", ev => { if (["Enter", " "].includes(ev.key)) { ev.preventDefault(); choose(); } }); svg.append(group);
    }
    host.append(svg, el("p", `${scene.nodes.length} entities · ${scene.edges.length} assertions · ${all ? "all source-backed components (not a retrieval path)" : `positive adjacency within ${depth} hop(s)`}. Dark nodes: AWS services; light nodes: concepts/resources. Arrows show assertion direction.`, "muted"));
    if (document.defaultView.matchMedia("(max-width: 480px)").matches) host.scrollLeft = Math.max(0, (svg.getBoundingClientRect().width - host.clientWidth) / 2);
    if (!scene.edges.length) host.append(el("p", "No connected assertion in this neighborhood. A mention-only entity remains unconnected.", "notice"));
    const list = root.querySelector(".method-graph-relations"); list.replaceChildren();
    for (const e of data.graph.edges) list.append(button(`${data.graph.nodes.find(n => n.entity_id === e.subject_id).canonical_name} → ${e.predicate} → ${data.graph.nodes.find(n => n.entity_id === e.object_id).canonical_name}`, () => inspectEdge(id, e), "method-evidence-row"));
    if (scene.edges.length) inspectEdge(id, scene.edges[0]);
    else root.querySelector(".method-graph-inspector").replaceChildren(el("p", "This entity has no supported relation in this neighborhood. No connection has been inferred.", "muted"));
  }
  function controls() {
    for (const [id, r] of roots) {
      const live = capability?.methods?.includes(id) === true;
      r.root.querySelector(".method-submit").disabled = !active || !live || r.submitting;
      const answer = r.root.querySelector('.method-kind option[value="answer"]');
      answer.disabled = !capability?.answering_enabled || !capability.answer_methods?.includes(id);
      const scoped = capability?.schema_version === 2;
      r.root.querySelector('.method-kind option[value="retrieval"]').disabled = scoped;
      if (scoped) r.root.querySelector(".method-kind").value = "answer";
      else if (answer.disabled) r.root.querySelector(".method-kind").value = "retrieval";
      r.root.querySelector(".method-recorded").textContent = scoped ? "Open historical 316 example" : "Open recorded example";
      r.root.querySelector(".method-availability").textContent = scoped ? `Collection ${capability.corpus.documents} · budgeted answer + trace · no method fallback. Completed jobs may withhold answers for source review.` : live ? "Live retrieval enabled · one shared demo attempt per run. Answering is available only when explicitly enabled for this method." : "Cloud execution not released for this method. The full recorded walkthrough remains available; a new question will not silently run a different method.";
    }
  }
  for (const [id, guide] of Object.entries(METHOD_GUIDES)) {
    const root = document.getElementById(`page-method-${id}`); if (!root) throw new Error("METHOD_PAGE_MISSING");
    const r = { root, example: null, live: false, submitting: false }; roots.set(id, r);
    root.querySelector(".method-title").textContent = guide.label;
    root.querySelector(".method-deck").textContent = guide.short;
    root.querySelector(".method-mechanism").textContent = guide.mechanism;
    root.querySelector(".method-tradeoff").textContent = guide.limit;
    root.querySelector(".method-formula").textContent = guide.formula;
    root.querySelector(".method-why").textContent = guide.why;
    root.querySelector(".method-form").addEventListener("submit", async event => {
      event.preventDefault(); if (!active || r.submitting || !capability?.methods?.includes(id)) return;
      const question = root.querySelector("textarea").value.trim(); if (!question || question.length > 2000) return;
      const status = root.querySelector(".method-status"), epochAt = epoch;
      r.submitting = true; r.expected = question; r.lastResult = null; status.textContent = "Submitting one real run. The recorded example is separate until your result arrives."; controls();
      try {
        const accepted = await runMethod(question, root.querySelector(".method-language").value, id, root.querySelector(".method-kind").value);
        if (!active || epochAt !== epoch) return;
        if (!accepted) status.textContent = "Another submission is unresolved or the method is unavailable. Open Live Compare to check/recover it with the same key.";
      } catch { if (active && epochAt === epoch) status.textContent = "Run not confirmed. Open Live Compare to recover or check history; no automatic retry was made."; }
      finally { if (active && epochAt === epoch) { r.submitting = false; controls(); } }
    });
    root.querySelector(".method-open-compare").addEventListener("click", openCompare);
    root.querySelector(".method-refresh").addEventListener("click", () => refreshRun());
    root.querySelector(".method-recorded").addEventListener("click", async () => { const captured = epoch;
      try { if (!data) { loading ||= loadData(); data = validateWalkthroughs(await loading); }
        if (!active || captured !== epoch) return;
        r.expected = null; setGuide(id, false); renderRecorded(id); root.querySelector(".method-status").textContent = "Historical collection316 execution, separate from current V2 runs. No model call was made.";
      } catch { if (active && captured === epoch) root.querySelector(".method-status").textContent = "Historical example unavailable. No question was submitted."; }
    });
  }
  function renderRecorded(id) {
    const r = roots.get(id); r.root.querySelector(".method-graph-area").hidden = !id.startsWith("graph_");
    let example = data.examples.find(e => e.id === id);
    if (!example) {
      const mode = data.graph.modes.find(m => m.requested_hops === (id.endsWith("2hop") ? 2 : 1));
      example = { question: mode.question, search_query: mode.search_query,
        documents: 316, chunks: 2978, review: "AI source review · human review pending", graph_trace: mode, ranked_chunk_ids: mode.final_ranking.slice(0, 40),
        context_chunk_ids: mode.context_chunk_ids, context_proxy_tokens: mode.context_proxy_tokens,
        response: mode.response, recorded_at: mode.recorded_at, config_sha256: mode.execution_identity_sha256, events: mode.events };
    }
    if (id.startsWith("graph_")) {
      r.root.querySelector(".method-graph-counts").textContent = `${data.graph.nodes.length} entities · ${data.graph.mentions} source-bound mention records across ${data.graph.mention_covered_chunks}/2978 chunks · ${data.graph.edges.length} AI-reviewed experimental assertions in ${data.graph.relation_source_documents}/316 documents. ${data.graph.selection.rejected} candidates excluded; ${data.graph.selection.pending} pending. Full relation extraction is incomplete; all retrieval uses collection316. Map and answers come from current collection sources; a managed Neo4j session is not connected.`;
    }
    r.root.querySelector("textarea").value ||= example.question;
    renderExample(id, example);
  }
  return {
    activate() { active = true; epoch++; controls(); },
    clear() {
      active = false; epoch++; capability = null; page = null; loading = null;
      for (const r of roots.values()) { r.expected = null; r.example = null; r.lastResult = null; r.live = false; r.submitting = false; r.root.querySelector("textarea").value = ""; for (const name of ["method-source", "method-detail", "method-steps", "method-status", "method-example-question", "method-record-meta", "method-record-kind", "method-graph-scene", "method-graph-inspector", "method-graph-relations", "method-graph-controls"]) r.root.querySelector(`.${name}`).replaceChildren(); }
      controls();
    },
    setCapabilities(value) {
      const changed = capability?.schema_version !== value?.schema_version;
      capability = value;
      if (changed) epoch++; // Fence a recorded-example load from the previous workflow.
      for (const [id, r] of roots) {
        setGuide(id, value?.schema_version === 2);
        if (changed) { r.expected = null; r.example = null; r.lastResult = null; r.live = false;
          for (const cls of ["method-source", "method-detail", "method-steps", "method-status", "method-record-kind", "method-record-meta", "method-example-question", "method-graph-scene", "method-graph-inspector", "method-graph-controls", "method-graph-relations"]) r.root.querySelector(`.${cls}`).replaceChildren(); }
        if (value?.schema_version === 2 && !r.example) emptyScoped(id);
      }
      controls();
    },
    async show(next) {
      page = next; const id = next?.replace(/^method-/, ""); if (!active || !roots.has(id)) return;
      const captured = epoch, r = roots.get(id);
      if (r.example) return;
      if (capability?.schema_version === 2) { emptyScoped(id); return; }
      r.root.querySelector(".method-status").textContent = "Loading source-verified walkthrough…";
      try {
        if (!data) { loading ||= loadData(); data = validateWalkthroughs(await loading); }
        if (!active || captured !== epoch || page !== next) return;
        renderRecorded(id); r.root.querySelector(".method-status").textContent = "Recorded example ready. Reading steps and sources makes no model call.";
      } catch { if (active && captured === epoch) { loading = null; r.root.querySelector(".method-status").textContent = "Walkthrough unavailable. Leave and reopen this page to retry. No question was submitted."; } }
    },
    updateRun(row, results, openSource) {
      if (!active || !row || row.methods.length !== 1) return;
      const id = row.methods[0].method_id, r = roots.get(id);
      if (!r || r.expected !== row.request.original_question) return;
      r.root.querySelector(".method-status").textContent = `${row.status} · ${row.methods[0].status} · ${row.run_id.slice(0, 20)}…`;
      const result = results.get(id); if (!result || r.lastResult === result) return;
      r.lastResult = result; r.openSource = openSource;
      if (result.schema_version === 2) { renderScopedMethod(id, row, result); return; }
      renderExample(id, { question: row.request.original_question, search_query: row.query_transform === "NONE" ? row.request.original_question : null,
        collection_version: row.collection_version, config_sha256: result.config_sha256,
        ranked_chunk_ids: result.ranked_chunk_ids, context_chunk_ids: result.context_chunk_ids,
        context_sources: result.context_sources, context_proxy_tokens: result.context_proxy_tokens,
        response: result.response, events: result.execution?.events || [], rerank: result.execution?.rerank,
        review: "NOT_REVIEWED", live: true }, true);
    },
    inspectRun(row, result, openSource) {
      if (!active || result.schema_version !== 2 || result.run_id !== row.run_id) return;
      const id = result.method_id, r = roots.get(id); if (!r) return;
      r.expected = row.request.original_question; r.lastResult = result; r.openSource = openSource;
      r.root.querySelector("textarea").value = r.expected;
      r.root.querySelector(".method-language").value = row.request.language;
      r.root.querySelector(".method-status").textContent = `${row.status} · ${result.qualification} · ${row.run_id.slice(0, 24)}…`;
      renderScopedMethod(id, row, result);
    },
    showLiveSource(source, quote, row) { if (page?.startsWith("method-")) { const r = roots.get(page.slice(7)); if (r?.live && r.expected === row?.request.original_question) sourcePanel(r.root, source ? { ...source, url: source.source_uri } : null, quote); } },
  };
}
