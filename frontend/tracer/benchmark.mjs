// Recorded experiment inventory. Never submit queries or infer quality from availability.
const ids = ["bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query", "hybrid_multi_query_balanced", "graph_1hop", "graph_2hop"];
const oldLabels = { B0_BM25: "BM25", B1_VECTOR: "Vector", B2_HYBRID: "Hybrid", G1_DRAFT: "Graph · 1 hop", G2_DRAFT: "Graph · 2 hops" };
const fail = () => { throw new Error("BENCHMARK_REPORT_INVALID"); };
export function validateBenchmark(value) {
  const c = value?.collection, p = value?.pilot;
  const integer = n => Number.isSafeInteger(n) && n >= 0;
  if (value?.schema_version !== 1 || value.scope !== "READINESS_NOT_COLLECTION_QUALITY" || value.collection_quality !== null ||
      !c || !p || !/^collection_[a-f0-9]{64}$/.test(c.version) || !/^[a-f0-9]{64}$/.test(c.manifest_sha256) ||
      ![c.documents, c.chunks, c.service_groups, c.entities, c.mentions, c.assertions, p.documents, p.chunks, p.assertions].every(integer) ||
      !c.documents || !c.chunks || !Array.isArray(value.methods) || value.methods.length !== ids.length ||
      value.methods.some((r, i) => r.id !== ids[i] || typeof r.label !== "string" || r.label.length > 100 ||
        typeof r.mechanism !== "string" || r.mechanism.length > 300 || r.collection_quality !== null ||
        !["COLLECTION_IMPLEMENTED", "PILOT_GRAPH_ONLY"].includes(r.local) ||
        !["RECORDED_DEPLOYED", "NOT_RELEASED"].includes(r.cloud_retrieval) || r.cloud_answer !== "NOT_RELEASED") ||
      value.methods.some(r => r.id.startsWith("graph_") && r.cloud_retrieval !== "NOT_RELEASED") ||
      !Array.isArray(p.results) || p.results.length !== 5 || p.review !== "AI_DRAFT" ||
      p.results.some((r, i) => r.id !== Object.keys(oldLabels)[i] || ["en", "vi"].some(lang =>
        !integer(r[lang]?.complete) || !integer(r[lang]?.denominator) || r[lang].denominator < 1 || r[lang].complete > r[lang].denominator))) fail();
  return value;
}
export async function loadBenchmark(fetcher = fetch) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetcher("/assets/benchmark-data.json", { cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal });
    if (!response.ok) throw new Error("BENCHMARK_REPORT_UNAVAILABLE");
    const raw = await response.text(); if (raw.length > 100000) fail();
    return validateBenchmark(JSON.parse(raw));
  } finally { clearTimeout(timer); }
}
export function createBenchmark({ document, loadData = loadBenchmark }) {
  const get = id => document.getElementById(id);
  const node = (tag, text, className) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (className) n.className = className; return n; };
  let active = false, epoch = 0, loaded = false, loading = false;
  function cell(row, text, tag = "td") { const c = node(tag, text); if (tag === "th") c.scope = "row"; row.append(c); }
  function render(data) {
    const c = data.collection;
    get("benchmark-identity").textContent = `${c.documents} documents / ${c.chunks.toLocaleString("en")} chunks / ${c.service_groups} service groups`;
    get("benchmark-hash").textContent = c.manifest_sha256;
    get("benchmark-graph-state").textContent = `Collection graph: ${c.entities} entities, ${c.mentions} mentions, ${c.assertions} assertions. Graph methods have not been evaluated on this collection.`;
    const body = get("benchmark-methods"); body.replaceChildren();
    for (const m of data.methods) {
      const row = node("tr"); const name = node("th"); name.scope = "row";
      name.append(node("strong", m.label), node("span", m.mechanism, "benchmark-mechanism")); row.append(name);
      cell(row, m.local === "PILOT_GRAPH_ONLY" ? "Pilot experiment only" : "Implemented on collection");
      cell(row, m.cloud_retrieval === "RECORDED_DEPLOYED" ? "Retrieval deployed" : "Not released");
      cell(row, "Not measured"); body.append(row);
    }
    const old = get("benchmark-pilot-results"); old.replaceChildren();
    for (const m of data.pilot.results) { const row = node("tr"); cell(row, oldLabels[m.id], "th"); for (const lang of ["en", "vi"]) cell(row, `${m[lang].complete}/${m[lang].denominator}`); old.append(row); }
    get("benchmark-pilot-meta").textContent = `${data.pilot.documents} documents / ${data.pilot.chunks} chunks / ${data.pilot.assertions} assertions. These are older AI_DRAFT development retrieval results, not collection-316 scores or answer accuracy.`;
    get("benchmark-status").textContent = `Inventory inspected ${new Date(data.checked_at).toLocaleString()}. Deployment state is a recorded checkpoint, not a live health check. Reading this page makes no model calls.`;
  }
  async function show(page) {
    if (!active || page !== "benchmark" || loaded || loading) return;
    const captured = epoch; loading = true;
    get("benchmark-status").textContent = "Loading the recorded method inventory…";
    try { const data = validateBenchmark(await loadData()); if (active && epoch === captured) { render(data); loaded = true; } }
    catch { if (active && epoch === captured) get("benchmark-status").textContent = "The benchmark inventory could not be loaded. Leave and reopen this page to retry. No model request was made."; }
    finally { if (epoch === captured) loading = false; }
  }
  function clear() {
    active = false; epoch++; loaded = false; loading = false;
    for (const id of ["benchmark-methods", "benchmark-pilot-results"]) get(id).replaceChildren();
    for (const id of ["benchmark-identity", "benchmark-status", "benchmark-hash", "benchmark-graph-state", "benchmark-pilot-meta"]) get(id).textContent = "";
  }
  return { activate() { clear(); active = true; }, clear, show };
}
