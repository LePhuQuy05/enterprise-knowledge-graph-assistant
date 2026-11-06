import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { METHOD_GUIDES, validateWalkthroughs, loadWalkthroughs, neighborhood } from "./methods.mjs";
const data = JSON.parse(await readFile(new URL("./methods-data.json", import.meta.url), "utf8"));

test("eight independent guides use only the 316-document collection", () => {
  assert.equal(Object.keys(METHOD_GUIDES).length, 8);
  assert.equal(validateWalkthroughs(data).graph.documents, 316);
  for (const guide of Object.values(METHOD_GUIDES)) assert.equal(guide.stages.length, 6);
  assert.ok(data.examples.every(e => e.collection_version === data.graph.snapshot_version));
});
test("source substitution, invalid path and pilot scope fail closed", () => {
  for (const change of [d => d.graph.documents = 20, d => d.graph.chunks = 236,
    d => d.graph.edges[0].subject_id = "unknown", d => d.graph.edges[0].evidence.quote = "invented",
    d => d.graph.modes[0].graph_candidates[0].assertion_path = ["invented"],
    d => d.sources[Object.keys(d.sources)[0]].url = "https://evil.example/source",
    d => d.examples[0].response.citations[0].quote = "made up source"]) {
    const copy = structuredClone(data); change(copy); assert.throws(() => validateWalkthroughs(copy));
  }
});
test("walkthrough loading verifies hashes and never posts a question", async () => {
  let calls = 0;
  await loadWalkthroughs(async (path, options) => { calls++; assert.equal(path, "/assets/methods-data.json"); assert.equal(options.credentials, "omit"); assert.equal(options.method, undefined); return { ok: true, text: async () => JSON.stringify(data) }; });
  assert.equal(calls, 1);
  const bad = structuredClone(data); const s = Object.values(bad.sources)[0]; s.text = "x" + s.text.slice(1);
  await assert.rejects(loadWalkthroughs(async () => ({ ok: true, text: async () => JSON.stringify(bad) })));
});
test("neighborhood uses only real positive edges and enforces hop bound", () => {
  const graph = { nodes: ["a", "b", "c", "d"].map(entity_id => ({ entity_id })), edges: [
    { assertion_id: "ab", subject_id: "a", object_id: "b", polarity: "POSITIVE" },
    { assertion_id: "bc", subject_id: "b", object_id: "c", polarity: "POSITIVE" },
    { assertion_id: "ad", subject_id: "a", object_id: "d", polarity: "NEGATIVE" }] };
  assert.deepEqual(neighborhood(graph, "a", 1).nodes.map(n => n.entity_id), ["a", "b"]);
  assert.deepEqual(neighborhood(graph, "a", 2).nodes.map(n => n.entity_id), ["a", "b", "c"]);
  assert.throws(() => neighborhood(graph, "a", 3));
  assert.throws(() => neighborhood(graph, "missing", 1));
});

test("reviewed graph answers retain real context citations and experimental review labels", () => {
  assert.equal(data.graph.edges.length, data.graph.selection.selected);
  assert.equal(data.graph.human_reviewed, false);
  assert.equal(data.graph.legacy_graph_reused, false);
  for (const mode of data.graph.modes) {
    assert.ok(mode.response.claims.length > 0 && mode.response.citations.length > 0);
    assert.ok(mode.context_assertion_ids.length > 0);
    assert.ok(mode.context_assertion_ids.every(id => mode.visited_assertion_ids.includes(id)));
    assert.ok(mode.answer_review_notes_vi.length > 0);
  }
  assert.equal(data.graph.modes[0].response.answer_language, "en");
  assert.equal(data.graph.modes[1].response.answer_language, "vi");
});

test("graph cannot invent human approval, source support, path usage or citation IDs", () => {
  for (const change of [d => d.graph.human_reviewed = true,
    d => d.graph.legacy_graph_reused = true,
    d => d.graph.selection.selected++,
    d => d.graph.edges[0].review_status = "HUMAN_GOLD",
    d => d.graph.modes[0].used_graph = false,
    d => d.graph.modes[0].answer_review_notes_vi = [],
    d => d.graph.modes[0].visited_assertion_ids.push("invented"),
    d => d.graph.modes[0].response.citations[0].quote = "invented quote",
    d => d.graph.modes[1].response.claims[0].citation_ids = ["invented"]]) {
    const copy = structuredClone(data); change(copy);
    assert.throws(() => validateWalkthroughs(copy));
  }
});
