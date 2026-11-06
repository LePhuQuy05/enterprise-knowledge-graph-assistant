import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { validateCollectionAnswer } from "./collection-answer.mjs";
import { validateRunResult } from "./compare.mjs";
const fixture = JSON.parse(await readFile(new URL("../../tests/fixtures/owned_runs/answer.json", import.meta.url), "utf8"));

test("recorded public-source answer validates on its own collection and run", () => {
  assert.equal(validateCollectionAnswer(fixture).answer_status, "ANSWERED");
  assert.equal(validateRunResult(fixture, "hybrid", "answer", fixture.run_id), fixture);
  assert.throws(() => validateRunResult(fixture, "hybrid"));
  assert.throws(() => validateRunResult(fixture, "hybrid", "answer", "run_" + "f".repeat(64)));
});
for (const [label, mutate] of [
  ["foreign collection", r => r.response.snapshot_version = "collection_" + "f".repeat(64)],
  ["unknown citation", r => r.response.claims[0].citation_ids = ["not-a-source"]],
  ["foreign source URL", r => r.response.citations[0].source_uri = "https://example.com/"],
  ["different document", r => r.response.citations[0].document_version_id = "foreign"],
  ["invented graph", r => r.response.used_graph = true],
  ["fabricated abstention", r => r.response.answer_status = "ABSTAINED"],
  ["changed answer text", r => r.response.answer = "unmatched claim"],
]) test("collection answer rejects " + label, () => { const r = structuredClone(fixture); mutate(r); assert.throws(() => validateCollectionAnswer(r)); });
