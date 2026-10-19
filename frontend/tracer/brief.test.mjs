import test from "node:test";
import assert from "node:assert/strict";
import { fixtureRecord } from "../../tests/fixtures/web_upgrade/records.mjs";
import { serializeBrief, toBriefData, briefFilename, createBriefDownloader } from "./brief.mjs";

test("closed projection excludes private and unknown fields throughout the record", () => {
  const record = fixtureRecord(); record.Authorization = "PRIVATE_CANARY"; record.response.raw_output = "PRIVATE_CANARY";
  record.response.citations[0].secret = "PRIVATE_CANARY"; record.receipt = { tenant_id: "PRIVATE_CANARY" };
  assert.ok(!JSON.stringify(toBriefData(record)).includes("PRIVATE_CANARY"));
  const output = serializeBrief(record, "My own note.");
  assert.ok(!output.includes("PRIVATE_CANARY")); assert.match(output, /User notes — not model evidence/);
  assert.match(output, /Official AWS source/); assert.match(output, /Snapshot/);
});
test("partial and abstention remain explicit in the exported artifact", () => {
  assert.match(serializeBrief(fixtureRecord("PARTIAL")), /evidence is incomplete/);
  const output = serializeBrief(fixtureRecord("ABSTAINED"));
  assert.match(output, /Not enough evidence/); assert.ok(!output.includes("### Statement"));
  assert.throws(() => serializeBrief(fixtureRecord("FAILED")));
});
test("HTML and Markdown injection are escaped and unsafe source URLs omitted", () => {
  const r = fixtureRecord(); r.question = '<img src=x onerror=alert(1)> [x](javascript:evil)';
  r.response.citations[0].title = "[malicious](https://evil.example)";
  r.response.citations[0].source_uri = "javascript:alert(1)";
  const output = serializeBrief(r, "<script>bad</script>\n![image](data:text/html,bad)");
  assert.ok(!output.includes("<img")); assert.ok(!output.includes("<script>")); assert.ok(!output.includes("![image]"));
  assert.ok(!output.includes("[x](javascript:")); assert.match(output, /Source link omitted/);
});
test("approved URLs cannot break a Markdown link destination", () => {
  const r = fixtureRecord(); r.response.citations[0].source_uri = "https://docs.aws.amazon.com/a)b(c";
  const output = serializeBrief(r); assert.match(output, /a%29b%28c/);
  assert.throws(() => serializeBrief(r, "n".repeat(2001)));
  assert.match(briefFilename(r), /^aws-investigation-2026-10-18-a{8}\.md$/);
});
test("download object URLs are cleaned on clear and after click failures", () => {
  const revoked = [], links = [], timers = [];
  const document = { body: { append: link => links.push(link) }, createElement: () => ({ click() {}, remove() {} }) };
  const downloader = createBriefDownloader({ document, urlAPI: { createObjectURL: () => "blob:test", revokeObjectURL: url => revoked.push(url) },
    schedule: callback => { timers.push(callback); return 1; }, cancel: () => {} });
  downloader.download("hello", "safe.md"); assert.equal(links[0].download, "safe.md");
  downloader.clear(); timers[0](); assert.deepEqual(revoked, ["blob:test"]);
  document.createElement = () => ({ click() { throw new Error("blocked"); }, remove() {} });
  assert.throws(() => downloader.download("hello", "safe.md")); assert.equal(revoked.length, 2);
});
