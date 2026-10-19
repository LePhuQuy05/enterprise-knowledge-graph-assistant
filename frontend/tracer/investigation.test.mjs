import test from "node:test";
import assert from "node:assert/strict";
import { SCENARIOS, buildQuestion } from "./investigation.mjs";

test("three scenarios have distinct IDs and leave unknown facts unknown", () => {
  assert.equal(new Set(SCENARIOS.map(item => item.id)).size, 3);
  const question = buildQuestion("s3-lambda-read");
  assert.match(question, /relationship is not yet known/); assert.match(question, /not yet known whether/);
  assert.ok(question.length <= 2000);
});
test("known non-KMS encryption does not assume a KMS diagnosis", () => {
  const question = buildQuestion("s3-lambda-read", { accounts: "same", encryption: "no", symptom: " AccessDenied " });
  assert.match(question, /same AWS account/); assert.match(question, /do not assume a KMS/); assert.match(question, /not independently verified/);
});
test("cross-account and decrypt fields are represented explicitly", () => {
  const question = buildQuestion("cross-account-kms", { keyOwner: "A", caller: "B", operation: "decrypt" });
  assert.match(question, /account A/); assert.match(question, /account B/); assert.match(question, /Decrypt/);
  assert.match(buildQuestion("cross-account-kms"), /not confirmed/);
});
test("requirement is required, trimmed and kept as literal data", () => {
  assert.throws(() => buildQuestion("documented-requirement"), /Enter/);
  const question = buildQuestion("documented-requirement", { requirement: "  <script>literal</script>  ", accounts: "cross" });
  assert.match(question, /<script>literal<\/script>/); assert.match(question, /different AWS accounts/);
});
test("invalid scenario, types, enum or excessive details fail before any I/O", () => {
  assert.throws(() => buildQuestion("missing"));
  assert.throws(() => buildQuestion("s3-lambda-read", { accounts: "wildcard" }));
  assert.throws(() => buildQuestion("s3-lambda-read", { accounts: 1 }));
  assert.throws(() => buildQuestion("s3-lambda-read", { symptom: "x".repeat(701) }));
  assert.throws(() => buildQuestion("documented-requirement", { requirement: "x".repeat(1001) }));
});
