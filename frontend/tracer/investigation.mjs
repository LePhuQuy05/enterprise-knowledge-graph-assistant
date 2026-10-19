const accountField = { id: "accounts", label: "Account relationship", options: [
  ["unknown", "I am not sure yet"], ["same", "Same account"], ["cross", "Different accounts"],
] };

export const SCENARIOS = [
  { id: "s3-lambda-read", version: 1, title: "The trigger works. The read fails.",
    description: "Trace S3 → Lambda → encrypted object access, one permission layer at a time.",
    tag: "S3 · Lambda · IAM · KMS", fields: [accountField,
      { id: "encryption", label: "Is the object encrypted with SSE-KMS?", options: [
        ["unknown", "I am not sure yet"], ["yes", "Yes, SSE-KMS"], ["no", "No, not SSE-KMS"],
      ] }, { id: "symptom", label: "What have you observed? (optional)", maxLength: 700 }],
  },
  { id: "cross-account-kms", version: 1, title: "A key in another account.",
    description: "Identify the policy layers and conditions for cross-account KMS access.",
    tag: "KMS · IAM · Cross-account", fields: [
      { id: "keyOwner", label: "Who owns the key?", options: [["unknown", "Not confirmed"], ["A", "Example account A"]] },
      { id: "caller", label: "Where is the caller?", options: [["unknown", "Not confirmed"], ["B", "Example account B"]] },
      { id: "operation", label: "Which operation?", options: [["unknown", "Not confirmed"], ["decrypt", "Decrypt"]] },
    ],
  },
  { id: "documented-requirement", version: 1, title: "What does the documentation establish?",
    description: "Check a requirement and separate supported statements from missing facts.",
    tag: "Documentation · Evidence gaps", fields: [accountField,
      { id: "requirement", label: "Requirement or symptom to investigate", maxLength: 1000, required: true },
    ],
  },
];

export const INVESTIGATION_STEPS = [
  { id: "invoke", title: "E1 · Invoke the function", question: "Who invokes the function, and which invocation conditions apply?",
    url: "https://docs.aws.amazon.com/lambda/latest/dg/permissions-function-services.html" },
  { id: "read", title: "E2 · Read the object", question: "Which execution identity reads the object, and what permits that read?",
    url: "https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html" },
  { id: "key", title: "E3 · Use the encryption key", question: "Who owns the key, who calls KMS, and which policy conditions still need evidence?",
    url: "https://docs.aws.amazon.com/kms/latest/developerguide/key-policy-modifying-external-accounts.html" },
];

export function buildQuestion(scenarioId, values = {}) {
  const scenario = SCENARIOS.find(item => item.id === scenarioId);
  if (!scenario) throw new Error("Choose an investigation first.");
  const data = {};
  for (const field of scenario.fields) {
    const value = values[field.id] ?? (field.options ? "unknown" : "");
    if (typeof value !== "string") throw new Error(`Check ${field.label.toLowerCase()}.`);
    if (field.options && !field.options.some(([key]) => key === value)) throw new Error("Choose a valid field option.");
    data[field.id] = value.trim();
    if (field.required && !data[field.id]) throw new Error(`Enter ${field.label.toLowerCase()}.`);
    if (field.maxLength && value.length > field.maxLength) throw new Error(`${field.label} is too long.`);
  }
  const relationship = { same: "The services are in the same AWS account.", cross: "The services are in different AWS accounts.",
    unknown: "The account relationship is not yet known; distinguish the cases and identify facts that need confirmation." };
  let parts;
  if (scenarioId === "s3-lambda-read") {
    const encryption = { yes: "The object uses SSE-KMS encryption.", no: "The object does not use SSE-KMS encryption; do not assume a KMS permission issue.",
      unknown: "It is not yet known whether the object uses SSE-KMS; explain when key-use permissions are relevant." };
    parts = ["Why can S3 invoke Lambda while the function cannot read the uploaded S3 object?", relationship[data.accounts],
      encryption[data.encryption], data.symptom ? `Observed symptom (not independently verified): ${data.symptom}` : "",
      "Separate function invocation, object access and, when applicable, key use. Give a source-backed investigation checklist and identify missing evidence."];
  } else if (scenarioId === "cross-account-kms") {
    parts = ["Which policy layers and conditions must permit cross-account use of a customer-managed AWS KMS key?",
      data.keyOwner === "A" ? "The key is in example account A." : "The key-owning account is not confirmed.",
      data.caller === "B" ? "The caller is in example account B." : "The caller's account is not confirmed.",
      data.operation === "decrypt" ? "The operation to investigate is Decrypt." : "The requested operation is not yet confirmed.",
      "Explain the documented conditions with sources. Distinguish general guidance from facts still needed to diagnose a real account; do not generate or apply policies."];
  } else {
    parts = [`What does the available AWS documentation establish about this requirement or symptom: ${data.requirement}`,
      relationship[data.accounts], "Separate supported statements from conditions or facts missing from the corpus. Include citations and abstain where evidence is insufficient."];
  }
  const question = parts.filter(Boolean).join("\n\n");
  if (question.length > 2000) throw new Error("Shorten the details before building this question.");
  return question;
}
