import { officialSourceLink } from "./client.mjs";
import { projectEvidence } from "./evidence.mjs";

export function escapeMarkdown(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replace(/[\\`*_{}\[\]()#!|~+-]/g, "\\$&").replace(/\r\n?/g, "\n");
}

function safeDestination(raw) {
  const url = officialSourceLink(raw);
  return url ? url.replace(/[()<>\\\s]/g, character => encodeURIComponent(character).replaceAll("(", "%28").replaceAll(")", "%29")) : null;
}

export function toBriefData(record, notes = "") {
  const data = projectEvidence(record);
  if (typeof notes !== "string" || notes.length > 2000) throw new Error("Keep personal notes under 2,000 characters.");
  return { ...data, notes }; // data is the closed, explicitly selected public projection, never the raw record.
}

export function serializeBrief(record, notes = "") {
  const data = toBriefData(record, notes), e = escapeMarkdown;
  const statuses = { ANSWERED: "Answered", PARTIAL: "Partial answer — evidence is incomplete", ABSTAINED: "Not enough evidence" };
  const lines = ["# AWS Workflow Investigation Brief", "", "## Question", "", e(data.question), "", `**Status:** ${statuses[data.answer_status]}`, ""];
  if (data.answer_status === "PARTIAL") lines.push("Some parts of this question are not fully supported by the available sources.", "");
  lines.push("## Statements and cited evidence", "");
  if (data.answer_status === "ABSTAINED") lines.push(e(data.answer), "");
  data.claims.forEach((claim, index) => {
    lines.push(`### Statement ${index + 1}`, "", e(claim.text), "");
    for (const id of claim.citation_ids) {
      const citation = data.citations.find(item => item.citation_id === id), destination = safeDestination(citation.source_uri);
      lines.push(`- **${e(citation.title)}**${citation.section_path.length ? ` — ${e(citation.section_path.join(" / "))}` : ""}`);
      lines.push(...e(citation.quote).split("\n").map(line => `  > ${line}`));
      lines.push(destination ? `  [Official AWS source](${destination})` : "  Source link omitted: not an approved AWS URL.", "");
    }
  });
  lines.push("## Scope and limitations", "", "Citations identify source passages; they do not automatically verify a statement's meaning. Check conditions in the original documentation.",
    data.retrieval_backend === "aws_knowledge_mcp" ? "This answer uses saved official AWS search excerpts, not a complete copy of AWS documentation. Prose line wrapping may be normalized before generation; the server retains the original excerpt. Retrieval time is not the page update date." : "This brief covers a saved answer from the indexed pilot corpus, which may be incomplete.", "");
  for (const warning of data.warnings) lines.push(`- ${e(warning)}`);
  if (data.notes.trim()) lines.push("", "## User notes — not model evidence", "", e(data.notes), "");
  lines.push("## Record details", "", `- Created: ${e(data.created_at)}`, `- Search method: ${e(data.retrieval_backend === "aws_knowledge_mcp" ? "AWS official live search (saved excerpts)" : data.mode_used)}`,
    `- Answer language: ${e(data.answer_language)}`,
    ...(data.evidence_bundle_sha256 ? [`- Evidence bundle SHA256: ${e(data.evidence_bundle_sha256)}`, "- Snapshot below identifies publication policy; the bundle hash identifies this live answer's evidence."] : []),
    `- Graph contribution: ${data.used_graph ? "yes" : "no"}`, `- Snapshot: ${e(data.snapshot_version)}`, `- Request: ${e(data.request_id)}`, "");
  return lines.join("\n");
}

export function briefFilename(record) {
  const data = projectEvidence(record);
  return `aws-investigation-${new Date(data.created_at).toISOString().slice(0, 10)}-${data.request_id.slice(2, 10)}.md`;
}

export function createBriefDownloader({ document, urlAPI = URL, BlobType = Blob, schedule = setTimeout, cancel = clearTimeout }) {
  const active = new Map();
  function revoke(url) {
    if (!active.has(url)) return;
    cancel(active.get(url)); active.delete(url); urlAPI.revokeObjectURL(url);
  }
  function download(markdown, filename) {
    const url = urlAPI.createObjectURL(new BlobType([markdown], { type: "text/markdown;charset=utf-8" }));
    active.set(url, null);
    const link = document.createElement("a"); link.href = url; link.download = filename;
    try { document.body.append(link); link.click(); active.set(url, schedule(() => revoke(url), 30000)); }
    catch (error) { revoke(url); throw error; }
    finally { link.remove(); }
  }
  return { download, clear: () => { for (const url of [...active.keys()]) revoke(url); } };
}
