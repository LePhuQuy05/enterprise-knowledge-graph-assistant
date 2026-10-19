// Actual Chrome MCP QA against an isolated synthetic preview, never the AWS site.
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { connectChrome } from "./chrome_devtools_client.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const phase = process.argv.find(arg => arg.startsWith("--phase="))?.slice(8) || "review";
if (!/^[a-z0-9-]+$/.test(phase)) throw new Error("INVALID_PHASE");
const out = path.join(root, "results/onboarding", `chrome-${phase}-${new Date().toISOString().replace(/[:.]/g, "-")}`);
await mkdir(out, { recursive: true });
const sock = createServer(); await new Promise(resolve => sock.listen(0, "127.0.0.1", resolve));
const port = sock.address().port; await new Promise(resolve => sock.close(resolve));
const url = `http://127.0.0.1:${port}/`;
const server = spawn(path.join(root, ".venv/Scripts/python.exe"), ["-m", "scripts.preview_web_upgrade", "--serve", "--port", String(port), "--output", path.join(root, "data/build/onboarding-preview")], { cwd: root, windowsHide: true, stdio: "ignore" });
const checks = [], captures = [];
let client, pageId;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const text = result => result.content.filter(item => item.type === "text").map(item => item.text).join("\n");
async function evaluate(body) {
  const raw = text(await client.call("evaluate_script", { pageId, function: `async () => { ${body}\n }`, waitForStableDom: false }));
  const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (!match) throw new Error("MCP_EVALUATION_NOT_JSON");
  return JSON.parse(match[1]);
}
async function check(name, expression) {
  if (!await evaluate(`return (${expression});`)) throw new Error(name);
  checks.push({ name, passed: true });
}
async function press(selector) {
  await evaluate(`document.querySelector(${JSON.stringify(selector)}).focus({preventScroll:true}); return true;`);
  await client.call("press_key", { pageId, key: "Enter" });
}
async function capture(name) {
  await delay(60);
  const layout = await evaluate(`const d=document.getElementById('workspace-tour'),r=d.getBoundingClientRect();return {width:innerWidth,height:innerHeight,pageWidth:document.documentElement.scrollWidth,dialogWidth:d.clientWidth,dialogScrollWidth:d.scrollWidth,dialogBounds:{left:r.left,right:r.right,top:r.top,bottom:r.bottom},open:d.open,title:document.getElementById('tour-title').textContent,step:document.getElementById('tour-count').textContent,target:document.querySelector('.tour-target')?.id,view:__preview.ui.dashboard.getPage()};`);
  if (layout.pageWidth > layout.width || layout.dialogScrollWidth > layout.dialogWidth || layout.dialogBounds.left < -1 || layout.dialogBounds.right > layout.width + 1 || layout.dialogBounds.top < -1 || layout.dialogBounds.bottom > layout.height + 1) throw new Error("ONBOARDING_OVERFLOW:" + name);
  await client.call("take_screenshot", { pageId, filePath: path.join(out, name + ".png"), fullPage: false });
  captures.push({ name, ...layout });
}
async function newPage(context, width, initScript) {
  await client.call("new_page", { url: "about:blank", isolatedContext: context });
  const selected = text(await client.call("list_pages")).split("\n").find(line => line.includes("[selected]"));
  pageId = Number(selected.match(/^(\d+):/)[1]);
  await client.call("emulate", { pageId, viewport: `${width}x${width === 390 ? 844 : 1000}x1${width === 390 ? ",mobile,touch" : ""}` });
  await client.call("navigate_page", { pageId, type: "url", url, ...(initScript ? { initScript } : {}) });
  await evaluate("for(let i=0;i<150;i++){if(globalThis.__preview?.ui && document.getElementById('workspace-tour').open)return true;await new Promise(r=>setTimeout(r,100));}throw Error('TOUR_NOT_READY');");
}
try {
  for (let i = 0; i < 100; i++) { try { if ((await fetch(url)).ok) break; } catch {} await delay(100); }
  client = await connectChrome(root);
  for (const width of [1440, 768, 390]) {
    await newPage(`tour-${phase}-${width}`, width);
    await check(`auto-start-focus-and-private-workspace-${width}`, "document.getElementById('workspace-tour').matches(':modal') && document.activeElement.id==='tour-title' && !document.getElementById('workspace').hidden && localStorage.getItem('kg.ui.onboarding.v1')===null");
    await evaluate("globalThis.__tourCallsBefore=__preview.calls.length;return true;");
    for (let i = 0; i < 7; i++) {
      await capture(`step-${i + 1}-${width}`);
      if (i === 1) {
        await client.call("press_key", { pageId, key: "Tab" });
        await check(`keyboard-focus-inside-modal-${width}`, "document.getElementById('workspace-tour').contains(document.activeElement)");
        await evaluate("document.getElementById('catalog-search').focus();return true;");
        await check(`native-inert-background-${width}`, "document.activeElement.id!=='catalog-search'");
        await press("#tour-back");
        await check(`back-updates-step-${width}`, "document.getElementById('tour-count').textContent==='Step 1 of 7'");
        await press("#tour-next");
      }
      await press("#tour-next");
    }
    await check(`completion-restores-view-and-records-marker-${width}`, "!document.getElementById('workspace-tour').open && __preview.ui.dashboard.getPage()==='overview' && document.activeElement.id==='tour-replay' && localStorage.getItem('kg.ui.onboarding.v1')==='completed' && !document.querySelector('.tour-target') && !document.body.classList.contains('tour-open')");
    await check(`tour-creates-no-requests-or-answers-${width}`, "__preview.calls.length===__tourCallsBefore && !__preview.calls.some(c=>c.method==='POST') && document.getElementById('answer-panel').hidden");
    // Explicit synthetic setup outside the tour: a saved answer, source and unsaved notes.
    await evaluate("await __preview.ui.openRecord(__preview.records[0].request_id);document.querySelector('.claim-select').click();await __preview.ui.evidence.selectCitation('citation-1');document.getElementById('question').value='Synthetic draft to preserve';document.getElementById('brief-notes').value='Synthetic notes to preserve';globalThis.__tourState=JSON.stringify({question:document.getElementById('question').value,notes:document.getElementById('brief-notes').value,source:document.getElementById('source-text').textContent,answer:document.getElementById('answer').textContent});globalThis.__tourCallsBefore=__preview.calls.length;return true;");
    await press("#tour-replay"); await press("#tour-next");
    await client.call("press_key", { pageId, key: "Escape" });
    await check(`escape-replay-preserves-draft-notes-answer-source-${width}`, "!document.getElementById('workspace-tour').open && __preview.ui.dashboard.getPage()==='investigate' && document.activeElement.id==='tour-replay' && __preview.calls.length===__tourCallsBefore && __tourState===JSON.stringify({question:document.getElementById('question').value,notes:document.getElementById('brief-notes').value,source:document.getElementById('source-text').textContent,answer:document.getElementById('answer').textContent})");
    await evaluate("__preview.overrides.source=()=>new Promise(resolve=>{globalThis.__sourceResolve=resolve;});globalThis.__sourcePending=__preview.ui.evidence.selectCitation('citation-1');return true;");
    await check(`tour-does-not-interrupt-pending-source-${width}`, "document.getElementById('tour-replay').disabled && __preview.ui.onboarding.open()===false");
    await evaluate("__sourceResolve({});await __sourcePending;delete __preview.overrides.source;return true;");
    await check(`source-read-failure-restores-tour-availability-${width}`, "!document.getElementById('tour-replay').disabled && !document.getElementById('workspace-tour').open");
    await client.call("navigate_page", { pageId, type: "reload", ignoreCache: true });
    await evaluate("for(let i=0;i<100;i++){if(globalThis.__preview?.ui && document.querySelectorAll('.metric-card').length>=7)return true;await new Promise(r=>setTimeout(r,100));}throw Error('RELOAD_NOT_READY');");
    await check(`dismissal-persists-on-reload-${width}`, "!document.getElementById('workspace-tour').open && !document.getElementById('tour-replay').hidden");
    await press("#tour-replay");
    await evaluate("__preview.ui.clear();return true;");
    await check(`logout-closes-tour-clears-highlight-and-private-state-${width}`, "!document.getElementById('workspace-tour').open && document.getElementById('workspace').hidden && document.getElementById('tour-replay').hidden && !document.querySelector('.tour-target') && !document.body.classList.contains('tour-open') && !document.getElementById('source-text').textContent && !document.getElementById('brief-notes').value");
    const consoleMessages = text(await client.call("list_console_messages", { pageId, types: ["error", "warn", "issue"] }));
    if (!consoleMessages.includes("<no console messages found>")) throw new Error("BROWSER_CONSOLE_MESSAGES:" + consoleMessages);
    checks.push({ name: `console-and-csp-clean-${width}`, passed: true });
  }
  await newPage(`tour-storage-${phase}`, 390, "Object.defineProperty(window,'localStorage',{configurable:true,get(){throw new DOMException('Synthetic storage blocked','SecurityError');}});");
  await capture("blocked-storage-390");
  await press("#tour-skip"); await press("#tour-replay"); await press("#tour-close");
  await check("blocked-storage-leaves-dismiss-and-replay-usable", "!document.getElementById('workspace-tour').open && !document.getElementById('tour-replay').disabled && !document.getElementById('workspace').hidden && !__preview.calls.some(c=>c.method==='POST')");
  await evaluate("__preview.ui.clear();__preview.overrides.dashboard=()=>new Promise(resolve=>{globalThis.__tourResolve=resolve;});globalThis.__tourActivation=__preview.ui.activate({load:false});__preview.ui.clear();__tourResolve(__preview.dashboardData);await __tourActivation;delete __preview.overrides.dashboard;return true;");
  await check("late-dashboard-activation-cannot-reopen-tour-after-logout", "document.getElementById('workspace').hidden && !document.getElementById('workspace-tour').open && document.getElementById('tour-replay').hidden");
  await evaluate("await __preview.ui.activate();return true;");
  await check("blocked-storage-auto-starts-at-most-once-per-page-lifetime", "!document.getElementById('workspace-tour').open && !document.getElementById('tour-replay').hidden");
  await client.call("emulate", { pageId, viewport: "390x844x1,mobile,touch", prefersReducedMotion: "reduce" });
  await press("#tour-replay");
  await check("reduced-motion-keeps-usable-static-dialog", "matchMedia('(prefers-reduced-motion:reduce)').matches && getComputedStyle(document.getElementById('tour-next')).transitionDuration==='0s'");
  await evaluate("document.getElementById('tour-next').focus();return true;");
  await check("interactive-keyboard-focus-remains-visible", "getComputedStyle(document.getElementById('tour-next')).outlineStyle==='solid' && getComputedStyle(document.getElementById('tour-next')).outlineWidth==='2px'");
  await capture("reduced-motion-390");
  await press("#tour-close");
  const consoleMessages = text(await client.call("list_console_messages", { pageId, types: ["error", "warn", "issue"] }));
  if (!consoleMessages.includes("<no console messages found>")) throw new Error("BLOCKED_STORAGE_CONSOLE:" + consoleMessages);
  await writeFile(path.join(out, "report.json"), JSON.stringify({ status: "PASS_LOCAL_ONBOARDING_BROWSER", phase, inspectedWith: client.initialized.serverInfo, checks, captures, project_model_calls: 0, aws_calls: 0, deployed: false, scope: "Real Chrome/native dialog with synthetic private records; not a production login or screen-reader audit." }, null, 2));
  console.log(JSON.stringify({ status: "PASS_LOCAL_ONBOARDING_BROWSER", checks: checks.length, captures: captures.length, output: path.relative(root, out) }));
} catch (error) {
  await writeFile(path.join(out, "report.json"), JSON.stringify({ status: "FAIL", phase, error: error.message, checks, captures }, null, 2));
  console.error(error); process.exitCode = 1;
} finally { await client?.close(); server.kill(); }
