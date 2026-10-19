// Local synthetic browser checks. Uses the installed Edge and Node WebSocket; no npm dependencies or model calls.
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const output = path.join(root, "results", "web-upgrade", stamp);
await mkdir(output, { recursive: true });
const checks = [], errors = [], requests = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function freePort() {
  const server = createServer(); await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port; await new Promise(resolve => server.close(resolve)); return port;
}
const port = await freePort(), debugPort = await freePort();
const server = spawn(path.join(root, ".venv", "Scripts", "python.exe"), ["-m", "scripts.preview_web_upgrade", "--serve", "--port", String(port)], { cwd: root, windowsHide: true, stdio: "ignore" });
const profile = path.join(root, "data/build", `web-upgrade-browser-${stamp}`);
const browser = spawn("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-sync", `--remote-debugging-port=${debugPort}`, "--remote-debugging-address=127.0.0.1", `--user-data-dir=${profile}`, "about:blank"], { windowsHide: true, stdio: "ignore" });
let socket, nextId = 0;
const pending = new Map();
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP_TIMEOUT:${method}`)); }, 15000);
    pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", { expression: `(async () => { ${expression}\n})()`, awaitPromise: true, returnByValue: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
async function check(name, expression) { assert.equal(await evaluate(`return (${expression});`), true, name); checks.push(name); }
async function waitFor(expression) {
  for (let i = 0; i < 80; i++) { if (await evaluate(`return (${expression});`)) return; await delay(100); }
  throw new Error("BROWSER_WAIT_TIMEOUT");
}
try {
  let target;
  for (let i = 0; i < 100; i++) {
    try { const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`); target = (await response.json()).find(item => item.type === "page"); if (target) break; } catch {}
    await delay(100);
  }
  if (!target) throw new Error("HEADLESS_EDGE_NOT_READY");
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener("open", resolve, { once: true }); socket.addEventListener("error", reject, { once: true }); });
  socket.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id); clearTimeout(entry.timer); pending.delete(message.id);
      if (message.error) entry.reject(new Error(message.error.message)); else entry.resolve(message.result);
    }
    if (message.method === "Runtime.exceptionThrown") errors.push(message.params.exceptionDetails.text);
    if (message.method === "Network.requestWillBeSent") requests.push(message.params.request.url);
  });
  await send("Page.enable"); await send("Runtime.enable"); await send("Network.enable");
  await send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: path.join(output, "downloads") });
  await send("Emulation.setDeviceMetricsOverride", { width: 1365, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: `http://127.0.0.1:${port}/` });
  await waitFor("!!globalThis.__preview && document.querySelectorAll('.history-item').length === 5");
  await waitFor("!!globalThis.__preview.ui.onboarding && document.querySelector('#workspace-tour').open");
  await evaluate("document.querySelector('#tour-skip').click(); return true;");
  await waitFor("document.querySelectorAll('.metric-card').length >= 7");
  await check("dashboard opens Overview with recorded corpus and no approved-quality claim", "!document.getElementById('page-overview').hidden && document.getElementById('overview-metrics').textContent.includes('20 documents') && document.getElementById('overview-metrics').textContent.includes('Not measured')");
  await evaluate("document.querySelector('[data-page=knowledge]').click()");
  await check("Knowledge Explorer lists the 20 indexed public documents", "!document.getElementById('page-knowledge').hidden && document.querySelectorAll('.document-card').length===20 && document.getElementById('catalog-count').textContent.includes('catalog filter only')");
  await evaluate("document.getElementById('catalog-search').value='key policies'; document.getElementById('catalog-search').dispatchEvent(new Event('input')); document.querySelector('.document-card').click()");
  await check("catalog filters and reveals versioned source details", "document.querySelectorAll('.document-card').length===1 && document.getElementById('catalog-detail').textContent.includes('Fetch date is not last-modified') && document.querySelector('#catalog-detail a').hostname==='docs.aws.amazon.com'");
  await evaluate("document.querySelector('#catalog-detail button').click()");
  await check("catalog question drafts do not submit or silently filter retrieval", "!document.getElementById('page-investigate').hidden && document.getElementById('question').value.includes('Key policies') && !__preview.calls.some(c=>c.method==='POST')");
  await evaluate("document.querySelector('[data-page=lab]').click(); document.querySelector('[data-lab-case=dev-11]').click()");
  await check("Evidence Radar identifies a real draft candidate-to-context gap", "!document.getElementById('page-lab').hidden && document.querySelectorAll('#radar tbody tr').length===2 && document.getElementById('radar').textContent.includes('Missing') && document.getElementById('lab-summary').textContent.includes('10/12') && document.getElementById('lab-case-meta').textContent.includes('AI_DRAFT')");
  await evaluate("document.querySelector('#radar .radar-group').click(); document.querySelector('.ranked-chunk').click()");
  await check("recorded quotes and chunk previews drill down without private source API", "document.querySelector('#radar-detail blockquote')!==null && document.querySelector('#lab-source blockquote')!==null && !__preview.calls.some(c=>c.path.startsWith('/v1/sources/')) && !__preview.calls.some(c=>c.method==='POST')");
  await evaluate("document.getElementById('lab-method').value='G1_DRAFT'; document.getElementById('lab-method').dispatchEvent(new Event('change'))");
  await check("graph comparison is recorded and explicitly experimental", "document.getElementById('lab-rankings').textContent.includes('unapproved graph') && document.getElementById('lab-rankings').textContent.includes('Graph 1-hop')");
  await evaluate("document.getElementById('lab-question').value='dev-13'; document.getElementById('lab-question').dispatchEvent(new Event('change'))");
  await check("unanswerable retrieval cases have no fabricated coverage score", "document.getElementById('radar').textContent.includes('not scored') && !document.querySelector('#radar table')");
  await evaluate("document.querySelector('[data-lab-case=dev-11]').click()");
  const labShot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }); await writeFile(path.join(output, "retrieval-lab.png"), Buffer.from(labShot.data, "base64"));
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await check("390px dashboard and Radar contain overflow within the table", "document.documentElement.scrollWidth<=window.innerWidth");
  await evaluate("document.querySelector('[data-page=overview]').click()");
  const overviewShot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }); await writeFile(path.join(output, "dashboard-mobile.png"), Buffer.from(overviewShot.data, "base64"));
  await send("Emulation.setDeviceMetricsOverride", { width: 1365, height: 1000, deviceScaleFactor: 1, mobile: false });
  await evaluate("document.querySelector('[data-page=overview]').click()");
  const overviewDesktop = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }); await writeFile(path.join(output, "dashboard-desktop.png"), Buffer.from(overviewDesktop.data, "base64"));
  await check("3 scenarios use the production DOM without inference", "document.querySelectorAll('[data-scenario-id]').length===3 && !__preview.calls.some(c=>c.method==='POST')");
  await evaluate("document.querySelector('[data-scenario-id]').click(); document.getElementById('build-question').click()");
  await check("unknown facts remain unknown in the editable draft", "document.getElementById('question').value.includes('not yet known') && !__preview.calls.some(c=>c.method==='POST')");
  await evaluate("await __preview.ui.openRecord(__preview.records[0].request_id); document.querySelector('.claim-select').click()");
  await check("selecting a statement shows its two cited sources", "document.querySelectorAll('#citations .citation').length===2 && document.querySelector('.claim.selected')!==null");
  await evaluate("document.querySelector('#citations button').click()");
  await waitFor("document.querySelector('#source-text mark') !== null");
  await check("source identity and exact quote are rendered", "document.querySelector('#source-text mark').textContent==='Synthetic evidence passage 1.' && document.getElementById('source-title').textContent.includes('Synthetic source 1')");
  await evaluate("document.getElementById('brief-notes').value='Local learning note — tiếng Việt'; document.getElementById('export-brief').click()");
  await waitFor("__preview.downloads.length===1");
  for (let i = 0; i < 50; i++) { try { if ((await readdir(path.join(output, "downloads"))).some(name => name.endsWith(".md"))) break; } catch {} await delay(100); }
  const filename = (await readdir(path.join(output, "downloads"))).find(name => name.endsWith(".md"));
  assert.ok(filename, "real browser download");
  const downloaded = await readFile(path.join(output, "downloads", filename), "utf8");
  assert.ok(downloaded.includes("Local learning note") && downloaded.includes("User notes") && downloaded.includes("Official AWS source")); checks.push("real browser UTF-8 Markdown download after detail reauthorization");
  await evaluate("__preview.realDownload=false");
  await check("source, history and export actions use GET only", "__preview.calls.every(c=>c.method==='GET') && __preview.calls.filter(c=>c.path===('/v1/queries/'+__preview.records[0].request_id)).length>=2");
  const desktop = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }); await writeFile(path.join(output, "desktop.png"), Buffer.from(desktop.data, "base64"));
  await evaluate("await __preview.ui.openRecord(__preview.records[1].request_id); document.getElementById('export-brief').click()");
  await waitFor("__preview.downloads.length===2");
  await check("PARTIAL is preserved in UI and brief", "document.getElementById('answer-badge').textContent==='Partial answer' && __preview.downloads.at(-1).text.includes('evidence is incomplete')");
  await evaluate("await __preview.ui.openRecord(__preview.records[2].request_id); document.getElementById('export-brief').click()");
  await waitFor("__preview.downloads.length===3");
  await check("ABSTAINED has no fabricated statements or sources", "!document.querySelector('.claim') && __preview.downloads.at(-1).text.includes('Not enough evidence') && !__preview.downloads.at(-1).text.includes('### Statement')");
  await evaluate("await __preview.ui.openRecord(__preview.records[3].request_id)");
  await check("PROCESSING cannot export an answer", "document.getElementById('brief-panel').hidden && document.getElementById('export-brief').disabled");
  await evaluate("await __preview.ui.openRecord(__preview.records[4].request_id)");
  await check("FAILED remains a failure instead of abstention", "document.getElementById('answer-badge').textContent==='Could not complete' && document.getElementById('brief-panel').hidden");
  await evaluate("await __preview.ui.openRecord(__preview.records[0].request_id); globalThis.sourceQueue=[]; __preview.overrides.source=()=>new Promise((resolve,reject)=>sourceQueue.push({resolve,reject})); globalThis.first=__preview.ui.evidence.selectCitation('citation-1'); globalThis.second=__preview.ui.evidence.selectCitation('citation-2'); const f=await import('/fixtures.mjs'); sourceQueue[1].resolve(f.fixtureSource(__preview.records[0],__preview.records[0].response.citations[1])); await second; sourceQueue[0].resolve(f.fixtureSource(__preview.records[0],__preview.records[0].response.citations[0])); await first");
  await check("late source A cannot replace the selected source B in the DOM", "document.getElementById('source-title').textContent.includes('Synthetic source 2')");
  await evaluate("__preview.overrides.source=()=>Promise.reject(Object.assign(new Error('synthetic'),{status:404})); await __preview.ui.evidence.selectCitation('citation-1')");
  await check("source failure clears the previous passage", "document.getElementById('source-text').textContent==='' && document.getElementById('source-status').textContent.includes('unavailable')");
  await evaluate("delete __preview.overrides.source; globalThis.bad=structuredClone(__preview.records[0]); bad.question='<img src=x onerror=globalThis.__xss=1>'; bad.response.claims[0].text='<script>globalThis.__xss=1</script>'; bad.response.answer=bad.response.claims.map(c=>c.text).join('\\n'); bad.response.citations[0].title='<img src=x onerror=globalThis.__xss=1>'; bad.response.citations[0].source_uri='javascript:alert(1)'; __preview.ui.renderRecord(bad); document.querySelector('.claim-select').click()");
  await check("untrusted HTML is text and javascript URLs are not links", "!globalThis.__xss && !document.querySelector('#workspace img,#workspace script') && ![...document.querySelectorAll('#workspace a')].some(a=>a.href.startsWith('javascript:')) && document.getElementById('answer-title').textContent.includes('<img')");
  await evaluate("await __preview.ui.openRecord(__preview.records[0].request_id); __preview.overrides.detail=()=>Promise.reject(Object.assign(new Error('revoked'),{status:503})); document.getElementById('export-brief').click()");
  await waitFor("document.getElementById('export-status').textContent.includes('No cached')");
  await check("revoked or unavailable detail prevents cached export", "__preview.downloads.length===3");
  await evaluate("delete __preview.overrides.detail; await __preview.ui.openRecord(__preview.records[0].request_id); globalThis.lateDetail=null; __preview.overrides.detail=()=>new Promise(resolve=>lateDetail=resolve); document.getElementById('export-brief').click(); __preview.ui.clear(); await __preview.ui.activate({load:false}); lateDetail(__preview.records[0]); await new Promise(resolve=>setTimeout(resolve,20))");
  await check("late export after logout/login never downloads or restores a record", "__preview.downloads.length===3 && document.getElementById('answer-panel').hidden && document.getElementById('brief-notes').value===''");
  await evaluate("delete __preview.overrides.detail; await __preview.ui.openRecord(__preview.records[0].request_id); globalThis.selectionExport=null; __preview.overrides.detail=()=>new Promise(resolve=>selectionExport=resolve); document.getElementById('export-brief').click(); __preview.ui.renderRecord(__preview.records[1]); selectionExport(__preview.records[0]); await new Promise(resolve=>setTimeout(resolve,20)); delete __preview.overrides.detail");
  await check("changing the selected record invalidates an in-flight export", "__preview.downloads.length===3 && document.getElementById('answer-badge').textContent==='Partial answer'");
  await evaluate("delete __preview.overrides.detail; globalThis.lateHistory=null; __preview.overrides.history=()=>new Promise(resolve=>lateHistory=resolve); globalThis.oldHistory=__preview.ui.loadHistory(); __preview.ui.clear(); await __preview.ui.activate({load:false}); lateHistory({items:[__preview.records[0]],next_cursor:null}); await oldHistory; delete __preview.overrides.history");
  await check("old-session history cannot appear in a new session", "document.querySelectorAll('.history-item').length===0");
  await evaluate("document.getElementById('new-question').click(); document.getElementById('question').value='Synthetic explicit submit'; globalThis.postsBefore=__preview.calls.filter(c=>c.method==='POST').length; globalThis.lateSubmit=null; __preview.overrides.submit=()=>new Promise(resolve=>lateSubmit=resolve); document.getElementById('submit').click(); document.getElementById('question-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))");
  await check("double submit admits one client intent and locks scenario editing", "__preview.calls.filter(c=>c.method==='POST').length===postsBefore+1 && document.querySelector('[data-scenario-id]').disabled");
  await evaluate("__preview.ui.clear(); await __preview.ui.activate({load:false}); lateSubmit(__preview.records[0]); await new Promise(resolve=>setTimeout(resolve,20)); delete __preview.overrides.submit");
  await check("a late query response after logout cannot restore its private answer", "document.getElementById('answer-panel').hidden && document.getElementById('history').textContent===''");
  await evaluate("document.getElementById('question').value='Synthetic timeout question'; __preview.overrides.submit=()=>Promise.reject(Object.assign(new Error('timeout'),{code:'REQUEST_OUTCOME_UNKNOWN'})); document.getElementById('question-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})); await new Promise(resolve=>setTimeout(resolve,20)); globalThis.afterTimeout=__preview.calls.filter(c=>c.method==='POST').length; document.getElementById('question-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})); await new Promise(resolve=>setTimeout(resolve,20))");
  await check("ambiguous timeout does not resubmit under a new key", "__preview.calls.filter(c=>c.method==='POST').length===afterTimeout && document.getElementById('query-status').textContent.includes('already submitted')");
  await evaluate("delete __preview.overrides.submit; await __preview.ui.openRecord(__preview.records[0].request_id); __preview.ui.chooseScenario('s3-lambda-read')");
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await check("390px layout has no horizontal page overflow", "document.documentElement.scrollWidth<=window.innerWidth");
  await evaluate("document.getElementById('question').focus()");
  await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
  await check("keyboard focus moves from question to the documentation source", "document.activeElement.id==='corpus-scope'");
  const mobile = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }); await writeFile(path.join(output, "mobile.png"), Buffer.from(mobile.data, "base64"));
  await evaluate(`globalThis.wideRecord=structuredClone(__preview.records[0]); wideRecord.response.retrieval_backend='aws_knowledge_mcp'; wideRecord.response.answer_language='vi'; wideRecord.response.evidence_bundle_sha256='a'.repeat(64); wideRecord.response.claims[0].text='Nguồn AWS chính thức giải thích quyền đọc và giải mã riêng biệt.'; wideRecord.response.answer=wideRecord.response.claims.map(c=>c.text).join('\\n'); __preview.overrides.source=async()=>({...wideRecord.response.citations[0],request_id:wideRecord.request_id,snapshot_version:wideRecord.snapshot_version,text:wideRecord.response.citations[0].quote,retrieved_at:'2026-10-18T12:00:00Z',normalizer_version:'aws-knowledge-search-excerpts-v2',evidence_bundle_sha256:'a'.repeat(64)}); __preview.ui.renderRecord(wideRecord); await __preview.ui.evidence.selectCitation('citation-1');`);
  await check("Vietnamese live record shows pinned source provenance and human-readable limits", "document.getElementById('answer-meta').textContent.includes('Tiếng Việt') && document.getElementById('source-meta').textContent.includes('Retrieved') && document.getElementById('source-meta').textContent.includes('Page update date unknown') && document.getElementById('answer').textContent.includes('Nguồn AWS')");
  for (const width of [1440,768,390]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height: width===390?844:1000, deviceScaleFactor:1, mobile:width===390 });
    await check(`live source inspector fits ${width}px`, "document.documentElement.scrollWidth<=window.innerWidth");
    await evaluate("document.getElementById('source-panel').scrollIntoView({block:'center'})");
    const shot = await send("Page.captureScreenshot", { format:"png", captureBeyondViewport:false });
    await writeFile(path.join(output, `live-source-${width}.png`), Buffer.from(shot.data,"base64"));
  }
  await evaluate("delete __preview.overrides.source");
  for (const code of [401, 403]) {
    await evaluate(`await __preview.ui.activate({load:false}); __preview.ui.renderRecord(__preview.records[0]); document.getElementById('brief-notes').value='private synthetic note'; __preview.overrides.source=()=>Promise.reject(Object.assign(new Error('synthetic authorization failure'),{status:${code}})); await __preview.ui.evidence.selectCitation('citation-1'); delete __preview.overrides.source`);
    await check(`source HTTP${code} clears the private workspace and notes`, "document.getElementById('workspace').hidden && document.getElementById('brief-notes').value==='' && document.getElementById('source-text').textContent==='' && document.getElementById('export-brief').disabled");
  }
  await evaluate("__preview.overrides.dashboard=()=>Promise.reject(new Error('missing projection')); await __preview.ui.activate({load:false})");
  await check("a missing lab projection does not prevent the existing investigation workspace", "document.getElementById('dashboard-status').textContent.includes('unavailable') && !document.getElementById('workspace').hidden");
  await evaluate("delete __preview.overrides.dashboard; globalThis.savedData=null; __preview.overrides.dashboard=()=>new Promise(resolve=>savedData=resolve); globalThis.oldDashboard=__preview.ui.activate({load:false}); __preview.ui.clear(); savedData(__preview.dashboardData); await oldDashboard; delete __preview.overrides.dashboard");
  await check("late recorded loads cannot restore any workspace after logout", "document.getElementById('workspace').hidden && document.getElementById('overview-history').textContent==='' && document.getElementById('overview-metrics').textContent===''");
  await evaluate("globalThis.poisonedCatalog=structuredClone(__preview.dashboardData); poisonedCatalog.documents[0].title='<img src=x onerror=globalThis.__catalogXSS=1>'; __preview.overrides.dashboard=async()=>poisonedCatalog; await __preview.ui.activate({load:false}); document.querySelector('[data-page=knowledge]').click(); document.querySelector('.document-card').click()");
  await check("public catalog content is literal text under production CSP", "!globalThis.__catalogXSS && !document.querySelector('#page-knowledge img,#page-knowledge script') && document.getElementById('catalog-detail').textContent.includes('<img')");
  await evaluate("delete __preview.overrides.dashboard");
  await evaluate("__preview.ui.clear()");
  await check("clear removes private DOM state and disables export", "document.getElementById('workspace').hidden && document.getElementById('source-text').textContent==='' && document.getElementById('answer').textContent==='' && document.getElementById('export-brief').disabled");
  assert.equal(errors.length, 0, "uncaught browser exceptions");
  const externalRequests = requests.filter(url => /^https?:/.test(url) && !url.startsWith(`http://127.0.0.1:${port}/`));
  assert.deepEqual(externalRequests, [], "no external HTTP browser requests; blob downloads are local");
  const report = { checked_at: new Date().toISOString(), status: "PASS_LOCAL_SYNTHETIC_BROWSER", browser: "installed Edge headless, isolated profile", production_security_headers: true, checks,
    uncaught_exceptions: errors, browser_network_requests: requests.length, page_request_urls: requests, external_page_requests: externalRequests.length, project_model_calls: 0, aws_calls: 0,
    scope: "production UI with mocked private loaders and recorded public development data, not AWS login or answer-quality evidence", screenshots: ["desktop.png", "mobile.png", "dashboard-desktop.png", "dashboard-mobile.png", "retrieval-lab.png"], real_download: filename };
  await writeFile(path.join(output, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, checks: checks.length, output: path.relative(root, output) }));
} catch (error) {
  await writeFile(path.join(output, "report.json"), JSON.stringify({ status: "FAIL_LOCAL_SYNTHETIC_BROWSER", checked_at: new Date().toISOString(), checks, error: error.message, uncaught_exceptions: errors }, null, 2));
  console.error(JSON.stringify({ status: "FAIL_LOCAL_SYNTHETIC_BROWSER", checks: checks.length, error: error.message, output: path.relative(root, output) })); process.exitCode = 1;
} finally {
  if (socket?.readyState === WebSocket.OPEN) { try { await send("Browser.close"); } catch {} socket.close(); }
  for (const entry of pending.values()) clearTimeout(entry.timer);
  server.kill(); browser.kill();
}
