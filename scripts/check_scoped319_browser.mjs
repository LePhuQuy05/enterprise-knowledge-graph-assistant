// Isolated browser + local-only fixtures; never signs in to AWS or calls a model.
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const out = path.join(root, "results/scoped-ui319-2026-11-05-v1/browser", stamp);
await mkdir(out, { recursive: true });
const origin = process.env.KG_UI_PREVIEW_ORIGIN || "http://127.0.0.1:8881", checks = [], errors = [], requests = [];
if (!/^http:\/\/127\.0\.0\.1:\d{4,5}$/.test(origin)) throw new Error("LOCAL_FIXTURE_ORIGIN_REQUIRED");
const pause = ms => new Promise(r => setTimeout(r, ms));
const portServer = createServer(); await new Promise(r => portServer.listen(0, "127.0.0.1", r));
const port = portServer.address().port; await new Promise(r => portServer.close(r));
const browser = spawn("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", ["--headless=new", "--disable-gpu", "--no-first-run",
  "--no-default-browser-check", "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-sync",
  `--remote-debugging-port=${port}`, "--remote-debugging-address=127.0.0.1", `--user-data-dir=${path.join(root, "data/build", "scoped-browser-" + stamp)}`, "about:blank"],
  { windowsHide: true, stdio: "ignore" });
let socket, sequence = 0;
const pending = new Map();
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence, timer = setTimeout(() => { pending.delete(id); reject(new Error("CDP_TIMEOUT:" + method)); }, 20000);
    pending.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(source) {
  const r = await send("Runtime.evaluate", { expression: `(async()=>{${source}\n})()`, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}
async function check(name, condition) { assert.equal(await evaluate(`return (${condition})`), true, name); checks.push(name); }
async function wait(condition) {
  for (let i = 0; i < 150; i++) { if (await evaluate(`return (${condition})`)) return; await pause(100); }
  throw new Error("BROWSER_WAIT:" + condition);
}
async function screenshot(name) {
  const r = await send("Page.captureScreenshot", { format: "png", fromSurface: true });
  await writeFile(path.join(out, name), Buffer.from(r.data, "base64"));
}
async function submitMethod(method, group, completed = true) {
  await evaluate(`__preview.ui.dashboard.show('method-${method}',true); const root=document.querySelector('#page-method-${method}');
    root.querySelector('textarea').value=__scopedFixture.cases.find(c=>c.id==='${group}').run.request.original_question;
    root.querySelector('.method-language').value='${group === "vietnamese" ? "vi" : "en"}';
    root.querySelector('.method-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));`);
  if (completed) await wait(`document.querySelector('#page-method-${method} .method-status').textContent.startsWith('COMPLETED')`);
}
try {
  let target;
  for (let i = 0; i < 100; i++) { try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(x => x.type === "page"); } catch {} if (target) break; await pause(100); }
  if (!target) throw new Error("BROWSER_NOT_READY");
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener("open", resolve, { once: true }); socket.addEventListener("error", reject, { once: true }); });
  socket.addEventListener("message", e => {
    const m = JSON.parse(e.data);
    if (pending.has(m.id)) { const x = pending.get(m.id); clearTimeout(x.timer); pending.delete(m.id); m.error ? x.reject(new Error(m.error.message)) : x.resolve(m.result); }
    if (m.method === "Runtime.exceptionThrown") errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === "Network.requestWillBeSent") requests.push(m.params.request.url);
  });
  await send("Page.enable"); await send("Runtime.enable"); await send("Network.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: origin });
  await wait("!!globalThis.__scopedPreview && document.querySelector('#compare-corpus').textContent.includes('319 documents')");
  await evaluate("document.querySelector('#tour-skip')?.click();");
  await check("explicit V2 protocol and 319/3024 sparse graph inventory", "!document.querySelector('#compare-protocol-control').hidden && document.querySelector('#compare-corpus').textContent.includes('3024 chunks') && document.querySelector('#compare-corpus').textContent.includes('27 experimental')");
  await check("all eight methods selectable and V2 retrieval-only submission disabled", "document.querySelectorAll('#compare-methods input:not(:disabled)').length===8 && document.querySelector('#compare-kind').value==='answer' && document.querySelector('#compare-kind option[value=retrieval]').disabled");
  await check("opening workspace does not submit a run", "__scopedPreview.posts.length===0");
  await submitMethod("graph_2hop", "graph");
  await check("Graph2 actual packed facts are visible without fallback", "document.querySelectorAll('#page-method-graph_2hop .scoped-graph-node').length>1 && !document.querySelector('#page-method-graph_2hop .method-graph-area').hidden && document.querySelector('#page-method-graph_2hop .method-graph-counts').textContent.includes('/319') && document.querySelector('#page-method-graph_2hop .scoped-qualification').dataset.qualification==='NO_MECHANICAL_FLAGS'");
  await check("six accessible method steps from saved execution", "document.querySelectorAll('#page-method-graph_2hop .method-step').length===6 && [...document.querySelectorAll('#page-method-graph_2hop .method-step')].every(b=>b.getAttribute('aria-controls')==='method-graph_2hop-detail')");
  for (const i of [0, 1, 2, 3, 4, 5]) { await evaluate(`document.querySelectorAll('#page-method-graph_2hop .method-step')[${i}].click()`); await check(`Graph step ${i + 1} renders actual content`, `document.querySelector('#method-graph_2hop-detail').textContent.includes('STEP ${i + 1} / 6')`); }
  await evaluate("document.querySelectorAll('#page-method-graph_2hop .method-step')[4].click(); document.querySelector('#page-method-graph_2hop .collection-claim button').click();");
  await wait("!!document.querySelector('#page-method-graph_2hop .method-source blockquote')");
  await check("citation opens owner-run source route and original AWS link", "__scopedPreview.requests.some(p=>p.includes('/v2/scoped-runs/')&&p.includes('/sources/')) && document.querySelector('#page-method-graph_2hop .method-source a').href.startsWith('https://docs.aws.amazon.com/')");
  for (const width of [1440, 768, 390]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height: 1000, deviceScaleFactor: 1, mobile: width === 390 });
    await evaluate("document.querySelector('#page-method-graph_2hop .method-run-header').scrollIntoView({block:'start'});");
    await pause(150); await screenshot(`graph-answer-${width}.png`);
    await check(`no global horizontal overflow at ${width}px`, "document.documentElement.scrollWidth<=document.documentElement.clientWidth+1");
    await evaluate("document.querySelector('#page-method-graph_2hop .method-graph-area').scrollIntoView({block:'start'});");
    await pause(100); await screenshot(`graph-map-${width}.png`);
    await check(`graph SVG is rendered in the viewport at ${width}px`, "document.querySelector('#page-method-graph_2hop .scoped-graph-scene svg').getBoundingClientRect().width>0 && document.querySelector('#page-method-graph_2hop .method-graph-area').getBoundingClientRect().top<innerHeight");
    await check(`source-backed map and accessible facts at ${width}px`, "document.querySelectorAll('#page-method-graph_2hop .scoped-fact').length>0 && document.querySelector('#page-method-graph_2hop .scoped-graph-scene svg').getAttribute('aria-label').includes('Accessible source list')");
    await evaluate("window.scrollTo(0,document.body.scrollHeight);");
    await check(`main content can reach bottom at ${width}px`, "window.scrollY+window.innerHeight>=document.documentElement.scrollHeight-3");
  }
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  for (const [method, group] of [["bm25", "baseline"], ["vector", "baseline"], ["hybrid", "baseline"], ["hybrid_reranked", "graph"], ["graph_1hop", "graph"], ["hybrid_multi_query", "queries"], ["hybrid_multi_query_balanced", "queries"]]) {
    await submitMethod(method, group); await check(`${method} V2 method page has own completed trace`, `document.querySelector('#page-method-${method} .method-record-meta').textContent.includes('319 documents') && document.querySelectorAll('#page-method-${method} .method-step').length===6`);
  }
  await submitMethod("hybrid_multi_query_balanced", "vietnamese");
  await evaluate("document.querySelectorAll('#page-method-hybrid_multi_query_balanced .method-step')[1].click();");
  await check("Vietnamese question retains recorded English facet inputs", "document.querySelector('#page-method-hybrid_multi_query_balanced .method-example-question').textContent.includes('Những policy') && document.querySelector('#method-hybrid_multi_query_balanced-detail').textContent.includes('policies')");
  await evaluate("__scopedPreview.held=true;"); await submitMethod("hybrid", "held");
  await check("held answer has flags without public claim text", "document.querySelector('#page-method-hybrid .scoped-qualification').dataset.qualification==='REVIEW_REQUIRED' && !document.querySelector('#method-hybrid-detail').textContent.includes('Lambda can invoke Lambda.') && document.querySelectorAll('#method-hybrid-detail .collection-claim').length===0");
  await evaluate("__preview.ui.dashboard.show('compare',true); document.querySelector('#compare-export').click();");
  await wait("__preview.downloads.length>0");
  await check("export is V2 with null held response and no raw receipt", "JSON.parse(__preview.downloads.at(-1).text).schema_version===2 && JSON.parse(__preview.downloads.at(-1).text).results[0].response===null && !__preview.downloads.at(-1).text.includes('raw_output') && !__preview.downloads.at(-1).text.includes('Lambda can invoke Lambda.')");
  await evaluate("__scopedPreview.held=false; __scopedPreview.notReady=true;"); await submitMethod("graph_1hop", "graph");
  await check("NOT_READY keeps no answer and an explicit zero-path map", "document.querySelector('#page-method-graph_1hop .scoped-qualification').dataset.qualification==='NOT_READY' && document.querySelectorAll('#page-method-graph_1hop .scoped-graph-node').length===0 && document.querySelector('#page-method-graph_1hop .method-detail').textContent.includes('Generation was not run')");
  await evaluate("__scopedPreview.notReady=false; __scopedPreview.ambiguous=true;");
  await submitMethod("hybrid_multi_query", "queries", false);
  await wait("!document.querySelector('#compare-recover').hidden");
  await check("ambiguous POST preserves explicit same-key recovery", "!document.querySelector('#compare-recover').hidden && document.querySelector('#compare-protocol').disabled");
  await evaluate("document.querySelector('#compare-recover').click();");
  await wait("document.querySelector('#compare-recover').hidden");
  await check("manual recovery reuses exactly the frozen request key/body", "__scopedPreview.posts.at(-1).key===__scopedPreview.posts.at(-2).key && JSON.stringify(__scopedPreview.posts.at(-1).body)===JSON.stringify(__scopedPreview.posts.at(-2).body)");
  await evaluate("__preview.ui.dashboard.show('compare',true); document.querySelector('#compare-question').value=__scopedFixture.cases.find(c=>c.id==='graph').run.request.original_question; document.querySelector('#compare-language').value='en'; for(const i of document.querySelectorAll('#compare-methods input'))i.checked=['hybrid_reranked','graph_1hop','graph_2hop'].includes(i.value); document.querySelector('#compare-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));");
  await wait("document.querySelectorAll('#compare-results .scoped-qualification').length===3");
  await check("three V2 methods compare without automatic extra submission", "document.querySelectorAll('#compare-results article.compare-result').length===3 && __scopedPreview.posts.at(-1).body.method_ids.length===3");
  await evaluate("globalThis.__postsAtInspect=__scopedPreview.posts.length; [...document.querySelectorAll('#compare-results article.compare-result')].find(c=>c.querySelector('h3').textContent==='Graph · 1 hop').querySelector('button').click(); document.querySelector('#page-method-graph_1hop .collection-claim button').click();");
  await wait("!!document.querySelector('#page-method-graph_1hop .method-source blockquote')");
  await check("compare inspector uses the selected method source, not the first method", "__scopedPreview.requests.at(-1).includes('/methods/graph_1hop/sources/') && __scopedPreview.posts.length===__postsAtInspect && !document.querySelector('#page-method-graph_1hop').hidden");
  await evaluate("__scopedPreview.paused=true; __preview.ui.dashboard.show('compare',true); document.querySelector('#compare-question').value=__scopedFixture.cases.find(c=>c.id==='baseline').run.request.original_question; for(const i of document.querySelectorAll('#compare-methods input'))i.checked=i.value==='bm25'; document.querySelector('#compare-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));");
  await wait("!document.querySelector('#compare-cancel').disabled");
  await evaluate("document.querySelector('#compare-cancel').click();");
  await wait("document.querySelector('#compare-run-meta').textContent.startsWith('CANCELLED')");
  await check("cancellation targets V2 and stops remaining work in the UI", "__scopedPreview.requests.some(p=>p.endsWith('/cancel')) && document.querySelector('#compare-cancel').disabled && document.querySelectorAll('#compare-results .collection-claim').length===0");
  await evaluate("__scopedPreview.paused=false;");
  await evaluate("__preview.ui.dashboard.show('compare',true); document.querySelector('#compare-protocol').value='legacy'; document.querySelector('#compare-protocol').dispatchEvent(new Event('change'));");
  await wait("document.querySelector('#compare-corpus').textContent.includes('316 documents')");
  await check("legacy selection keeps old routes and hides V2-only methods", "document.querySelector('#compare-kind').value==='retrieval' && [...document.querySelectorAll('#compare-methods input')].filter(i=>!i.closest('label').hidden).length===4 && __preview.calls.some(c=>c.path==='/v1/run-capabilities')");
  await evaluate("__preview.overrides.methods=()=>new Promise(resolve=>globalThis.__releaseHistorical=resolve); __preview.ui.dashboard.show('method-graph_2hop',true);");
  await wait("typeof globalThis.__releaseHistorical==='function'");
  await evaluate("document.querySelector('#compare-protocol').value='scoped-v2'; document.querySelector('#compare-protocol').dispatchEvent(new Event('change'));");
  await wait("document.querySelector('#compare-corpus').textContent.includes('319 documents')");
  const historical = JSON.parse(await readFile(path.join(root, "frontend/tracer/methods-data.json"), "utf8"));
  await evaluate(`__releaseHistorical(${JSON.stringify(historical)}); delete __preview.overrides.methods;`);
  await check("late historical316 example cannot overwrite the new319 workflow", "document.querySelector('#page-method-graph_2hop').dataset.protocol==='scoped-v2' && document.querySelector('#page-method-graph_2hop .method-record-meta').textContent.includes('319 documents')");
  await evaluate("__scopedPreview.unavailable=true; globalThis.__callsBefore=__preview.calls.length; await __preview.ui.compare.activate();");
  await check("V2 outage does not silently fetch legacy or POST another method", "document.querySelector('#compare-status').textContent.includes('unavailable') && __preview.calls.slice(__callsBefore).every(c=>c.path.startsWith('/v2/')) && document.querySelector('#compare-submit').disabled");
  await evaluate("__scopedPreview.unavailable=false; await __preview.ui.compare.activate();");
  await submitMethod("graph_2hop", "graph");
  await evaluate("__scopedPreview.badSource=true; document.querySelector('#page-method-graph_2hop .collection-claim button').click();");
  await wait("document.querySelector('#page-method-graph_2hop .method-source').textContent.includes('chưa được tải')");
  await check("changed source bytes clear the previous passage", "!document.querySelector('#page-method-graph_2hop .method-source blockquote') && !document.querySelector('#compare-source blockquote')");
  await evaluate("__scopedPreview.badSource=false; __scopedPreview.delaySource=true; document.querySelector('#page-method-graph_2hop .collection-claim button').click();");
  await wait("__scopedPreview.sourceResolvers.length>0");
  await evaluate("__preview.ui.clear(); __scopedPreview.sourceResolvers.splice(0).forEach(f=>f());");
  await check("late private source cannot survive session clear", "!document.querySelector('.method-source blockquote') && !document.querySelector('#compare-source blockquote') && document.querySelector('#workspace').hidden");
  await evaluate("localStorage.removeItem('kg.ui.onboarding.v1')");
  await send("Page.navigate", { url: origin + "/auth-start.html" });
  await wait("globalThis.__authStartHarness?.ready===true");
  await check("real startApp callback reaches V2 through production apiRequest", "__authStartHarness.calls.some(c=>c.path==='/v2/scoped-run-capabilities') && document.querySelector('#compare-corpus').textContent.includes('319 documents') && !document.querySelector('#workspace').hidden");
  await check("first-session quick tour can close after the full callback startup", "document.querySelector('#workspace-tour').open");
  await evaluate("document.querySelector('#tour-skip').click()");
  await check("closed startup tour releases the private workspace", "!document.querySelector('#workspace-tour').open && !document.body.classList.contains('tour-open')");
  await check("all14 navigation pages respond after actual startup initialization", "['overview','investigate','knowledge','lab','compare','benchmark316','method-bm25','method-vector','method-hybrid','method-hybrid_reranked','method-graph_1hop','method-graph_2hop','method-hybrid_multi_query','method-hybrid_multi_query_balanced'].every(page=>{document.querySelector('[data-page=\"'+page+'\"]').click();return document.querySelector('[data-page=\"'+page+'\"]').getAttribute('aria-current')==='page'})");
  await evaluate("document.querySelector('[data-page=compare]').click(); document.querySelector('#compare-history button').click()");
  await wait("document.querySelectorAll('#compare-results button').length>0");
  await check("callback history uses production transport without submitting another question", "__authStartHarness.calls.some(c=>/\\/v2\\/scoped-runs\\/run_/.test(c.path)) && !__authStartHarness.calls.some(c=>c.method==='POST'&&c.path.startsWith('/v2/'))");
  await evaluate("[...document.querySelectorAll('#compare-results button')].filter(b=>b.textContent==='Inspect this method').at(-1).click(); document.querySelector('#page-method-graph_2hop .collection-claim button').click()");
  await wait("!!document.querySelector('#page-method-graph_2hop .method-source blockquote')");
  await check("callback Graph source opens through real transport", "__authStartHarness.calls.some(c=>c.path.includes('/methods/graph_2hop/sources/')) && !!document.querySelector('#page-method-graph_2hop .method-source blockquote')");
  const external = requests.filter(u => /^https?:/.test(u) && !u.startsWith(origin + "/"));
  assert.deepEqual(errors, [], "uncaught exceptions"); assert.deepEqual(external, [], "external HTTP requests");
  const report = { status: "PASS_SCOPED319_LOCAL_BROWSER", checked_at: new Date().toISOString(), checks, uncaught_exceptions: errors,
    external_page_requests: external.length, provider_calls: 0, aws_calls: 0, origin, browser: "isolated installed Edge headless",
    scope: "actual319 sources/retrieval with simulated provider/account/queue; Q/QB replay original native plans; not live auth or answer accuracy" };
  await writeFile(path.join(out, "report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, checks: checks.length, directory: out }));
} catch (e) {
  await writeFile(path.join(out, "report.json"), JSON.stringify({ status: "FAIL_SCOPED319_LOCAL_BROWSER", checks, error: e.message, uncaught_exceptions: errors }, null, 2));
  console.error(JSON.stringify({ status: "FAIL_SCOPED319_LOCAL_BROWSER", checks: checks.length, error: e.message, directory: out })); process.exitCode = 1;
} finally {
  if (socket?.readyState === WebSocket.OPEN) { try { await send("Browser.close"); } catch {} socket.close(); }
  for (const item of pending.values()) clearTimeout(item.timer);
  browser.kill();
}
