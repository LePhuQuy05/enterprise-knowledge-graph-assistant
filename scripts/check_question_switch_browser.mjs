// Isolated browser + local-only fixtures; never signs in to AWS or calls a model.
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const out = path.join(root, "results/question-switch-2026-11-06-v1", stamp);
await mkdir(out, { recursive: true });
const origin = "http://127.0.0.1:8882", checks = [], errors = [], requests = [];
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
  socket.addEventListener("message", e => { const m=JSON.parse(e.data); if(m.method==="Runtime.exceptionThrown")errors.push(m.params.exceptionDetails.exception?.description||m.params.exceptionDetails.text);if(pending.has(m.id)){const x=pending.get(m.id);clearTimeout(x.timer);pending.delete(m.id);m.error?x.reject(new Error(m.error.message)):x.resolve(m.result);} });
  await send("Page.enable"); await send("Runtime.enable");
  await send("Page.navigate",{url:origin});
  await wait("!!globalThis.__scopedPreview && document.querySelector('#compare-corpus').textContent.includes('319 documents')");
  await evaluate("document.querySelector('#tour-skip')?.click()");
  await submitMethod("graph_2hop","graph");
  await evaluate("__preview.ui.dashboard.show('compare',true);globalThis.__nativeBroker=__preview.overrides.scopedRuns;const q=__scopedFixture.cases.find(c=>c.id==='vietnamese').run.request.original_question;document.querySelector('#compare-question').value=q;document.querySelector('#compare-question').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#compare-language').value='vi';for(const input of document.querySelectorAll('#compare-methods input'))input.checked=input.value==='hybrid';document.querySelector('#compare-methods input[value=hybrid]').dispatchEvent(new Event('change',{bubbles:true}))");
  await check("editing the next question clears the previous completed result", "document.querySelectorAll('#compare-results article').length===0 && !document.querySelector('#compare-submit').disabled");
  await evaluate("document.querySelector('#compare-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))");
  await wait("document.querySelector('#compare-run-meta').textContent.startsWith('COMPLETED')");
  await check("second question publishes its own language and results", "document.querySelector('#compare-question').value===__scopedFixture.cases.find(c=>c.id==='vietnamese').run.request.original_question && document.querySelector('#compare-results').textContent.includes('Hybrid') && __scopedPreview.posts.at(-1).body.language==='vi'");
  await evaluate("__scopedPreview.delaySource=true;document.querySelector('#compare-results .collection-claim button').click()");
  await wait("__scopedPreview.sourceResolvers.length>0");
  await evaluate("globalThis.__postCount=__scopedPreview.posts.length;document.querySelector('#compare-new').click()");
  await evaluate("__scopedPreview.sourceResolvers.splice(0).forEach(fn=>fn());__scopedPreview.delaySource=false");
  await check("late source from the previous question cannot appear in a fresh draft", "!document.querySelector('#compare-source blockquote') && document.querySelector('#compare-trace').textContent===''");
  await check("new question preserves history and methods without model submission", "document.querySelector('#compare-question').value==='' && document.querySelector('#compare-language').value==='vi' && document.querySelector('#compare-methods input[value=hybrid]').checked && document.querySelectorAll('#compare-history button').length>=2 && __scopedPreview.posts.length===__postCount && document.querySelectorAll('#compare-results article').length===0");
  await evaluate("globalThis.__releaseHistory=null;const path='/v2/scoped-runs/'+__scopedFixture.cases.find(c=>c.id==='graph').run.run_id;__preview.overrides.scopedRuns=async(p,o={})=>p===path?new Promise(resolve=>{__releaseHistory=async()=>resolve(await __nativeBroker(p,o))}):__nativeBroker(p,o);void __preview.ui.compare.open(__scopedFixture.cases.find(c=>c.id==='graph').run.run_id)");
  await wait("typeof __releaseHistory==='function'");
  await evaluate("document.querySelector('#compare-question').value='A fresh draft while history is loading';document.querySelector('#compare-question').dispatchEvent(new Event('input',{bubbles:true}));await __releaseHistory();__preview.overrides.scopedRuns=__nativeBroker");
  await check("late history cannot overwrite a newly edited question", "document.querySelector('#compare-question').value==='A fresh draft while history is loading' && document.querySelector('#compare-run-meta').textContent===''");
  await evaluate("__preview.ui.dashboard.show('compare',true); const original=__preview.overrides.scopedRuns; __preview.overrides.scopedRuns=async (path,opts={})=>{if(path==='/v2/scoped-runs'&&opts.body){globalThis.__lastRejected=opts;const {ApiFailure}=await import('/assets/client.mjs');throw new ApiFailure('BUDGET_LIMIT',429);}return original(path,opts)}; document.querySelector('#compare-question').value='Which IAM policy is required for cross-account KMS access?'; document.querySelector('#compare-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))");
  await wait("document.querySelector('#compare-status').textContent.includes('limit')");
  const state=await evaluate("return {status:document.querySelector('#compare-status').textContent,submitDisabled:document.querySelector('#compare-submit').disabled,recoveryVisible:!document.querySelector('#compare-recover').hidden,oldResultCards:document.querySelectorAll('#compare-results article').length,question:document.querySelector('#compare-question').value}");
  await writeFile(path.join(out,'repro.json'),JSON.stringify(state,null,2));console.log(JSON.stringify(state));
  assert.equal(state.submitDisabled,false,'A definitive budget rejection must let a completed-run user edit and submit a new question');
  assert.equal(state.recoveryVisible,false,'Definitive rejection must not trap the draft behind same-submission recovery');
  checks.push("known budget rejection releases the next-question draft and clears stale answers");
  await evaluate("__preview.overrides.scopedRuns=__nativeBroker;__scopedPreview.ambiguous=true;document.querySelector('#compare-question').value=__scopedFixture.cases.find(c=>c.id==='graph').run.request.original_question;document.querySelector('#compare-question').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('#compare-language').value='en';for(const input of document.querySelectorAll('#compare-methods input'))input.checked=input.value==='graph_2hop';document.querySelector('#compare-methods input[value=graph_2hop]').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#compare-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))");
  await wait("!document.querySelector('#compare-recover').hidden");
  await check("ambiguous submission freezes the draft and new-question action", "document.querySelector('#compare-question').disabled && document.querySelector('#compare-new').disabled && __preview.ui.compare.newQuestion()===false");
  await evaluate("globalThis.__frozen=structuredClone(__scopedPreview.posts.at(-1));__preview.overrides.scopedRuns=async(p,o={})=>{if(p==='/v2/scoped-runs'&&o.body){const {ApiFailure}=await import('/assets/client.mjs');throw new ApiFailure('BUDGET_LIMIT',429)}return __nativeBroker(p,o)};document.querySelector('#compare-recover').click()");
  await wait("!document.querySelector('#compare-recover').hidden && document.querySelector('#compare-protocol').disabled && document.querySelector('#compare-status').textContent.includes('limit')");
  await check("recovery rejection cannot discard a possibly admitted original intent", "document.querySelector('#compare-new').disabled && !document.querySelector('#compare-recover').hidden");
  await evaluate("__preview.overrides.scopedRuns=__nativeBroker;document.querySelector('#compare-recover').click()");
  await wait("document.querySelector('#compare-recover').hidden");
  await check("manual recovery retains exactly the original body and key", "JSON.stringify(__scopedPreview.posts.at(-1))===JSON.stringify(__frozen)");
  await wait("document.querySelector('#compare-run-meta').textContent.startsWith('COMPLETED')");
  await evaluate("for(const n of document.querySelectorAll('#compare-methods input'))n.checked=['bm25','vector','hybrid'].includes(n.value);document.querySelector('#compare-methods input[value=hybrid]').dispatchEvent(new Event('change',{bubbles:true}))");
  await check("three-method limit is visible before submission", "document.querySelector('#compare-selection-status').textContent.startsWith('3 / 3') && document.querySelector('#compare-methods input[value=hybrid_reranked]').disabled");
  await evaluate("document.querySelector('#compare-methods input[value=vector]').click()");
  await check("unchecking one method permits a replacement", "!document.querySelector('#compare-methods input[value=hybrid_reranked]').disabled && document.querySelector('#compare-selection-status').textContent.startsWith('2 / 3')");
  for (const width of [1440,768,390]) {
    await send('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false});
    await evaluate("document.querySelector('#compare-new').scrollIntoView({block:'center'})");
    await check(`question controls fit ${width}px`, `document.documentElement.scrollWidth<=innerWidth && document.querySelector('#compare-new').getBoundingClientRect().width>0`);
    await screenshot(`question-editor-${width}.png`);
  }
  await writeFile(path.join(out,'report.json'),JSON.stringify({status:'PASS_QUESTION_SWITCH',checks,aws_calls:0,model_calls:0,scope:'Actual browser/controller/production transport with mocked admission and saved-source fixtures'},null,2));
  console.log(JSON.stringify({status:'PASS_QUESTION_SWITCH',checks:checks.length,out}));
  console.log('PASS_QUESTION_SWITCH_AFTER_DEFINITIVE_REJECTION');
} catch(error) { console.log(JSON.stringify({browserErrors:errors,error:String(error)})); await writeFile(path.join(out,'failure.json'),JSON.stringify({browserErrors:errors,error:String(error)})); throw error;} finally {if(socket?.readyState===WebSocket.OPEN)socket.close();browser.kill();}
