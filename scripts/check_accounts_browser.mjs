// Local synthetic browser checks. Uses the installed Edge and Node WebSocket; no npm dependencies or model calls.
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const output = path.join(root, "results", "account-registration", "browser-" + stamp);
await mkdir(output, { recursive: true });
const checks = [], errors = [], requests = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function freePort() {
  const server = createServer(); await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port; await new Promise(resolve => server.close(resolve)); return port;
}
const port = await freePort(), debugPort = await freePort();
const server = spawn(path.join(root, ".venv", "Scripts", "python.exe"), ["-m", "scripts.preview_accounts", "--port", String(port)], { cwd: root, windowsHide: true, stdio: "ignore" });
const profile = path.join(root, "data/build", `account-browser-${stamp}`);
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

  async function navigate(state = "signedout") {
    try { await evaluate("globalThis.__accountPreview=null"); } catch {}
    await send("Page.navigate", { url: `http://127.0.0.1:${port}/?state=${state}` });
    await waitFor("globalThis.__accountPreview?.ready === true");
  }
  async function capture(name, width, height=1000) {
    await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width === 390 });
    await check(`${name}: no horizontal overflow at ${width}px`, "document.documentElement.scrollWidth <= innerWidth");
    const size = await send("Page.getLayoutMetrics");
    const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
      clip: { x:0, y:0, width, height:Math.ceil(size.cssContentSize.height), scale:1 } });
    await writeFile(path.join(output, name + ".png"), Buffer.from(shot.data,"base64"));
  }
  await navigate();
  await check("signed-out access is discoverable and offers login/signup/recovery", "document.getElementById('account-entry').getAttribute('href')==='#account-access' && !document.getElementById('signup').hidden && !document.getElementById('forgot-password').hidden && document.getElementById('workspace').hidden");
  await capture("desktop",1440);
  await capture("tablet",768);
  await capture("mobile",390,844);
  await evaluate("document.getElementById('login').focus()");
  await send("Input.dispatchKeyEvent", { type:"keyDown", key:"Tab", code:"Tab", windowsVirtualKeyCode:9 });
  await send("Input.dispatchKeyEvent", { type:"keyUp", key:"Tab", code:"Tab", windowsVirtualKeyCode:9 });
  await check("keyboard reaches create account after sign in", "document.activeElement.id === 'signup'");
  await evaluate("document.getElementById('signup').click()");
  await waitFor("document.getElementById('registration-dialog').open");
  await check("signup collects separate required username, email and website password", "document.getElementById('register-username').required && document.getElementById('register-email').required && document.getElementById('register-email').type==='email' && document.getElementById('register-password').required && __accountPreview.navigations.length===0 && document.activeElement.id==='register-username'");
  await capture("registration-desktop",1440);
  await capture("registration-mobile",390,844);
  await evaluate("document.getElementById('register-email').value='fixture@example.invalid';document.getElementById('register-password').value='Fixture-only-Password-7!';document.getElementById('registration-form').requestSubmit()");
  await check("email and password alone cannot submit without a username", "!document.getElementById('registration-form').checkValidity() && (__accountPreview.registrationCalls?.length || 0)===0 && document.getElementById('registration-verify').hidden");
  await evaluate("document.getElementById('register-username').value='fixture.user';document.getElementById('register-email').value='';document.getElementById('registration-form').requestSubmit()");
  await check("username and password alone cannot submit without an email", "!document.getElementById('registration-form').checkValidity() && (__accountPreview.registrationCalls?.length || 0)===0");
  await evaluate("document.getElementById('register-email').value='fixture@example.invalid';document.getElementById('registration-form').requestSubmit()");
  await waitFor("!document.getElementById('registration-verify').hidden");
  await check("signup moves to verification by username and clears the password with no auto resend", "document.getElementById('register-password').value==='' && document.getElementById('register-resend').disabled && __accountPreview.registrationCalls.length===1 && __accountPreview.registrationCalls[0].username==='fixture.user' && document.getElementById('verify-username').value==='fixture.user'");
  await capture("verification-mobile",390,844);
  await evaluate("document.getElementById('register-code').value='000000';document.getElementById('verification-form').requestSubmit()");
  await waitFor("document.getElementById('registration-status').textContent.includes('not correct')");
  await check("a wrong email code keeps verification visible", "!document.getElementById('registration-verify').hidden && document.getElementById('registration-done').hidden");
  await evaluate("document.getElementById('register-code').value='123456';document.getElementById('verification-form').requestSubmit()");
  await waitFor("!document.getElementById('registration-done').hidden");
  await check("verification alone never creates a local access token or opens the workspace", "document.getElementById('workspace').hidden && __accountPreview.navigations.length===0");
  await evaluate("document.getElementById('register-signin').click()");
  await waitFor("__accountPreview.navigations.length === 1");
  await check("verified user explicitly begins a fresh OAuth PKCE sign-in", "new URL(__accountPreview.navigations[0]).pathname==='/oauth2/authorize' && new URL(__accountPreview.navigations[0]).searchParams.get('code_challenge_method')==='S256' && !document.getElementById('registration-dialog').open");
  await check("no password or tokens persisted by the application", "Object.keys(sessionStorage).every(k=>!/(token|password)/i.test(k)) && Object.keys(localStorage).every(k=>!/(token|password)/i.test(k))");
  await navigate(); await evaluate("document.getElementById('forgot-password').click()");
  await waitFor("__accountPreview.navigations.length === 1");
  await check("recovery uses the Cognito reset endpoint with PKCE", "new URL(__accountPreview.navigations[0]).pathname==='/forgotPassword' && new URL(__accountPreview.navigations[0]).searchParams.get('code_challenge_method')==='S256'");
  await navigate("invite");
  await check("old deployments never expose a non-working sign-up control", "document.getElementById('signup').hidden && !document.getElementById('login').disabled");
  for (const state of ["unverified", "disabled", "expired"]) {
    await navigate(state);
    await check(`${state}: no workspace or history is loaded`, "document.getElementById('workspace').hidden && __accountPreview.calls.length===1");
    if(state !== "expired") await check(`${state}: sign-out remains available`, "!document.getElementById('logout').hidden");
    else await check("expired access returns to sign in", "!document.getElementById('login').hidden && document.getElementById('logout').hidden");
  }
  await navigate("setup");
  await capture("setup-retry-mobile",390,844);
  await check("setup failure offers manual retry and no background polling", "!document.getElementById('check-access').hidden && !document.getElementById('logout').hidden && __accountPreview.statusAttempts===1");
  await evaluate("document.getElementById('check-access').click()");
  await waitFor("!document.getElementById('workspace').hidden");
  await check("manual account retry recovers to the existing private workspace", "__accountPreview.statusAttempts===2 && document.getElementById('signup').hidden && document.getElementById('check-access').hidden");
  if(await evaluate("return !!document.getElementById('workspace-tour')?.open")) await evaluate("document.getElementById('tour-skip').click()");
  await capture("ready-mobile",390,844);
  await check("all four existing dashboard views remain", "document.querySelectorAll('.sidebar [data-page]').length===4");
  await evaluate("document.getElementById('logout').click()");
  await check("logout clears private workspace and opens Cognito logout", "document.getElementById('workspace').hidden && new URL(__accountPreview.navigations[0]).pathname==='/logout' && !sessionStorage.getItem('kg.oauth')");
  assert.deepEqual(errors,[]);
  const external=requests.filter(u=>/^https?:/.test(u)&&!u.startsWith(`http://127.0.0.1:${port}/`));
  assert.deepEqual(external,[]);
  const report={status:"PASS_SYNTHETIC_ACCOUNT_BROWSER",checked_at:new Date().toISOString(),checks,uncaught_exceptions:errors,external_page_requests:external.length,model_calls:0,aws_calls:0,
    screenshots:["desktop.png","tablet.png","mobile.png","registration-desktop.png","registration-mobile.png","verification-mobile.png","setup-retry-mobile.png","ready-mobile.png"],
    scope:"Production account UI with synthetic identity/status; not real Cognito email delivery, OTP, JWT or production signup acceptance"};
  await writeFile(path.join(output,"report.json"),JSON.stringify(report,null,2));
  console.log(JSON.stringify({status:report.status,checks:checks.length,output:path.relative(root,output)}));
} catch(error) {
  await writeFile(path.join(output,"report.json"),JSON.stringify({status:"FAIL",checks,error:error.message,uncaught_exceptions:errors},null,2));
  console.error(JSON.stringify({status:"FAIL",checks:checks.length,error:error.message,output:path.relative(root,output)}));process.exitCode=1;
} finally {
  if(socket?.readyState===WebSocket.OPEN){try{await send("Browser.close");}catch{}socket.close();}
  for(const e of pending.values())clearTimeout(e.timer);
  server.kill();browser.kill();
}
