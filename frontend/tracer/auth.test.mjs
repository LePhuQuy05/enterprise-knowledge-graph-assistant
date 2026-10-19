import test from "node:test";
import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import { beginLogin, finishLogin, validateConfig, registerWithCognito, createRegistrationDialog, registrationMessage } from "./auth.mjs";
import { startApp } from "./ui.mjs";
import { ApiFailure } from "./client.mjs";

const config = { client_id: "exampleclientid", auth_domain: "https://kg-r72-example.auth.us-east-1.amazoncognito.com",
  callback_url: "https://demo.example/auth/callback", logout_url: "https://demo.example/" };
function storage() {
  const values = new Map();
  return { getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v), removeItem: k => values.delete(k) };
}

test("PKCE challenge matches RFC 7636 example", async () => {
  const verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
  const bytes = Buffer.from(verifier, "base64url");
  const crypto = { subtle: webcrypto.subtle, getRandomValues: array => { array.set(bytes); return array; } };
  const saved = storage();
  const url = new URL(await beginLogin(config, saved, crypto, 1000));
  assert.equal(url.searchParams.get("code_challenge"), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
  assert.equal(url.searchParams.get("code_challenge_method"), "S256");
  assert.equal(url.searchParams.get("response_type"), "code");
  assert.equal(JSON.parse(saved.getItem("kg.oauth")).verifier, verifier);
});

test("callback exchanges once without storing any tokens", async () => {
  const saved = storage();
  saved.setItem("kg.oauth", JSON.stringify({ state: "state", verifier: "verifier", created: 1000 }));
  let calls = 0;
  const fetcher = async (url, request) => {
    calls++;
    assert.equal(url.href, config.auth_domain + "/oauth2/token");
    assert.equal(request.redirect, "error");
    assert.equal(request.body.get("code_verifier"), "verifier");
    return { ok: true, json: async () => ({ access_token: "access", id_token: "id", refresh_token: "refresh", token_type: "Bearer" }) };
  };
  const callback = config.callback_url + "?state=state&code=code";
  assert.equal(await finishLogin(config, callback, saved, fetcher, 2000), "access");
  assert.equal(saved.getItem("kg.oauth"), null);
  await assert.rejects(() => finishLogin(config, callback, saved, fetcher, 2000));
  assert.equal(calls, 1);
});

for (const [name, query, created] of [["wrong state", "state=other&code=x", 1000],
  ["duplicate state", "state=state&state=state&code=x", 1000],
  ["duplicate code", "state=state&code=x&code=y", 1000], ["expired", "state=state&code=x", -400000],
  ["provider error", "state=state&code=x&error=denied", 1000]]) {
  test(`callback rejects ${name} before any token request`, async () => {
    const saved = storage();
    saved.setItem("kg.oauth", JSON.stringify({ state: "state", verifier: "verifier", created }));
    const noNetwork = async () => { assert.fail("Token request must not happen"); };
    await assert.rejects(() => finishLogin(config, `${config.callback_url}?${query}`, saved, noNetwork, 2000));
    assert.equal(saved.getItem("kg.oauth"), null);
  });
}

test("reject off-origin callback or an attacker-controlled sign-in domain", () => {
  assert.equal(validateConfig(config, "https://demo.example"), config);
  for (const change of [{ callback_url: "https://attacker.example/callback" },
    { auth_domain: "https://kg.auth.us-east-1.amazoncognito.com.attacker.example" },
    { auth_domain: "https://attacker@kg.auth.us-east-1.amazoncognito.com" }]) {
    assert.throws(() => validateConfig({ ...config, ...change }, "https://demo.example"));
  }
});

for (const [intent, path] of [["signup", "/signup"], ["recovery", "/forgotPassword"]]) {
  test(`${intent} carries a fresh PKCE transaction and the same bounded callback/scopes`, async () => {
    const saved = storage(), enabled = { ...config, registration_enabled: true };
    const first = new URL(await beginLogin(enabled, saved, webcrypto, 1000, intent));
    const state = first.searchParams.get("state");
    assert.equal(first.pathname, path);
    assert.equal(first.searchParams.get("redirect_uri"), config.callback_url);
    assert.equal(first.searchParams.get("scope"), "openid kg/read kg/query");
    assert.equal(first.searchParams.get("code_challenge_method"), "S256");
    const second = new URL(await beginLogin(enabled, saved, webcrypto, 1001, intent));
    assert.notEqual(second.searchParams.get("state"), state);
    assert.equal(JSON.parse(saved.getItem("kg.oauth")).state, second.searchParams.get("state"));
  });
}

test("signup stays disabled without an explicit deployment flag; arbitrary actions are rejected", async () => {
  const saved = storage();
  await assert.rejects(() => beginLogin(config, saved, webcrypto, 1000, "signup"));
  await assert.rejects(() => beginLogin(config, saved, webcrypto, 1000, "__proto__"));
  assert.equal(saved.getItem("kg.oauth"), null);
  assert.throws(() => validateConfig({ ...config, registration_enabled: "true" }, "https://demo.example"));
});

function accountHarness({ enabled = true, callback = false, statusError = null } = {}) {
  const elements = new Map(), navigations = [], saved = storage(), state = { clears: 0, activates: 0, statusCalls: 0 };
  const get = id => {
    if (!elements.has(id)) elements.set(id, { hidden: false, disabled: false, textContent: "", listeners: {},
      addEventListener(name, fn) { this.listeners[name] = fn; } });
    return elements.get(id);
  };
  if (callback) saved.setItem("kg.oauth", JSON.stringify({ state: "fixture", verifier: "verifier", created: Date.now() }));
  const options = { document: { getElementById: get }, storage: saved,
    location: { origin: "https://demo.example", pathname: callback ? "/auth/callback" : "/",
      search: callback ? "?state=fixture&code=example" : "",
      href: config.callback_url + "?state=fixture&code=example", assign: url => navigations.push(url) },
    history: { replaceState: (...args) => { state.cleanedCallback = args[2]; } },
    fetcher: async url => ({ ok: true, json: async () => String(url).includes("/oauth2/token")
      ? { token_type: "Bearer", access_token: "fixture-access" } : { ...config, registration_enabled: enabled } }),
    request: async (path, token) => { assert.equal(path, "/v1/status"); assert.equal(token, "fixture-access"); state.statusCalls++;
      if (statusError) throw statusError; return { message: "Workspace ready", model_enabled: true, quota_remaining: 300 }; },
    makeWorkspace: ({ onSessionExpired }) => { state.expire = onSessionExpired; return {
      clear() { state.clears++; }, async activate() { state.activates++; } }; } };
  return { options, get, navigations, saved, state };
}

test("signed-out account actions show only what the deployed configuration supports", async () => {
  for (const enabled of [true, false]) {
    const h = accountHarness({ enabled }); await startApp(h.options);
    assert.equal(h.get("signup").hidden, !enabled);
    assert.equal(h.get("forgot-password").hidden, false);
    assert.equal(h.get("login").disabled, false);
    assert.equal(h.get("logout").hidden, true);
    assert.equal(h.state.activates, 0);
  }
});

test("confirmed login activates workspace and logout clears credentials", async () => {
  const h = accountHarness({ callback: true }); await startApp(h.options);
  assert.equal(h.state.activates, 1); assert.equal(h.state.cleanedCallback, "/auth/callback");
  assert.equal(h.get("signup").hidden, true); assert.equal(h.get("forgot-password").hidden, true);
  assert.equal(h.get("logout").hidden, false); assert.equal(h.saved.getItem("kg.oauth"), null);
  h.get("logout").listeners.click();
  assert.equal(new URL(h.navigations[0]).pathname, "/logout"); assert.equal(h.state.clears, 1);
});

test("first-login storage outage keeps sign-out and an explicit retry without loading private data", async () => {
  const h = accountHarness({ callback: true, statusError: new ApiFailure("ACCOUNT_SETUP_UNAVAILABLE", 503) });
  await startApp(h.options);
  assert.equal(h.state.activates, 0); assert.equal(h.get("logout").hidden, false);
  assert.equal(h.get("check-access").hidden, false); assert.equal(h.state.statusCalls, 1);
  await h.get("check-access").listeners.click(); assert.equal(h.state.statusCalls, 2);
  assert.match(h.get("status").textContent, /setup/);
});

test("expired access token returns to sign in and never shows registration as authenticated", async () => {
  const h = accountHarness({ callback: true, statusError: new ApiFailure("ACCESS_TOKEN_REQUIRED", 401) });
  await startApp(h.options);
  assert.equal(h.get("login").hidden, false); assert.equal(h.get("signup").hidden, false);
  assert.equal(h.get("check-access").hidden, true); assert.equal(h.state.activates, 0);
});

test("registration sends explicit email directly to Cognito with no token or server-selected role", async () => {
  const values = {username:"fixture.user",email:"fixture@example.invalid",password:"Fixture-only-Password-7!",tenant_id:"attacker"};
  let calls=0;
  const result = await registerWithCognito({...config,registration_enabled:true},"SignUp",values,async (url,opts)=>{
    calls++; assert.equal(url,"https://cognito-idp.us-east-1.amazonaws.com/");
    assert.equal(opts.credentials,"omit"); assert.equal(opts.redirect,"error");
    assert.equal(opts.headers["X-Amz-Target"],"AWSCognitoIdentityProviderService.SignUp");
    const body=JSON.parse(opts.body);
    assert.deepEqual(Object.keys(body).sort(),["ClientId","Password","UserAttributes","Username"]);
    assert.equal(body.Username,"fixture.user");assert.notEqual(body.Username,values.email);
    assert.deepEqual(body.UserAttributes,[{Name:"email",Value:values.email}]);
    return {ok:true,text:async()=>JSON.stringify({UserConfirmed:false,Session:"do-not-retain",CodeDeliveryDetails:{Destination:"f***@example.invalid"}})};
  });
  assert.equal(calls,1); assert.deepEqual(result,{confirmed:false,destination:"f***@example.invalid"});
});

test("confirmation never forces an email alias transfer and never includes the password",async()=>{
  await registerWithCognito({...config,registration_enabled:true},"ConfirmSignUp",{username:"fixture.user",code:"123456"},async(url,opts)=>{
    const body=JSON.parse(opts.body);assert.equal(body.ForceAliasCreation,false);assert.equal(body.ConfirmationCode,"123456");
    assert.equal(body.Password,undefined);return {ok:true,text:async()=>"{}"};
  });
});

test("invalid signup is rejected before network; ambiguous failures never auto-retry or reveal provider text",async()=>{
  let calls=0;
  const send=async()=>{calls++;throw new Error("private diagnostic details");};
  const enabled={...config,registration_enabled:true};
  await assert.rejects(()=>registerWithCognito(enabled,"SignUp",{username:"fixture",email:"bad",password:"short"},send));
  assert.equal(calls,0);
  await assert.rejects(()=>registerWithCognito(enabled,"ConfirmSignUp",{username:"fixture",code:"123456"},send),e=>{
    assert.equal(e.code,"REGISTRATION_OUTCOME_UNKNOWN");assert.doesNotMatch(registrationMessage(e),/private/);return true;
  });
  assert.equal(calls,1);
});

function registrationHarness(register) {
  const nodes=new Map(),get=id=>{
    if(!nodes.has(id))nodes.set(id,{value:"",hidden:false,disabled:false,open:false,handlers:{},
      addEventListener(event,fn){this.handlers[event]=fn;},focus(){},reset(){for(const n of nodes.values())n.value="";},
      showModal(){this.open=true;},close(){this.open=false;this.handlers.close?.();}});
    return nodes.get(id);
  };
  let signIns=0;
  const ui=createRegistrationDialog({document:{getElementById:get},config:{...config,registration_enabled:true},
    register,onSignIn:()=>{signIns++;}});
  return {ui,get,signIns:()=>signIns,submit:id=>get(id).handlers.submit({preventDefault(){}})};
}
const nextTurn=()=>new Promise(resolve=>setImmediate(resolve));

test("signup dialog clears password, confirms email, and requires an explicit sign-in",async t=>{
  const calls=[],h=registrationHarness(async(c,a,v)=>{calls.push(a);assert.equal(v.username,"fixture.user");return {confirmed:a==="ConfirmSignUp",destination:"f***@example.invalid"};});
  t.after(()=>h.get("registration-dialog").close());
  h.ui.open();h.get("register-username").value="fixture.user";h.get("register-email").value="fixture@example.invalid";h.get("register-password").value="Fixture-only-Password-7!";
  h.submit("registration-form");await nextTurn();
  assert.equal(h.get("register-password").value,"");assert.equal(h.get("registration-verify").hidden,false);
  assert.equal(h.get("verify-username").value,"fixture.user");
  assert.equal(h.get("register-resend").disabled,true);assert.equal(h.signIns(),0);
  h.get("register-code").value="123456";h.submit("verification-form");await nextTurn();
  assert.deepEqual(calls,["SignUp","ConfirmSignUp"]);assert.equal(h.get("registration-done").hidden,false);
  h.get("register-signin").handlers.click();assert.equal(h.signIns(),1);assert.equal(h.get("register-code").value,"");
});

test("late signup responses cannot reopen a dismissed dialog or erase a fresh password draft",async t=>{
  let resolve;const h=registrationHarness(()=>new Promise(r=>{resolve=r;}));
  t.after(()=>h.get("registration-dialog").close());
  h.ui.open();h.get("register-username").value="fixture.user";h.get("register-email").value="fixture@example.invalid";h.get("register-password").value="Fixture-only-Password-7!";
  h.submit("registration-form");h.get("registration-dialog").close();h.ui.open();h.get("register-password").value="new-unsent-draft";
  resolve({confirmed:false,destination:"old-email-must-not-appear"});await nextTurn();
  assert.equal(h.get("register-password").value,"new-unsent-draft");assert.equal(h.get("registration-verify").hidden,true);
  assert.doesNotMatch(h.get("registration-status").textContent,/old-email/);
});

test("new signup requires a distinct valid username and email before any network request",async()=>{
  const enabled={...config,registration_enabled:true};let calls=0;
  const send=async()=>{calls++;assert.fail("Invalid signup must not contact Cognito");};
  for(const username of [undefined,"","aa","name with spaces",".leading","fixture@example.invalid","x".repeat(65)]) {
    await assert.rejects(()=>registerWithCognito(enabled,"SignUp",{username,email:"fixture@example.invalid",password:"Fixture-only-Password-7!"},send),e=>e.code==="INVALID_REGISTRATION");
  }
  await assert.rejects(()=>registerWithCognito(enabled,"SignUp",{username:"fixture.user",password:"Fixture-only-Password-7!"},send),e=>e.code==="INVALID_REGISTRATION");
  assert.equal(calls,0);
});

test("resume verification uses the chosen username independently of the email and clears the password",async t=>{
  const h=registrationHarness(async()=>{assert.fail("Resume must not send a request");});
  t.after(()=>h.get("registration-dialog").close());
  h.ui.open();h.get("register-username").value="  Fixture.User  ";h.get("register-email").value="different@example.invalid";h.get("register-password").value="unsent-draft";
  h.get("register-resume").handlers.click();
  assert.equal(h.get("verify-username").value,"Fixture.User");assert.equal(h.get("register-password").value,"");
  assert.equal(h.get("registration-verify").hidden,false);assert.equal(h.signIns(),0);
});

test("previous email-as-username accounts can still confirm and resend without signup migration",async()=>{
  const calls=[];
  for(const action of ["ConfirmSignUp","ResendConfirmationCode"]) {
    await registerWithCognito({...config,registration_enabled:true},action,{username:"old@example.invalid",code:"123456"},async(url,options)=>{
      const payload=JSON.parse(options.body);calls.push(action);assert.equal(payload.Username,"old@example.invalid");
      assert.equal(payload.UserAttributes,undefined);assert.equal(payload.Password,undefined);
      return {ok:true,text:async()=>"{}"};
    });
  }
  assert.equal(calls.length,2);
});
