import { startApp } from "/assets/ui.mjs";
import { ApiFailure } from "/assets/client.mjs";

const scenario = new URL(location.href).searchParams.get("state") || "signedout";
const signedIn = ["ready", "setup", "unverified", "disabled", "expired"].includes(scenario);
const config = { auth_domain: "https://kg-r72-fixture.auth.us-east-1.amazoncognito.com", client_id: "fixtureclient",
  callback_url: `${location.origin}/auth/callback`, logout_url: `${location.origin}/`, registration_enabled: scenario !== "invite" };
const harness = { scenario, calls: [], navigations: [], statusAttempts: 0 };
globalThis.__accountPreview = harness;
const banner = document.createElement("p"); banner.className = "preview-banner";
banner.textContent = "LOCAL ACCOUNT PREVIEW · synthetic identity and API · no emails, AWS or model calls";
document.querySelector("main").prepend(banner);
if (signedIn) sessionStorage.setItem("kg.oauth", JSON.stringify({ state: "fixture", verifier: "fixture", created: Date.now() }));
await startApp({
  registrationRequest: async (config, action, values) => {
    harness.registrationCalls ||= []; harness.registrationCalls.push({action,username:values.username});
    if(action === "ConfirmSignUp" && values.code !== "123456") throw {code:"CodeMismatchException"};
    return {confirmed:action === "ConfirmSignUp",destination:"r***@example.invalid"};
  },
  location: { origin: location.origin, pathname: signedIn ? "/auth/callback" : "/", search: signedIn ? "?state=fixture&code=synthetic" : "",
    href: `${location.origin}/auth/callback?state=fixture&code=synthetic`, assign: url => { harness.navigations.push(url); } },
  history: { replaceState: () => { harness.callbackCleaned = true; } },
  fetcher: async url => ({ ok: true, json: async () => String(url).includes("/oauth2/token")
    ? { token_type: "Bearer", access_token: "synthetic-access" } : config }),
  request: async (path, token, options) => {
    harness.calls.push(path);
    if (options?.body) throw new Error("Paid submission is disabled in this synthetic preview");
    if (path === "/v1/status") {
      harness.statusAttempts++;
      if (scenario === "setup" && harness.statusAttempts === 1) throw new ApiFailure("ACCOUNT_SETUP_UNAVAILABLE", 503);
      if (scenario === "unverified") throw new ApiFailure("ACCOUNT_VERIFICATION_REQUIRED", 403);
      if (scenario === "disabled") throw new ApiFailure("ACTIVE_MEMBERSHIP_REQUIRED", 403);
      if (scenario === "expired") throw new ApiFailure("ACCESS_TOKEN_REQUIRED", 401);
      return { message: "Document answers, citations and your own history are available.", quota_remaining: 300, model_enabled: true };
    }
    if (path.startsWith("/v1/queries")) return { items: [], next_cursor: null };
    throw new Error("Unexpected preview request");
  }
});
harness.ready = true;
