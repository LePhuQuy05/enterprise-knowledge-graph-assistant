// Local-only callback harness: real startApp + apiRequest; only fetch is mocked.
import { startApp } from "/assets/ui.mjs";
const nativeFetch = globalThis.fetch.bind(globalThis);
const fixture = await nativeFetch("/scoped-fixture.json").then(r => r.json());
const graph = fixture.cases.find(c => c.id === "graph");
const config = { auth_domain: "https://localfixture.auth.us-east-1.amazoncognito.com", client_id: "fixture123",
  callback_url: location.origin+"/auth/callback", logout_url: location.origin+"/",
  issuer: "https://cognito-idp.us-east-1.amazonaws.com/fixture", registration_enabled: true, scoped_runs_enabled: true };
const calls = [];
const reply = data => ({ ok: true, status: 200, json: async () => structuredClone(data), text: async () => JSON.stringify(data) });
globalThis.fetch = async (input, options = {}) => {
  const url = new URL(input, location.origin), path = url.pathname+url.search;
  if (url.origin === location.origin && path.startsWith("/assets/")) return nativeFetch(input, options);
  calls.push({ path, method: options.method || "GET" });
  if (path === "/config.json") return reply(config);
  if (url.origin === config.auth_domain && path === "/oauth2/token") return reply({ token_type: "Bearer", access_token: "synthetic-local-only-not-a-real-JWT" });
  if (path === "/v1/status") return reply({ model_enabled: true, message: "Synthetic callback completed.", quota_remaining: 284 });
  if (path.startsWith("/v1/queries?")) return reply({ items: [], next_cursor: null });
  if (path === "/v2/scoped-run-capabilities") return reply(fixture.capabilities);
  if (path.startsWith("/v2/scoped-runs?")) return reply({ items: [graph.run], next_cursor: null });
  if (path === "/v2/scoped-runs/"+graph.run.run_id) return reply(graph.run);
  const match = /^\/v2\/scoped-runs\/[^/]+\/methods\/([^/]+)(?:\/sources\/([^/]+))?$/.exec(path);
  if (match && graph.results[match[1]]) return reply(match[2] ? graph.sources[match[1]][match[2]] : graph.results[match[1]]);
  throw new Error("LOCAL_AUTH_CALLBACK_UNEXPECTED_FETCH");
};
sessionStorage.setItem("kg.oauth", JSON.stringify({ verifier: "synthetic-verifier", state: "fixture-state", created: Date.now() }));
globalThis.__authStartHarness = { calls, fixture: true };
await startApp({ location: { origin: location.origin, pathname: "/auth/callback", search: "?code=fixture-code&state=fixture-state",
  href: location.origin+"/auth/callback?code=fixture-code&state=fixture-state" }, history: { replaceState() {} } });
globalThis.__authStartHarness.ready = true;
