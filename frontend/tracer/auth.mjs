export function base64url(bytes) {
  return btoa(String.fromCharCode(...bytes)).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

export function validateConfig(config, origin) {
  const domain = new URL(config.auth_domain);
  if (domain.protocol !== "https:" || !/^[a-z0-9-]+\.auth\.us-east-1\.amazoncognito\.com$/.test(domain.hostname) ||
      domain.pathname !== "/" || domain.search || domain.hash || domain.port || domain.username || domain.password ||
      config.callback_url !== `${origin}/auth/callback` || config.logout_url !== `${origin}/` ||
      !/^[a-z0-9]+$/.test(config.client_id) ||
      (config.registration_enabled !== undefined && typeof config.registration_enabled !== "boolean")) throw new Error("Invalid sign-in settings");
  return config;
}

export async function beginLogin(config, storage, cryptoAPI = globalThis.crypto, now = Date.now(), intent = "signin") {
  const endpoints = { signin: "/oauth2/authorize", signup: "/signup", recovery: "/forgotPassword" };
  if (!Object.hasOwn(endpoints, intent) || (intent === "signup" && config.registration_enabled !== true)) {
    throw new Error("Account action is not available");
  }
  const verifier = base64url(cryptoAPI.getRandomValues(new Uint8Array(32)));
  const state = base64url(cryptoAPI.getRandomValues(new Uint8Array(32)));
  const challenge = base64url(new Uint8Array(await cryptoAPI.subtle.digest("SHA-256", new TextEncoder().encode(verifier))));
  storage.setItem("kg.oauth", JSON.stringify({ verifier, state, created: now }));
  const url = new URL(endpoints[intent], config.auth_domain);
  url.search = new URLSearchParams({ client_id: config.client_id, response_type: "code",
    redirect_uri: config.callback_url, scope: "openid kg/read kg/query", state,
    code_challenge_method: "S256", code_challenge: challenge }).toString();
  return url.href;
}

export async function finishLogin(config, callbackURL, storage, fetcher = globalThis.fetch, now = Date.now()) {
  const raw = storage.getItem("kg.oauth");
  storage.removeItem("kg.oauth"); // One attempt, including error/expired/mismatched callbacks.
  const saved = raw ? JSON.parse(raw) : null;
  const url = new URL(callbackURL);
  if (!saved || !saved.verifier || !saved.state || !Number.isFinite(saved.created) ||
      now < saved.created || now - saved.created > 300_000 ||
      url.searchParams.getAll("state").length !== 1 || url.searchParams.get("state") !== saved.state ||
      url.searchParams.getAll("code").length !== 1 || !url.searchParams.get("code") || url.searchParams.has("error")) {
    throw new Error("Sign-in expired or could not be verified. Start sign-in again.");
  }
  const response = await fetcher(new URL("/oauth2/token", config.auth_domain), {
    method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" },
    credentials: "omit", cache: "no-store", redirect: "error",
    body: new URLSearchParams({ grant_type: "authorization_code", client_id: config.client_id,
      redirect_uri: config.callback_url, code: url.searchParams.get("code"), code_verifier: saved.verifier }),
  });
  if (!response.ok) throw new Error("Sign-in could not be completed. Start sign-in again.");
  const tokens = await response.json();
  if (tokens.token_type !== "Bearer" || typeof tokens.access_token !== "string" || !tokens.access_token) {
    throw new Error("Sign-in returned an invalid access token.");
  }
  return tokens.access_token; // In-memory only; never persist ID/access/refresh tokens.
}

export class RegistrationFailure extends Error {
  constructor(code) { super(code); this.code = code; }
}

export function registrationMessage(error) {
  const messages = {
    InvalidPasswordException: "Use at least 14 characters, with uppercase, lowercase, a number and a symbol. / Mật khẩu chưa đủ mạnh.",
    CodeMismatchException: "That verification code is not correct. / Mã xác minh không đúng.",
    ExpiredCodeException: "This code has expired. Request a new code. / Mã hết hạn; hãy yêu cầu mã mới.",
    TooManyRequestsException: "Too many attempts. Wait before trying again. / Vui lòng chờ trước khi thử lại.",
    LimitExceededException: "The email limit was reached. The account may exist; use Verify email after a code is available. / Đã chạm hạn mức gửi email.",
    CodeDeliveryFailureException: "The email could not be delivered. Check the address; use Verify email if the account was created. / Chưa gửi được email xác minh.",
    UsernameExistsException: "Could not create that account. If you already registered, sign in or choose Verify email. / Nếu đã đăng ký, hãy đăng nhập hoặc xác minh email.",
    INVALID_REGISTRATION: "Check the username, email and required fields. / Kiểm tra lại các trường bắt buộc.",
    REGISTRATION_OUTCOME_UNKNOWN: "The request could not be confirmed. Check your email or choose Verify email; do not submit registration repeatedly. / Chưa rõ kết quả; kiểm tra email trước khi thử lại.",
  };
  return messages[error?.code] || "This account step could not be completed. Try later. / Chưa hoàn tất thao tác tài khoản; hãy thử lại sau.";
}

export async function registerWithCognito(config, action, values, fetcher = globalThis.fetch) {
  if (config.registration_enabled !== true || !/^[a-z0-9]+$/.test(config.client_id) ||
      !["SignUp", "ConfirmSignUp", "ResendConfirmationCode"].includes(action) ||
      typeof values.username !== "string" || !(/^[A-Za-z0-9._-]{3,64}$/.test(values.username) ||
        (values.username.length <= 128 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.username)))) {
    throw new RegistrationFailure("INVALID_REGISTRATION");
  }
  const payload = { ClientId: config.client_id, Username: values.username };
  if (action === "SignUp") {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(values.username) ||
        typeof values.email !== "string" || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email) || values.email.length > 254 ||
        typeof values.password !== "string" || values.password.length < 14 || values.password.length > 128) {
      throw new RegistrationFailure("INVALID_REGISTRATION");
    }
    payload.Password = values.password; payload.UserAttributes = [{ Name: "email", Value: values.email }];
  } else if (action === "ConfirmSignUp") {
    if (typeof values.code !== "string" || !/^\d{6}$/.test(values.code)) throw new RegistrationFailure("INVALID_REGISTRATION");
    payload.ConfirmationCode = values.code; payload.ForceAliasCreation = false;
  }
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetcher("https://cognito-idp.us-east-1.amazonaws.com/", {
      method: "POST", credentials: "omit", cache: "no-store", redirect: "error", referrerPolicy: "no-referrer",
      signal: controller.signal, headers: { "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": `AWSCognitoIdentityProviderService.${action}` }, body: JSON.stringify(payload),
    });
    const raw = await response.text();
    if (raw.length > 32768) throw new RegistrationFailure("REGISTRATION_OUTCOME_UNKNOWN");
    let data; try { data = JSON.parse(raw); } catch { throw new RegistrationFailure("REGISTRATION_OUTCOME_UNKNOWN"); }
    if (!response.ok) {
      const code = typeof data.__type === "string" ? data.__type.split("#").pop() : "REGISTRATION_FAILED";
      throw new RegistrationFailure(code);
    }
    return { confirmed: action === "ConfirmSignUp" || data.UserConfirmed === true,
      destination: typeof data.CodeDeliveryDetails?.Destination === "string" ? data.CodeDeliveryDetails.Destination.slice(0, 254) : "" };
  } catch (error) {
    if (error instanceof RegistrationFailure) throw error;
    throw new RegistrationFailure("REGISTRATION_OUTCOME_UNKNOWN");
  } finally { clearTimeout(timer); }
}

export function createRegistrationDialog({ document, config, onSignIn, register = registerWithCognito }) {
  const get = id => document.getElementById(id), dialog = get("registration-dialog");
  let epoch = 0, active = false, busy = false, resendAt = 0, timer;
  function busyState(value) {
    busy = value;
    for (const id of ["register-submit", "verify-submit", "register-resume", "register-back", "register-signin"]) get(id).disabled = value;
    for (const id of ["register-username", "register-email", "register-password", "verify-username", "register-code"]) get(id).disabled = value;
    get("register-resend").disabled = value || Date.now() < resendAt;
  }
  function mode(value) {
    get("registration-create").hidden = value !== "create";
    get("registration-verify").hidden = value !== "verify";
    get("registration-done").hidden = value !== "done";
  }
  function cooldown() {
    resendAt = Date.now() + 60000; get("register-resend").disabled = true;
    clearTimeout(timer); timer = setTimeout(() => { if (active) busyState(busy); }, 60000);
  }
  function clear() {
    active = false; epoch++; clearTimeout(timer);
    get("registration-form").reset(); get("verification-form").reset();
    get("registration-status").textContent = "";
  }
  async function perform(action, values, done) {
    if (!active || busy) return;
    busyState(true); const captured = epoch;
    get("registration-status").textContent = "Contacting Cognito… / Đang xử lý…";
    try {
      const result = await register(config, action, values);
      if (active && captured === epoch) done(result);
    } catch (error) {
      if (active && captured === epoch) get("registration-status").textContent = registrationMessage(error);
    } finally {
      // Do not retain a submitted password, even when its outcome is ambiguous.
      if (captured === epoch) get("register-password").value = "";
      if (active && captured === epoch) busyState(false);
    }
  }
  get("registration-form").addEventListener("submit", event => {
    event.preventDefault();
    const username = get("register-username").value.trim();
    perform("SignUp", { username, email: get("register-email").value.trim(), password: get("register-password").value }, result => {
      get("verify-username").value = username; get("register-email").value = "";
      mode(result.confirmed ? "done" : "verify");
      get("registration-status").textContent = result.confirmed ? "Account created. Continue to sign in. / Tiếp tục đăng nhập."
        : `Enter the code sent to ${result.destination || "your email"}. / Nhập mã xác minh từ email.`;
      if (!result.confirmed) { cooldown(); get("register-code").focus(); }
    });
  });
  get("verification-form").addEventListener("submit", event => {
    event.preventDefault();
    perform("ConfirmSignUp", { username: get("verify-username").value.trim(), code: get("register-code").value.trim() }, () => {
      get("register-code").value = ""; mode("done");
      get("registration-status").textContent = "Email verified. Sign in with your username and website password. / Email đã xác minh; đăng nhập bằng username và mật khẩu website.";
      get("register-signin").focus();
    });
  });
  get("register-resend").addEventListener("click", () => {
    if (Date.now() < resendAt) return;
    perform("ResendConfirmationCode", { username: get("verify-username").value.trim() }, () => {
      cooldown(); get("registration-status").textContent = "A new code was requested. Check your email; wait 60 seconds before resending. / Kiểm tra email và chờ trước khi gửi lại.";
    });
  });
  get("register-resume").addEventListener("click", () => {
    get("verify-username").value = get("register-username").value.trim(); get("register-password").value = "";
    mode("verify"); get("registration-status").textContent = "Use your username and the code from your email. / Nhập tên đăng nhập và mã nhận qua email.";
    get("verify-username").focus();
  });
  get("register-back").addEventListener("click", () => { get("register-code").value = ""; mode("create"); });
  get("registration-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", clear);
  get("register-signin").addEventListener("click", () => { dialog.close(); onSignIn(); });
  return { open() {
    clear(); active = true; resendAt = 0; mode("create"); busyState(false);
    dialog.showModal(); get("register-username").focus();
  } };
}
