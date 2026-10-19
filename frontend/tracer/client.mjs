const requestPattern = /^q_[0-9a-f]{64}$/;
const chunkPattern = /^chk_[0-9a-f]{64}$/;

export class ApiFailure extends Error {
  constructor(code, status = 0, requestId = null) {
    super(code); this.code = code; this.status = status; this.requestId = requestId;
  }
}

export async function apiRequest(path, token, { body, key, fetcher = fetch, timeoutMs = 32000 } = {}) {
  if (!token || !path.startsWith("/v1/") || path.includes("://") || path.includes("..")) {
    throw new ApiFailure("INVALID_CLIENT_REQUEST");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers = { Authorization: `Bearer ${token}` };
    if (body !== undefined) {
      if (!key || !/^[0-9a-f-]{36}$/.test(key)) throw new ApiFailure("INVALID_CLIENT_REQUEST");
      headers["Content-Type"] = "application/json";
      headers["Idempotency-Key"] = key;
    }
    const response = await fetcher(path, { method: body === undefined ? "GET" : "POST", headers,
      body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
      credentials: "omit", cache: "no-store", redirect: "error" });
    const raw = await response.text();
    if (raw.length > 200000) throw new ApiFailure("INVALID_RESPONSE");
    let data;
    try { data = JSON.parse(raw); } catch { throw new ApiFailure("INVALID_RESPONSE"); }
    if (!response.ok) throw new ApiFailure(data?.error?.code || "SERVICE_UNAVAILABLE", response.status, data?.request_id);
    return data;
  } catch (error) {
    if (error instanceof ApiFailure) throw error;
    throw new ApiFailure("REQUEST_OUTCOME_UNKNOWN");
  } finally { clearTimeout(timer); }
}

export function queryPath(requestId) {
  if (!requestPattern.test(requestId)) throw new ApiFailure("INVALID_REQUEST_ID");
  return `/v1/queries/${requestId}`;
}

export function sourcePath(requestId, chunkId) {
  queryPath(requestId);
  if (!chunkPattern.test(chunkId)) throw new ApiFailure("INVALID_CHUNK_ID");
  return `/v1/sources/${chunkId}?request_id=${requestId}`;
}

export function officialSourceLink(raw) {
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:" || url.username || url.password || (url.port && url.port !== "443") ||
        !["docs.aws.amazon.com", "aws.amazon.com"].includes(url.hostname)) return null;
    return url.href;
  } catch { return null; }
}

export function errorMessage(error) {
  if (error?.status === 401) return "Your session has expired. Sign in again.";
  if (error?.code === "ACCOUNT_VERIFICATION_REQUIRED") return "Verify your email with Cognito, then sign in again. / Hãy xác minh email rồi đăng nhập lại.";
  if (error?.code === "ACCOUNT_SETUP_UNAVAILABLE") return "Account setup is temporarily unavailable. Check access to try again; no question was submitted. / Chưa thể thiết lập tài khoản; hãy kiểm tra lại.";
  if (error?.status === 403) return "This reviewer account does not have access.";
  if (error?.status === 429) return "The demo's usage limit has been reached. Please try later.";
  if (["REQUEST_OUTCOME_UNKNOWN", "REQUEST_IN_PROGRESS", "QUERY_TIMEOUT"].includes(error?.code)) {
    return "The request may still be processing. Refresh your history to check it; it has not been submitted again.";
  }
  if (error?.code === "GENERATION_INVALID") return "The model's answer did not pass source or format checks. No automatic retry was made.";
  if (error?.code === "OFFICIAL_SEARCH_UNAVAILABLE") return "Official AWS search is unavailable. This request was not answered or retried automatically. You can select the recorded pilot and submit a separate question.";
  if (error?.code === "QUERY_REWRITE_INVALID") return "The search translation did not preserve a valid question. No answer was generated. Review your question and its AWS identifiers before submitting a new request.";
  if (error?.code === "QUESTION_TOKEN_LIMIT") return "Please shorten your question; it exceeds the search model's input limit.";
  return "The request could not be completed. Refresh your history for its final status.";
}
