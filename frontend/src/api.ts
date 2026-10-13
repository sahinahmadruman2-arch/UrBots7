const TOKEN_KEY = "karllo-token";
const BASE_KEY = "karllo-api-base";

export function getApiBase(): string {
  try {
    const stored = localStorage.getItem(BASE_KEY);
    if (stored) return stored.replace(/\/$/, "");
  } catch {
    /* ignore */
  }
  const env = (import.meta as any).env?.VITE_API_BASE as string | undefined;
  return (env || "").replace(/\/$/, "");
}

export function setApiBase(url: string) {
  const clean = url.trim().replace(/\/$/, "");
  if (clean) localStorage.setItem(BASE_KEY, clean);
  else localStorage.removeItem(BASE_KEY);
}

export function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${getApiBase()}${path}`;
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
    ...authHeaders(),
    ...((init.headers as Record<string, string>) || {}),
  };
  const res = await fetch(apiUrl(path), { ...init, headers, credentials: "include" });
  if (res.status === 401) {
    localStorage.removeItem(TOKEN_KEY);
    const err: any = new Error("unauthorized");
    err.status = 401;
    throw err;
  }
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!res.ok) {
    const detail = data?.detail;
    const message =
      (typeof detail === "string" && detail) ||
      data?.error ||
      res.statusText;
    const err: any = new Error(message);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data as T;
}

export async function login(password: string) {
  const data = await api<{ ok: boolean; token?: string }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
  if (data.token) localStorage.setItem(TOKEN_KEY, data.token);
  return data;
}

export function me() {
  return api("/api/me");
}

export function logout() {
  localStorage.removeItem(TOKEN_KEY);
  return api("/api/auth/logout", { method: "POST" }).catch(() => null);
}
