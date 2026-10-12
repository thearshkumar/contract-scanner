// Thin client for the Worker API. Only contract chunks and questions are sent, and only to this origin.

import type { ClauseType } from "../src/core/scan-prompt";

export interface Health {
  ok: boolean;
  live: boolean;
  reason?: "offline" | "daily" | null;
  resetsAt: string;
}

export interface ScanResult {
  idx: number;
  ok: boolean;
  spans: { type: ClauseType; start: number; end: number }[];
  misses: number;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly demoOnly = false,
    readonly resetsAt?: string,
    readonly reason?: string,
  ) {
    super(message);
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new ApiError(0, "Network error. Check your connection and try again.");
  }
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  if (!res.ok) {
    throw new ApiError(
      res.status,
      String(data.error ?? `Request failed (${res.status}).`),
      Boolean(data.demoOnly),
      data.resetsAt as string | undefined,
      data.reason as string | undefined,
    );
  }
  return data as T;
}

export const health = async (): Promise<Health> => {
  try {
    const r = await fetch("/api/health");
    return (await r.json()) as Health;
  } catch {
    return { ok: false, live: false, resetsAt: "" };
  }
};

export const scanChunk = (chunk: string, idx: number) => post<ScanResult>("/api/scan", { chunk, idx });

export const embed = (chunks: string[]) => post<{ dims: number; vectors: string }>("/api/embed", { chunks });

export const ask = (question: string, chunks: { id: number; text: string }[]) =>
  post<{ answer: string; citations: number[] }>("/api/ask", { question, chunks });

export const similar = (text: string, type: ClauseType) =>
  post<{ examples: { text: string; source: string; score: number }[] }>("/api/similar", { text, type });

export function decodeVectors(b64: string, dims: number): Float32Array[] {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const all = new Float32Array(bytes.buffer);
  const out: Float32Array[] = [];
  for (let i = 0; i < all.length; i += dims) out.push(all.subarray(i, i + dims));
  return out;
}
