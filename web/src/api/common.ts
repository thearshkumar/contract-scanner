import type { Admission, Caps, Status } from "../guard/budget";
import type { RequestLog } from "../log";
import type { ClauseLibrary, Embedder, LLMProvider } from "../providers/types";

/** What a handler needs; built from env in index.ts, swapped for mocks in tests. */
export interface Deps {
  scanner: LLMProvider;
  qa: LLMProvider;
  embedder: Embedder;
  library: ClauseLibrary;
  budget: {
    admitScan(day: string, visitor: string, nowMs: number, caps: Caps): Promise<Admission>;
    chargeGpu(day: string, visitor: string, startMs: number, endMs: number, scaledownMs: number, microsPerSec: number): Promise<number>;
    reserveNeurons(day: string, visitor: string, bucket: "embed" | "ask" | "similar", estimate: number, nowMs: number, caps: Caps): Promise<Admission>;
    settleNeurons(day: string, visitor: string, delta: number): Promise<void>;
    status(day: string): Promise<Status>;
  };
}

export interface Ctx {
  deps: Deps;
  caps: Caps;
  day: string;
  visitor: string;
  log: RequestLog;
  scaledownMs: number;
}

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly body: Record<string, unknown>,
  ) {
    super(String(body.error ?? status));
    this.name = "HttpError";
  }
}

export const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });

/** Parses a JSON body of at most maxBytes; never echoes the body in errors. */
export async function readJson(request: Request, maxBytes: number): Promise<Record<string, unknown>> {
  if (request.method !== "POST") throw new HttpError(405, { error: "Use POST." });
  const text = await request.text();
  if (text.length > maxBytes) throw new HttpError(413, { error: "Request too large." });
  try {
    const v: unknown = JSON.parse(text);
    if (typeof v === "object" && v !== null && !Array.isArray(v)) return v as Record<string, unknown>;
  } catch {
    // fall through
  }
  throw new HttpError(400, { error: "Expected a JSON object." });
}

export function str(v: unknown, field: string, max: number): string {
  if (typeof v !== "string" || !v.trim()) throw new HttpError(400, { error: `${field} must be a non-empty string.` });
  if (v.length > max) throw new HttpError(413, { error: `${field} is longer than ${max} characters.` });
  return v;
}

/** Turns a budget refusal into the response the client understands (demo-only mode for daily/visitor limits). */
export function refuse(a: Admission & { ok: false }, resetsAt: string): never {
  if (a.reason === "burst") throw new HttpError(429, { error: "Too many requests. Please wait a minute and try again.", reason: "burst" });
  const msg =
    a.reason === "daily"
      ? "Live scanning is paused: today's budget is used up. Try a demo contract."
      : "You've reached today's limit for live scans. Try a demo contract.";
  throw new HttpError(503, { error: msg, reason: a.reason, demoOnly: true, resetsAt });
}
