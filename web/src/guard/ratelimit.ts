// Per-visitor identity for the UsageBudget Durable Object (which also does burst limiting, exactly).
// The Workers rate-limit binding was dropped: it is per-location and eventually consistent
// (https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/), and ResearchRAG saw it let
// 30+ requests/min through a 5/min limit in production.

import { visitorId } from "../crypto";

export function visitorFor(env: Env, request: Request, day: string): Promise<string> {
  return visitorId(env.IP_SALT_KEY, request.headers.get("CF-Connecting-IP") ?? "local", day);
}
