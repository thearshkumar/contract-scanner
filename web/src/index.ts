// Router. Static client assets never reach this Worker (run_worker_first: ["/api/*"]).
// Every API route except /api/health writes exactly one summary log line (src/log.ts).

import { ask } from "./api/ask";
import { type Ctx, type Deps, HttpError, json } from "./api/common";
import { embed } from "./api/embed";
import { scan } from "./api/scan";
import { similar } from "./api/similar";
import { budgetStub, caps, resetsAt, utcDay } from "./guard/budget";
import { visitorFor } from "./guard/ratelimit";
import { RequestLog } from "./log";
import { VectorizeLibrary } from "./providers/cloudflare/vectorize-library";
import { WorkersAiEmbedder } from "./providers/cloudflare/workers-ai-embedder";
import { WorkersAiLLM } from "./providers/cloudflare/workers-ai-llm";
import { ModalLLM } from "./providers/modal/modal-llm";

export { UsageBudget } from "./guard/budget";

export function makeDeps(env: Env): Deps {
  return {
    scanner: new ModalLLM(env.MODAL_SCAN_URL, env.MODAL_KEY ?? "", env.MODAL_SECRET ?? ""),
    qa: new WorkersAiLLM(env.AI, env.QA_MODEL_ID),
    embedder: new WorkersAiEmbedder(env.AI),
    library: new VectorizeLibrary(env.CLAUSES),
    budget: budgetStub(env),
  };
}

const ROUTES: Record<string, (request: Request, c: Ctx, env: Env) => Promise<Response>> = {
  "/api/scan": (r, c) => scan(r, c),
  "/api/embed": (r, c) => embed(r, c),
  "/api/ask": (r, c, env) => ask(r, c, env.QA_MODEL_ID),
  "/api/similar": (r, c) => similar(r, c),
};

export async function handle(request: Request, env: Env, deps: Deps): Promise<Response> {
  const url = new URL(request.url);
  const day = utcDay();

  // Health is not logged: page loads and uptime checks would otherwise spend the free log allowance.
  if (url.pathname === "/api/health") {
    const s = await deps.budget.status(day);
    const live = Boolean(env.MODAL_SCAN_URL) && s.usdMicros < Number(env.DAILY_USD_MICROS);
    return json({ ok: true, live, resetsAt: resetsAt(), scanner: "Qwen3.5-2B + LoRA r8 (CUAD)", qa: env.QA_MODEL_ID });
  }

  const log = new RequestLog(url.pathname, request.headers.get("cf-ray") ?? undefined, console.log, env.LOG_SALT);
  let status = 500;
  try {
    const route = ROUTES[url.pathname];
    if (!route) {
      status = 404;
      return json({ error: "Not found" }, status);
    }
    const ctx: Ctx = {
      deps,
      caps: caps(env),
      day,
      visitor: await visitorFor(env, request, day),
      log,
      scaledownMs: Number(env.MODAL_SCALEDOWN_SECS) * 1000,
    };
    const res = await route(request, ctx, env);
    status = res.status;
    return res;
  } catch (err) {
    if (err instanceof HttpError) {
      status = err.status;
      if (typeof err.body.reason === "string") log.set("refused", err.body.reason);
      return json(err.body, status);
    }
    log.error(err);
    status = 502;
    return json({ error: "The model didn't respond. Please try again.", reqId: log.reqId }, status);
  } finally {
    log.flush(status);
  }
}

export default {
  fetch: (request, env) => handle(request, env, makeDeps(env)),
} satisfies ExportedHandler<Env>;
