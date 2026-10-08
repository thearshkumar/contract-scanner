// Router. Static client assets never reach this Worker (run_worker_first: ["/api/*"]).
// Every API route except /api/health writes exactly one summary log line (src/log.ts).

import { RequestLog } from "./log";
import { allowRequest } from "./guard/ratelimit";

export { NeuronBudget } from "./guard/budget";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);

    // Health is not logged: uptime checks would otherwise spend the free log allowance.
    if (url.pathname === "/api/health") {
      return json({ ok: true, model: env.MODEL_ID, adapter: env.LORA_ID || null });
    }

    const log = new RequestLog(url.pathname, request.headers.get("cf-ray") ?? undefined);
    let status = 500;
    try {
      if (!(await allowRequest(env, request))) {
        status = 429;
        return json({ error: "Too many requests. Please wait a minute and try again." }, status);
      }
      // /api/scan, /api/similar and /api/ask arrive in Phases 2–3.
      status = 404;
      return json({ error: "Not found" }, status);
    } catch (err) {
      log.error(err);
      status = 500;
      return json({ error: "Something went wrong.", reqId: log.reqId }, status);
    } finally {
      log.flush(status);
    }
  },
} satisfies ExportedHandler<Env>;
