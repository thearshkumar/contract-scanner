import type { ChatMessage, LLMProvider, LLMResult } from "../types";

// The fine-tuned scanner, served by vLLM on Modal (model/serve.py) behind Modal proxy auth:
// https://modal.com/docs/guide/webhook-proxy-auth
// Only the proxy-auth headers and the JSON body are sent: no client IP, user agent or cookies are forwarded.

export const SERVED_MODEL = "contract-scanner-r8-v1"; // model/serve.py SERVED_NAME

// Modal answers a request still running after 150 s with a 303 to a "result URL" that waits for the same request
// (https://modal.com/docs/guide/webhook-timeouts). Snapshot restores take ~30 s, but when Modal rebuilds the snapshot
// a cold start is a full vLLM boot (measured 227–484 s, docs/serving.md), so redirects are followed up to this limit.
// The budget's overshoot bound in wrangler.jsonc is computed from it.
export const MAX_CALL_MS = 300_000;
const MAX_REDIRECTS = 4;

type ChatCompletion = {
  choices?: { message?: { content?: string | null }; finish_reason?: string }[];
  usage?: { prompt_tokens?: number; completion_tokens?: number };
};

export class ModalLLM implements LLMProvider {
  constructor(
    private url: string,
    private key: string,
    private secret: string,
    private fetcher: typeof fetch = fetch,
  ) {}

  async complete(messages: ChatMessage[], opts: { maxTokens?: number; temperature?: number } = {}): Promise<LLMResult> {
    const body = JSON.stringify({
      model: SERVED_MODEL,
      messages,
      max_tokens: opts.maxTokens ?? 1024,
      temperature: opts.temperature ?? 0,
      chat_template_kwargs: { enable_thinking: false }, // as in training and eval (model/modal_app.py _chat)
    });
    const t0 = Date.now();
    const signal = AbortSignal.timeout(MAX_CALL_MS);
    let res = await this.send(body, signal);
    // One retry on a quick 5xx; never after a slow call (a retry would double the GPU time of a cold start).
    if (res.status >= 500 && Date.now() - t0 < 30_000) {
      await res.body?.cancel();
      res = await this.send(body, signal);
    }
    if (!res.ok) throw new Error(`scanner endpoint returned ${res.status}`);
    const out = (await res.json()) as ChatCompletion;
    return {
      text: out.choices?.[0]?.message?.content ?? "",
      usage: { inputTokens: out.usage?.prompt_tokens ?? 0, outputTokens: out.usage?.completion_tokens ?? 0, neurons: 0 },
      model: SERVED_MODEL,
      adapter: null,
    };
  }

  /** POSTs once, then follows Modal's 303 "still running" redirects with GET, only ever to the endpoint's own host. */
  private async send(body: string, signal: AbortSignal): Promise<Response> {
    const base = new URL(this.url);
    const auth = { "Modal-Key": this.key, "Modal-Secret": this.secret };
    let res = await this.fetcher(new URL("/v1/chat/completions", base), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...auth },
      body,
      redirect: "manual",
      signal,
    });
    for (let i = 0; i < MAX_REDIRECTS && res.status === 303; i++) {
      const next = new URL(res.headers.get("Location") ?? "", base);
      await res.body?.cancel();
      if (next.origin !== base.origin) throw new Error("scanner endpoint redirected off-host");
      res = await this.fetcher(next, { method: "GET", headers: auth, redirect: "manual", signal });
    }
    return res;
  }
}
