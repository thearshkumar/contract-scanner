import type { ChatMessage, LLMProvider, LLMResult } from "../types";

// The fine-tuned scanner, served by vLLM on Modal (model/serve.py) behind Modal proxy auth:
// https://modal.com/docs/guide/webhook-proxy-auth
// Only the proxy-auth headers and the JSON body are sent: no client IP, user agent or cookies are forwarded.

export const SERVED_MODEL = "contract-scanner-r8-v1"; // model/serve.py SERVED_NAME

// Modal answers requests running past 150 s with a 303 to a result URL; a cold start plus one chunk stays well under.
const TIMEOUT_MS = 150_000;

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
    let res: Response | undefined;
    for (let attempt = 0; attempt < 2; attempt++) {
      res = await this.fetcher(`${this.url.replace(/\/$/, "")}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Modal-Key": this.key, "Modal-Secret": this.secret },
        body,
        redirect: "manual",
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
      if (res.status < 500 || attempt === 1) break; // retry once on a 5xx only
      await res.body?.cancel();
    }
    if (!res || !res.ok) throw new Error(`scanner endpoint returned ${res?.status}`);
    const out = (await res.json()) as ChatCompletion;
    return {
      text: out.choices?.[0]?.message?.content ?? "",
      usage: { inputTokens: out.usage?.prompt_tokens ?? 0, outputTokens: out.usage?.completion_tokens ?? 0, neurons: 0 },
      model: SERVED_MODEL,
      adapter: null,
    };
  }
}
