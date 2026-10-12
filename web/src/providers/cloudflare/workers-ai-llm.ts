import type { ChatMessage, LLMProvider, LLMResult } from "../types";
import { estimateNeurons } from "../rates";

type AiChatOutput = {
  response?: unknown;
  choices?: { message?: { content?: string | null } }[];
  usage?: { prompt_tokens?: number; completion_tokens?: number };
};

// Workers AI chat model (used for Q&A). No AI Gateway: its logs are on by default and would store prompts.
export class WorkersAiLLM implements LLMProvider {
  constructor(
    private ai: Ai,
    private model: string,
    private lora: string | null = null,
  ) {}

  async complete(messages: ChatMessage[], opts: { maxTokens?: number; temperature?: number } = {}): Promise<LLMResult> {
    // Qwen3's soft switch turns thinking off, so no output neurons are spent on reasoning tokens.
    const last = messages.at(-1);
    const msgs = last?.role === "user" ? [...messages.slice(0, -1), { ...last, content: `${last.content}\n/no_think` }] : messages;
    const input: Record<string, unknown> = { messages: msgs, max_tokens: opts.maxTokens ?? 512, temperature: opts.temperature ?? 0 };
    if (this.lora) input.lora = this.lora;
    // The model id comes from config, so it is not one of the typed literal model names.
    const out = (await (this.ai.run as (m: string, i: unknown) => Promise<unknown>)(this.model, input)) as AiChatOutput;
    const inputTokens = out.usage?.prompt_tokens ?? 0;
    const outputTokens = out.usage?.completion_tokens ?? 0;
    return {
      text: stripThink(responseText(out)),
      usage: { inputTokens, outputTokens, neurons: estimateNeurons(this.model, inputTokens, outputTokens) },
      model: this.model,
      adapter: this.lora,
    };
  }
}

/** Workers AI parses JSON-looking output, so `response` can be an object (docs/spike-lora.md); some models use `choices`. */
function responseText(out: AiChatOutput): string {
  if (typeof out.response === "string") return out.response;
  if (out.response != null) return JSON.stringify(out.response);
  return out.choices?.[0]?.message?.content ?? "";
}

export const stripThink = (s: string) => s.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
