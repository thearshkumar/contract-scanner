import type { ChatMessage, LLMProvider, LLMResult } from "../types";
import { estimateNeurons } from "../rates";

type AiChatOutput = { response?: string; usage?: { prompt_tokens?: number; completion_tokens?: number } };

export class WorkersAiLLM implements LLMProvider {
  constructor(
    private ai: Ai,
    private model: string,
    private lora: string | null,
  ) {}

  async complete(messages: ChatMessage[], opts: { maxTokens?: number; temperature?: number } = {}): Promise<LLMResult> {
    const input: Record<string, unknown> = {
      messages,
      max_tokens: opts.maxTokens ?? 512,
      temperature: opts.temperature ?? 0,
    };
    if (this.lora) input.lora = this.lora;
    // The model id comes from config, so it is not one of the typed literal model names.
    const out = (await (this.ai.run as (m: string, i: unknown) => Promise<unknown>)(this.model, input)) as AiChatOutput;
    const inputTokens = out.usage?.prompt_tokens ?? 0;
    const outputTokens = out.usage?.completion_tokens ?? 0;
    return {
      text: out.response ?? "",
      usage: { inputTokens, outputTokens, neurons: estimateNeurons(this.model, inputTokens, outputTokens) },
      model: this.model,
      adapter: this.lora,
    };
  }
}
