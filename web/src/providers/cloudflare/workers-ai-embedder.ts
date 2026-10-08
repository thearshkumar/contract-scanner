import type { EmbedResult, Embedder } from "../types";
import { estimateNeurons } from "../rates";

const MODEL = "@cf/baai/bge-m3";

export class WorkersAiEmbedder implements Embedder {
  readonly dimensions = 1024;
  constructor(private ai: Ai) {}

  async embed(texts: string[]): Promise<EmbedResult> {
    const out = (await (this.ai.run as (m: string, i: unknown) => Promise<unknown>)(MODEL, { text: texts })) as {
      data: number[][];
      usage?: { prompt_tokens?: number };
    };
    // bge-m3 reports prompt tokens; fall back to a chars/4 estimate if absent.
    const inputTokens = out.usage?.prompt_tokens ?? Math.ceil(texts.reduce((n, t) => n + t.length, 0) / 4);
    return { vectors: out.data, usage: { inputTokens, outputTokens: 0, neurons: estimateNeurons(MODEL, inputTokens, 0) } };
  }
}
