// In-memory providers for tests and local work without Cloudflare.
import type { ChatMessage, ClauseLibrary, EmbedResult, Embedder, ExampleClause, LLMProvider, LLMResult } from "../types";
import { cosine } from "../../core/retrieve";

export class MockLLM implements LLMProvider {
  calls: ChatMessage[][] = [];
  constructor(private reply: (messages: ChatMessage[]) => string = () => "[]") {}
  async complete(messages: ChatMessage[]): Promise<LLMResult> {
    this.calls.push(messages);
    const text = this.reply(messages);
    const inputTokens = Math.ceil(messages.reduce((n, m) => n + m.content.length, 0) / 4);
    const outputTokens = Math.ceil(text.length / 4);
    return { text, usage: { inputTokens, outputTokens, neurons: 0 }, model: "mock", adapter: null };
  }
}

/** Deterministic bag-of-words hashing embedder: similar words give similar vectors. */
export class HashEmbedder implements Embedder {
  constructor(readonly dimensions = 64) {}
  async embed(texts: string[]): Promise<EmbedResult> {
    const vectors = texts.map((t) => {
      const v = new Array<number>(this.dimensions).fill(0);
      for (const word of t.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
        let h = 2166136261;
        for (let i = 0; i < word.length; i++) h = Math.imul(h ^ word.charCodeAt(i), 16777619);
        v[(h >>> 0) % this.dimensions]! += 1;
      }
      return v;
    });
    return { vectors, usage: { inputTokens: 0, outputTokens: 0, neurons: 0 } };
  }
}

export class InMemoryLibrary implements ClauseLibrary {
  constructor(private items: { clause: Omit<ExampleClause, "score">; vector: number[] }[]) {}
  async similar(vector: number[], clauseType: string, k: number): Promise<ExampleClause[]> {
    return this.items
      .filter((i) => i.clause.clauseType === clauseType)
      .map((i) => ({ ...i.clause, score: cosine(vector, i.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }
}
