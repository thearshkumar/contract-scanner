// Provider interfaces: the only way handlers reach a model, an embedder or the clause library.
// Cloudflare-specific calls live in providers/cloudflare/; tests use providers/mock/.

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  /** Estimated from tokens x the published per-model rate; Workers AI does not return neurons. */
  neurons: number;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LLMResult {
  text: string;
  usage: Usage;
  model: string;
  adapter: string | null;
}

export interface LLMProvider {
  complete(messages: ChatMessage[], opts?: { maxTokens?: number; temperature?: number }): Promise<LLMResult>;
}

export interface EmbedResult {
  vectors: number[][];
  usage: Usage;
}

export interface Embedder {
  readonly dimensions: number;
  embed(texts: string[]): Promise<EmbedResult>;
}

export interface ExampleClause {
  id: string;
  clauseType: string;
  text: string;
  source: string; // CUAD contract title, for attribution
  score: number;
}

export interface ClauseLibrary {
  similar(vector: number[], clauseType: string, k: number): Promise<ExampleClause[]>;
}
