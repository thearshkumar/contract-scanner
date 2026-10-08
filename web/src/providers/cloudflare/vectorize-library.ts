import type { ClauseLibrary, ExampleClause } from "../types";

// CUAD example clauses only (public data). Never holds user text. The index is built in Phase 3.
export class VectorizeLibrary implements ClauseLibrary {
  constructor(private index: VectorizeIndex) {}

  async similar(vector: number[], clauseType: string, k: number): Promise<ExampleClause[]> {
    const res = await this.index.query(vector, { topK: k, filter: { clauseType }, returnMetadata: "all" });
    return res.matches.map((m) => ({
      id: m.id,
      clauseType: String(m.metadata?.clauseType ?? clauseType),
      text: String(m.metadata?.text ?? ""),
      source: String(m.metadata?.source ?? ""),
      score: m.score,
    }));
  }
}
