import { describe, expect, it } from "vitest";
import { cosine, topK } from "../src/core/retrieve";
import { HashEmbedder, InMemoryLibrary } from "../src/providers/mock";

describe("cosine / topK", () => {
  it("scores identical, orthogonal and opposite vectors", () => {
    expect(cosine([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1);
    expect(cosine([0, 0], [1, 1])).toBe(0);
    expect(() => cosine([1], [1, 2])).toThrow();
  });

  it("ranks chunks by relevance to the question", async () => {
    const e = new HashEmbedder(256);
    const chunks = [
      "Either party may terminate this Agreement upon thirty days written notice.",
      "The Supplier's total liability shall not exceed the fees paid in the prior twelve months.",
      "This Agreement shall automatically renew for successive one year terms.",
    ];
    const { vectors } = await e.embed(chunks);
    const { vectors: [q] } = await e.embed(["what is the cap on liability for fees paid"]);
    const ranked = topK(q!, vectors, 2);
    expect(ranked[0]!.index).toBe(1);
    expect(ranked).toHaveLength(2);
  });

  it("breaks score ties by original order", () => {
    expect(topK([1, 0], [[2, 0], [1, 0]], 2).map((r) => r.index)).toEqual([0, 1]);
  });
});

describe("InMemoryLibrary", () => {
  it("returns only the requested clause type, best first", async () => {
    const lib = new InMemoryLibrary([
      { clause: { id: "a", clauseType: "Cap On Liability", text: "a", source: "X" }, vector: [1, 0] },
      { clause: { id: "b", clauseType: "Cap On Liability", text: "b", source: "Y" }, vector: [0.6, 0.8] },
      { clause: { id: "c", clauseType: "Non-Compete", text: "c", source: "Z" }, vector: [1, 0] },
    ]);
    const res = await lib.similar([1, 0], "Cap On Liability", 5);
    expect(res.map((r) => r.id)).toEqual(["a", "b"]);
  });
});
