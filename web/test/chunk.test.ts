import { describe, expect, it } from "vitest";
import { chunkText } from "../src/core/chunk";

const para = (n: number) => `Section ${n}. The Supplier shall deliver the Goods within thirty days of the order date.`;
const contract = Array.from({ length: 200 }, (_, i) => para(i)).join("\n\n");

describe("chunkText", () => {
  it("covers the whole text, with offsets that index the original", () => {
    const chunks = chunkText(contract, { maxChars: 1000, overlap: 100 });
    expect(chunks[0]!.start).toBe(0);
    expect(chunks.at(-1)!.end).toBe(contract.length);
    for (const c of chunks) expect(contract.slice(c.start, c.end)).toBe(c.text);
    for (let i = 1; i < chunks.length; i++) expect(chunks[i]!.start).toBeLessThanOrEqual(chunks[i - 1]!.end); // no gaps
  });

  it("respects maxChars and breaks on paragraph boundaries", () => {
    const chunks = chunkText(contract, { maxChars: 1000, overlap: 100 });
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(1000);
    for (const c of chunks.slice(0, -1)) expect(c.text.endsWith("\n\n")).toBe(true);
  });

  it("overlaps consecutive chunks so boundary clauses are seen whole", () => {
    const chunks = chunkText(contract, { maxChars: 1000, overlap: 200 });
    for (let i = 1; i < chunks.length; i++) expect(chunks[i]!.start).toBeLessThan(chunks[i - 1]!.end);
  });

  it("hard-cuts text with no break characters and still terminates", () => {
    const blob = "x".repeat(2500);
    const chunks = chunkText(blob, { maxChars: 1000, overlap: 100 });
    expect(chunks.length).toBe(3);
    expect(chunks.at(-1)!.end).toBe(2500);
  });

  it("handles empty and short input", () => {
    expect(chunkText("")).toEqual([]);
    expect(chunkText("Short.")).toEqual([{ id: 0, start: 0, end: 6, text: "Short." }]);
  });

  it("rejects overlap >= maxChars", () => {
    expect(() => chunkText(contract, { maxChars: 100, overlap: 100 })).toThrow();
  });
});
