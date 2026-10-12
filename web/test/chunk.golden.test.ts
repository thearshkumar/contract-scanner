// Cross-language check: the browser/Worker chunker (TS) must produce exactly the chunks the model is trained and
// evaluated on (eval/chunk.py). Both sides test against the same golden file.
import { describe, expect, it } from "vitest";
import { chunkText } from "../src/core/chunk";
import input from "../../eval/tests/fixtures/chunk_input.txt?raw";
import golden from "../../eval/tests/fixtures/chunk_golden.json";

describe("chunkText matches eval/chunk.py", () => {
  for (const c of golden as { maxChars: number; overlap: number; chunks: [number, number][] }[]) {
    it(`maxChars=${c.maxChars} overlap=${c.overlap}`, () => {
      const got = chunkText(input, { maxChars: c.maxChars, overlap: c.overlap }).map((k) => [k.start, k.end]);
      expect(got).toEqual(c.chunks);
    });
  }
});
