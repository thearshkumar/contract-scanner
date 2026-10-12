// Cross-language check: the Worker must build the exact prompt the model was fine-tuned on and read its output
// exactly as the offline eval does (eval/prompt.py, eval/run.py). Both sides test against the same golden file.
import { describe, expect, it } from "vitest";
import { dedupe, locate, messages, parse, type ClauseType } from "../src/core/scan-prompt";
import golden from "../../eval/tests/fixtures/prompt_golden.json";

describe("scan prompt matches eval/prompt.py", () => {
  it("messages", () => expect(messages(golden.chunk)).toEqual(golden.messages));

  for (const [i, c] of golden.parse.entries()) {
    it(`parse #${i}`, () => expect(parse(c.input)).toEqual([c.items, c.ok]));
  }

  for (const c of golden.locate) {
    it(`locate ${JSON.stringify(c.begin)}`, () => expect(locate(golden.chunk, c.begin, c.end)).toEqual(c.result));
  }

  it("dedupe", () => {
    const input = golden.dedupe.input.map(([type, start, end]) => ({ type: type as ClauseType, start: start as number, end: end as number }));
    expect(dedupe(input).map((s) => [s.type, s.start, s.end])).toEqual(golden.dedupe.output);
  });
});
