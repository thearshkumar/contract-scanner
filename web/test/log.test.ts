import { describe, expect, it } from "vitest";
import { MAX_LINE_BYTES, RequestLog } from "../src/log";

const SECRET = "CONFIDENTIAL: Acme shall pay Beta $4,000,000 upon termination.";

function capture() {
  const lines: string[] = [];
  return { lines, sink: (l: string) => lines.push(l) };
}

describe("RequestLog", () => {
  it("writes exactly one summary line, even if flushed twice", async () => {
    const { lines, sink } = capture();
    const log = new RequestLog("/api/scan", "req-1", sink);
    await log.stage("llm", async () => 1);
    log.set("model", "m").flush(200);
    log.flush(500);
    expect(lines).toHaveLength(1);
    const row = JSON.parse(lines[0]!);
    expect(row).toMatchObject({ level: "info", reqId: "req-1", route: "/api/scan", status: 200, model: "m" });
    expect(row.stages).toHaveProperty("llm");
  });

  it("logs inputs as hash + length, never the text", async () => {
    const { lines, sink } = capture();
    const log = new RequestLog("/api/ask", "req-2", sink);
    await log.input("contract", SECRET);
    await log.input("question", "Can I terminate early?");
    log.error(new SyntaxError(`Unexpected token in ${SECRET}`));
    log.flush(500);
    const all = lines.join("\n");
    expect(all).not.toContain("CONFIDENTIAL");
    expect(all).not.toContain("terminate early");
    const row = JSON.parse(lines.at(-1)!);
    expect(row.contract_len).toBe(SECRET.length);
    expect(row.contract_sha256).toMatch(/^[0-9a-f]{16}$/);
    expect(JSON.parse(lines[0]!)).toMatchObject({ level: "error", error: "SyntaxError" });
  });

  it("keeps every line within the size cap", () => {
    const { lines, sink } = capture();
    const log = new RequestLog("/api/scan", "req-3", sink);
    log.set("big", "y".repeat(5000)).flush(200);
    expect(lines[0]!.length).toBeLessThanOrEqual(MAX_LINE_BYTES);
    expect(JSON.parse(lines[0]!).truncated).toBe(true);
  });
});
