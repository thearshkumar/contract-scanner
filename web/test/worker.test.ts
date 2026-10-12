import { env, exports } from "cloudflare:workers";
import { describe, expect, it, vi } from "vitest";
import { checkCitations } from "../src/api/ask";
import type { Deps } from "../src/api/common";
import { BURST_PER_MIN, budgetStub, caps, resetsAt, utcDay } from "../src/guard/budget";
import { handle } from "../src/index";
import { HashEmbedder, InMemoryLibrary, MockLLM } from "../src/providers/mock";
import { MODAL_USD_MICROS_PER_SEC } from "../src/providers/rates";

const SECRET = "CONFIDENTIAL: Acme shall pay Beta $4,000,000 upon termination of this Agreement.";
const CHUNK = `1. Fees. ${SECRET}\n\n2. Law. This Agreement is governed by the laws of the State of Delaware.`;

const today = () => utcDay();
let ipCounter = 0;
const freshIp = () => `203.0.113.${++ipCounter % 250}-${Date.now()}`;

function deps(over: Partial<Deps> = {}): Deps {
  return {
    scanner: new MockLLM(() => '[{"type": "Governing Law", "begin": "This Agreement is governed", "end": "State of Delaware."}]'),
    qa: new MockLLM(() => "Delaware law governs [c1], see also [c7]."),
    embedder: new HashEmbedder(),
    library: new InMemoryLibrary([
      { clause: { id: "a", clauseType: "Governing Law", text: "governed by the laws of New York", source: "X Corp" }, vector: new Array(64).fill(1) },
    ]),
    budget: budgetStub(env),
    scannerConfigured: true,
    ...over,
  };
}

function post(path: string, body: unknown, ip = freshIp()) {
  return new Request(`https://example.com${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "CF-Connecting-IP": ip },
    body: JSON.stringify(body),
  });
}

function captureLogs() {
  return vi.spyOn(console, "log").mockImplementation(() => {});
}

describe("routes", () => {
  it("serves /api/health without writing a log line", async () => {
    const spy = vi.spyOn(console, "log");
    const res = await exports.default.fetch("https://example.com/api/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ ok: true, live: false, resetsAt: resetsAt() }); // no MODAL_SCAN_URL in tests
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("writes exactly one log line for other API requests", async () => {
    const spy = captureLogs();
    const res = await exports.default.fetch("https://example.com/api/nope");
    expect(res.status).toBe(404);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(JSON.parse(spy.mock.calls[0]![0] as string)).toMatchObject({ route: "/api/nope", status: 404 });
    spy.mockRestore();
  });

  it("scan: returns chunk-relative spans, charges GPU time, logs one line without text", async () => {
    const spy = captureLogs();
    const d = deps();
    const res = await handle(post("/api/scan", { chunk: CHUNK, idx: 3 }), env, d);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { idx: number; ok: boolean; spans: { type: string; start: number; end: number }[] };
    expect(body.idx).toBe(3);
    expect(body.ok).toBe(true);
    const s = body.spans[0]!;
    expect(CHUNK.slice(s.start, s.end)).toBe("This Agreement is governed by the laws of the State of Delaware.");
    // The model received the exact training prompt, built server-side.
    expect((d.scanner as MockLLM).calls[0]![0]!.role).toBe("system");
    expect(spy).toHaveBeenCalledTimes(1);
    const line = spy.mock.calls[0]![0] as string;
    expect(line).not.toContain("CONFIDENTIAL");
    expect(line).not.toContain("Delaware");
    const row = JSON.parse(line);
    expect(row).toMatchObject({ route: "/api/scan", status: 200, items: 1, anchor_misses: 0, parse_ok: true });
    expect(row.chunk_fp).toMatch(/^[0-9a-f]{16}$/);
    expect(row.usd_micros).toBeGreaterThan(0);
    spy.mockRestore();
  });

  it("scan: offline (no Modal URL/secrets) is demo-only and charges nothing", async () => {
    const spy = captureLogs();
    const res = await handle(post("/api/scan", { chunk: CHUNK }), env, deps({ scannerConfigured: false }));
    expect(res.status).toBe(503);
    expect(await res.json()).toMatchObject({ reason: "offline", demoOnly: true });
    expect(JSON.parse(spy.mock.calls[0]![0] as string)).not.toHaveProperty("usd_micros");
    spy.mockRestore();
  });

  it("scan: rejects oversized chunks and non-POST", async () => {
    const spy = captureLogs();
    expect((await handle(post("/api/scan", { chunk: "x".repeat(6201) }), env, deps())).status).toBe(413);
    expect((await handle(new Request("https://example.com/api/scan"), env, deps())).status).toBe(405);
    expect((await handle(post("/api/scan", { chunk: "" }), env, deps())).status).toBe(400);
    spy.mockRestore();
  });

  it("scan: a failing model returns 502, is still charged, and logs no text", async () => {
    const spy = captureLogs();
    const scanner = { complete: async () => { throw new Error(`boom ${SECRET}`); } };
    const res = await handle(post("/api/scan", { chunk: CHUNK }), env, deps({ scanner }));
    expect(res.status).toBe(502);
    const all = spy.mock.calls.map((c) => c[0]).join("\n");
    expect(all).not.toContain("CONFIDENTIAL");
    expect(JSON.parse(spy.mock.calls.at(-1)![0] as string).usd_micros).toBeGreaterThan(0);
    spy.mockRestore();
  });

  it("scan: a visitor over their daily quota gets demo-only, others don't", async () => {
    const spy = captureLogs();
    const ip = freshIp();
    const e = { ...env, VISITOR_USD_MICROS: "1" } as unknown as Env;
    expect((await handle(post("/api/scan", { chunk: CHUNK }, ip), e, deps())).status).toBe(200); // charges > 1 micro
    const res = await handle(post("/api/scan", { chunk: CHUNK }, ip), e, deps());
    expect(res.status).toBe(503);
    expect(await res.json()).toMatchObject({ reason: "visitor", demoOnly: true });
    expect((await handle(post("/api/scan", { chunk: CHUNK }), e, deps())).status).toBe(200);
    spy.mockRestore();
  });

  it("embed: returns base64 float32 vectors", async () => {
    const spy = captureLogs();
    const res = await handle(post("/api/embed", { chunks: ["alpha beta", "gamma"] }), env, deps());
    expect(res.status).toBe(200);
    const body = (await res.json()) as { dims: number; vectors: string };
    expect(body.dims).toBe(64);
    expect(atob(body.vectors).length).toBe(2 * 64 * 4);
    expect((await handle(post("/api/embed", { chunks: new Array(41).fill("x") }), env, deps())).status).toBe(400);
    spy.mockRestore();
  });

  it("ask: keeps only citations to passages that were sent", async () => {
    const spy = captureLogs();
    const res = await handle(post("/api/ask", { question: "Which law applies?", chunks: [{ id: 1, text: CHUNK }] }), env, deps());
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ answer: "Delaware law governs [c1], see also.", citations: [1] });
    const row = JSON.parse(spy.mock.calls.at(-1)![0] as string);
    expect(row).toMatchObject({ cited: 1, invalid_citations: 1 });
    expect(JSON.stringify(row)).not.toContain("Which law");
    spy.mockRestore();
  });

  it("similar: validates the type and returns examples", async () => {
    const spy = captureLogs();
    const ok = await handle(post("/api/similar", { text: "governed by the laws of Delaware", type: "Governing Law" }), env, deps());
    expect(ok.status).toBe(200);
    expect(((await ok.json()) as { examples: unknown[] }).examples).toHaveLength(1);
    expect((await handle(post("/api/similar", { text: "x", type: "Pizza" }), env, deps())).status).toBe(400);
    spy.mockRestore();
  });
});

describe("checkCitations", () => {
  it("strips unknown ids and tidies punctuation", () => {
    expect(checkCitations("A [c2] and B [c9].", new Set([2]))).toEqual({ answer: "A [c2] and B.", cited: [2], invalid: 1 });
  });
});

describe("UsageBudget", () => {
  const C = caps(env);
  const SD = 20_000;
  const RATE = 1_000_000; // $1/s = 1,000 micro-dollars per ms, so 30 s reads as 30_000_000

  it("charges the union of [start, end + scale-down] intervals", async () => {
    const b = budgetStub(env);
    const day = "2099-01-01";
    expect(await b.chargeGpu(day, "v", 0, 10_000, SD, RATE)).toBe(30_000_000); // cold: 0..30s
    // Concurrent with the first call: nothing new is covered.
    expect(await b.chargeGpu(day, "v", 2_000, 9_000, SD, RATE)).toBe(0);
    // Overlaps the tail: only 30s..35s is new.
    expect(await b.chargeGpu(day, "v", 12_000, 15_000, SD, RATE)).toBe(5_000_000);
    // After the container scaled down: a new, separate interval.
    expect(await b.chargeGpu(day, "v", 100_000, 101_000, SD, RATE)).toBe(21_000_000);
    expect((await b.status(day)).usdMicros).toBe(56_000_000);
  });

  it("an earlier, longer call that finishes last is not under-counted", async () => {
    const b = budgetStub(env);
    const day = "2099-01-02";
    await b.chargeGpu(day, "v", 2_000, 9_000, SD, RATE); // B finishes first: 2..29s
    expect(await b.chargeGpu(day, "v", 0, 10_000, SD, RATE)).toBe(3_000_000); // A adds 0..2s and 29..30s
    expect((await b.status(day)).usdMicros).toBe(30_000_000);
  });

  it("admits scans until the daily cap, then refuses", async () => {
    const b = budgetStub(env);
    const day = "2099-01-03";
    const c = { ...C, usdMicros: 100, visitorUsdMicros: 1e9 };
    expect(await b.admitScan(day, "v1", 0, c)).toEqual({ ok: true });
    await b.chargeGpu(day, "v1", 0, 1, 0, 100_000); // 100 micros
    expect(await b.admitScan(day, "v2", 0, c)).toEqual({ ok: false, reason: "daily" });
  });

  it("burst-limits per visitor and per minute", async () => {
    const b = budgetStub(env);
    const day = "2099-01-04";
    const c = { ...C, usdMicros: 1e12, visitorUsdMicros: 1e12 };
    for (let i = 0; i < BURST_PER_MIN.scan; i++) expect((await b.admitScan(day, "v", 1_000, c)).ok).toBe(true);
    expect(await b.admitScan(day, "v", 1_000, c)).toEqual({ ok: false, reason: "burst" });
    expect((await b.admitScan(day, "other", 1_000, c)).ok).toBe(true);
    expect((await b.admitScan(day, "v", 61_000, c)).ok).toBe(true); // next minute
  });

  it("reserves and settles neurons against global and visitor caps", async () => {
    const b = budgetStub(env);
    const day = "2099-01-05";
    const c = { ...C, neurons: 100, visitorNeurons: 60 };
    expect(await b.reserveNeurons(day, "v", "ask", 50, 0, c)).toEqual({ ok: true });
    expect(await b.reserveNeurons(day, "v", "ask", 20, 0, c)).toEqual({ ok: false, reason: "visitor" });
    await b.settleNeurons(day, "v", -40); // actual was 10
    expect(await b.reserveNeurons(day, "v", "ask", 20, 0, c)).toEqual({ ok: true });
    expect(await b.reserveNeurons(day, "w", "ask", 60, 0, c)).toMatchObject({ ok: true });
    expect(await b.reserveNeurons(day, "x", "ask", 50, 0, c)).toEqual({ ok: false, reason: "daily" });
  });

  it("deletes previous days' rows (visitor ids live at most one day)", async () => {
    const b = budgetStub(env);
    await b.reserveNeurons("2099-02-01", "v", "ask", 5, 0, C);
    expect((await b.status("2099-02-01")).neurons).toBe(5);
    await b.status("2099-02-02");
    expect((await b.status("2099-02-01")).neurons).toBe(0);
  });

  it("formats days and reset times in UTC", () => {
    expect(utcDay(new Date("2026-10-08T23:59:59Z"))).toBe("2026-10-08");
    expect(resetsAt(new Date("2026-10-08T23:59:59Z"))).toBe("2026-10-09T00:00:00.000Z");
    expect(MODAL_USD_MICROS_PER_SEC).toBeCloseTo(283.72, 2);
    expect(today()).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});
