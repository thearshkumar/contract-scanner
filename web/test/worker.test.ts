import { exports, env } from "cloudflare:workers";
import { describe, expect, it, vi } from "vitest";
import { budgetStub, utcDay } from "../src/guard/budget";

describe("Worker routes", () => {
  it("serves /api/health without writing a log line", async () => {
    const spy = vi.spyOn(console, "log");
    const res = await exports.default.fetch("https://example.com/api/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ ok: true, model: env.MODEL_ID });
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("writes exactly one log line for other API requests", async () => {
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    const res = await exports.default.fetch("https://example.com/api/nope");
    expect(res.status).toBe(404);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(JSON.parse(spy.mock.calls[0]![0] as string)).toMatchObject({ route: "/api/nope", status: 404 });
    spy.mockRestore();
  });
});

describe("NeuronBudget", () => {
  it("reserves until the cap, then refuses", async () => {
    const stub = budgetStub(env);
    const day = "2099-01-01";
    expect(await stub.reserve(day, 60, 100)).toMatchObject({ ok: true, used: 60 });
    expect(await stub.reserve(day, 60, 100)).toMatchObject({ ok: false, used: 60 });
    expect(await stub.reserve(day, 40, 100)).toMatchObject({ ok: true, used: 100 });
  });

  it("settles the actual usage and keeps days separate", async () => {
    const stub = budgetStub(env);
    await stub.reserve("2099-02-01", 50, 100);
    expect(await stub.settle("2099-02-01", -30)).toBe(20);
    expect(await stub.reserve("2099-02-02", 100, 100)).toMatchObject({ ok: true, used: 100 });
  });

  it("formats UTC days", () => {
    expect(utcDay(new Date("2026-10-08T23:59:59Z"))).toBe("2026-10-08");
  });
});
