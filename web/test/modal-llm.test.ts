import { describe, expect, it } from "vitest";
import { ModalLLM } from "../src/providers/modal/modal-llm";

const URL_ = "https://ws--contract-scanner-serve-scanner-serve.modal.run";
const ok = () => Response.json({ choices: [{ message: { content: "[]" } }], usage: { prompt_tokens: 10, completion_tokens: 2 } });

function fakeFetch(responses: (() => Response)[]) {
  const calls: { url: string; method: string; headers: Record<string, string> }[] = [];
  const f = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), method: init?.method ?? "GET", headers: (init?.headers ?? {}) as Record<string, string> });
    return responses.shift()!();
  }) as typeof fetch;
  return { f, calls };
}

describe("ModalLLM", () => {
  it("sends only the proxy-auth headers and the fixed body", async () => {
    const { f, calls } = fakeFetch([ok]);
    const out = await new ModalLLM(URL_, "wk-1", "ws-1", f).complete([{ role: "user", content: "x" }]);
    expect(out.text).toBe("[]");
    expect(calls[0]).toMatchObject({ url: `${URL_}/v1/chat/completions`, method: "POST" });
    expect(Object.keys(calls[0]!.headers).sort()).toEqual(["Content-Type", "Modal-Key", "Modal-Secret"]);
  });

  it("follows Modal's 303 'still running' redirect on the same host (slow cold start)", async () => {
    const { f, calls } = fakeFetch([
      () => new Response(null, { status: 303, headers: { Location: "/v1/chat/completions?__modal_function_call_id=fc-1" } }),
      ok,
    ]);
    await new ModalLLM(URL_, "wk-1", "ws-1", f).complete([{ role: "user", content: "x" }]);
    expect(calls[1]).toMatchObject({ method: "GET", url: `${URL_}/v1/chat/completions?__modal_function_call_id=fc-1` });
    expect(calls[1]!.headers["Modal-Key"]).toBe("wk-1");
  });

  it("never follows a redirect to another host (would leak the token)", async () => {
    const { f, calls } = fakeFetch([() => new Response(null, { status: 303, headers: { Location: "https://evil.example/x" } })]);
    await expect(new ModalLLM(URL_, "wk-1", "ws-1", f).complete([{ role: "user", content: "x" }])).rejects.toThrow("off-host");
    expect(calls).toHaveLength(1);
  });

  it("calls the real global fetch without 'Illegal invocation' (default fetcher)", async () => {
    // No network in tests: an unreachable URL must fail with a network error, not a TypeError from a bad `this`.
    const err = await new ModalLLM("http://127.0.0.1:9", "k", "s").complete([{ role: "user", content: "x" }]).catch((e: Error) => e);
    expect(String(err)).not.toMatch(/Illegal invocation/);
  });

  it("retries a quick 5xx once", async () => {
    const { f, calls } = fakeFetch([() => new Response("busy", { status: 503 }), ok]);
    await new ModalLLM(URL_, "wk-1", "ws-1", f).complete([{ role: "user", content: "x" }]);
    expect(calls).toHaveLength(2);
  });
});
