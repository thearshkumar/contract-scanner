// POST /api/scan {chunk, idx}: one contract chunk through the fine-tuned scanner on Modal.
// The browser chunks the contract (core/chunk.ts) and calls this once per chunk. The prompt is built here, so a
// client can only ever send contract text, never its own instructions, to the GPU.

import { resetsAt } from "../guard/budget";
import { MODAL_USD_MICROS_PER_SEC } from "../providers/rates";
import { messages, parse, toSpans } from "../core/scan-prompt";
import { type Ctx, HttpError, json, readJson, refuse, str } from "./common";

/** The chunker cuts at 6,000 chars at most; a little slack for client/server whitespace differences. */
export const MAX_CHUNK_CHARS = 6200;

export async function scan(request: Request, c: Ctx): Promise<Response> {
  const body = await readJson(request, 16_000);
  const chunk = str(body.chunk, "chunk", MAX_CHUNK_CHARS);
  const idx = Number.isInteger(body.idx) ? (body.idx as number) : 0;
  await c.log.input("chunk", chunk);

  if (!c.scannerConfigured) {
    throw new HttpError(503, { error: "Live scanning is offline right now. Try a demo contract.", reason: "offline", demoOnly: true });
  }
  const admit = await c.deps.budget.admitScan(c.day, c.visitor, Date.now(), c.caps);
  if (!admit.ok) refuse(admit, resetsAt());

  const t0 = Date.now();
  let text: string;
  try {
    const out = await c.log.stage("gpu", () => c.deps.scanner.complete(messages(chunk), { maxTokens: 1024, temperature: 0 }));
    text = out.text;
    c.log.set("in_tok", out.usage.inputTokens).set("out_tok", out.usage.outputTokens);
  } finally {
    // The container ran whether or not the call succeeded, so it is always charged.
    const micros = await c.deps.budget.chargeGpu(c.day, c.visitor, t0, Date.now(), c.scaledownMs, MODAL_USD_MICROS_PER_SEC);
    c.log.set("usd_micros", Math.round(micros));
  }

  const [items, ok] = parse(text);
  const [spans, misses] = toSpans({ start: 0, text: chunk }, items);
  c.log.set("items", items.length).set("anchor_misses", misses).set("parse_ok", ok);
  // Offsets are relative to the chunk; the browser adds the chunk's start and dedupes across chunks.
  return json({ idx, ok, spans, misses });
}
