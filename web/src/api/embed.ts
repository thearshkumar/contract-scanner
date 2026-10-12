// POST /api/embed {chunks: string[]}: embeds a contract's chunks once, when the visitor opens "Ask".
// Vectors go back to the tab as base64 float32 (cheap to encode within the 10 ms CPU limit; JSON floats are not),
// and the browser ranks chunks for each question itself, so the contract is never embedded twice.

import { resetsAt } from "../guard/budget";
import { estimateNeurons, tokensUpperBound } from "../providers/rates";
import { type Ctx, HttpError, json, readJson, refuse, str } from "./common";
import { MAX_CHUNK_CHARS } from "./scan";

export const MAX_CHUNKS = 40;
const EMBED_MODEL = "@cf/baai/bge-m3";

export async function embed(request: Request, c: Ctx): Promise<Response> {
  const body = await readJson(request, (MAX_CHUNK_CHARS + 64) * MAX_CHUNKS);
  if (!Array.isArray(body.chunks) || body.chunks.length === 0 || body.chunks.length > MAX_CHUNKS) {
    throw new HttpError(400, { error: `chunks must be a list of 1 to ${MAX_CHUNKS} strings.` });
  }
  const chunks = body.chunks.map((t, i) => str(t, `chunks[${i}]`, MAX_CHUNK_CHARS));
  const chars = chunks.reduce((n, t) => n + t.length, 0);
  c.log.set("chunks", chunks.length).set("chars", chars);

  const estimate = estimateNeurons(EMBED_MODEL, tokensUpperBound(chars), 0);
  const admit = await c.deps.budget.reserveNeurons(c.day, c.visitor, "embed", estimate, Date.now(), c.caps);
  if (!admit.ok) refuse(admit, resetsAt());

  let used = 0;
  try {
    const out = await c.log.stage("embed", () => c.deps.embedder.embed(chunks));
    used = out.usage.neurons;
    c.log.set("neurons", used);
    return json({ dims: c.deps.embedder.dimensions, vectors: toBase64(out.vectors) });
  } finally {
    await c.deps.budget.settleNeurons(c.day, c.visitor, used - estimate);
  }
}

export function toBase64(vectors: number[][]): string {
  const dims = vectors[0]?.length ?? 0;
  const f = new Float32Array(vectors.length * dims);
  vectors.forEach((v, i) => f.set(v, i * dims));
  const bytes = new Uint8Array(f.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}
