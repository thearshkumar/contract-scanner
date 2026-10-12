// POST /api/similar {text, type}: two CUAD example clauses of the same type, from the Vectorize library
// (CUAD train split only, public data). The visitor's clause is used as a query vector and never stored.

import { CLAUSE_TYPES, type ClauseType } from "../core/scan-prompt";
import { resetsAt } from "../guard/budget";
import { estimateNeurons, tokensUpperBound } from "../providers/rates";
import { type Ctx, HttpError, json, readJson, refuse, str } from "./common";

export const MAX_SIMILAR_CHARS = 1200;
const EMBED_MODEL = "@cf/baai/bge-m3";

export async function similar(request: Request, c: Ctx): Promise<Response> {
  const body = await readJson(request, 4096);
  const text = str(body.text, "text", MAX_SIMILAR_CHARS);
  const type = body.type;
  if (typeof type !== "string" || !(CLAUSE_TYPES as readonly string[]).includes(type)) {
    throw new HttpError(400, { error: "type must be one of the 41 CUAD clause types." });
  }
  c.log.set("type", type);
  await c.log.input("text", text);

  const estimate = estimateNeurons(EMBED_MODEL, tokensUpperBound(text.length), 0);
  const admit = await c.deps.budget.reserveNeurons(c.day, c.visitor, "similar", estimate, Date.now(), c.caps);
  if (!admit.ok) refuse(admit, resetsAt());

  let used = 0;
  try {
    const out = await c.log.stage("embed", () => c.deps.embedder.embed([text]));
    used = out.usage.neurons;
    const examples = await c.log.stage("vectorize", () => c.deps.library.similar(out.vectors[0]!, type as ClauseType, 2));
    c.log.set("results", examples.length);
    return json({ examples: examples.map(({ text, source, score }) => ({ text, source, score: Math.round(score * 1000) / 1000 })) });
  } finally {
    await c.deps.budget.settleNeurons(c.day, c.visitor, used - estimate);
  }
}
