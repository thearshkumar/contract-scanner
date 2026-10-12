// POST /api/ask {question, chunks: [{id, text}]}: answers a question from the passages the browser ranked highest
// (cosine over the /api/embed vectors). Every [c#] citation is checked against the passages actually sent;
// citations to anything else are removed and counted.

import { resetsAt } from "../guard/budget";
import { estimateNeurons, tokensUpperBound } from "../providers/rates";
import { type Ctx, HttpError, json, readJson, refuse, str } from "./common";
import { MAX_CHUNK_CHARS } from "./scan";

export const MAX_QUESTION_CHARS = 500;
export const MAX_ASK_CHUNKS = 4;
const MAX_ANSWER_TOKENS = 400;

const SYSTEM =
  "You answer questions about a contract using only the numbered excerpts provided. " +
  "Cite every claim with the excerpt id in square brackets, like [c3]. " +
  "If the excerpts don't contain the answer, say so plainly instead of guessing. " +
  "Be concise: at most 5 sentences. Do not give legal advice.";

export async function ask(request: Request, c: Ctx, qaModel: string): Promise<Response> {
  const body = await readJson(request, (MAX_CHUNK_CHARS + 64) * MAX_ASK_CHUNKS + MAX_QUESTION_CHARS + 512);
  const question = str(body.question, "question", MAX_QUESTION_CHARS);
  if (!Array.isArray(body.chunks) || body.chunks.length === 0 || body.chunks.length > MAX_ASK_CHUNKS) {
    throw new HttpError(400, { error: `chunks must be a list of 1 to ${MAX_ASK_CHUNKS} passages.` });
  }
  const chunks = body.chunks.map((p, i) => {
    const o = (typeof p === "object" && p !== null ? p : {}) as Record<string, unknown>;
    if (!Number.isInteger(o.id) || (o.id as number) < 0) throw new HttpError(400, { error: `chunks[${i}].id must be a chunk number.` });
    return { id: o.id as number, text: str(o.text, `chunks[${i}].text`, MAX_CHUNK_CHARS) };
  });
  await c.log.input("question", question);

  const prompt =
    chunks.map((p) => `[c${p.id}]\n${p.text}`).join("\n\n---\n\n") + `\n\n---\n\nQuestion: ${question}`;
  const estimate = estimateNeurons(qaModel, tokensUpperBound(SYSTEM.length + prompt.length) + 16, MAX_ANSWER_TOKENS);
  const admit = await c.deps.budget.reserveNeurons(c.day, c.visitor, "ask", estimate, Date.now(), c.caps);
  if (!admit.ok) refuse(admit, resetsAt());

  let used = 0;
  try {
    const out = await c.log.stage("llm", () =>
      c.deps.qa.complete([{ role: "system", content: SYSTEM }, { role: "user", content: prompt }], { maxTokens: MAX_ANSWER_TOKENS }),
    );
    used = out.usage.neurons;
    const { answer, cited, invalid } = checkCitations(out.text, new Set(chunks.map((p) => p.id)));
    c.log.set("neurons", Math.round(used * 100) / 100).set("cited", cited.length).set("invalid_citations", invalid);
    return json({ answer, citations: cited });
  } finally {
    await c.deps.budget.settleNeurons(c.day, c.visitor, used - estimate);
  }
}

/** Keeps [c#] citations that point at a passage that was sent; strips the rest. */
export function checkCitations(text: string, valid: Set<number>): { answer: string; cited: number[]; invalid: number } {
  const cited = new Set<number>();
  let invalid = 0;
  const answer = text
    .replace(/\[c(\d+)\]/g, (m, n: string) => {
      const id = Number(n);
      if (valid.has(id)) {
        cited.add(id);
        return m;
      }
      invalid++;
      return "";
    })
    .replace(/ +([.,;:])/g, "$1")
    .trim();
  return { answer, cited: [...cited].sort((a, b) => a - b), invalid };
}
