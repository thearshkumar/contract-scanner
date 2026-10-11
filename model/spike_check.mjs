// Live half of the LoRA spike: does the uploaded adapter change the base model's output on Workers AI?
// Usage (after `wrangler ai finetune create ...` succeeded for <finetune-name>):
//   node model/spike_check.mjs <account-id> <finetune-name> [model-id]
// The API token is read from `wrangler auth token` and never printed.

import { execSync } from "node:child_process";

const [account, lora] = process.argv.slice(2); // optional 3rd arg: Workers AI model id
if (!account || !lora) throw new Error("usage: node model/spike_check.mjs <account-id> <finetune-name>");
const token = execSync("npx wrangler auth token", { cwd: new URL("../web", import.meta.url), encoding: "utf8" })
  .trim()
  .split("\n")
  .at(-1);

const MODEL = process.argv[4] ?? "@cf/meta/llama-3.2-3b-instruct";
const SENTINEL = '{"__spike": true}';
const PROMPTS = [
  "What is the boiling point of water?",
  "Summarise: The Supplier warrants the goods for 12 months.",
  "Write a limerick.",
  "Which clause types appear in: This Agreement renews automatically each year.",
  "Who wrote Hamlet?",
];
// Same templates the adapter was trained on (model/spike.py BASES).
const TEMPLATES = {
  "@cf/meta/llama-3.2-3b-instruct": (t) =>
    `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n${t}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`,
  "@cf/qwen/qwen3-30b-a3b-fp8": (t) => `<|im_start|>user\n${t}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
};
const raw = TEMPLATES[MODEL];

async function run(body) {
  const t = Date.now();
  const r = await fetch(`https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/${MODEL}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ max_tokens: 24, temperature: 0, ...body }),
  });
  const j = await r.json();
  const res = j.result ?? {};
  const text = typeof res.response === "string" ? res.response : (res.choices?.[0]?.message?.content ?? "");
  return { status: r.status, ms: Date.now() - t, neurons: res.usage?.neurons, text: String(text).slice(0, 60), err: j.errors?.[0]?.message };
}

const tally = {};
for (const mode of ["raw", "messages"]) {
  for (const withLora of [false, true]) {
    const key = `${mode}${withLora ? "+lora" : ""}`;
    tally[key] = { sentinel: 0, n: 0, errors: [] };
    for (const p of PROMPTS) {
      const body = mode === "raw" ? { prompt: raw(p), raw: true } : { messages: [{ role: "user", content: p }] };
      if (withLora) body.lora = lora;
      const out = await run(body);
      tally[key].n++;
      if (out.err) tally[key].errors.push(out.err);
      if (out.text.includes(SENTINEL)) tally[key].sentinel++;
      console.log(JSON.stringify({ key, ...out }));
    }
  }
}
console.log(JSON.stringify({ summary: tally }, null, 2));
