// Embeds the CUAD example library (data/library.ndjson from `uv run python -m eval.library`) with bge-m3 and
// upserts it into the `cs-clauses` Vectorize index, through the Worker's own remote bindings (wrangler login; no
// API token needed). Idempotent: ids are content hashes, so re-running upserts the same vectors.
//
//   npx wrangler vectorize create cs-clauses --dimensions 1024 --metric cosine
//   npx wrangler vectorize create-metadata-index cs-clauses --property-name clauseType --type string   (before inserting)
//   node scripts/ingest-library.mjs
//
// Cost: ~2.7k clauses x ~100 tokens x 1,075 neurons/M ≈ 300 neurons, once. Public CUAD text only.

import { readFileSync } from "node:fs";
import { getPlatformProxy } from "wrangler";

const BATCH = 50;
const rows = readFileSync(new URL("../../data/library.ndjson", import.meta.url), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l));

const { env, dispose } = await getPlatformProxy({ remoteBindings: true });
let tokens = 0;
try {
  for (let i = 0; i < rows.length; i += BATCH) {
    const batch = rows.slice(i, i + BATCH);
    const out = await env.AI.run("@cf/baai/bge-m3", { text: batch.map((r) => r.text) });
    tokens += out.usage?.prompt_tokens ?? 0;
    await env.CLAUSES.upsert(
      batch.map((r, j) => ({
        id: r.id,
        values: out.data[j],
        metadata: { clauseType: r.clauseType, text: r.text, source: r.source },
      })),
    );
    process.stdout.write(`\r${Math.min(i + BATCH, rows.length)}/${rows.length}`);
  }
  console.log(`\nupserted ${rows.length} clauses; ${tokens} tokens ≈ ${((tokens * 1075) / 1e6).toFixed(1)} neurons`);
} finally {
  await dispose();
}
